import torch
import torch.nn as nn
import torch.nn.functional as F

class LossComputer:
    def __init__(self, config):
        self.config = config
        self.lambda_orth = config['lambda_orth']
        self.lambda_contrast = config['lambda_contrast']
        self.lambda_causal = config['lambda_causal']

    def safe_normalize(self, x, eps=1e-8):
        """
        Normalize vector safely, handling zero vectors.
        """
        norm = x.norm(dim=-1, keepdim=True)
        # Avoid division by zero
        return x / (norm + eps)

    def calc_orthogonality(self, z_inv, z_spec):
        """
        Minimize cosine similarity between invariant and specific features.
        """
        # Safe Normalize
        z_inv_norm = self.safe_normalize(z_inv)
        z_spec_norm = self.safe_normalize(z_spec)
        
        # Cosine similarity
        cosine = (z_inv_norm * z_spec_norm).sum(dim=-1)
        
        # Minimize squared cosine
        loss = torch.mean(cosine ** 2)
        return loss

    def calc_contrastive(self, z_inv_src, z_inv_tgt, overlap_flag):
        """
        InfoNCE loss to align z_inv_src and z_inv_tgt for the SAME user.
        Only for overlapping users.
        """
        mask = overlap_flag.bool()
        if not mask.any():
            return z_inv_src.sum() * 0.0 # Connect to graph
            
        z_src = z_inv_src[mask]
        z_tgt = z_inv_tgt[mask]
        
        # Safe Normalize
        z_src = self.safe_normalize(z_src)
        z_tgt = self.safe_normalize(z_tgt)
        
        # Similarity matrix
        logits = torch.matmul(z_src, z_tgt.T) # (B_overlap, B_overlap)
        
        # Temperature
        temperature = 0.1
        logits /= temperature
        
        # Labels are diagonal (i-th src corresponds to i-th tgt in this filtered batch)
        labels = torch.arange(z_src.size(0), device=z_src.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

    def calc_causal_bpr(self, pos_scores, neg_scores, propensity=None, confidence=None, mask=None):
        """
        Weighted BPR Loss.
        loss = -log(sigmoid(pos - neg)) * weight
        """
        # Numerical Stability: Clamp scores
        diff = pos_scores - neg_scores
        diff = torch.clamp(diff, min=-10.0, max=10.0)
        loss = -F.logsigmoid(diff)
        
        # Apply IPW
        if self.config.get('enable_causal', True) and propensity is not None:
            # Inverse Propensity Weighting
            # Clamp propensity to be positive to avoid division by zero or negative weights
            propensity = torch.clamp(propensity, min=0.0)
            weight = 1.0 / (propensity + 1e-6)
            
            # Clip weights to prevent instability from rare items
            weight = torch.clamp(weight, max=10.0)
            
            loss = loss * weight
        
        # Apply Confidence Weighting (Lasso)
        if confidence is not None:
             loss = loss * confidence
             
        # Apply Curriculum Mask
        if mask is not None:
             loss = loss * mask
             # Normalize by active samples?
             # If mask is 0/1, we just sum and divide by (sum(mask) + epsilon)
             # But mean() divides by batch size.
             # So effectively we treat masked samples as 0 loss.
             # This reduces gradient magnitude.
             # To keep scale, we should divide by mask.sum().
             # But simplistic 'mean' is standard for curriculum (drop samples).
             
        return loss.mean()

    def compute_total_loss(self, outputs, epoch=None):
        pos_scores = outputs['pos_scores']
        neg_scores = outputs['neg_scores']
        z_inv_src = outputs['z_inv_src']
        z_spec_src = outputs['z_spec_src']
        z_inv_tgt = outputs['z_inv_tgt']
        z_spec_tgt = outputs['z_spec_tgt']
        overlap_flag = outputs['overlap_flag']
        propensity = outputs.get('propensity', None)
        confidence = outputs.get('confidence', None)
        
        # Curriculum Logic (Lasso-Specific)
        mask = None
        if confidence is not None and not self.config.get('disable_curriculum', False):
             # Import here or rely on passed function?
             # Simplest: use thresholding here
             # Default Schedule: Warmup 5, Threshold 0.8 -> 0.5
             threshold = 0.5
             if epoch is not None and epoch < 5:
                  threshold = 0.8
             
             mask = (confidence > threshold).float()
        
        # 1. Recommendation Loss (BPR)
        rec_loss = self.calc_causal_bpr(pos_scores, neg_scores, propensity, confidence, mask)
        
        # 2. Orthogonality Loss (Source & Target)
        orth_loss_src = self.calc_orthogonality(z_inv_src, z_spec_src)
        orth_loss_tgt = self.calc_orthogonality(z_inv_tgt, z_spec_tgt)
        orth_loss = (orth_loss_src + orth_loss_tgt) / 2
        
        if not self.config.get('enable_contrast', True):
           contrast_loss = z_inv_src.sum() * 0.0
        else:
           contrast_loss = self.calc_contrastive(z_inv_src, z_inv_tgt, overlap_flag)
        
        # Check for NaNs and fallback
        # Use sum()*0.0 pattern to preserve gradient graph even if value is 0
        if torch.isnan(rec_loss):
            rec_loss = pos_scores.sum() * 0.0
        if torch.isnan(orth_loss):
            orth_loss = z_inv_src.sum() * 0.0
        if torch.isnan(contrast_loss):
            contrast_loss = z_inv_src.sum() * 0.0

        # Total Loss
        total_loss = (
            rec_loss + 
            self.lambda_orth * orth_loss + 
            self.lambda_contrast * contrast_loss
        )
        
        return {
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'orth_loss': orth_loss,
            'contrast_loss': contrast_loss
        }
