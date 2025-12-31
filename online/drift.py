
import torch
import torch.nn.functional as F

def embedding_drift(old_emb, new_emb, threshold=0.1):
    """
    Detects significant drift in user embeddings using KL Divergence.
    
    Args:
        old_emb: [batch_size, dim] Logits/Embeddings from previous time step.
        new_emb: [batch_size, dim] Current Logits/Embeddings.
        threshold: KL divergence threshold to trigger drift.
        
    Returns:
        Boolean tensor [batch_size] indicating if drift occurred.
    """
    # We treat the embeddings as logits of a distribution over latent features.
    # KL Divergence isn't symmetric. standard practice is KL(P || Q) aka KL(old || new)
    # measuring how much information is lost when approximating old with new, or vice-versa.
    # Here we check if the new embedding has diverged FROM the old (reference).
    
    # log_softmax expects logits
    log_probs_new = F.log_softmax(new_emb, dim=-1)
    probs_old = F.softmax(old_emb, dim=-1)
    
    # KL Div: sum(p(x) * (log p(x) - log q(x))) = sum(p(x) * log(p(x)/q(x)))
    # PyTorch F.kl_div expects (log_target, input) order if reduction is batchmean, 
    # BUT actually signature is input (log_probs), target (probs).
    # "The input given is expected to contain log-probabilities and the target is given the probabilities."
    
    # So we pass (new, old) to measure how 'new' differs from 'old' distribution.
    # Note: reduction='none' allows us to get per-sample KL div (after summing over dim).
    
    kl = F.kl_div(
        log_probs_new,
        probs_old,
        reduction='none'
    ).sum(dim=-1) # Sum over feature dimension
    
    return kl > threshold
