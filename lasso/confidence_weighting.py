
import torch

def weighted_loss(loss, confidence):
    """
    Applies confidence weighting to the loss.
    
    Args:
        loss: [batch_size, ...] Tensor containing per-sample loss values.
        confidence: [batch_size, ...] Tensor containing confidence scores (0 to 1) for each sample.
        
    Returns:
        Scalar mean weighted loss.
    """
    # Ensure confidence is structurally compatible (e.g., same device, broadcastable)
    if confidence.device != loss.device:
        confidence = confidence.to(loss.device)
        
    # We can perform a simple element-wise multiplication.
    # High confidence -> High weight (Trust this sample)
    # Low confidence -> Low weight (Don't trust this sample much)
    weighted = loss * confidence
    
    return weighted.mean()
