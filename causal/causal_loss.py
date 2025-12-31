
import torch

def causal_bpr_loss(
    pos_score,
    neg_score,
    propensity,
    clip_max: float = 10.0,
    eps: float = 1e-6
):
    """
    Computes BPR loss with Inverse Propensity Weighting (IPW) for causal debiasing.
    
    Args:
        pos_score: Scores for positive items (batch_size, 1) or (batch_size,)
        neg_score: Scores for negative items (batch_size, 1) or (batch_size,)
        propensity: Propensity scores P(E=1|u, i) for the positive interactions.
                    Shape should match pos_score (batch_size, 1) or (batch_size,).
        clip_max: Maximum value for IPW weights to prevent variance explosion.
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar mean loss.
    """
    # Propensity Clipping (Critical for Stability)
    # Avoid extremely small propensities causing massive weights
    propensity = torch.clamp(propensity, min=0.01, max=0.95)
    
    # Inverse propensity weighting
    # We want to weight the loss by 1/P(E=1) to simulate a randomized experiment
    ipw = 1.0 / (propensity + eps)
    
    # Clipping IPW weights to further prevent variance explosion
    ipw = torch.clamp(ipw, max=clip_max)
    
    loss = -ipw.detach() * torch.nn.functional.logsigmoid(pos_score - neg_score)
    
    return loss.mean()
