
import torch
import torch.nn.functional as F

def transfer_gate_entropy(gate_values):
    """
    Calculates entropy of the transfer gate.
    High entropy means the model is uncertain or mixing domains equally.
    Low entropy means it's relying heavily on one domain.
    
    Args:
        gate_values: [batch_size, 1] values in [0, 1] (sigmoid output).
                     Representing P(domain=Source).
    
    Returns:
        Scalar average entropy.
    """
    # p is prob of source, 1-p is prob of target
    # Entropy H(p) = -p log p - (1-p) log (1-p)
    
    # Clip for stability
    p = torch.clamp(gate_values, min=1e-6, max=1-1e-6)
    
    entropy = -p * torch.log(p) - (1-p) * torch.log(1-p)
    return entropy.mean()

def exposure_bias_metric(propensity_scores):
    """
    Tracks statistics of exposure propensity to monitor bias.
    
    Args:
        propensity_scores: [batch_size, 1] values in [0, 1].
    
    Returns:
        Dict with mean, std, and min/max.
    """
    return {
        "mean_exposure": propensity_scores.mean().item(),
        "std_exposure": propensity_scores.std().item(),
        "min_exposure": propensity_scores.min().item(),
        "max_exposure": propensity_scores.max().item()
    }
