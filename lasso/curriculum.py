
import numpy as np

def curriculum_schedule(confidence, epoch, warmup_epochs=5, initial_threshold=0.8, final_threshold=0.5):
    """
    Determines which samples to keep based on a curriculum schedule.
    
    Args:
        confidence: Numpy array or Tensor of confidence scores.
        epoch: Current training epoch (0-indexed).
        warmup_epochs: Number of epochs to keep strict threshold.
        initial_threshold: Threshold during warmup.
        final_threshold: Threshold after warmup.
        
    Returns:
        Boolean mask of samples to keep.
    """
    # Determine the threshold
    if epoch < warmup_epochs:
        threshold = initial_threshold
    else:
        # Step function for simplicity as per requirements, 
        # but could be linear decay: threshold = max(final_threshold, initial_threshold - decay * (epoch - warmup))
        threshold = final_threshold
        
    # Return boolean mask
    return confidence > threshold
