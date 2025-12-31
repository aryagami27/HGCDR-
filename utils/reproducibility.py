
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic algos
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        # MPS doesn't strictly support manual_seed same way, but setting torch seed covers it mostly
        pass
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔒 Reproducibility: Seed set to {seed}")
