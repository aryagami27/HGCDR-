import torch
import sys
import os

path = 'models/saved/best_model.pth'
if not os.path.exists(path):
    print(f"Checkpoint not found at {path}")
    sys.exit(1)

try:
    state = torch.load(path, map_location='cpu')
    print(f"Checkpoint loaded. Keys: {len(state.keys())}")
    
    disentangle_keys = [k for k in state.keys() if 'disentangle' in k]
    print(f"Disentangle keys found: {len(disentangle_keys)}")
    if disentangle_keys:
        print(f"Sample: {disentangle_keys[0]}")
    
    hgt_p_rel = state.get('hgt.layers.0.p_rel.user_src__rates__item_src')
    if hgt_p_rel is not None:
        print(f"HGT p_rel shape: {hgt_p_rel.shape}") # Expect [1, 8] if 8 heads
    
    hgt_k_rel = state.get('hgt.layers.0.k_rel.weight')
    if hgt_k_rel is not None:
        print(f"HGT k_rel shape: {hgt_k_rel.shape}")

except Exception as e:
    print(f"Error loading: {e}")
