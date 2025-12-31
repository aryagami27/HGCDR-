
import torch
from torch.utils.data import Dataset
import numpy as np
import random

def build_exposure_dataset(interactions_df, item_pool, neg_ratio=4):
    """
    Constructs a dataset for training the Exposure Model using explicit assumptions.
    
    Assumption 1 (Positive): Any interaction (rating/click) implies EXPOSURE (E=1).
    Assumption 2 (Negative): A random item not interacted with is likely UNEXPOSED (E=0).
    
    Args:
        interactions_df: DataFrame with ['user_id', 'item_id']
        item_pool: List/Array of all available item IDs.
        neg_ratio: Number of negative (unexposed) samples per positive interaction.
        
    Returns:
        exposure_data: List of (user_id, item_id, label)
    """
    user_interacted = interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()
    all_items = set(item_pool)
    
    data = []
    
    for user_id, items in user_interacted.items():
        # Positive Samples (E=1)
        for item_id in items:
            data.append((user_id, item_id, 1.0))
            
        # Negative Samples (E=0)
        # Sample 'neg_ratio' unexposed items per positive
        num_neg = len(items) * neg_ratio
        
        # Optimized Sampling
        # Instead of set difference which is slow, just sample and check
        # For very small pools, this might retry a lot, but for RecSys usually fine.
        count = 0
        while count < num_neg:
            neg_item = random.choice(item_pool)
            if neg_item not in items:
                data.append((user_id, neg_item, 0.0))
                count += 1
                
    return data

class ExposureDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        user, item, label = self.data[idx]
        return {
            'user_id': torch.tensor(user, dtype=torch.long),
            'item_id': torch.tensor(item, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }
