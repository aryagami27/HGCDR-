import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MetaDataset(Dataset):
    def __init__(self, df, min_interactions=5, support_ratio=0.5, neg_sample_num=1, item_pool=None):
        """
        Meta-Learning Dataset that yields Tasks (User Support/Query Sets).
        
        Args:
            df (pd.DataFrame): Dataframe containing 'user_id', 'item_id'.
            min_interactions (int): Minimum interactions required to be a valid task.
            support_ratio (float): Ratio of interactions to use for support set.
            neg_sample_num (int): Number of negative samples per positive interaction.
            item_pool (list): List of all available item IDs for negative sampling.
        """
        self.neg_sample_num = neg_sample_num
        self.support_ratio = support_ratio
        
        # Group by user
        self.user_groups = df.groupby('user_id')
        self.valid_users = [u for u, group in self.user_groups if len(group) >= min_interactions]
        
        if item_pool is None:
            self.item_pool = df['item_id'].unique().tolist()
        else:
            self.item_pool = item_pool
            
    def __len__(self):
        return len(self.valid_users)
    
    def __getitem__(self, idx):
        user_id = self.valid_users[idx]
        user_data = self.user_groups.get_group(user_id)
        
        # Split into support and query
        n_interactions = len(user_data)
        n_support = max(1, int(n_interactions * self.support_ratio))
        
        # Shuffle
        user_data = user_data.sample(frac=1).reset_index(drop=True)
        
        support_data = user_data.iloc[:n_support]
        query_data = user_data.iloc[n_support:]
        
        # If query is empty (rare due to min_interactions logic but possible), use support as query or error?
        # Standard approach: ensure at least 1 query.
        if len(query_data) == 0:
            # Steal one from support if possible
            if len(support_data) > 1:
                query_data = support_data.iloc[-1:]
                support_data = support_data.iloc[:-1]
            else:
                # Fallback: duplicate
                query_data = support_data
        
        return {
            'support': self._process_set(support_data, user_id),
            'query': self._process_set(query_data, user_id)
        }
        
    def _process_set(self, data, user_id):
        """
        Converts a dataframe subset into tensor format with negative samples.
        """
        # Positives
        pos_items = data['item_id'].values
        
        # Negatives
        neg_items = []
        for _ in pos_items:
            neg_samples = []
            while len(neg_samples) < self.neg_sample_num:
                neg = random.choice(self.item_pool)
                if neg not in pos_items: # Simple collision check against POSITIVES of this user
                     neg_samples.append(neg)
            neg_items.append(neg_samples)
        
        return {
            'user_id': torch.tensor([user_id] * len(pos_items), dtype=torch.long),
            'item_id': torch.tensor(pos_items, dtype=torch.long),
            'neg_item_id': torch.tensor(neg_items, dtype=torch.long)
        }

def meta_collate_fn(batch):
    # Batch is a list of task dicts: [{'support': ..., 'query': ...}, ...]
    # We want to collate them effectively. 
    # Usually in meta-learning, we process tasks sequentially or batched.
    # HGCDR++ Meta-learning (Reptile) usually takes ONE task (or a batch of tasks) and updates.
    # Let's keep the list structure so the Trainer can iterate over tasks.
    return batch
