import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class CrossDomainDataset(Dataset):
    def __init__(self, source_df, target_df, user_mapping, tokenizer=None, item_mapping_src=None, item_mapping_tgt=None, max_length=128, neg_sample_num=1, driver='source'):
        """
        Args:
            source_df (pd.DataFrame): Dataframe for source domain interactions.
            target_df (pd.DataFrame): Dataframe for target domain interactions.
            user_mapping (dict): Mapping from source user IDs to target user IDs.
            tokenizer: Tokenizer for text data (optional).
            item_mapping_src: Mapping for source items (optional).
            item_mapping_tgt: Mapping for target items (optional).
            max_length (int): Max length for text tokenization.
            neg_sample_num (int): Number of negative samples.
            driver (str): 'source' or 'target'. Determines which DF to iterate over.
        """
        self.source_df = source_df
        self.target_df = target_df
        self.user_mapping = user_mapping
        # Reverse mapping for target->source lookup
        self.user_mapping_rev = {v: k for k, v in user_mapping.items()}
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.neg_sample_num = neg_sample_num
        self.item_mapping_src = item_mapping_src
        self.item_mapping_tgt = item_mapping_tgt
        self.driver = driver

        # Create a set of all items in target domain for negative sampling
        self.target_item_ids = self.target_df['item_id'].unique().tolist()
        
        if self.driver == 'source':
            self.data = self.source_df.reset_index(drop=True)
        else:
            self.data = self.target_df.reset_index(drop=True)
        
        # Group target interactions by user for fast lookup (used when driver='source')
        self.target_user_groups = self.target_df.groupby('user_id')
        
        # Compute Item Popularity for Propensity
        self.item_pop = self.target_df['item_id'].value_counts().to_dict()
        self.total_target_interactions = len(self.target_df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.driver == 'source':
            source_user_id = row['user_id']
            source_item_id = row['item_id']
            # Look up target
            target_user_id = self.user_mapping.get(source_user_id, source_user_id) # Default to same if not mapped
            
            # Source Text/Image
            text_raw = row.get('text', "")
            image_feat = row.get('image_feat', torch.zeros(128))
            
            # Target Data (Validation/Overlap)
            overlap_flag = 0
            target_pos_item_id = 0
            target_neg_item_ids = [0] * self.neg_sample_num
            
            if target_user_id in self.target_user_groups.groups:
                overlap_flag = 1
                user_target_interactions = self.target_user_groups.get_group(target_user_id)
                target_row = user_target_interactions.sample(1).iloc[0]
                target_pos_item_id = target_row['item_id']
                
                sampled_negs = []
                while len(sampled_negs) < self.neg_sample_num:
                    neg_item = random.choice(self.target_item_ids)
                    if neg_item != target_pos_item_id:
                        sampled_negs.append(neg_item)
                target_neg_item_ids = sampled_negs
            
        else: # driver == 'target'
            target_user_id = row['user_id']
            target_pos_item_id = row['item_id']
            
            # Look up source
            source_user_id = self.user_mapping_rev.get(target_user_id, target_user_id)
            
            # Since we iterate target, we KNOW the target interaction exists (it's this row!)
            # But we need negatives for evaluation
            overlap_flag = 1 # We are in target domain
            # Wait, overlap_flag usually means "User exists in Both". 
            # If we are iterating target, we check if they exist in source.
            if target_user_id not in self.user_mapping_rev:
                 overlap_flag = 0 
                 
            # Negative Sampling
            sampled_negs = []
            while len(sampled_negs) < self.neg_sample_num:
                neg_item = random.choice(self.target_item_ids)
                if neg_item != target_pos_item_id:
                    sampled_negs.append(neg_item)
            target_neg_item_ids = sampled_negs
            
            # Source Input (Dummy/Placeholder since we drive by target)
            # We don't have a specific source interaction to pair with.
            # Just use 0s or look up random history?
            # Model uses 'src_user_node_idx' derived from 'source_user_id'.
            source_item_id = 0
            text_raw = ""
            image_feat = torch.zeros(128)

        # Tokenize (Common)
        if self.tokenizer:
            text_tokens = self.tokenizer(
                text_raw, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors='pt'
            )
            input_ids = text_tokens['input_ids'].squeeze(0)
            attention_mask = text_tokens['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            
        source_input = {
            'user_id': source_user_id,
            'item_id': source_item_id,
            'text_tokens': input_ids,
            'attention_mask': attention_mask,
            'image_feat': image_feat
        }

        target_input = {
            'user_id': target_user_id,
            'item_id': target_pos_item_id,
            'neg_item_id': target_neg_item_ids
        }

        # Propensity
        target_item_pop = self.item_pop.get(target_pos_item_id, 0)
        p_global = (target_item_pop + 1) / (self.total_target_interactions + 1)
        
        alpha = 1.0 if overlap_flag == 0 else 0.5
        propensity = alpha * p_global + (1 - alpha) * 0.5
        
        return {
            'source_input': source_input,
            'target_input': target_input,
            'overlap_flag': overlap_flag,
            'propensity': propensity
        }

def collate_fn(batch):
    # Custom collate to stack items and handle text padding if not done in __getitem__
    
    source_inputs = [b['source_input'] for b in batch]
    target_inputs = [b['target_input'] for b in batch]
    overlap_flags = [b['overlap_flag'] for b in batch]
    propensities = [b['propensity'] for b in batch]
    
    # Collate source
    collated_source = {
        'user_id': torch.tensor([x['user_id'] for x in source_inputs], dtype=torch.long),
        'item_id': torch.tensor([x['item_id'] for x in source_inputs], dtype=torch.long),
        'text_tokens': torch.stack([x['text_tokens'] for x in source_inputs]),
        'attention_mask': torch.stack([x['attention_mask'] for x in source_inputs]),
        # Handle image_feat which might be tensor or list
        'image_feat': torch.stack([torch.tensor(x['image_feat'], dtype=torch.float32) if not isinstance(x['image_feat'], torch.Tensor) else x['image_feat'].float() for x in source_inputs])
    }
    
    # Collate target
    # neg_item_id is [Batch, NegSampleNum] list of lists
    neg_items_stacked = torch.tensor([x['neg_item_id'] for x in target_inputs], dtype=torch.long)
    
    collated_target = {
        'user_id': torch.tensor([x['user_id'] for x in target_inputs], dtype=torch.long),
        'item_id': torch.tensor([x['item_id'] for x in target_inputs], dtype=torch.long),
        'neg_item_id': neg_items_stacked
    }
    
    return {
        'source_input': collated_source,
        'target_input': collated_target,
        'overlap_flag': torch.tensor(overlap_flags, dtype=torch.float32),
        'propensity': torch.tensor(propensities, dtype=torch.float32)
    }

def prepare_subgraph_batch(batch, global_data, device):
    """
    Samples a subgraph manually (K-hop) to avoid NeighborLoader dict issues.
    This functionality is critical for both Main Training and Verification.
    """
    # 1. Seeds
    src_users = batch['source_input']['user_id'].cpu()
    tgt_users = batch['target_input']['user_id'].cpu()
    tgt_pos_items = batch['target_input']['item_id'].cpu()
    tgt_neg_items = batch['target_input']['neg_item_id'].cpu().view(-1)
    
    # Current active nodes per type (Global IDs)
    active_nodes_initial = {
        'user_src': src_users,
        'user_tgt': tgt_users,
        'item_tgt': torch.cat([tgt_pos_items, tgt_neg_items])
    }
    
    # Make Robust: Filter out-of-bounds IDs
    # (Lasso/Synthetic data might yield IDs > num_nodes in graph)
    active_nodes = {}
    for k, v in active_nodes_initial.items():
        if hasattr(global_data, 'num_nodes_dict') and k in global_data.num_nodes_dict:
             limit = global_data.num_nodes_dict[k]
             mask = (v >= 0) & (v < limit)
             valid_v = v[mask]
             active_nodes[k] = torch.unique(valid_v)
        else:
             active_nodes[k] = torch.unique(v)
             
    # 2. Manual K-Hop Expansion (2 Hops for GNN)
    final_nodes = {k: v for k, v in active_nodes.items()}
    data_cpu = global_data.cpu()
    
    with torch.no_grad():
        for hop in range(2): 
            new_nodes = {}
            if hasattr(data_cpu, 'edge_index_dict'):
                for (src_type, rel, dst_type), edge_index in data_cpu.edge_index_dict.items():
                    if src_type in final_nodes:
                        seeds = final_nodes[src_type]
                        # Robust Mask
                        mask = torch.isin(edge_index[0], seeds)
                        neighbors = edge_index[1, mask]
                        if dst_type not in new_nodes: new_nodes[dst_type] = []
                        new_nodes[dst_type].append(neighbors)
            
            # Merge new nodes
            for ntype, tensors in new_nodes.items():
                if tensors:
                    merged = torch.cat(tensors)
                    merged = torch.unique(merged)
                    if ntype in final_nodes:
                         final_nodes[ntype] = torch.unique(torch.cat([final_nodes[ntype], merged]))
                    else:
                         final_nodes[ntype] = merged
    
    # 3. Create Subgraph (Safe)
    safe_final_nodes = {}
    for k, v in final_nodes.items():
        if hasattr(data_cpu, 'num_nodes_dict') and k in data_cpu.num_nodes_dict:
            limit = data_cpu.num_nodes_dict[k]
            safe_final_nodes[k] = v[v < limit]
        else:
            safe_final_nodes[k] = v
            
    subgraph = data_cpu.subgraph(safe_final_nodes)
    subgraph = subgraph.to(device)
    
    # 4. Map Global -> Local (for Model input)
    def map_ids(global_ids, type_name):
        # Maps global indices to local indices in 'subgraph' (which is ordered by safe_final_nodes)
        if type_name not in safe_final_nodes:
             return torch.zeros_like(global_ids)
        
        mapping_tensor = safe_final_nodes[type_name] # Sorted
        
        # Robust Mapping: 
        # If ID was filtered (e.g. 1337), searchsorted returns insertion index.
        # We must check if value actually matches.
        
        # Clamp input to avoid crash? No, input is arbitrary.
        # searchsorted works on any input.
        indices = torch.searchsorted(mapping_tensor, global_ids)
        
        # Check validity
        # Indices >= len are invalid
        # Indices < len but mapping_tensor[indices] != global_ids are invalid
        # Clone indices to clamp or mask
        
        is_valid = (indices < len(mapping_tensor))
        # Refine validity
        # We can't index mapping_tensor with invalid indices to check equality.
        valid_indices = indices[is_valid]
        valid_global_ids = global_ids[is_valid]
        
        # Check equality for potentially valid ones
        # mapping_tensor[valid_indices] == valid_global_ids
        # Wait, efficient check:
        # map back
        
        # Simple Logic:
        # Just clamp to 0 and rely on Model to handle 0 (Unknown) or Mask?
        # Model doesn't take mask.
        # We map invalid -> 0.
        
        indices[~is_valid] = 0
        # Check equality for in-bound indices
        matched = mapping_tensor[indices] == global_ids
        indices[~matched] = 0
        
        return indices
        
    batch['src_user_local_idx'] = map_ids(batch['source_input']['user_id'].cpu(), 'user_src').to(device)
    batch['tgt_user_local_idx'] = map_ids(batch['target_input']['user_id'].cpu(), 'user_tgt').to(device)
    batch['tgt_pos_item_local_idx'] = map_ids(batch['target_input']['item_id'].cpu(), 'item_tgt').to(device)
    
    neg_flat = batch['target_input']['neg_item_id'].cpu().view(-1)
    neg_local_flat = map_ids(neg_flat, 'item_tgt').to(device)
    batch['tgt_neg_item_local_idx'] = neg_local_flat.view(batch['target_input']['neg_item_id'].shape)
    
    # Store Subgraph
    batch['graph_data'] = subgraph
    
    # Add Node Indices (Global) just in case model wants them
    # batch['tgt_user_node_idx'] = ... (already in input)
    
    return batch
