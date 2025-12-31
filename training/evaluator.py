import torch
import numpy as np
import math

class Evaluator:
    def __init__(self, model, device='cpu', k_list=[10, 20]):
        """
        Args:
            model: The trained HGCDRPlus model.
            device: Device to run evaluation on.
            k_list: List of K values for HR@K and NDCG@K.
        """
        self.model = model
        self.device = device
        self.k_list = k_list
        
    def evaluate(self, dataloader, cold_start_users=None, cold_item_ids=None):
        self.model.eval()
        
        hr = {k: 0.0 for k in self.k_list}
        ndcg = {k: 0.0 for k in self.k_list}
        num_users = 0
        
        # Cold User Metrics
        hr_cold = {k: 0.0 for k in self.k_list}
        ndcg_cold = {k: 0.0 for k in self.k_list}
        num_cold = 0
        
        # Cold Item Metrics
        hr_item_cold = {k: 0.0 for k in self.k_list}
        ndcg_item_cold = {k: 0.0 for k in self.k_list}
        num_item_cold = 0
        
        cold_start_set = set(cold_start_users) if cold_start_users is not None else set()
        cold_item_set = set(cold_item_ids) if cold_item_ids is not None else set()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = self._to_device(batch)
                
                # Extract User IDs for Cold-Start Check
                try:
                    uids = batch['target_input']['user_id']
                    if hasattr(uids, 'tolist'):
                        uids = uids.tolist()
                except:
                    uids = [] 
                
                # Extract Ground Truth Item IDs for Item-Cold-Start Check (Target Items)
                # 'pos_item_id' is usually in 'target_input'
                try:
                    iids = batch['target_input']['item_id']
                    if hasattr(iids, 'tolist'):
                        iids = iids.tolist()
                except:
                    iids = []
                
                outputs = self.model(batch)
                
                pos_scores = outputs['pos_scores'] # [Batch, 1]
                neg_scores = outputs['neg_scores'] # [Batch, Num_Negatives]
                
                all_scores = torch.cat([pos_scores, neg_scores], dim=1)
                _, indices = torch.sort(all_scores, dim=1, descending=True)
                hits = (indices == 0)
                rank = hits.nonzero(as_tuple=True)[1] 
                
                batch_size = all_scores.shape[0]
                num_users += batch_size
                
                # Check for Cold Start
                is_cold_user = [uid in cold_start_set for uid in uids]
                num_cold += sum(is_cold_user)
                
                # Check for Cold Item
                is_cold_item = [iid in cold_item_set for iid in iids]
                num_item_cold += sum(is_cold_item)
                
                # Compute Metrics
                for i in range(batch_size):
                    r = rank[i].item()
                    c_user = is_cold_user[i] if i < len(is_cold_user) else False
                    c_item = is_cold_item[i] if i < len(is_cold_item) else False
                    
                    for k in self.k_list:
                        # HR
                        if r < k:
                            hr[k] += 1
                            if c_user: hr_cold[k] += 1
                            if c_item: hr_item_cold[k] += 1
                        
                        # NDCG
                        if r < k:
                            val = 1.0 / math.log2(r + 2.0)
                            ndcg[k] += val
                            if c_user: ndcg_cold[k] += val
                            if c_item: ndcg_item_cold[k] += val
                            
        # Average
        metrics = {}
        for k in self.k_list:
            metrics[f'HR@{k}'] = hr[k] / num_users if num_users > 0 else 0.0
            metrics[f'NDCG@{k}'] = ndcg[k] / num_users if num_users > 0 else 0.0
            
            metrics[f'Cold_HR@{k}'] = hr_cold[k] / num_cold if num_cold > 0 else 0.0
            metrics[f'Cold_NDCG@{k}'] = ndcg_cold[k] / num_cold if num_cold > 0 else 0.0
            
            metrics[f'ItemCold_HR@{k}'] = hr_item_cold[k] / num_item_cold if num_item_cold > 0 else 0.0
            metrics[f'ItemCold_NDCG@{k}'] = ndcg_item_cold[k] / num_item_cold if num_item_cold > 0 else 0.0
            
        metrics['Num_Users'] = num_users
        metrics['Num_Cold_Users'] = num_cold
        metrics['Num_Cold_Items'] = num_item_cold
            
        return metrics

    def _to_device(self, batch):
        # Helper to move batch to device (same as in Trainer)
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._to_device(v) for v in batch]
        elif hasattr(batch, 'to'): # For HeteroData
             return batch.to(self.device)
        return batch
