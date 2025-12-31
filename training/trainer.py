import torch
import torch.optim as optim
from training.losses import LossComputer
# import wandb # Optional, but good practice

from data.dataloader import prepare_subgraph_batch

class Trainer:
    def __init__(self, model, config, device='cpu', exposure_model=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.exposure_model = exposure_model
        if self.exposure_model:
            self.exposure_model.to(device)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.loss_computer = LossComputer(config)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss_avg = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._to_device(batch)
            
            # [SCALABILITY] Sample Subgraph & Map Indices
            # We check if graph_data is present (provided by collate_wrapper)
            if 'graph_data' in batch:
                # Use robust subgraph sampling from dataloader
                try:
                    batch = prepare_subgraph_batch(batch, batch['graph_data'], self.device)
                except Exception as e:
                    # Fallback or Log?
                    # If this fails, model forward might crash or produce NaN.
                    # We log and skip to match robust behavior.
                    print(f"Skipping training batch {batch_idx} due to sampling error: {e}")
                    continue
            
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(batch)
            
            # [CAUSAL] Calculate Propensity if module exists
            if self.exposure_model and self.config.get('enable_causal', True):
                with torch.no_grad():
                    # Exposure model takes raw IDs (local or global?)
                    # ExposureModel (causal/exposure_model.py) usually uses its own Embeddings.
                    # It was trained on `target_user2id` mappings.
                    # The `batch['target_input']['user_id']` carries these IDs.
                    uids = batch['target_input']['user_id']
                    iids = batch['target_input']['item_id']
                    # Ensure they are on device
                    if self.config.get('disable_ipw', False):
                         # [ABLATION] Causal without IPW (Inverse Weight = 1.0)
                         # Set propensity to 1.0 so 1/p = 1
                         propensity = torch.ones_like(uids, dtype=torch.float)
                    else:
                         propensity = self.exposure_model(uids, iids)
                    
                    # [ABLATION] Randomize Propensity (Exposure Quality Check)
                    if self.config.get('randomize_propensity', False):
                         # Shuffle propensity scores across the batch
                         idx = torch.randperm(propensity.size(0))
                         propensity = propensity[idx]
                         
                    outputs['propensity'] = propensity
            
            # [LASSO] Confidence (if available)
            if 'confidence' in batch:
                outputs['confidence'] = batch['confidence']

            # Loss
            loss_dict = self.loss_computer.compute_total_loss(outputs, epoch=epoch_idx)
            loss = loss_dict['total_loss']
            
            # 2.3 Training Stability Checks
            # If loss is NaN, it might be due to bad data.
            # LossComputer tries to return 0.0 but if inputs are Inf, it fails.
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is NaN/Inf at epoch {epoch_idx}, batch {batch_idx}. Skipping update.")
                continue
            
            # Backward
            loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss_avg += loss.item()
            
            if batch_idx % 10 == 0:
                rec_val = loss_dict['rec_loss'].item() if isinstance(loss_dict['rec_loss'], torch.Tensor) else loss_dict['rec_loss']
                orth_val = loss_dict['orth_loss'].item() if isinstance(loss_dict['orth_loss'], torch.Tensor) else loss_dict['orth_loss']
                
                prop_stats = ""
                if 'propensity' in outputs:
                     p = outputs['propensity']
                     prop_stats = f" | Prop[Min:{p.min():.4f}, Mean:{p.mean():.4f}, Max:{p.max():.4f}]"

                print(f"Epoch {epoch_idx} | Batch {batch_idx} | Loss: {loss.item():.4f} | Rec: {rec_val:.4f} | Orth: {orth_val:.4f}{prop_stats}")
                
                # DIAGNOSTIC: Check for Oversmoothing (User Src vs Tgt alignment)
                # Helps verify GNN depth impact
                if 'z_src_spec' in outputs and 'z_tgt_spec' in outputs:
                    z_src = outputs['z_src_spec'].detach() # Detach to avoid graph retention
                    z_tgt = outputs['z_tgt_spec'].detach()
                    
                    # Compute Cosine Sim between centroids
                    src_mean = z_src.mean(dim=0)
                    tgt_mean = z_tgt.mean(dim=0)
                    cos_sim = torch.nn.functional.cosine_similarity(src_mean.unsqueeze(0), tgt_mean.unsqueeze(0)).item()
                    print(f"  [Diagnostic] Latent Alignment CosSim: {cos_sim:.4f}")
                
        # Handle empty loop
        if len(dataloader) > 0:
            return total_loss_avg / len(dataloader)
        return 0.0

    def meta_train_epoch(self, meta_dataloader, graph_data, inner_lr=0.01, inner_steps=1):
        """
        Meta-Training Epoch.
        meta_dataloader yields batches of tasks: (support_batch, query_batch)
        graph_data: The HeteroData object containing the graph structure.
        """
        from models.meta_learner import MetaWrapper
        
        # Wrap model
        meta_model = MetaWrapper(self.model, inner_lr, inner_steps, device=self.device)
        
        # Freeze Encoders (Text & HGT)
        self.model.train()
        
        # Freeze Encoders
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False
        for param in self.model.hgt.parameters():
            param.requires_grad = False
            
        total_loss_avg = 0
        
        for batch_idx, task_batch in enumerate(meta_dataloader):
            # unpack list of dicts
            if isinstance(task_batch, list):
                support_batch = [t['support'] for t in task_batch]
                query_batch = [t['query'] for t in task_batch]
            elif isinstance(task_batch, dict):
                 # Handle case where collate_fn stacked them into a single dict (not likely with our implementation)
                 support_batch = task_batch['support']
                 query_batch = task_batch['query']
            else:
                # Fallback
                support_batch, query_batch = task_batch
            
            # Move to device (recursively handles lists of dicts)
            support_batch = self._to_device(support_batch)
            query_batch = self._to_device(query_batch)
            
            self.optimizer.zero_grad()
            
            # Meta-Forward (Outer Loop Loss)
            # Pass graph_data too
            graph_data_device = graph_data.to(self.device) if hasattr(graph_data, 'to') else graph_data
            loss = meta_model(support_batch, query_batch, graph_data_device)
            
            # Backward (Outer Loop Update)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            total_loss_avg += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Meta-Epoch | Batch {batch_idx} | Query Loss: {loss.item():.4f}")
        
        # Unfreeze Encoders
        for param in self.model.text_encoder.parameters():
            param.requires_grad = True
        for param in self.model.hgt.parameters():
            param.requires_grad = True
        
        # Ensure model is on the correct device after meta-training
        self.model.to(self.device)
            
        if batch_idx + 1 > 0:
            return total_loss_avg / (batch_idx + 1)
        return 0.0

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._to_device(v) for v in batch]
        # Handle HeteroData or other objects
        if hasattr(batch, 'to'):
             return batch.to(self.device)
        return batch

    def evaluate(self, dataloader, k=10, cold_user_ids=None):
        """
        Evaluates the model on the given dataloader.
        Returns Top-K Metrics (HR, NDCG).
        If cold_user_ids is provided, also returns Cold-Start metrics.
        """
        self.model.eval()
        
        hr_sum = 0
        ndcg_sum = 0
        count = 0
        
        cold_hr_sum = 0
        cold_ndcg_sum = 0
        cold_count = 0
        
        import math
        
        with torch.no_grad():
            for batch in dataloader:
                # Need graph data? 
                # If using Scalability: 'prepare_subgraph_batch' should be called or assuming FULL graph in 'batch'?
                # For Evaluation, if we use Full Graph (Transductive), we might just pass `graph_data` externally?
                # But Trainer.evaluate assumes `batch` contains everything needed.
                # If `main.py` handles graph injection (via collate), we are good.
                
                batch = self._to_device(batch)
                
                # Check if we need to prepare subgraph (Validation/Test should also use subgraph if Training did?)
                # Or use Full Graph?
                # If batch has 'graph_data', use it. 
                # If 'prepare_subgraph_batch' logic is needed, it should be applied here too.
                # BUT Trainer shouldn't depend on 'main.py' function.
                # We assume batch is ready-to-run (collate handled it or caller modified it).
                # NOTE: If main.py has `prepare_subgraph_batch`, we might skip subgraphs here and crash if missing.
                # Ideally, dataloader collate does it.
                # Given current state, we assume 'graph_data' is present (Full or Sub).
                
                outputs = self.model(batch)
                
                pos_scores = outputs['pos_scores'] # (B, 1)
                neg_scores = outputs['neg_scores'] # (B, Neg)
                
                # Combine
                # Target is at index 0
                all_scores = torch.cat([pos_scores, neg_scores], dim=1) # (B, 1+Neg)
                
                # Rank
                # We want to see if Index 0 is in Top K
                _, indices = torch.topk(all_scores, k, dim=1)
                
                # HR
                hits = (indices == 0).sum(dim=1) # (B,) 1 if hit, 0 if not
                
                # NDCG
                # If hit, what rank?
                # Using torch.where usually
                # Simple implementation:
                # Rank of index 0?
                # sort -> find index 0
                # But topk is faster.
                # indices contains the INDICES of top k items.
                # if 0 is in indices[i], ndcg = 1 / log2(rank + 2)
                
                ndcgs = torch.zeros_like(hits, dtype=torch.float)
                for i in range(len(hits)):
                    if hits[i] > 0:
                        # Find rank
                        rank = (indices[i] == 0).nonzero(as_tuple=True)[0].item()
                        ndcgs[i] = 1.0 / math.log2(rank + 2)
                
                hr_sum += hits.sum().item()
                ndcg_sum += ndcgs.sum().item()
                count += len(hits)
                
                # Cold Start Split
                if cold_user_ids is not None:
                    uids = batch['target_input']['user_id'].cpu().tolist()
                    for i, uid in enumerate(uids):
                        if uid in cold_user_ids:
                            cold_hr_sum += hits[i].item()
                            cold_ndcg_sum += ndcgs[i].item()
                            cold_count += 1
                            
        metrics = {
            f'HR@{k}': hr_sum / count if count > 0 else 0.0,
            f'NDCG@{k}': ndcg_sum / count if count > 0 else 0.0
        }
        
        if cold_user_ids is not None:
            metrics.update({
                f'Cold_HR@{k}': cold_hr_sum / cold_count if cold_count > 0 else 0.0,
                f'Cold_NDCG@{k}': cold_ndcg_sum / cold_count if cold_count > 0 else 0.0
            })
            
        return metrics
