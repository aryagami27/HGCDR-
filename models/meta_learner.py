import torch
import torch.nn as nn
import copy
from models.recommender import HGCDRPlus

from models.disentangle import DisentangleNet

class MetaWrapper(nn.Module):
    def __init__(self, model: HGCDRPlus, inner_lr=0.01, inner_steps=1, device=None):
        super(MetaWrapper, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device if device else next(model.parameters()).device
        
    def _inner_loop(self, user_support_data, graph_data):
        """
        Performs the inner loop adaptation for a single user.
        """
        # Step A: Clone the projection heads
        fast_model = copy.copy(self.model)
        
        # Helper to get device
        device = self.device
        config = self.model.config
        
        # Reconstruct and load state for DisentangleNet
        fast_model.disentangle_src = DisentangleNet(
            config['embedding_dim'], config['z_inv_dim'], config['z_spec_dim']
        ).to(device)
        fast_model.disentangle_src.load_state_dict(self.model.disentangle_src.state_dict())
        
        fast_model.disentangle_tgt = DisentangleNet(
            config['embedding_dim'], config['z_inv_dim'], config['z_spec_dim']
        ).to(device)
        fast_model.disentangle_tgt.load_state_dict(self.model.disentangle_tgt.state_dict())
        
        # Reconstruct and load state for ItemProj
        fast_model.item_proj = nn.Linear(config['embedding_dim'], config['z_inv_dim']).to(device)
        fast_model.item_proj.load_state_dict(self.model.item_proj.state_dict())
        
        # Ensure they are in training mode
        fast_model.disentangle_src.train()
        fast_model.disentangle_tgt.train()
        fast_model.item_proj.train()
        
        # Modules to adapt
        modules_to_adapt = [fast_model.disentangle_src, fast_model.disentangle_tgt, fast_model.item_proj]
        
        # Optimization Loop
        for _ in range(self.inner_steps):
            # Step B: Forward pass on support set
            # Construct Batch Data compatible with HGCDRPlus
            batch_data = self._construct_batch(user_support_data, graph_data)
            
            outputs = fast_model(batch_data)
            
            # Step C: Calculate Rec Loss
            pos_scores = outputs['pos_scores']
            neg_scores = outputs['neg_scores']
            
            # Numerical Stability: Clamp scores to prevent overflow in logsigmoid
            # logsigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
            # If x is very large positive, exp(-x) -> 0, log(1) -> 0. Safe.
            # If x is very large negative, exp(-x) -> inf. Unsafe if not handled by stable impl.
            # PyTorch logsigmoid is generally stable, but extreme values can still cause issues in gradients.
            # Clamping the difference (pos - neg) is a good safety measure.
            diff = pos_scores - neg_scores
            diff = torch.clamp(diff, min=-10.0, max=10.0)
            
            rec_loss = -torch.nn.functional.logsigmoid(diff).mean()
            
            # Check for NaN loss
            if torch.isnan(rec_loss):
                # print("Warning: NaN loss in inner loop. Skipping update.")
                break # Skip this step or the whole loop
            
            # Step D: Compute gradients
            all_params = []
            for m in modules_to_adapt:
                all_params.extend(list(m.parameters()))
                
            grads = torch.autograd.grad(rec_loss, all_params, create_graph=False, allow_unused=True)
            
            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            # Step E: Update parameters manually
            param_to_name = {}
            for m in modules_to_adapt:
                for n, p in m.named_parameters():
                    param_to_name[p] = (m, n)
            
            for param, grad in zip(all_params, grads):
                if grad is None:
                    continue
                
                # Double check for NaN in grad
                if torch.isnan(grad).any():
                    # print("Warning: NaN gradient in inner loop. Skipping parameter update.")
                    continue

                if param in param_to_name:
                    module, name = param_to_name[param]
                    new_val = param - self.inner_lr * grad
                    
                    # Recursively set the attribute
                    parts = name.split('.')
                    sub_mod = module
                    for part in parts[:-1]:
                        sub_mod = getattr(sub_mod, part)
                    
                    if hasattr(sub_mod, parts[-1]):
                         delattr(sub_mod, parts[-1])
                    setattr(sub_mod, parts[-1], new_val)
                    
        return fast_model

    def _construct_batch(self, task_data, graph_data):
        """
        Wraps separate task data into HGCDRPlus batch format.
        """
        # task_data keys: user_id, item_id, neg_item_id
        # HGCDRPlus keys: source_input, target_input, overlap_flag, graph_data
        
        # Dummy source
        batch_size = task_data['user_id'].shape[0]
        # We assume 0 is a safe dummy index if vocab size > 0
        
        # Target Input
        target_input = {
            'user_id': task_data['user_id'], # [B]
            'item_id': task_data['item_id'], # [B]
            'neg_item_id': task_data['neg_item_id'] # [B, Negs]
        }
        
        # Construct Batch
        batch_data = {
            'source_input': None, # Model might fail if accessed?
            # HGCDRPlus.forward accesses:
            # batch_data['src_user_node_idx']
            # batch_data['tgt_user_node_idx']
            # batch_data['tgt_pos_item_node_idx']
            # batch_data['tgt_neg_item_node_idx']
            # It also accesses 'overlap_flag'
            
            'target_input': target_input,
            'overlap_flag': torch.zeros(batch_size, device=task_data['user_id'].device), # Assume no overlap info for meta-tasks
            'graph_data': graph_data,
            
            # Node Indices (Assuming encoded IDs match node indices)
            'src_user_node_idx': torch.zeros(batch_size, dtype=torch.long, device=task_data['user_id'].device), # Dummy
            'tgt_user_node_idx': task_data['user_id'],
            'tgt_pos_item_node_idx': task_data['item_id'],
            'tgt_neg_item_node_idx': task_data['neg_item_id']
        }
        return batch_data

    def forward(self, support_batch, query_batch, graph_data):
        """
        Meta-Training Step.
        support_batch: List of user_support_data dicts used for inner loop
        query_batch: List of user_query_data dicts used for outer loop
        graph_data: HeteroData structure
        """
        # support_batch is a LIST of task dictionaries
        # query_batch is a LIST of task dictionaries
        
        num_tasks = len(support_batch)
        total_query_loss = 0
        
        for i in range(num_tasks):
            user_support = support_batch[i]
            user_query = query_batch[i]
            
            # 1. Inner Loop (Fast Adapt)
            fast_model = self._inner_loop(user_support, graph_data)
            
            # 2. Outer Loop (Evaluate on Query)
            batch_data = self._construct_batch(user_query, graph_data)
            outputs = fast_model(batch_data)
            
            # Calculate Loss
            pos_scores = outputs['pos_scores']
            neg_scores = outputs['neg_scores']
            diff = pos_scores - neg_scores
            diff = torch.clamp(diff, min=-10.0, max=10.0)
            loss = -torch.nn.functional.logsigmoid(diff).mean()
            
            total_query_loss += loss
            
        if num_tasks > 0:
            return total_query_loss / num_tasks
        return torch.tensor(0.0, device=graph_data.x_dict['user_src'].device)
