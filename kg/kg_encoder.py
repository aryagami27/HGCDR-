
import torch
import torch.nn as nn
import torch.nn.functional as F

class KGEncoder(nn.Module):
    """
    Multi-hop Knowledge Graph Encoder.
    Propagates information over the KG using relation-specific transformations 
    and attention-based aggregation.
    """
    def __init__(self, emb_dim: int, num_relations: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        # Relation-specific linear transformations
        self.rel_transform = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim) for _ in range(num_relations)
        ])
        
        # Self-attention scoring vector
        self.attn = nn.Linear(emb_dim, 1)

    def forward(self, node_emb, edge_index, edge_type):
        """
        Args:
            node_emb: [num_nodes, emb_dim] Initial node embeddings (entities + items)
            edge_index: [2, num_edges] Source and destination indices
            edge_type: [num_edges] Relation type for each edge
            
        Returns:
            Refined node embeddings after 'num_layers' hops.
        """
        h = node_emb
        for _ in range(self.num_layers):
            h = self.message_passing(h, edge_index, edge_type)
        return h

    def message_passing(self, h, edge_index, edge_type):
        src, dst = edge_index
        num_nodes = h.size(0)
        
        # Prepare to collect all messages and attention scores
        # We process all relations in a loop, but effectively accumulate messages for each node
        
        # We need to construct a tensor of shape [num_edges, emb_dim] containing the transformed messages
        messages = torch.zeros(src.size(0), self.emb_dim, device=h.device)
        
        unique_rels = torch.unique(edge_type)
        
        for r in unique_rels:
            mask = (edge_type == r)
            if not mask.any(): 
                continue
                
            # Get source features for edges of type r
            h_src = h[src[mask]]
            
            # Apply relation-specific transformation: W_r * h_src
            # r is a tensor, convert to int for indexing
            r_idx = r.item()
            msg_r = self.rel_transform[r_idx](h_src)
            
            # Store messages
            messages[mask] = msg_r

        # Calculate attention scores: a = LeakyReLU(attn(message)) or similar
        # Here using simple Linear -> Softmax logic as requested
        # We need to normalize attention scores per destination node
        
        # 1. Compute unnormalized scores
        attn_weights = self.attn(messages).squeeze(-1) # [num_edges]
        
        # 2. Subtract max for stability before exp (optional but good practice)
        # For simplicity in pure pytorch without scatter_max, we might skip, but let's try to be stable if possible.
        # Standard GAT approach: exp(score) / sum(exp(score))
        
        attn_exp = torch.exp(attn_weights)
        
        # Sum of exponentials for each destination: denominator
        # We use index_add_ or scatter_add
        denom = torch.zeros(num_nodes, device=h.device)
        denom.index_add_(0, dst, attn_exp)
        
        # Avoid division by zero for isolated nodes
        denom = denom + 1e-10
        
        # Expand denominator to edges
        denom_expanded = denom[dst]
        
        # Normalized attention coefficients
        alphas = attn_exp / denom_expanded
        
        # 3. Aggregate messages: h_dst = sum(alpha * msg)
        weighted_messages = messages * alphas.unsqueeze(-1)
        
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, weighted_messages)
        
        # Add residual connection
        return h + agg
