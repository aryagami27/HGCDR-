
import torch
import torch.nn as nn

class KGFusion(nn.Module):
    """
    Fuses the learned item ID embedding with the knowledge graph embedding
    using a learned gating mechanism.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, 1)

    def forward(self, item_id_emb, kg_emb):
        """
        Args:
            item_id_emb: [batch_size, dim] - Embedding from ID lookup
            kg_emb: [batch_size, dim] - Embedding from KGEncoder for the corresponding items
            
        Returns:
            Fused embedding: [batch_size, dim]
        """
        # Concatenate and compute gate value in [0, 1]
        concat = torch.cat([item_id_emb, kg_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(concat))
        
        # Convex combination
        return alpha * item_id_emb + (1 - alpha) * kg_emb
