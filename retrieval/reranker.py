
import torch
import torch.nn as nn

class NeuralReRanker(nn.Module):
    """
    Neural Re-Ranking Model.
    Scores (User, Item) pairs to refine the initial candidate list from retrieval stage.
    """
    def __init__(self, user_dim, item_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(user_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_emb, item_emb):
        """
        Args:
            user_emb: [batch_size, dim] or [batch_size, num_candidates, dim]
            item_emb: [batch_size, dim] or [batch_size, num_candidates, dim]
            
        Returns:
            scores: [batch_size] or [batch_size, num_candidates]
        """
        # Concatenate along the last dimension
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # Forward pass and squeeze the last output dimension (which is 1)
        return self.net(x).squeeze(-1)
