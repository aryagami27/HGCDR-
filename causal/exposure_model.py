
import torch
import torch.nn as nn

class ExposureModel(nn.Module):
    """
    Predicts the probability of exposure P(E=1 | user, item) using an MLP.
    This model is trained to distinguish between observed interactions and
    unobserved ones (which are treated as missing at random or not exposed).
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_id, item_id):
        """
        Args:
            user_id: (batch_size,) Int Tensor
            item_id: (batch_size,) Int Tensor
        Returns:
            Propensity score (probability of exposure) in [0, 1]
        """
        u_emb = self.user_embedding(user_id)
        i_emb = self.item_embedding(item_id)
        x = torch.cat([u_emb, i_emb], dim=-1)
        return torch.sigmoid(self.net(x))
