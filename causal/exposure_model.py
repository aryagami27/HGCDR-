
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
        # ROBUSTNESS: Clamp indices to valid range to prevent CUDA device-side asserts
        # User Embeddings
        num_users = self.user_embedding.num_embeddings
        if (user_id >= num_users).any() or (user_id < 0).any():
            import logging
            # Log once per batch to avoid flooding? Or just log.
            # Using a simple print or logging.warning might flood if it happens often.
            # But essential for debugging.
            if (user_id >= num_users).any():
                 logging.warning(f"[ExposureModel] OOB User IDs detected! Max: {user_id.max().item()}, Limit: {num_users}. Clamping...")
            user_id = user_id.clamp(0, num_users - 1)
            
        # Item Embeddings
        num_items = self.item_embedding.num_embeddings
        if (item_id >= num_items).any() or (item_id < 0).any():
            import logging
            if (item_id >= num_items).any():
                 logging.warning(f"[ExposureModel] OOB Item IDs detected! Max: {item_id.max().item()}, Limit: {num_items}. Clamping...")
            item_id = item_id.clamp(0, num_items - 1)

        u_emb = self.user_embedding(user_id)
        i_emb = self.item_embedding(item_id)
        x = torch.cat([u_emb, i_emb], dim=-1)
        return torch.sigmoid(self.net(x))
