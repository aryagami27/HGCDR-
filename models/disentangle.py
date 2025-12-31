import torch
import torch.nn as nn

class DisentangleNet(nn.Module):
    def __init__(self, input_dim, z_inv_dim, z_spec_dim):
        super(DisentangleNet, self).__init__()
        self.mlp_inv = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, z_inv_dim)
        )
        self.mlp_spec = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, z_spec_dim)
        )
        
        # Projection for specific features before adding to invariant
        self.spec_proj = nn.Linear(z_spec_dim, z_inv_dim)
        
        self.gate = TransferGate(z_inv_dim, z_spec_dim)

    def forward(self, u_emb, other_z_spec=None):
        """
        Args:
            u_emb: Input user embedding (from HGT/Fusion)
            other_z_spec: Specific features from the OTHER domain (for transfer)
                          If None, we are just extracting features.
        """
        z_inv = self.mlp_inv(u_emb)
        z_spec = self.mlp_spec(u_emb)
        
        z_final = z_inv
        gate_val = None
        
        if other_z_spec is not None:
            # Transfer mechanism
            # z_final = z_inv + g * Project(z_spec_source)
            gate_val = self.gate(z_inv, other_z_spec)
            z_final = z_inv + gate_val * self.spec_proj(other_z_spec)
            
        return z_inv, z_spec, z_final, gate_val

class TransferGate(nn.Module):
    def __init__(self, z_inv_dim, z_spec_dim):
        super(TransferGate, self).__init__()
        self.fc = nn.Linear(z_inv_dim + z_spec_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_inv_target, z_spec_source):
        combined = torch.cat([z_inv_target, z_spec_source], dim=-1)
        return self.sigmoid(self.fc(combined))
