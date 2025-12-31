import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

class HGTModule(nn.Module):
    def __init__(self, metadata, hidden_channels, num_heads=4, num_layers=2, in_channels=None):
        """
        Args:
            metadata: Tuple (node_types, edge_types) from PyG HeteroData.metadata()
            hidden_channels: Output dimension (embedding_dim)
            num_heads: Number of attention heads
            num_layers: Number of HGT layers
            in_channels: (Optional) Input dimension(s). Can be int or Dict[str, int].
                         If None, defaults to hidden_channels.
        """
        super(HGTModule, self).__init__()
        
        if in_channels is None:
            in_channels = hidden_channels
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer handles diverse input dims (Dict), subsequent layers use hidden_channels
            layer_in = in_channels if i == 0 else hidden_channels
            
            self.layers.append(
                HGTConv(
                    in_channels=layer_in,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads
                )
            )

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Dictionary of node features {node_type: x}
            edge_index_dict: Dictionary of edge indices {edge_type: edge_index}
        """
        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict
