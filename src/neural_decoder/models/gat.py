import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
import numpy as np


class EEGToLatentGAT(nn.Module):
    """
    A Spatiotemporal Graph Attention Network (ST-GAT) for EEG data.

    This model processes EEG data by first applying a Graph Attention Network (GAT)
    to capture spatial dependencies between channels at each time step.
    Then, it uses a 1D Temporal CNN to learn patterns across the time sequence.

    Input Shape: (batch_size, num_channels, sequence_length)
                 e.g., (64, 17, 80)

    Output Shape: (batch_size, latent_dim)
                  e.g., (64, 1024) for a CLIP latent embedding
    """
    def __init__(self, num_channels: int = 17,
                       sequence_length: int = 80,
                       latent_dim: int = 1024,
                       gat_hidden_dim: int = 32,
                       num_gat_heads: int = 4,
                       tcn_out_channels: int = 64,
                       mlp_hidden_dim: int = 256,
                       adjacency_matrix: np.ndarray | None = None):
        """
        Initializes the SpatioTemporalGAT model.

        Args:
            num_channels (int): Number of EEG channels (nodes).
            sequence_length (int): Length of the EEG sequence.
            latent_dim (int): The dimension of the final output latent space.
            gat_hidden_dim (int): The feature dimension of each GAT head.
            num_gat_heads (int): The number of attention heads in the GAT layer.
            tcn_out_channels (int): The number of output channels for the temporal CNN.
            mlp_hidden_dim (int): The hidden dimension for the final MLP head.
            adjacency_matrix (np.ndarray, optional): A (num_channels, num_channels)
                adjacency matrix. If None, a fully-connected graph is used.
        """
        super(EEGToLatentGAT, self).__init__()

        self.num_channels = num_channels
        self.sequence_length = sequence_length
        
        if adjacency_matrix is None:
            adj_matrix = torch.ones(num_channels, num_channels) - torch.eye(num_channels)
        else:
            adj_matrix = torch.from_numpy(adjacency_matrix).float()
        
        self.register_buffer("edge_index", dense_to_sparse(adj_matrix)[0])

        self.gat_conv = GATConv(in_channels=1,
                                out_channels=gat_hidden_dim,
                                heads=num_gat_heads,
                                concat=True,
                                dropout=0.3)

        gat_output_dim = gat_hidden_dim * num_gat_heads

        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=gat_output_dim,
                      out_channels=tcn_out_channels,
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(tcn_out_channels),
            nn.Dropout(0.3),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(tcn_out_channels * num_channels, mlp_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, latent_dim).
        """
        batch_size = x.size(0)

        x_reshaped = x.permute(0, 2, 1).contiguous()
        x_reshaped = x_reshaped.view(batch_size * self.sequence_length, self.num_channels, 1)

        num_nodes_per_step = self.num_channels

        edge_indices = [torch.as_tensor(self.edge_index) + i * num_nodes_per_step for i in range(batch_size * self.sequence_length)]
        batched_edge_index = torch.cat(edge_indices, dim=1)

        x_flat = x_reshaped.view(-1, 1)

        spatial_features = self.gat_conv(x_flat, batched_edge_index)
        spatial_features = F.elu(spatial_features)

        gat_output_dim = spatial_features.size(1)
        x_spatial_sequence = spatial_features.view(batch_size, self.sequence_length, self.num_channels, gat_output_dim)
        x_spatial_sequence = x_spatial_sequence.permute(0, 2, 3, 1).contiguous()
        x_spatial_sequence = x_spatial_sequence.view(batch_size * self.num_channels, gat_output_dim, self.sequence_length)

        temporal_features = self.tcn(x_spatial_sequence)

        readout_features = temporal_features.mean(dim=2)
        readout_features = readout_features.view(batch_size, -1)

        latent_embedding = self.mlp_head(readout_features)

        return latent_embedding