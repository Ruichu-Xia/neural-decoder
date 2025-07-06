import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np


class EEGToLatentGCN(nn.Module):
    def __init__(self, num_channels: int = 17,
                       sequence_length: int = 80,
                       latent_dim: int = 1024,
                       hidden_dim: int = 256,
                       num_gnn_layers: int = 3,
                       node_feature_dim: int = 64,
                       adjacency_matrix: np.ndarray | None = None):
        """
        Initializes the GNN model.

        Args:
            num_channels (int): Number of EEG channels (nodes in the graph).
            sequence_length (int): Length of the EEG sequence (initial feature dim per node).
            latent_dim (int): The desired dimension of the output latent space.
            hidden_dim (int): Hidden dimension for GNN layers and MLP head.
            num_gnn_layers (int): Number of Graph Convolutional layers.
            node_feature_dim (int): The dimension of the feature vector for each node
                                    after the initial linear projection from sequence_length.
            adjacency_matrix (np.ndarray, optional): A (num_channels, num_channels)
                                                      adjacency matrix defining graph connectivity.
                                                      If None, a default fully-connected graph is used.
        """
        super(EEGToLatentGCN, self).__init__()

        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim

        self.node_embedding = nn.Sequential(
            nn.Linear(sequence_length, node_feature_dim),
            nn.LeakyReLU()
        )

        self.gnn_layers = nn.ModuleList()
        current_in_features = node_feature_dim
        for i in range(num_gnn_layers):
            # GCNConv takes (in_channels, out_channels) for node features
            self.gnn_layers.append(GCNConv(current_in_features, hidden_dim))
            current_in_features = hidden_dim

        self.mlp_head = nn.Sequential(
            nn.Linear(current_in_features, hidden_dim), # Input from global pooling
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        if not adjacency_matrix:
            adj_matrix = torch.ones(num_channels, num_channels) - torch.eye(num_channels)
        else:
            adj_matrix = torch.from_numpy(adjacency_matrix).float()

        self.register_buffer('edge_index', dense_to_sparse(adj_matrix)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the GNN.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, latent_dim).
        """
        batch_size = x.size(0)

        node_features_flat = x.view(-1, self.sequence_length)
        node_features_embedded = self.node_embedding(node_features_flat)
        batch_tensor = torch.arange(batch_size, device=x.device).repeat_interleave(self.num_channels)
        assert batch_tensor.size(0) == node_features_embedded.size(0), \
            "Batch tensor size mismatch with node features."

        current_node_features = node_features_embedded
        for gnn_layer in self.gnn_layers:
            current_node_features = gnn_layer(current_node_features, self.edge_index)
            current_node_features = nn.LeakyReLU()(current_node_features)

        graph_features = global_mean_pool(current_node_features, batch_tensor)

        latent_representation = self.mlp_head(graph_features)

        return latent_representation


