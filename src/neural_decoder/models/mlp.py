import torch
import torch.nn as nn


class EEGToLatentMLP(nn.Module):
    """
    A simple feed-forward neural network to map EEG features to a latent space.

    Input Shape: (batch_size, num_channels, sequence_length)
                 e.g., (32, 17, 80) for EEG data
    Output Shape: (batch_size, latent_dim)
                  e.g., (32, 1024) for image latent space
    """
    def __init__(self, num_channels: int = 17, sequence_length: int = 80,
                 latent_dim: int = 1024, hidden_dim: int = 512, num_hidden_layers: int = 1):
        """
        Initializes the neural network.

        Args:
            num_channels (int): Number of EEG channels (e.g., 17).
            sequence_length (int): Length of the EEG sequence (e.g., 80).
            latent_dim (int): The desired dimension of the output latent space (e.g., 1024).
            hidden_dim (int): The size of each hidden layer.
            num_hidden_layers (int): The number of hidden layers between the input and output.
                                     Must be at least 0. If 0, it's a direct linear mapping.
        """
        super(EEGToLatentMLP, self).__init__()

        # Calculate the flattened input size from EEG data
        self.input_size = num_channels * sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        # Define the layers of the neural network
        layers = []
        current_input_dim = self.input_size

        # Add the first hidden layer
        if self.num_hidden_layers > 0:
            layers.append(nn.Linear(current_input_dim, self.hidden_dim))
            layers.append(nn.LeakyReLU())
            current_input_dim = self.hidden_dim

            # Add additional hidden layers
            for _ in range(self.num_hidden_layers - 1):
                layers.append(nn.Linear(current_input_dim, self.hidden_dim))
                layers.append(nn.LeakyReLU())
        
        # Output layer: connects the last hidden layer (or input if no hidden layers)
        # to the latent_dim.
        # If num_hidden_layers is 0, current_input_dim will still be self.input_size
        layers.append(nn.Linear(current_input_dim, self.latent_dim))

        # Use nn.Sequential to chain the layers, making the forward pass cleaner
        self.mlp = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, latent_dim).
        """
        # Flatten the input tensor from (batch_size, 17, 80) to (batch_size, 17 * 80)
        # The -1 in .view() automatically infers the batch size.
        x = x.view(-1, self.input_size)

        # Pass through the dynamically created MLP
        x = self.mlp(x)

        return x