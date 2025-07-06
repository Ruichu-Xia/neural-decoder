import torch
import torch.nn as nn


class EEGToLatentCNN(nn.Module):
    def __init__(self, num_channels: int = 17, 
                       sequence_length: int = 80,
                       latent_dim: int = 1024, 
                       hidden_dim: int = 512, 
                       num_conv_layers: int = 3,
                       kernel_size: int = 3,
                       stride: int = 1,
                       padding: int = 2,
                       pool_size: int = 2):
        super(EEGToLatentCNN, self).__init__()
        
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        conv_layers = []
        in_channels = num_channels
        current_seq_len = sequence_length

        for i in range(num_conv_layers):
            out_channels = hidden_dim if i == 0 else hidden_dim // (2**i)
            if out_channels < 16: out_channels = 16

            conv_layers.append(nn.Conv1d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.LeakyReLU())

            conv_layers.append(nn.MaxPool1d(pool_size, stride=pool_size))

            in_channels = out_channels

            current_seq_len = (current_seq_len + 2 * padding - kernel_size) // stride + 1

            current_seq_len = current_seq_len // pool_size

        self.conv_block = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy_input = torch.randn(1, num_channels, sequence_length)
            dummy_output = self.conv_block(dummy_input)
            self.flattened_conv_output_size = dummy_output.numel() // 1 

        self.mlp_head = nn.Sequential(
            nn.Linear(self.flattened_conv_output_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_head(x)
        return x
        