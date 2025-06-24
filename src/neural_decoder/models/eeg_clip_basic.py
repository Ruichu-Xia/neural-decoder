import torch.nn as nn


class EEGToCLIPNet(nn.Module):
    def __init__(self, eeg_input_dim, clip_embedding_dim=1024, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super().__init__()
        layers = []
        input_dim = eeg_input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, clip_embedding_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)