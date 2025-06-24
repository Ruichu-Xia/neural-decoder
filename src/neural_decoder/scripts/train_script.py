from data_utils import get_lazy_dataloaders
from model_utils import train_model
from eeg_clip_basic import EEGToCLIPNet


def train_eeg_clip_basic(sub_id: int, batch_size: int = 32, shuffle: bool = True, val_split_ratio: float = 0.2):
    train_loader, val_loader, test_loader = get_lazy_dataloaders(sub_id=sub_id, batch_size=batch_size, shuffle=shuffle, val_split_ratio=val_split_ratio)
    model = EEGToCLIPNet(eeg_input_dim=1360, clip_embedding_dim=1024, hidden_dims=[512, 256, 128], dropout_rate=0.3)
    train_model(model, train_loader, val_loader, device='cuda', num_epochs=100, lr=0.001, weight_decay=1e-5)

if __name__ == "__main__":
    train_eeg_clip_basic(sub_id=1, batch_size=32, shuffle=True, val_split_ratio=0.2)