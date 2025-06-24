import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from configs.config import config

class EEGEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for EEG and Embedding data using memory-mapped numpy arrays.
    Normalization is done on-the-fly.
    """
    def __init__(self, eeg_path: str, embedding_path: str, norm_mean: np.ndarray, norm_std: np.ndarray, flatten_eeg: bool = True):
        self.eeg = np.load(eeg_path, mmap_mode='r')
        self.embedding = np.load(embedding_path, mmap_mode='r')
        assert len(self.eeg) == len(self.embedding), "EEG and Embedding must have the same number of samples"
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.flatten_eeg = flatten_eeg

        if not self.flatten_eeg:
            self.norm_mean = self.norm_mean.reshape(-1, 1)
            self.norm_std = self.norm_std.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flatten_eeg:
            # Flatten EEG data to 1D    
            eeg = self.eeg[idx].reshape(-1)
            eeg = (eeg - self.norm_mean) / self.norm_std
        else:
            # Keep spatial dimensions: normalize each channel separately
            eeg = self.eeg[idx]  # Keep original shape (e.g., [4, 63, 250])

            eeg = (eeg - self.norm_mean) / self.norm_std
        
        embedding = self.embedding[idx]
        return torch.from_numpy(eeg.copy()).float(), torch.from_numpy(embedding.copy()).float()

def get_dataloaders(
    sub_id: int,
    batch_size: int = 32,
    shuffle: bool = True,
    val_split_ratio: float = 0.2,
    embedding_type: str = 'clip',
    flatten_eeg: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]  :
    """
    Get lazy dataloaders for EEG and CLIP data.

    Args:
        sub_id: The subject ID.
        batch_size: The batch size.
        shuffle: Whether to shuffle the data.
        eeg_dir: The directory containing the EEG data.
        clip_dir: The directory containing the CLIP data.
        val_split_ratio: The ratio of the validation set.
        flatten_eeg: Whether to flatten EEG data to 1D or keep spatial dimensions.

    Returns:
        A tuple containing the train, validation, and test dataloaders.
    """
    eeg_train_path = f"{config.data.eeg_dir}sub-{sub_id:02d}/{config.data.train_path}"
    eeg_test_path = f"{config.data.eeg_dir}sub-{sub_id:02d}/{config.data.test_path}"
    eeg_train = np.load(eeg_train_path)
    norm_mean, norm_std = _calculate_norm_mean_std(eeg_train, flatten_eeg)

    embedding_train_path = f"{config.data.embedding_dir}train_{embedding_type}.npy"
    embedding_test_path = f"{config.data.embedding_dir}test_{embedding_type}.npy"

    train_dataset = EEGEmbeddingDataset(eeg_train_path, embedding_train_path, norm_mean, norm_std, flatten_eeg=flatten_eeg)
    total_train_samples = len(train_dataset)
    val_samples = int(total_train_samples * val_split_ratio)
    train_samples = total_train_samples - val_samples
    train_dataset, val_dataset = random_split(train_dataset, [train_samples, val_samples], 
    generator=torch.Generator().manual_seed(42)) 

    test_dataset = EEGEmbeddingDataset(eeg_test_path, embedding_test_path, norm_mean, norm_std, flatten_eeg=flatten_eeg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def get_numpy_from_loader(loader):
    """Helper function to convert a PyTorch DataLoader to flattened NumPy arrays."""
    all_eeg = []
    all_embeddings = []
    for eeg_batch, embedding_batch in loader:
        all_eeg.append(eeg_batch.numpy())
        all_embeddings.append(embedding_batch.numpy())
    
    eeg_np = np.concatenate(all_eeg, axis=0)
    eeg_np_flat = eeg_np.reshape(eeg_np.shape[0], -1)
    embeddings_np = np.concatenate(all_embeddings, axis=0)
    
    return eeg_np_flat, embeddings_np

def load_normalized_numpy_data(eeg_path: str, 
                    embedding_path: str, 
                    norm_mean: np.ndarray, 
                    norm_std: np.ndarray, 
                    flatten_eeg: bool = True):
    """
    Load the EEG and embedding data from the given paths.

    Args:
        eeg_path: The path to the EEG data.
        embedding_path: The path to the embedding data.
        norm_mean: The mean of the EEG data.
        norm_std: The standard deviation of the EEG data.
        flatten_eeg: Whether to flatten the EEG data.

    Returns:
        A tuple containing the EEG and embedding data.
    """
    eeg_train_path = f"{config.data.eeg_dir}sub-{sub_id:02d}/{config.data.train_path}"
    eeg_train = np.load(eeg_path)
    norm_mean, norm_std = _calculate_norm_mean_std(eeg_train, flatten_eeg)

    if flatten_eeg:

def _calculate_norm_mean_std(eeg_train: np.ndarray, flatten_eeg: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to calculate the mean and standard deviation of the EEG data.

    Args:
        eeg_train: The EEG data.
        flatten_eeg: Whether to flatten the EEG data.

    Returns:
        A tuple containing the mean and standard deviation of the EEG data.
    """
    if flatten_eeg:
        eeg_train_reshaped = eeg_train.reshape(eeg_train.shape[0], -1)
        norm_mean = np.mean(eeg_train_reshaped, axis=0)
        norm_std = np.std(eeg_train_reshaped, axis=0, ddof=1)
    else:
        norm_mean = np.mean(eeg_train, axis=(0, 2))
        norm_std = np.std(eeg_train, axis=(0, 2), ddof=1)

        norm_std[norm_std == 0] = 1e-8

    return norm_mean, norm_std
    
