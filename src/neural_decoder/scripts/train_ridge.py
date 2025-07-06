import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from configs.config import config
from data_utils import get_dataloaders, get_numpy_from_loader
from models.ridge import RidgeWrapper
from model_utils import evaluate_model


def main():
    args = parse_args()
    sub_id = args.sub_id

    # Configs
    device = "cpu"
    ridge_params = config.model.ridge.model_dump()

    # Load data
    clip_train_loader, clip_val_loader, clip_test_loader = get_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='clip')
    vae_train_loader, vae_val_loader, vae_test_loader = get_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='vae')

    clip_eeg_train, clip_embedding_train = get_numpy_from_loader(clip_train_loader)
    clip_eeg_val, clip_embedding_val = get_numpy_from_loader(clip_val_loader)
    clip_train_full = np.concatenate((clip_eeg_train, clip_eeg_val), axis=0)
    clip_embedding_full = np.concatenate((clip_embedding_train, clip_embedding_val), axis=0)

    vae_eeg_train, vae_embedding_train = get_numpy_from_loader(vae_train_loader)
    vae_eeg_val, vae_embedding_val = get_numpy_from_loader(vae_val_loader)
    vae_train_full = np.concatenate((vae_eeg_train, vae_eeg_val), axis=0)
    vae_embedding_full = np.concatenate((vae_embedding_train, vae_embedding_val), axis=0)

    # Fit ridge model for clip
    ridge_model_clip = RidgeWrapper(**ridge_params)
    ridge_model_clip.fit(clip_train_full, clip_embedding_full)

    loss_fn = nn.MSELoss()
    test_loss_clip = evaluate_model(ridge_model_clip, clip_test_loader, device, loss_fn)
    print(f"Test loss for clip: {test_loss_clip:.6f}")

    # Fit ridge model for vae
    ridge_model_vae = RidgeWrapper(**ridge_params)
    ridge_model_vae.fit(vae_train_full, vae_embedding_full)

    test_loss_vae = evaluate_model(ridge_model_vae, vae_test_loader, device, loss_fn)
    print(f"Test loss for vae: {test_loss_vae:.6f}")

    # Save models
    save_dir = f"{config.model.checkpoint_dir}ridge/sub-{sub_id:02d}/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(ridge_model_clip.state_dict(), f"{save_dir}clip.pth")
    torch.save(ridge_model_vae.state_dict(), f"{save_dir}vae.pth")

    print(f"Models saved to {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_id", type=int, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()