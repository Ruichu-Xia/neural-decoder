import os
import argparse
import numpy as np
import torch
from data_utils import get_dataloaders, get_numpy_from_loader, match_latent_distribution
from models.ridge import RidgeWrapper
from models.mlp import EEGToLatentMLP
from configs.config import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_args()
    sub_id = args.sub_id
    model_name = args.model_name
    num_channels = config.model.num_channels
    sequence_length = config.model.sequence_length
    clip_dim = config.model.clip_dim
    vae_dim = config.model.vae_dim
    
    save_dir = f"{config.model.checkpoint_dir}{model_name}/sub-{sub_id:02d}/"

    # Load data
    clip_train_loader, clip_val_loader, clip_test_loader = get_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='clip')
    vae_train_loader, vae_val_loader, vae_test_loader = get_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='vae')

    _, clip_embedding_train = get_numpy_from_loader(clip_train_loader)
    _, clip_embedding_val = get_numpy_from_loader(clip_val_loader)
    clip_target_latent = np.concatenate((clip_embedding_train, clip_embedding_val), axis=0)

    _, vae_embedding_train = get_numpy_from_loader(vae_train_loader)
    _, vae_embedding_val = get_numpy_from_loader(vae_val_loader)
    vae_target_latent = np.concatenate((vae_embedding_train, vae_embedding_val), axis=0)

    clip_eeg_test, _ = get_numpy_from_loader(clip_test_loader)
    vae_eeg_test, _ = get_numpy_from_loader(vae_test_loader)

    print("Data loaded successfully.")

    if model_name == "ridge":
        print("Performing inference with Ridge model...")
        clip_model = RidgeWrapper()
        clip_model.load_state_dict(torch.load(f"{save_dir}clip.pth"))
        vae_model = RidgeWrapper()
        vae_model.load_state_dict(torch.load(f"{save_dir}vae.pth"))

        clip_pred_latent = clip_model(clip_eeg_test).numpy()
        vae_pred_latent = vae_model(vae_eeg_test).numpy()
        print("Ridge inference complete.")

    elif model_name == "mlp":
        print("Performing inference with MLP model...")
        clip_model = EEGToLatentMLP(num_channels=num_channels,
                                   sequence_length=sequence_length,
                                   latent_dim=clip_dim,
                                   hidden_dim=512,
                                   num_hidden_layers=2)
        vae_model = EEGToLatentMLP(num_channels=num_channels,
                                   sequence_length=sequence_length,
                                   latent_dim=vae_dim,
                                   hidden_dim=512,
                                   num_hidden_layers=2)
        clip_model.load_state_dict(torch.load(f"{save_dir}clip.pth")['model_state_dict'])
        vae_model.load_state_dict(torch.load(f"{save_dir}vae.pth")['model_state_dict'])

        clip_model.to(device)
        vae_model.to(device)

        clip_model.eval()
        vae_model.eval()

        clip_eeg_test = torch.from_numpy(clip_eeg_test).float().to(device)
        vae_eeg_test = torch.from_numpy(vae_eeg_test).float().to(device)

        with torch.no_grad():
            clip_pred_latent = clip_model(clip_eeg_test).cpu().numpy()
            vae_pred_latent = vae_model(vae_eeg_test).cpu().numpy()
        print("MLP inference complete.")
    else:
        raise ValueError(f"Model {model_name} does not exist.")

    print("Matching latent distributions...")
    clip_pred_latent = match_latent_distribution(clip_pred_latent, clip_target_latent)
    vae_pred_latent = match_latent_distribution(vae_pred_latent, vae_target_latent)

    pred_dir = f"{config.data.predicted_embedding_dir}{model_name}/sub-{sub_id:02d}/"
    os.makedirs(pred_dir, exist_ok=True)
    pred_clip_path = f"{pred_dir}pred_clip.npy"
    pred_vae_path = f"{pred_dir}pred_vae.npy"
    np.save(pred_clip_path, clip_pred_latent)
    np.save(pred_vae_path, vae_pred_latent)

    print(f"Predicted embeddings saved to: {pred_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_id", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["ridge", "mlp"])
    return parser.parse_args()

if __name__ == "__main__":
    main()