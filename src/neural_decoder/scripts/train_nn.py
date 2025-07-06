import gc
import argparse
import torch
import torch.nn as nn

from models.mlp import EEGToLatentMLP
from models.cnn import EEGToLatentCNN
from models.gcn import EEGToLatentGCN
from models.gat import EEGToLatentGAT
from model_utils import train_model, evaluate_model, save_loss_plot
from data_utils import get_dataloaders
from configs.config import config 


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    args = parse_args()
    sub_id = args.sub_id
    model_name = args.model_name
    num_channels = config.model.num_channels
    sequence_length = config.model.sequence_length
    clip_dim = config.model.clip_dim
    vae_dim = config.model.vae_dim

    save_dir = f"{config.model.checkpoint_dir}{model_name}/sub-{sub_id:02d}/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_train_loader, clip_val_loader, clip_test_loader = get_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='clip', flatten_eeg=False)
    vae_train_loader, vae_val_loader, vae_test_loader = get_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='vae', flatten_eeg=False)

    print("EEG to CLIP:")
    model = EEGToLatentMLP(num_channels=num_channels, 
                           sequence_length=sequence_length, 
                           latent_dim=clip_dim, 
                           hidden_dim=512, 
                           num_hidden_layers=2)
    train_losses, val_losses = train_model(model, 
                                           clip_train_loader, 
                                           clip_val_loader, 
                                           device=device, 
                                           num_epochs=100, 
                                           lr=0.001, 
                                           weight_decay=1e-5,
                                           sub_id=sub_id,
                                           embedding_type="clip",
                                           model_name=model_name)

    plot_path = f"{config.model.checkpoint_dir}{model_name}/sub-{sub_id:02d}/clip_loss_plot.png"
    save_loss_plot(train_losses, val_losses, plot_path)

    loss_fn = nn.MSELoss()
    model.load_state_dict(torch.load(f"{save_dir}/clip.pth")['model_state_dict'])
    test_loss = evaluate_model(model, clip_test_loader, device, loss_fn)

    print(f"Finished training EEG to CLIP. Test loss: {test_loss:.6}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    print("EEG to VAE:")
    model = EEGToLatentMLP(num_channels=num_channels, 
                           sequence_length=sequence_length, 
                           latent_dim=vae_dim, 
                           hidden_dim=512, 
                           num_hidden_layers=2)
    train_losses, val_losses = train_model(model, 
                                           vae_train_loader, 
                                           vae_val_loader, 
                                           device=device, 
                                           num_epochs=100, 
                                           lr=0.001, 
                                           weight_decay=1e-5,
                                           sub_id=sub_id,
                                           embedding_type="vae",
                                           model_name=model_name)

    plot_path = f"{config.model.checkpoint_dir}{model_name}/sub-{sub_id:02d}/vae_loss_plot.png"
    save_loss_plot(train_losses, val_losses, plot_path)

    loss_fn = nn.MSELoss()
    model.load_state_dict(torch.load(f"{save_dir}/vae.pth")['model_state_dict'])
    test_loss = evaluate_model(model, vae_test_loader, device, loss_fn)

    print(f"Finished training EEG to VAE. Test loss: {test_loss:.6}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_id", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["mlp", "cnn"])
    return parser.parse_args()


if __name__ == "__main__":
    main()