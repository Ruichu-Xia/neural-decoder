from data_utils import get_lazy_dataloaders

def main():
    # Load the data
    sub_id = 1
    clip_train_loader, clip_val_loader, clip_test_loader = get_lazy_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='clip')
    vae_train_loader, vae_val_loader, vae_test_loader = get_lazy_dataloaders(sub_id=sub_id, batch_size=32, shuffle=True, embedding_type='vae')

    model = RidgeWrapper()
    
    train_model(model, clip_train_loader, clip_val_loader, device)

if __name__ == "__main__":
    main()