import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from configs.config import config 

    
def train_model(model, 
                train_loader, 
                val_loader, 
                device, 
                num_epochs=100, 
                lr=0.01, 
                weight_decay=1e-5, 
                sub_id=1, 
                embedding_type="clip", 
                model_name='best_model'):
    """Train the neural network"""
    save_dir = f"{config.model.checkpoint_dir}{model_name}/sub-{sub_id:02d}/"
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    print("Starting model training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_eeg, batch_embedding in train_loader:
            batch_eeg, batch_embedding = batch_eeg.to(device), batch_embedding.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_eeg)
            loss = criterion(outputs, batch_embedding)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_eeg, batch_embedding in val_loader:
                batch_eeg, batch_embedding = batch_eeg.to(device), batch_embedding.to(device)
                outputs = model(batch_eeg)
                loss = criterion(outputs, batch_embedding)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model checkpoint
            print(f'Epoch {epoch+1}/{num_epochs}: Val Loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f'{save_dir}{embedding_type}.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1},  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses


def evaluate_model(model, data_loader, device, loss_fn):
    """Calculates the average MSE for a model on a given data loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for eeg_batch, embedding_batch in data_loader:
            eeg_batch = eeg_batch.to(device)
            embedding_batch = embedding_batch.to(device)
            
            prediction = model(eeg_batch)
            
            loss = loss_fn(prediction, embedding_batch)
            total_loss += loss.item() * eeg_batch.size(0)

    return total_loss / len(data_loader.dataset)


def save_loss_plot(train_losses, val_losses, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path) 