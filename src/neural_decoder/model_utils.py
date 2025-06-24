import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def fit_sklearn_model(model, train_x, train_y):
    
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.01, weight_decay=1e-5, model_name='best_model'):
    """Train the neural network"""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_eeg, batch_clip in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_eeg, batch_clip = batch_eeg.to(device), batch_clip.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_eeg)
            loss = criterion(outputs, batch_clip)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_eeg, batch_clip in val_loader:
                batch_eeg, batch_clip = batch_eeg.to(device), batch_clip.to(device)
                outputs = model(batch_eeg)
                loss = criterion(outputs, batch_clip)
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f'{model_name}.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, data_loader, device, loss_fn):
    """Calculates the average MSE for a model on a given data loader."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        for eeg_batch, embedding_batch in data_loader:
            eeg_batch = eeg_batch.to(device)
            embedding_batch = embedding_batch.to(device)
            
            # Get model prediction
            prediction = model(eeg_batch)
            
            # Calculate loss
            loss = loss_fn(prediction, embedding_batch)
            total_loss += loss.item() * eeg_batch.size(0) # Multiply by batch size

    return total_loss / len(data_loader.dataset)