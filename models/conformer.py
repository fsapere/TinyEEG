import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ================================
# 1. Data Management and Integration
# ================================
# Add the project root to the path to import data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import windows_dataset

# Split the dataset using split("session")
print("Splitting dataset into train and validation sets...")
splitted = windows_dataset.split("session")
train_set = splitted["0train"]
valid_set = splitted["1test"]

# Dynamically extract signal properties
print("Inferring signal properties from the training set...")

# We extract the properties directly from the dataset instead of relying on
# infer_signal_properties, which might be missing in some braindecode versions.
sample_x, sample_y, _ = train_set[0]
n_chans = sample_x.shape[0]
n_times = sample_x.shape[1]

# Assuming BCI IV-2a which has 4 classes (0, 1, 2, 3)
n_outputs = 4

print(f"Extracted properties -> n_chans: {n_chans}, n_outputs: {n_outputs}, n_times: {n_times}")


# ================================
# 2. Network Architecture (EEG Conformer) & 3. Quantization Rules
# ================================
class EEGConformer(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times):
        super(EEGConformer, self).__init__()
        
        # ----------------------------
        # Convolution Module
        # ----------------------------
        self.temporal_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=40, 
            kernel_size=(1, 25), 
            stride=(1, 1), 
            bias=False
        )
        
        self.spatial_conv = nn.Conv2d(
            in_channels=40, 
            out_channels=40, 
            kernel_size=(n_chans, 1), 
            stride=(1, 1), 
            bias=False
        )
        
        self.batch_norm = nn.BatchNorm2d(40)
        self.elu = nn.ELU()
        
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(1, 75), 
            stride=(1, 15)
        )
        
        # Dynamic calculation of the Sequence Length output by pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_chans, n_times)
            x = self.temporal_conv(dummy_input)
            x = self.spatial_conv(x)
            x = self.avg_pool(x)
            self.seq_len = x.shape[-1]
            
        # ----------------------------
        # Self-Attention Module
        # ----------------------------
        # Standard PyTorch TransformerEncoderLayer (Brevitas-ready implementation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=40, 
            nhead=10, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # ----------------------------
        # Classifier Module
        # ----------------------------
        self.flatten = nn.Flatten()
        
        # Two nn.Linear layers as specified for the final classifier module
        self.fc1 = nn.Linear(40 * self.seq_len, 64)
        self.fc1_act = nn.ELU()
        self.fc2 = nn.Linear(64, n_outputs)

    def forward(self, x):
        # Add the channel dimension if necessary
        # Expected input shape: (Batch, 1, n_chans, n_times)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # --- Convolution Module Forward ---
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = self.elu(x)
        x = self.avg_pool(x)
        
        # Rearrange
        # from (Batch, 40, 1, seq_len) to (Batch, seq_len, 40)
        x = x.squeeze(2)          
        x = x.transpose(1, 2)     
        
        # --- Self-Attention Module Forward ---
        x = self.transformer(x)
        
        # --- Classifier Module Forward ---
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        
        return x


# ================================
# 4. Training Loop
# ================================
def main():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Instantiate the model
    model = EEGConformer(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times).to(device)
    print("EEGConformer Model Initialized successfully.")
    
    # DataLoader preparation
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    epochs = 500
    
    # Early Stopping Configuration
    best_val_loss = float('inf')
    patience = 50
    early_stop_counter = 0
    save_path = '/scratch/fsapere/TinyEEG/models/conformer_best.pth'
    
    print(f"\nStarting Training Loop (max {epochs} epochs)...")
    for epoch in range(epochs):
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # --- VALIDATION ---
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch, _ in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                valid_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += y_batch.size(0)
                valid_correct += (predicted == y_batch).sum().item()
                
        valid_loss /= valid_total
        valid_acc = valid_correct / valid_total
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
              f"Val Loss: {valid_loss:.4f} - Val Acc: {valid_acc:.4f}")
              
        # --- EARLY STOPPING & SAVING ---
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best Val Loss! Model saved to {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}! Best Val Loss was {best_val_loss:.4f}.")
                break

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
