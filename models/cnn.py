import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
# Append parent directory to sys.path to find 'data' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dataset from our data loader
from data.data_loader import windows_dataset

# ================================
# 1. EEGNET ARCHITECTURE
# ================================
class EEGNet(nn.Module):
    """
    EEGNet architecture based on Table 2 of the original paper.
    Strictly defined using __init__ modules for Brevitas quantization compatibility.
    """
    def __init__(self, n_chans, n_outputs, n_times, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # --- Block 1 ---
        # Temporal convolution: F1 filters of size (1, 32), bias=False
        self.conv1 = nn.Conv2d(1, F1, (1, 32), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise spatial convolution: D * F1 filters, size (C, 1)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        # Activation, Pooling and Dropout instantiated as class members
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=0.5)
        
        # --- Block 2 ---
        # Separable Convolution = Depthwise + Pointwise
        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding='same', bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        
        # Activation, Pooling and Dropout
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=0.5)
        
        # --- Classifier ---
        self.flatten = nn.Flatten()
        
        # The time dimension shrinks by a factor of 4 then 8 (total 32)
        out_time = n_times // 32
        self.dense = nn.Linear(F2 * out_time, n_outputs)

    def forward(self, x):
        # Ensure 4D tensor (Batch, 1, Channels, Time)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        return x

# ================================
# 2. TRAINING PIPELINE
# ================================
def main():
    # Force CUDA as requested by the user. Fail fast if GPU is not properly supported.
    device = torch.device('cuda')
    print(f"Device in use: {device}")
    
    # Dataset Splitting
    print("Splitting dataset into train and validation sets...")
    splitted = windows_dataset.split("session")
    train_set = splitted['0train']
    val_set = splitted['1test']
    
    # Infer properties
    try:
        from braindecode.util import infer_signal_properties
        n_chans, n_outputs, n_times = infer_signal_properties(train_set)
    except ImportError:
        # Fallback to manual tensor inspection
        sample_x, sample_y, _ = train_set[0]
        n_chans = sample_x.shape[0]
        n_times = sample_x.shape[1]
        n_outputs = 4  # BNCI2014_001 4-class motor imagery
        
    print(f"Data properties inferred: Channels={n_chans}, Time={n_times}, Outputs={n_outputs}")
    
    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Model Initialization
    model = EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times).to(device)
    print("EEGNet Model Initialized successfully.")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 500
    
    # Early Stopping Configuration
    best_val_loss = float('inf')
    patience = 50
    early_stop_counter = 0
    save_path = '/scratch/fsapere/TinyEEG/models/eegnet_best.pth'
    
    print(f"\nStarting Training Loop (max {epochs} epochs)...")
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for batch_X, batch_y, _ in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_y).sum().item()
            total_train += batch_y.size(0)
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_y).sum().item()
                total_val += batch_y.size(0)
                
        train_acc = train_correct / total_train
        val_acc = val_correct / total_val
        avg_val_loss = val_loss / total_val
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss/total_train:.4f} - Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")
              
        # --- EARLY STOPPING & SAVING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best Val Loss! Model saved to {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}! Best Val Loss was {best_val_loss:.4f}.")
                break

if __name__ == '__main__':
    main()
