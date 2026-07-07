import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import Brevitas modules
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat, Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int32Bias
from brevitas.export import export_onnx_qcdq

# Map the requested name to the actual Brevitas class
LearnedIntWeightPerChannelFloat = Int8WeightPerChannelFloat

# Append parent directory to sys.path to find 'data' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import windows_dataset


# ==========================================
# 1. QUANTIZED EEGNET ARCHITECTURE
# ==========================================
class QuantizedEEGNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, F1=8, D=2, F2=16):
        super(QuantizedEEGNet, self).__init__()
        
        # --- Block 1 ---
        # Temporal convolution: F1 filters of size (1, 32), bias=False
        self.conv1 = qnn.QuantConv2d(
            1, F1, (1, 32), padding='same', bias=False,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise spatial convolution: D * F1 filters, size (C, 1)
        self.depthwise_conv = qnn.QuantConv2d(
            F1, F1 * D, (n_chans, 1), groups=F1, bias=False,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=0.5)
        
        # --- Block 2 ---
        # Separable Convolution = Depthwise + Pointwise
        self.sep_depthwise = qnn.QuantConv2d(
            F1 * D, F1 * D, (1, 16), groups=F1 * D, padding='same', bias=False,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        self.sep_pointwise = qnn.QuantConv2d(
            F1 * D, F2, (1, 1), bias=False,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        self.bn3 = nn.BatchNorm2d(F2)
        
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=0.5)
        
        # --- Classifier ---
        self.flatten = nn.Flatten()
        
        # The time dimension shrinks by a factor of 4 then 8 (total 32)
        out_time = n_times // 32
        self.dense = qnn.QuantLinear(
            F2 * out_time, n_outputs, bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            bias_quant=Int32Bias
        )

    def forward(self, x):
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


# ==========================================
# 2. QUANTIZED EEG CONFORMER ARCHITECTURE
# ==========================================
class QuantizedEEGConformer(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times):
        super(QuantizedEEGConformer, self).__init__()
        
        # --- Convolution Module ---
        self.temporal_conv = qnn.QuantConv2d(
            in_channels=1, 
            out_channels=40, 
            kernel_size=(1, 25), 
            stride=(1, 1), 
            bias=False,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        
        self.spatial_conv = qnn.QuantConv2d(
            in_channels=40, 
            out_channels=40, 
            kernel_size=(n_chans, 1), 
            stride=(1, 1), 
            bias=False,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
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
            
        # --- Self-Attention Module (FP32) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=40, 
            nhead=10, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # --- Classifier Module ---
        self.flatten = nn.Flatten()
        
        self.fc1 = qnn.QuantLinear(
            40 * self.seq_len, 64, bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            bias_quant=Int32Bias
        )
        self.fc1_act = nn.ELU()
        self.fc2 = qnn.QuantLinear(
            64, n_outputs, bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            bias_quant=Int32Bias
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # Convolution Module Forward
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = self.elu(x)
        x = self.avg_pool(x)
        
        # Rearrange
        x = x.squeeze(2)          
        x = x.transpose(1, 2)     
        
        # Self-Attention Module Forward
        x = self.transformer(x)
        
        # Classifier Module Forward
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        
        return x


# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def calculate_model_footprint(model):
    """
    Calculates total parameters and the theoretical weight memory footprint in KB.
    Weights of quantized layers: 1 byte (INT8)
    Biases of quantized layers: 4 bytes (INT32)
    All other parameters / buffers: 4 bytes (FP32)
    """
    total_params = 0
    total_bytes = 0
    
    # Process parameters
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        
        is_quant_weight = False
        is_quant_bias = False
        
        # Check if the parameter belongs to a quantized layer
        quant_layers = ['conv1', 'depthwise_conv', 'sep_depthwise', 'sep_pointwise', 'dense', 
                        'temporal_conv', 'spatial_conv', 'fc1', 'fc2']
        if any(ql in name for ql in quant_layers):
            if 'weight' in name:
                is_quant_weight = True
            elif 'bias' in name:
                is_quant_bias = True
                
        if is_quant_weight:
            total_bytes += numel * 1  # 8-bit weight = 1 byte
        elif is_quant_bias:
            total_bytes += numel * 4  # 32-bit bias = 4 bytes
        else:
            total_bytes += numel * 4  # Other parameters in FP32 = 4 bytes
            
    # Process buffers (like BatchNorm statistics)
    for name, buf in model.named_buffers():
        if 'running_mean' in name or 'running_var' in name:
            total_bytes += buf.numel() * 4  # FP32 buffer = 4 bytes
            
    return total_params, total_bytes / 1024.0


def log_to_csv(model_name, max_acc, total_params, footprint_kb, csv_path="/scratch/fsapere/TinyEEG/ablation_results.csv"):
    """
    Appends execution statistics to the CSV logging file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model Name", "Max Accuracy", "Total Parameters", "Footprint (KB)"])
        writer.writerow([model_name, f"{max_acc:.4f}", total_params, f"{footprint_kb:.2f}"])
    print(f"[*] Successfully logged {model_name} metrics to {csv_path}")


# ==========================================
# 4. TRAINING FUNCTION
# ==========================================
def train_model(model, train_loader, val_loader, device, save_path, lr, patience=50, epochs=500):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    
    print(f"\nStarting training loop for {model.__class__.__name__}...")
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
            
        train_loss /= total_train
        train_acc = train_correct / total_train
        
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
                
        val_loss /= total_val
        val_acc = val_correct / total_val
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # --- CHECKPOINTING & EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best validation loss! Saved model to {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                break
                
    return best_val_acc


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device in use: {device}")
    
    # Dataset Splitting
    print("Splitting dataset into train and validation sets...")
    splitted = windows_dataset.split("session")
    train_set = splitted['0train']
    val_set = splitted['1test']
    
    # Extract channel, time, and class information
    sample_x, sample_y, _ = train_set[0]
    n_chans = sample_x.shape[0]
    n_times = sample_x.shape[1]
    n_outputs = 4
    
    print(f"Signal properties: Channels={n_chans}, Time={n_times}, Outputs={n_outputs}")
    
    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # ----------------------------
    # A. Train QuantizedEEGNet
    # ----------------------------
    print("\n" + "="*50)
    print("A. TRAINING QUANTIZED EEGNET FROM SCRATCH")
    print("="*50)
    eegnet_model = QuantizedEEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times).to(device)
    eegnet_save_path = '/scratch/fsapere/TinyEEG/models/eegnet_int8_best.pth'
    
    eegnet_best_acc = train_model(
        model=eegnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path=eegnet_save_path,
        lr=0.001,
        patience=50,
        epochs=500
    )
    
    # Calculate footprint and log EEGNet metrics
    eegnet_params, eegnet_footprint = calculate_model_footprint(eegnet_model)
    log_to_csv("QuantizedEEGNet", eegnet_best_acc, eegnet_params, eegnet_footprint)
    
    # ----------------------------
    # B. Train QuantizedEEGConformer
    # ----------------------------
    print("\n" + "="*50)
    print("B. TRAINING QUANTIZED EEG CONFORMER FROM SCRATCH")
    print("="*50)
    conformer_model = QuantizedEEGConformer(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times).to(device)
    conformer_save_path = '/scratch/fsapere/TinyEEG/models/conformer_int8_best.pth'
    
    conformer_best_acc = train_model(
        model=conformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path=conformer_save_path,
        lr=0.0002,
        patience=50,
        epochs=500
    )
    
    # Calculate footprint and log EEGConformer metrics
    conformer_params, conformer_footprint = calculate_model_footprint(conformer_model)
    log_to_csv("QuantizedEEGConformer", conformer_best_acc, conformer_params, conformer_footprint)
    
    # ----------------------------
    # C. ONNX Export (QuantizedEEGNet Only)
    # ----------------------------
    print("\n" + "="*50)
    print("C. EXPORTING QUANTIZED EEGNET TO ONNX (QCDQ)")
    print("="*50)
    
    # Load the best state dict
    print(f"Loading best weights from {eegnet_save_path}...")
    eegnet_model.load_state_dict(torch.load(eegnet_save_path, map_location=device))
    eegnet_model.eval()
    
    # Move model to CPU for ONNX export (standard safety procedure)
    eegnet_model.to('cpu')
    
    dummy_input = torch.randn(1, 1, n_chans, n_times)
    onnx_export_path = '/scratch/fsapere/TinyEEG/models/eegnet_int8.onnx'
    
    try:
        print(f"Exporting model to {onnx_export_path}...")
        export_onnx_qcdq(
            eegnet_model,
            args=dummy_input,
            export_path=onnx_export_path,
            opset_version=14
        )
        print("[*] ONNX export completed successfully!")
    except Exception as e:
        print(f"[!] ONNX export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
