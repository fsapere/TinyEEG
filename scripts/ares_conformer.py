import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat, Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int32Bias

import sys
sys.path.append("/scratch/fsapere/ARES/ARES")
import ares.nn as ann

LearnedIntWeightPerChannelFloat = Int8WeightPerChannelFloat

class AresQuantizedEEGConformer(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times):
        super(AresQuantizedEEGConformer, self).__init__()
        
        # --- Convolution Module ---
        self.temporal_conv = qnn.QuantConv2d(
            in_channels=1, 
            out_channels=40, 
            kernel_size=(1, 25), 
            stride=(1, 1), 
            bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        
        self.spatial_conv = qnn.QuantConv2d(
            in_channels=40, 
            out_channels=40, 
            kernel_size=(n_chans, 1), 
            stride=(1, 1), 
            bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        
        self.relu = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        
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
            
        # --- Self-Attention Module (ARES Quantized) ---
        self.transformer = nn.Sequential(*[
            ann.TransformerBlock(embed_dim=40, num_heads=10, ff_dim=2048, seq_len=self.seq_len, bit_width=8) 
            for _ in range(6)
        ])
        
        # --- Classifier Module ---
        self.flatten = nn.Flatten()
        
        self.fc1 = qnn.QuantLinear(
            40 * self.seq_len, 64, bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            bias_quant=Int32Bias
        )
        self.fc1_act = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
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
        x = self.relu(x)
        x = self.avg_pool(x)
        
        # Rearrange
        if hasattr(x, 'value'):
            x = x.value
        x = x.squeeze(2)          
        x = x.transpose(1, 2)     
        
        # Self-Attention Module Forward
        x = self.transformer(x)
        
        # Classifier Module Forward
        if hasattr(x, 'value'):
            x = x.value
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        
        return x
