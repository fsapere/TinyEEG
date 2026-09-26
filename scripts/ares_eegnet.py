import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat, Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int32Bias

LearnedIntWeightPerChannelFloat = Int8WeightPerChannelFloat

class AresQuantizedEEGNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, F1=8, D=2, F2=16):
        super(AresQuantizedEEGNet, self).__init__()
        
        # BN1 folded into conv1: bias=True
        self.conv1 = qnn.QuantConv2d(
            1, F1, (1, 32), padding='same', bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        
        # BN2 folded into depthwise_conv: bias=True
        self.depthwise_conv = qnn.QuantConv2d(
            F1, F1 * D, (n_chans, 1), groups=F1, bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        
        self.relu1 = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        # Dropout removed for ARES compatibility
        
        # We need a QuantIdentity here if we had concatenation/residual, but it's sequential.
        self.sep_depthwise = qnn.QuantConv2d(
            F1 * D, F1 * D, (1, 16), groups=F1 * D, padding='same', bias=False,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        
        # BN3 folded into sep_pointwise: bias=True
        self.sep_pointwise = qnn.QuantConv2d(
            F1 * D, F2, (1, 1), bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat
        )
        
        self.relu2 = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        # Dropout removed for ARES compatibility
        
        self.flatten = nn.Flatten()
        
        out_time = n_times // 32
        self.dense = qnn.QuantLinear(
            F2 * out_time, n_outputs, bias=True,
            weight_quant=LearnedIntWeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
            bias_quant=Int32Bias
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.relu1(x)
        x = self.avg_pool1(x)
        
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.relu2(x)
        x = self.avg_pool2(x)
        
        if hasattr(x, 'value'):
            x = x.value
            
        x = self.flatten(x)
        x = self.dense(x)
        return x
