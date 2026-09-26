# /scratch/fsapere/TinyEEG/deploy_ares.py
import sys
import os
import torch
from pathlib import Path

# 1. Add ARES to the Python path to import its tools
ares_path = "/scratch/fsapere/ARES/ARES"
if ares_path not in sys.path:
    sys.path.append(ares_path)

from tools.pytorch_extractor import BrevitasExtractor
from tools.generate_golden_outputs import GoldenOutputGenerator
from codegen.generate_c_code import CCodeGenerator
# NOTE: Import your QuantizedEEGNet class from your file (assuming it's called test_export.py or similar)
from test_export import QuantizedEEGNet 

def main():
    target_hardware = "siracusa" # or "gap9"
    artifacts_dir = Path("ares_artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    print("[1/3] Loading PyTorch model...")
    # Recreate the model with the same dimensions as training
    base_model = QuantizedEEGNet(22, 4, 576) 
    
    # Load the best weights saved by the training script
    checkpoint = torch.load("models/eegnet_int8_best.pth", map_location='cpu')
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_model.load_state_dict(checkpoint)
    base_model.eval()

    # ARES requires an explicit QuantIdentity layer at the beginning to handle the FP32 input.
    # Brevitas did this internally in conv1, but ARES wants a standalone layer. We add it:
    import torch.nn as nn
    from brevitas.nn import QuantIdentity
    from brevitas.quant import Int8ActPerTensorFloat
    class ARESWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.input_quant = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
            self.model = m
        def forward(self, x):
            return self.model(self.input_quant(x))
            
    model = ARESWrapper(base_model)
    # Execute a dummy forward pass to initialize the dummy QuantIdentity
    with torch.no_grad():
        model(torch.randn(1, 1, 22, 576))
    
    print(f"\n[2/3] Extracting parameters with ARES (BrevitasExtractor)...")
    extractor = BrevitasExtractor(model)
    
    # Generate a dummy input with the real EEG dimensions
    dummy_input = torch.randn(1, 1, 22, 576)
    
    # Pass the dummy input to the extractor to correctly compute the shapes
    network_info = extractor.extract_all(sample_input=dummy_input)
    extractor._compute_residency_policy() # Calculate whether weights go to L2 or L3
    
    # PATCH FOR ARES: ARES does not support PyTorch's native padding='same' strings.
    # Even kernels (32 and 16) with asymmetric 'same' in PyTorch cause a dimension reduction
    # if rounded down in ARES (which only supports symmetric padding).
    # We round up (k // 2) to ensure the final tensor reaches exactly 288 features.
    for layer_name, layer in network_info.items():
        if 'padding' in layer and layer['padding'] == 'same':
            k_h, k_w = layer['kernel_size']
            layer['padding'] = (k_h // 2, k_w // 2)
    
    # Save json and weights for C generation
    extractor.save_to_json(artifacts_dir / "network_info.json")
    extractor.save_weights(artifacts_dir / "weights")
    
    # PATCH FOR ARES (BIAS): CCodeGenerator assumes each convolution has a bias.
    # We create dummy biases (zero arrays) for layers with bias=False to avoid KeyErrors.
    for layer_name, layer in network_info.items():
        if layer.get('type') in ['QuantConv2d', 'QuantLinear']:
            bias_path = artifacts_dir / "weights" / f"{layer_name}_bias_fp32.npy"
            if not bias_path.exists():
                out_ch = layer.get('out_channels', layer.get('out_features', 0))
                import numpy as np
                np.save(bias_path, np.zeros(out_ch, dtype=np.float32))
    
    print(f"\n[2.5/3] Generating Golden Outputs...")
    np_input = dummy_input.numpy()
    golden_gen = GoldenOutputGenerator(network_info)
    # Pass a list containing our single test case
    golden_gen.generate_golden_outputs([np_input], output_dir=artifacts_dir)
    test_case_dir = artifacts_dir / "test_case_1"
    
    # CCodeGenerator expects files to be named input0_* even if there's only one input
    import shutil
    if (test_case_dir / "input_fp32.npy").exists():
        shutil.copy(test_case_dir / "input_fp32.npy", test_case_dir / "input0_fp32.npy")
    if (test_case_dir / "intermediate_int8" / "input_int8.npy").exists():
        shutil.copy(test_case_dir / "intermediate_int8" / "input_int8.npy", test_case_dir / "intermediate_int8" / "input0_int8.npy")
    
    print(f"\n[3/3] Generating C Code for {target_hardware}...")
    # Initialize the generator by passing the newly created folders
    generator = CCodeGenerator(
        target_name=target_hardware,
        network_info_path=artifacts_dir / "network_info.json",
        weights_dir=artifacts_dir / "weights",
        test_case_dir=test_case_dir,
        output_dir=artifacts_dir / "c_project"
    )
    generator.generate_all()
    
    print("\nGeneration completed! The C project is located in ares_artifacts/c_project")

if __name__ == "__main__":
    main()
