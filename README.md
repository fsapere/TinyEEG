# TinyEEG
Microscaling & Quantization on EEG-related CNN and Transformers.

This repository contains the workflow for training, quantizing, and deploying EEG-based neural networks (EEGNet and Conformer) on the Siracusa hardware platform.

## Project Structure and Workflow

1. Training and Quantization
The models are defined and trained using PyTorch and Brevitas. The training scripts apply INT8/sub-byte quantization to prepare the network for edge deployment.

2. Code Generation
The `deploy_ares.py` and `gen_only.py` scripts extract the quantized network parameters and leverage the ARES toolchain to generate the optimized C project.

3. Hardware Simulation
The generated C code is compiled and simulated on GVSoC (the Siracusa platform simulator). The `run_sim_full.sh` script handles the execution and extracts the architectural profiling metrics, including clock cycles and memory allocation (L1/L2).

4. Profiling and Plotting
The execution benchmarks and ablation results are collected in `ablation_results.csv` and the `hardware_logs/` directory. The `run_all_plots.sh` script parses this data to generate presentation-ready plots (latency, hardware efficiency, memory footprint) saved in the `plots/` folder.

## Key Optimizations
- Resolved L1 memory leak during CHW convolution tiling by implementing a safe fallback for single-tile execution.
- Bypassed hardware FPU limitations by converting runtime floating-point requantization into integer-only arithmetic using pre-calculated offline multipliers and bit-shifts.

## Hardware Profiling Results (GVSoC)
The fully quantized INT8 pipeline has been profiled on the Siracusa simulated platform, yielding the following performance metrics:

- **Total Inference Latency:** 19.84 Million clock cycles
- **L1 Memory Utilization:** 115.4 KB (TCDM buffering for tiles and DMA)
- **L2 Memory Utilization:** ~117 KB (114.2 KB activation arena + static weights)
- **Peak Hardware Efficiency:** Up to 12.8 MACs/cycle utilizing target accelerators (NE16) vs 2.5 MACs/cycle on standard cluster cores.
