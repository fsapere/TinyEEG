#!/bin/bash

# Define paths
ONNX_PATH="/scratch/fsapere/TinyEEG/models/eegnet_int8"
LOG_DIR="/scratch/fsapere/TinyEEG/hardware_logs"

# Create log directory if it does not exist
mkdir -p "$LOG_DIR"

# Navigate to Deeploy installation directory
cd /app/Deeploy || { echo "Error: /app/Deeploy directory not found. Please run this script inside the Singularity container."; exit 1; }

echo "========================================"
echo "Starting benchmark execution on Siracusa"
echo "========================================"

echo ""
echo "[1/4] Run 1: Baseline (1 Core)..."
python DeeployTest/deeployRunner_siracusa.py -t "$ONNX_PATH" --cores=1 > "$LOG_DIR/log_1core.txt" 2>&1
echo "Run 1 completed. Log saved in $LOG_DIR/log_1core.txt"

echo ""
echo "[2/4] Run 2: Parallelism (8 Cores)..."
python DeeployTest/deeployRunner_siracusa.py -t "$ONNX_PATH" --cores=8 > "$LOG_DIR/log_8cores.txt" 2>&1
echo "Run 2 completed. Log saved in $LOG_DIR/log_8cores.txt"

echo ""
echo "[3/4] Run 3: L1 Memory Hierarchy (Tiling, 8 Cores)..."
python DeeployTest/deeployRunner_tiled_siracusa.py -t "$ONNX_PATH" --cores=8 --profileTiling > "$LOG_DIR/log_8cores_tiled.txt" 2>&1
echo "Run 3 completed. Log saved in $LOG_DIR/log_8cores_tiled.txt"

echo ""
echo "[4/4] Run 4: Hardware Accelerator (NEUREKA, 8 Cores, Tiled)..."
python DeeployTest/deeployRunner_tiled_siracusa_w_neureka.py -t "$ONNX_PATH" --cores=8 --neureka-wmem > "$LOG_DIR/log_neureka.txt" 2>&1
echo "Run 4 completed. Log saved in $LOG_DIR/log_neureka.txt"

echo ""
echo "========================================"
echo "All benchmarks have been completed!"
echo "Log files are located in: $LOG_DIR"
echo "========================================"
