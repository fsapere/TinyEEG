#!/bin/bash

# =======================================================
# Script to generate all final presentation plots
# =======================================================

echo "======================================"
echo " Starting Plot Generation Pipeline... "
echo "======================================"

echo "[1/9] Generating Pareto Curve..."
python scripts/plot_pareto.py
if [ $? -ne 0 ]; then echo "Error in plot_pareto.py"; exit 1; fi

echo "[2/9] Generating Microscaling Impact Plot..."
python scripts/plot_microscaling.py
if [ $? -ne 0 ]; then echo "Error in plot_microscaling.py"; exit 1; fi

echo "[3/9] Generating Training Curves Plot..."
python scripts/plot_training_curves.py
if [ $? -ne 0 ]; then echo "Error in plot_training_curves.py"; exit 1; fi

echo "[4/9] Generating Confusion Matrix Plot..."
python scripts/plot_confusion_matrix.py
if [ $? -ne 0 ]; then echo "Error in plot_confusion_matrix.py"; exit 1; fi

echo "[5/9] Generating Model Memory Breakdown Plot..."
python scripts/plot_memory_breakdown.py
if [ $? -ne 0 ]; then echo "Error in plot_memory_breakdown.py"; exit 1; fi

echo "[6/9] Generating Toolchain Gap Diagram..."
python scripts/plot_toolchain_gap.py
if [ $? -ne 0 ]; then echo "Error in plot_toolchain_gap.py"; exit 1; fi

echo "[7/9] Generating EEGNet Spatial Filters Topoplot..."
python scripts/plot_eegnet_topomap.py
if [ $? -ne 0 ]; then echo "Error in plot_eegnet_topomap.py"; exit 1; fi

echo "[8/9] Generating Conformer Attention Map (CAT)..."
python scripts/plot_conformer_attention.py
if [ $? -ne 0 ]; then echo "Error in plot_conformer_attention.py"; exit 1; fi

echo "[9/9] Generating t-SNE Feature Embeddings..."
python scripts/plot_tsne_embeddings.py
if [ $? -ne 0 ]; then echo "Error in plot_tsne_embeddings.py"; exit 1; fi

echo "======================================"
echo " All 9 plots generated successfully! "
echo " Check the TinyEEG/ folder for PNGs."
echo "======================================"
