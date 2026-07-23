import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the style for the presentation
plt.style.use('seaborn-v0_8-whitegrid')

# Output folder (relative to the script position in TinyEEG/scripts)
output_dir = '../plots'
os.makedirs(output_dir, exist_ok=True)

print("[*] Generating presentation plots...")

# ==========================================
# 1. Accuracy vs Footprint Trade-off (Ablation Study)
# ==========================================
try:
    models_data = {}
    with open('../ablation_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mname = row['Model Name']
            acc = float(row['Max Accuracy'])
            footprint = float(row['Footprint (KB)'])
            
            if mname not in models_data or acc > models_data[mname]['acc']:
                models_data[mname] = {'acc': acc, 'footprint': footprint}

    eegnet_names = []
    eegnet_accs = []
    eegnet_footprints = []
    
    conformer_names = []
    conformer_accs = []
    conformer_footprints = []
    
    for mname, data in models_data.items():
        if 'EEGNet' in mname and 'Conformer' not in mname:
            eegnet_names.append(mname.replace('QuantizedEEGNet', 'EEGNet_INT8'))
            eegnet_accs.append(data['acc'])
            eegnet_footprints.append(data['footprint'])
        elif 'Conformer' in mname:
            conformer_names.append(mname.replace('QuantizedEEGConformer', 'Conformer_INT8'))
            conformer_accs.append(data['acc'])
            conformer_footprints.append(data['footprint'])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy Plot
    ax1 = axes[0]
    ax1.bar(eegnet_names, eegnet_accs, color='#4a90e2', label='EEGNet')
    ax1.bar(conformer_names, conformer_accs, color='#e74c3c', label='Conformer')
    ax1.set_ylabel('Max Accuracy')
    ax1.set_title('Accuracy Comparison by Quantization')
    ax1.set_ylim(0.5, 0.9)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Memory Footprint Plot
    ax2 = axes[1]
    ax2.bar(eegnet_names, eegnet_footprints, color='#4a90e2')
    ax2.bar(conformer_names, conformer_footprints, color='#e74c3c')
    ax2.set_ylabel('Memory Footprint (KB)')
    ax2.set_title('Memory Impact (KB)')
    ax2.set_yscale('log') # Log scale because Conformer is huge
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_footprint.png', dpi=300)
    print("Saved: accuracy_footprint.png")
except Exception as e:
    print(f"Error loading ablation_results.csv: {e}")

# ==========================================
# 2. Latency per Layer (Clock Cycles) - Simulated Data (TO REPLACE WITH GVSOC)
# ==========================================
# Extract these from GVSoC (e.g. make clean all run)
layers = ['Spatial Conv', 'Temporal Conv', 'Depthwise Sep', 'Conformer Attention', 'Dense Out']
cycles_fp32 = [150000, 300000, 120000, 2500000, 50000]
cycles_int8 = [ 45000,  85000,  40000,  450000, 15000] # Much faster due to NE16/RISC-V

x = np.arange(len(layers))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, cycles_fp32, width, label='FP32 (Baseline)', color='#95a5a6')
rects2 = ax.bar(x + width/2, cycles_int8, width, label='INT8 (Quantized)', color='#2ecc71')

ax.set_ylabel('Clock Cycles (GVSOC)')
ax.set_title('Latency Breakdown per Layer')
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/latency_breakdown.png', dpi=300)
print("Saved: latency_breakdown.png")

# ==========================================
# 3. Hardware Efficiency (MACs / Cycle) - Simulated Data
# ==========================================
# How well we are saturating the target accelerator
models = ['EEGNet INT8 (Cores)', 'Conformer INT8 (NE16)']
macs_per_cycle = [2.5, 12.8] # Example: Cluster cores vs NE16 acceleration

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, macs_per_cycle, color=['#3498db', '#9b59b6'])
ax.set_ylabel('MACs / Cycle')
ax.set_title('Hardware Efficiency on GAP9/Siracusa')
ax.set_ylim(0, 16) # NE16 theoretical max depends on architecture

# Add labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval} MAC/c', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/hardware_efficiency.png', dpi=300)
print("Saved: hardware_efficiency.png")

print(f"[*] All plots have been generated in the '{output_dir}/' folder")
