import matplotlib.pyplot as plt

def main():
    # Data manually curated
    # EEGNet data
    eegnet_mem = [1.77, 1.82, 2.89, 9.23]
    eegnet_acc = [76.74, 79.17, 80.56, 82.50]
    eegnet_labels = ['INT4_Standard', 'MXFP4', 'INT8', 'FP32']
    
    # EEGConformer data
    conf_mem = [4105.14, 4108.74, 4162.95, 4509.4]
    conf_acc = [62.50, 63.54, 64.93, 68.06]
    conf_labels = ['INT4_Standard', 'MXFP4', 'INT8', 'FP32']
    
    plt.figure(figsize=(10, 6))
    
    # Plot lines and points
    plt.plot(eegnet_mem, eegnet_acc, marker='o', markersize=8, linewidth=2, label='EEGNet', color='#1f77b4')
    plt.plot(conf_mem, conf_acc, marker='o', markersize=8, linewidth=2, label='EEGConformer', color='#ff7f0e')
    
    # Set X axis to log scale
    plt.xscale('log')
    
    # Add text labels for EEGNet
    for x, y, label in zip(eegnet_mem, eegnet_acc, eegnet_labels):
        plt.text(x, y + 0.3, label, fontsize=9, ha='center', va='bottom', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # Add text labels for EEGConformer
    for x, y, label in zip(conf_mem, conf_acc, conf_labels):
        plt.text(x, y + 0.3, label, fontsize=9, ha='center', va='bottom', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.title('Pareto Curve: Accuracy vs Memory Footprint', fontsize=14, fontweight='bold')
    plt.xlabel('Memory Footprint (KB) [Log Scale]', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.legend(title='Architecture', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    out_path = '/scratch/fsapere/TinyEEG/pareto_curve.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
