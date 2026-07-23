import matplotlib.pyplot as plt

def main():
    # Mock data based on the parameters
    labels = ['EEGNet', 'EEGConformer']
    
    # Convolutional parameters
    conv_params = [1200, 2000]
    
    # Linear and Attention parameters
    linear_attn_params = [1160, 1152400]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    width = 0.5
    ax.bar(labels, conv_params, width, label='Convolutional Layers', color='#1f77b4')
    ax.bar(labels, linear_attn_params, width, bottom=conv_params, label='Linear & Attention Layers', color='#ff7f0e')
    
    ax.set_ylabel('Number of Parameters (Log Scale)', fontsize=12)
    ax.set_title('Model Parameter Breakdown: EEGNet vs Conformer', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left')
    
    # Set y-axis to log scale because Conformer is much larger
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with exact numbers
    for i in range(len(labels)):
        ax.text(i, conv_params[i] / 2, f"{conv_params[i]}", ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, conv_params[i] + linear_attn_params[i] / 2, f"{linear_attn_params[i]}", ha='center', va='center', color='black' if linear_attn_params[i] < 10000 else 'white', fontweight='bold')

    plt.tight_layout()
    out_path = '/scratch/fsapere/TinyEEG/memory_breakdown.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
