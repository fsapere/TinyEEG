import matplotlib.pyplot as plt

def main():
    # Filtered Conformer data
    formats = ['FP32 (Baseline)', 'INT4 Standard', 'MXFP4 (Microscaling)']
    accuracies = [68.06, 62.50, 63.54]
    
    # Strategic colors: Grey for baseline, Red for drop, Green for recovery
    colors = ['#8c8c8c', '#d62728', '#2ca02c']
    
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(formats, accuracies, color=colors, width=0.6)
    
    # Add labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.title('Impact of Microscaling on EEGConformer', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.ylim(55, 75) # Set limits to highlight the differences
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    out_path = '/scratch/fsapere/TinyEEG/conformer_mx_impact.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
