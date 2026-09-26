import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

def main():
    # Hardware constraints and actual usage
    memories = ['L1 Cache (TCDM)', 'L2 Cache (SRAM)']
    usage = [115.4, 117.0]
    limits = [128.0, 2048.0]  # Hardware physical limits (Siracusa/GAP9 specs)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate percentages
    percentages = [u/l * 100 for u, l in zip(usage, limits)]
    
    # Plot bars
    bars = ax.bar(memories, usage, width=0.4, color=['#e74c3c', '#3498db'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Draw limit lines
    for i, limit in enumerate(limits):
        ax.hlines(y=limit, xmin=i-0.25, xmax=i+0.25, color='black', linestyle='--', linewidth=2.5, zorder=3)
        ax.text(i+0.28, limit, f'HW Limit\n({limit} KB)', va='center', ha='left', fontsize=11, fontweight='bold', color='black')

    ax.set_ylabel('Memory Utilization (KB)', fontsize=13, fontweight='bold')
    ax.set_title('Deployed EEGNet: Hardware Memory Footprint (GVSoC)', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, max(limits) * 1.1)
    
    # Annotate bars
    for bar, percent, u in zip(bars, percentages, usage):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{u} KB\n({percent:.1f}%)',
                ha='center', va='center', color='white', fontweight='bold', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3))

    plt.tight_layout()
    out_path = f'{output_dir}/hardware_memory_utilization.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
