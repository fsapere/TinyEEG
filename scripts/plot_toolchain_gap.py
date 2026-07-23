import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Define blocks
    blocks = [
        {'name': 'ONNX\nModel', 'xy': (0.05, 0.4), 'color': '#ccebc5'},
        {'name': 'Deeploy\nParser', 'xy': (0.25, 0.4), 'color': '#b3cde3'},
        {'name': 'Mapper\n(Binding)', 'xy': (0.45, 0.4), 'color': '#fbb4ae'},
        {'name': 'C Kernel\nGeneration', 'xy': (0.75, 0.4), 'color': '#decbe4'}
    ]
    
    # Draw blocks
    for b in blocks:
        rect = patches.FancyBboxPatch(b['xy'], 0.15, 0.2, boxstyle="round,pad=0.05", 
                                      edgecolor='black', facecolor=b['color'], lw=2)
        ax.add_patch(rect)
        ax.text(b['xy'][0] + 0.075, b['xy'][1] + 0.1, b['name'], 
                ha='center', va='center', fontsize=11, fontweight='bold')
                
    # Draw arrows
    style = "Simple, tail_width=0.5, head_width=6, head_length=8"
    kw = dict(arrowstyle=style, color="k")
    
    # ONNX -> Parser
    arr1 = patches.FancyArrowPatch((0.2, 0.5), (0.25, 0.5), **kw)
    ax.add_patch(arr1)
    
    # Parser -> Mapper
    arr2 = patches.FancyArrowPatch((0.4, 0.5), (0.45, 0.5), **kw)
    ax.add_patch(arr2)
    
    # Mapper -> C Kernel
    arr3 = patches.FancyArrowPatch((0.6, 0.5), (0.75, 0.5), **kw)
    ax.add_patch(arr3)
    
    # NEMO branch (Success)
    kw_nemo = dict(arrowstyle=style, color="green", connectionstyle="arc3,rad=-0.3")
    arr_nemo = patches.FancyArrowPatch((0.2, 0.65), (0.5, 0.65), **kw_nemo)
    ax.add_patch(arr_nemo)
    ax.text(0.35, 0.8, "NEMO (RequantizedConv)", ha='center', va='center', color='green', fontweight='bold', fontsize=10)
    ax.text(0.51, 0.63, "✓", ha='center', va='center', color='green', fontweight='bold', fontsize=18)
    
    # Brevitas branch (Fail)
    kw_brev = dict(arrowstyle=style, color="red", connectionstyle="arc3,rad=0.3")
    arr_brev = patches.FancyArrowPatch((0.2, 0.35), (0.5, 0.35), **kw_brev)
    ax.add_patch(arr_brev)
    ax.text(0.35, 0.2, "Brevitas (QOps/QLinearConv)", ha='center', va='center', color='red', fontweight='bold', fontsize=10)
    ax.text(0.51, 0.37, "✗", ha='center', va='center', color='red', fontweight='bold', fontsize=18)
    
    plt.title('Deeploy Toolchain Gap for Quantized Operations', fontsize=15, fontweight='bold', y=1.05)
    
    out_path = '/scratch/fsapere/TinyEEG/toolchain_gap_diagram.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
