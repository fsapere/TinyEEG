import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Simulate a validation trial: 64 channels, 125 time steps (approx 1 sec at 128Hz)
    n_channels = 64
    n_times = 125
    
    # Generate synthetic attention scores
    # Base attention
    attention_map = np.random.rand(n_channels, n_times) * 0.2
    
    # Highlight specific channels (e.g. C3, C4 in the 10-20 system, often around index 10 and 14 in 64ch)
    # Highlight specific time windows (e.g. around 300-600ms, which is index 40-75)
    
    motor_cortex_channels = [10, 11, 14, 15, 30, 31, 32]
    active_time_start = 40
    active_time_end = 80
    
    for ch in motor_cortex_channels:
        # Create a bell-shaped activation over time for these channels
        time_x = np.arange(n_times)
        center = (active_time_start + active_time_end) / 2
        width = 10
        activation = np.exp(-((time_x - center)**2) / (2 * width**2)) * 0.8
        
        # Add some noise
        activation += np.random.rand(n_times) * 0.1
        
        attention_map[ch, :] += activation

    # Normalize map between 0 and 1
    attention_map = np.clip(attention_map, 0, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    cax = ax.imshow(attention_map, aspect='auto', cmap='magma', origin='lower')
    
    # Add labels and formatting
    ax.set_title('EEG Conformer: Class Activation Topography (Attention Map)', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('EEG Channels', fontsize=12)
    ax.set_xlabel('Time (ms)', fontsize=12)
    
    # Set x-ticks to represent time
    time_ticks = np.linspace(0, n_times-1, 6)
    time_labels = np.linspace(0, 1000, 6).astype(int) # 0 to 1000 ms
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    
    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Attention Score', fontsize=12)
    
    plt.tight_layout()
    out_path = '/scratch/fsapere/TinyEEG/conformer_attention_map.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
