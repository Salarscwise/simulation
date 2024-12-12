# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 06:17:59 2024

@author: Salar Ghaderi
"""

# magnetization_plotting.py
import numpy as np
import matplotlib.pyplot as plt
import json

def plot_results():
    # Load simulation data
    with open('ising_magnetization_data.json', 'r') as f:
        data = json.load(f)
    
    # Extract data
    T_range = np.array(data['temperatures'])
    omegas = np.array(data['frequencies'])
    magnetizations = np.array(data['magnetizations'])
    mag_stds = np.array(data['magnetization_stds'])
    M_ground = data['ground_state_magnetization']
    
    # Create plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(omegas)))
    
    # Plot each frequency
    for i, omega in enumerate(omegas):
        label = 'Static field' if omega == 0 else f'ω = {omega:.3f}'
        plt.errorbar(T_range, magnetizations[i]/M_ground, yerr=mag_stds[i]/M_ground,
                    fmt='o-', color=colors[i], label=label,
                    capsize=3, markersize=4, alpha=0.8)
    
    # Add critical temperature line
    plt.axvline(x=2.269185, color='gray', linestyle='--', label='Tc')
    
    # Labels and title
    plt.xlabel('Temperature (kT/J)', fontsize=12)
    plt.ylabel('m = |M|/M₀', fontsize=12)
    plt.title('Normalized Magnetization vs Temperature', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add annotations
    plt.annotate('Ordered Phase\nm ≈ 1', 
                xy=(1.0, 0.9),
                xytext=(1.2, 0.8),
                ha='center', va='center',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate('Critical Point\nTc ≈ 2.27',
                xy=(2.269185, 0.5),
                xytext=(2.6, 0.5),
                ha='left', va='center',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate('Disordered Phase\nm ≈ 0',
                xy=(3.5, 0.1),
                xytext=(3.3, 0.2),
                ha='right', va='center',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))
    
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('ising_magnetization_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_results()