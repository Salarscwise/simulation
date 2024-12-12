# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 06:08:44 2024

@author: Salar Ghaderi
"""

# plotting.py
import numpy as np
import matplotlib.pyplot as plt
import json

def plot_results():
    # Load simulation data
    with open('ising_simulation_data.json', 'r') as f:
        data = json.load(f)
    
    # Extract data
    T_range = np.array(data['temperatures'])
    omegas = np.array(data['frequencies'])
    energies = np.array(data['energies'])
    energy_stds = np.array(data['energy_stds'])
    E_ground = data['ground_state_energy']
    
    # Create plot
    plt.figure(figsize=(14, 10))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(omegas)))
    
    # Plot each frequency
    for i, omega in enumerate(omegas):
        label = 'Static field' if omega == 0 else f'ω = {omega:.3f}'
        plt.errorbar(T_range, energies[i], yerr=energy_stds[i],
                    fmt='o-', color=colors[i], label=label,
                    capsize=3, markersize=4, alpha=0.8)
    
    # Add critical temperature line
    plt.axvline(x=2.269185, color='gray', linestyle='--', label='Tc')
    
    # Labels and title
    plt.xlabel('Temperature (kT/J)', fontsize=12)
    plt.ylabel('Total Lattice Energy (J)', fontsize=12)
    plt.title('Total System Energy vs Temperature for Different Frequencies', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add annotations
    plt.annotate('Ground State\nPerfectly ordered spins\nE = -2N²J - NH', 
                xy=(0.2, E_ground), 
                xytext=(0.5, E_ground*0.9),
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))
                
    plt.annotate('Critical Region\nTc ≈ 2.27',
                xy=(2.269185, -3000),
                xytext=(2.5, -2800),
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))
                
    plt.annotate('High T Region\nDisordered state',
                xy=(3.5, -1800),
                xytext=(3.2, -1500),
                ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))
                
    plt.annotate('Frequency effects\nmost pronounced',
                xy=(2.0, -2500),
                xytext=(1.7, -2200),
                ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('ising_energy_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_results()