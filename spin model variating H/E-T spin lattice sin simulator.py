# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 06:08:03 2024

@author: Salar Ghaderi
"""

# simulation.py
import numpy as np
from numba import jit
from tqdm import tqdm
import json

@jit(nopython=True)
def calculate_total_energy(lattice, J, H):
    """Calculate total lattice energy correctly."""
    N = len(lattice)
    E = 0.0
    for i in range(N):
        for j in range(N):
            E -= J * lattice[i,j] * (
                lattice[i, (j+1)%N] + 
                lattice[(i+1)%N, j]
            )
    E -= H * np.sum(lattice)
    return E

@jit(nopython=True)
def metropolis_step(lattice, beta, J, H):
    """Single Metropolis step with external field."""
    N = len(lattice)
    for _ in range(N*N):
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        spin = lattice[i,j]
        local_field = (lattice[i, (j+1)%N] + 
                      lattice[i, (j-1)%N] + 
                      lattice[(i+1)%N, j] + 
                      lattice[(i-1)%N, j])
        dE = 2 * spin * (J * local_field + H)
        if beta == np.inf:
            if dE < 0:
                lattice[i,j] = -spin
        else:
            if dE <= 0 or np.random.random() < np.exp(-beta * dE):
                lattice[i,j] = -spin

def run_simulation():
    # System parameters
    N = 50
    J = 1.0
    H_amp = 0.1
    
    # Temperature range
    T_low = np.linspace(0, 0.5, 5)
    T_mid = np.linspace(0.5, 2.0, 15)
    T_high = np.linspace(2.0, 4.0, 20)
    T_range = np.unique(np.concatenate([T_low, T_mid, T_high]))
    beta_range = np.where(T_range != 0, 1/T_range, np.inf)
    
    # Frequencies
    omegas = np.array([0.0, 0.01, 0.02, 0.05])
    
    # Simulation steps
    thermalization = 6000
    measure_steps = 9000
    measures_per_T = 5
    
    # Results storage
    energies = np.zeros((len(omegas), len(T_range)))
    energy_stds = np.zeros((len(omegas), len(T_range)))
    E_ground = -2 * N * N * J - N * N * H_amp
    
    # Main simulation loop
    for w_idx, omega in enumerate(tqdm(omegas, desc="Frequencies")):
        for t_idx, (T, beta) in enumerate(tqdm(zip(T_range, beta_range), 
                                              desc="Temperature", 
                                              total=len(T_range))):
            energy_samples = np.zeros(measures_per_T)
            
            for m in range(measures_per_T):
                lattice = np.ones((N, N))
                
                if T == 0:
                    if omega == 0:
                        energy_samples[m] = E_ground
                    else:
                        energy_samples[m] = -2 * N * N * J
                else:
                    # Thermalization
                    for step in range(thermalization):
                        H = H_amp * np.sin(omega * step) if omega > 0 else H_amp
                        metropolis_step(lattice, beta, J, H)
                    
                    # Measurement
                    E_total = 0
                    measure_count = 0
                    for step in range(measure_steps):
                        H = H_amp * np.sin(omega * (step + thermalization)) if omega > 0 else H_amp
                        metropolis_step(lattice, beta, J, H)
                        if step % 10 == 0:
                            E = calculate_total_energy(lattice, J, H)
                            E_total += E
                            measure_count += 1
                    
                    energy_samples[m] = E_total / measure_count
            
            energies[w_idx, t_idx] = np.mean(energy_samples)
            energy_stds[w_idx, t_idx] = np.std(energy_samples)
    
    # Save results
    simulation_data = {
        'temperatures': T_range.tolist(),
        'frequencies': omegas.tolist(),
        'energies': energies.tolist(),
        'energy_stds': energy_stds.tolist(),
        'ground_state_energy': E_ground,
        'parameters': {
            'N': N,
            'J': J,
            'H_amp': H_amp,
            'Tc': 2.269185
        }
    }
    
    with open('ising_simulation_data.json', 'w') as f:
        json.dump(simulation_data, f)

if __name__ == "__main__":
    run_simulation()