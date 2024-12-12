# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 06:17:04 2024

@author: Salar Ghaderi
"""

# magnetization_simulation.py
import numpy as np
from numba import jit
from tqdm import tqdm
import json

@jit(nopython=True)
def calculate_magnetization(lattice):
    """Calculate absolute magnetization."""
    return abs(np.sum(lattice))

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
    
    # Temperature range with focus on critical region
    T_range = np.concatenate([
        np.array([0.0]),  # Explicitly include T=0
        np.linspace(0.2, 2.0, 20),  # More points before Tc
        np.linspace(2.0, 2.5, 15),  # Dense around Tc
        np.linspace(2.5, 4.0, 15)   # Fewer points after Tc
    ])
    beta_range = np.where(T_range != 0, 1/T_range, np.inf)
    
    # Selected frequencies
    omegas = np.array([0.0, 0.01, 0.02, 0.05])
    
    # Simulation steps
    thermalization = 5000
    measure_steps = 5000
    measures_per_T = 5
    
    # Store results
    magnetizations = np.zeros((len(omegas), len(T_range)))
    mag_stds = np.zeros((len(omegas), len(T_range)))
    
    # Ground state magnetization
    M_ground = N * N
    
    for w_idx, omega in enumerate(tqdm(omegas, desc="Frequencies")):
        for t_idx, (T, beta) in enumerate(tqdm(zip(T_range, beta_range), 
                                              desc="Temperature", 
                                              total=len(T_range))):
            mag_samples = np.zeros(measures_per_T)
            
            for m in range(measures_per_T):
                lattice = np.ones((N, N))
                
                if T == 0:
                    mag_samples[m] = M_ground
                    continue
                    
                # Thermalization
                for step in range(thermalization):
                    H = H_amp if omega == 0 else H_amp * np.sin(omega * step)
                    metropolis_step(lattice, beta, J, H)
                
                # Measurement with proper period handling
                M_total = 0
                measure_count = 0
                
                if omega > 0:
                    field_period = int(2 * np.pi / omega)
                    num_periods = max(10, measure_steps // field_period)
                    actual_steps = num_periods * field_period
                else:
                    actual_steps = measure_steps
                
                for step in range(actual_steps):
                    H = H_amp if omega == 0 else H_amp * np.sin(omega * (step + thermalization))
                    metropolis_step(lattice, beta, J, H)
                    
                    if step % 10 == 0:
                        M = calculate_magnetization(lattice)
                        M_total += M
                        measure_count += 1
                
                mag_samples[m] = M_total / measure_count
            
            magnetizations[w_idx, t_idx] = np.mean(mag_samples)
            mag_stds[w_idx, t_idx] = np.std(mag_samples)
    
    # Save results
    simulation_data = {
        'temperatures': T_range.tolist(),
        'frequencies': omegas.tolist(),
        'magnetizations': magnetizations.tolist(),
        'magnetization_stds': mag_stds.tolist(),
        'ground_state_magnetization': M_ground,
        'parameters': {
            'N': N,
            'J': J,
            'H_amp': H_amp,
            'Tc': 2.269185
        }
    }
    
    with open('ising_magnetization_data.json', 'w') as f:
        json.dump(simulation_data, f)

if __name__ == "__main__":
    run_simulation()