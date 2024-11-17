# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:48:19 2024

@author: Salar
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Simulation settings
L = 20  # Size of the lattice (LxL grid)
J = 1.0  # Interaction strength
k_B = 1.0  # Boltzmann constant
H_0 = 0.1  # Magnetic field amplitude
num_steps = 100000  # Number of simulation steps
T_min = 1.0
T_max = 5.0
num_T = 50  # Number of temperature steps
temperatures = np.linspace(T_min, T_max, num_T)  # Temperature range

# Function to initialize the lattice
def init_lattice(L, random=False):
    return np.random.choice([-1, 1], size=(L, L)) if random else np.ones((L, L), dtype=int)

# Function to compute the total energy
def compute_energy(lattice, J, H):
    energy = 0
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L]
            energy += -J * S * neighbors
    energy -= H * np.sum(lattice)
    return energy

# Metropolis-Hastings algorithm
def metropolis_step(lattice, T, J, H):
    L = lattice.shape[0]
    i, j = np.random.randint(0, L, 2)
    S = lattice[i, j]
    neighbors = (
        lattice[(i + 1) % L, j] +
        lattice[(i - 1) % L, j] +
        lattice[i, (j + 1) % L] +
        lattice[i, (j - 1) % L]
    )
    dE = 2 * J * S * neighbors + 2 * H * S
    if dE <= 0 or np.random.rand() < np.exp(-dE / (k_B * T)):
        lattice[i, j] *= -1
        return dE, -2 * S
    else:
        return 0, 0

# Function to calculate the total magnetization
def compute_magnetization(lattice):
    return np.sum(lattice)

# Function to compute the time-dependent field
def time_dependent_field(t, period):
    return H_0 * np.sin(2 * np.pi * t / period)

# Simulate magnetization and energy for different field periods and temperatures
periods = [50, 100, 200]  # Field periods to simulate
results = {}
colors = {50: "purple", 100: "green", 200: "red"}  # Field period colors

start_time = time.time()  # Track start time

# Total number of steps in the entire simulation
total_steps = len(periods) * num_T * num_steps

step_count = 0  # Counter for overall progress

for period in periods:
    magnetization_avg = []
    energy_avg = []
    for T_idx, T in enumerate(temperatures):
        lattice = init_lattice(L)
        E = compute_energy(lattice, J, 0)
        M = compute_magnetization(lattice)

        M_total = 0
        E_total = 0

        # Loop over time steps
        for t in range(num_steps):
            H = time_dependent_field(t, period)
            dE, dM = metropolis_step(lattice, T, J, H)
            E += dE
            M += dM
            M_total += M
            E_total += E

            # Update progress display
            step_count += 1
            elapsed = time.time() - start_time
            est_total_time = elapsed / step_count * total_steps
            print(f"\rTemp: {T:.2f} | {((step_count) / total_steps) * 100:.1f}% Complete | Time: {est_total_time:.1f}s", end="")

        # Average over all time steps
        M_avg = M_total / (num_steps * L * L)
        E_avg = E_total / (num_steps * L * L)
        magnetization_avg.append(M_avg)
        energy_avg.append(E_avg)

    results[period] = {
        "temperature": temperatures,
        "magnetization": magnetization_avg,
        "energy": energy_avg,
    }

# Plot magnetization vs temperature for different periods
plt.figure(figsize=(12, 6))
for period, result in results.items():
    plt.plot(result["temperature"], result["magnetization"], label=f"Period = {period}", color=colors[period])
plt.xlabel("Temperature (T / k_BJ)")
plt.ylabel("Magnetization per spin")
plt.title("Magnetization vs Temperature for Different Field Periods")
plt.legend()
plt.savefig("SinH E-T (S)")
plt.show()

# Plot energy vs temperature for different periods
plt.figure(figsize=(12, 6))
for period, result in results.items():
    plt.plot(result["temperature"], result["energy"], label=f"Period = {period}", color=colors[period])
plt.xlabel("Temperature (T / k_BJ)")
plt.ylabel("Energy per spin")
plt.title("Energy vs Temperature for Different Field Periods")
plt.legend()
plt.savefig("SinH E-T (S)")
plt.show()
