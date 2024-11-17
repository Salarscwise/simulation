

"""
@author: Salar
"""

import numpy as np
import pickle
import time

# Simulation parameters
grid_size = 20  # Grid dimensions (grid_size x grid_size)
interaction_strength = 1.0  # Interaction strength (rescaled units)
boltzmann_const = 1.0  # Boltzmann constant (rescaled units)
steps_per_temp = 1000000  # Simulation steps per temperature

# Critical temperature for the 2D Ising model in rescaled units
critical_temp = 2.0 / np.log(1 + np.sqrt(2))  # ~2.269185

# Temperature range and steps
temp_min = 1.0
temp_max = 5.0
temp_steps_count = 50
temp_steps = np.linspace(temp_min, temp_max, temp_steps_count)

# Initialize lattice spins
def init_lattice(size, random_spins=False):
    # Create a lattice with either random spins or all +1
    return (
        np.random.choice([-1, 1], size=(size, size))
        if random_spins
        else np.ones((size, size), dtype=int)
    )

# Compute total energy of the lattice
def compute_energy(lattice, J, ext_field):
    total_energy = 0
    size = lattice.shape[0]
    for x in range(size):
        for y in range(size):
            spin = lattice[x, y]
            neighbors = lattice[(x + 1) % size, y] + lattice[x, (y + 1) % size]+ lattice[x, (y - 1) % size]+lattice[(x - 1) % size, y]
            total_energy += -J * spin * neighbors
    total_energy -= ext_field * np.sum(lattice)
    return total_energy

# Metropolis-Hastings update step
def metropolis_update(lattice, temp, J, ext_field):
    size = lattice.shape[0]
    x, y = np.random.randint(0, size, 2)
    spin = lattice[x, y]
    neighbors = (
        lattice[(x + 1) % size, y]
        + lattice[(x - 1) % size, y]
        + lattice[x, (y + 1) % size]
        + lattice[x, (y - 1) % size]
    )
    energy_change = 2 * J * spin * neighbors + 2 * ext_field * spin
    if energy_change <= 0 or np.random.rand() < np.exp(-energy_change / (boltzmann_const * temp)):
        lattice[x, y] *= -1
        return energy_change, -2 * spin
    return 0, 0

# Calculate total magnetization
def calc_magnetization(lattice):
    return np.sum(lattice)

# External field values for testing
field_values = [0.0, 0.1, -0.1]

# Simulation results
results = {}

for field in field_values:
    magnetization_data = []
    energy_data = []

    # Initialize lattice
    lattice = init_lattice(grid_size, random_spins=(field < 0))
    energy = compute_energy(lattice, interaction_strength, field)
    magnetization = calc_magnetization(lattice)

    print(f"\nSimulating for field H = {field:.2f}")
    start = time.time()

    for idx, temp in enumerate(temp_steps):
        total_energy = 0
        total_magnetization = 0

        for _ in range(steps_per_temp):
            delta_energy, delta_magnetization = metropolis_update(lattice, temp, interaction_strength, field)
            energy += delta_energy
            magnetization += delta_magnetization
            total_energy += energy
            total_magnetization += magnetization

        avg_energy = total_energy / steps_per_temp / (grid_size ** 2)
        avg_magnetization = total_magnetization / steps_per_temp / (grid_size ** 2)
        energy_data.append(avg_energy)
        magnetization_data.append(avg_magnetization)

        # Progress feedback
        elapsed = time.time() - start
        est_total_time = elapsed / (idx + 1) * temp_steps_count
        print(f"\rTemp: 0 to 5 | {((idx + 1) / temp_steps_count) * 100:.1f}% Complete | Time: {est_total_time:.1f}s", end="")

    results[field] = {"temp": temp_steps, "magnetization": magnetization_data, "energy": energy_data}

# Save results to a file
output_file = "ising_results.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"\nSimulation results saved to {output_file}")
