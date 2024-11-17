"""
@author: Salar
"""

import numpy as np
import time

# Simulation parameters
grid_size = 20
interaction_strength = 1.0
boltzmann_const = 1.0
steps_per_temp = 100000
temp_high = 10.0
temp_low = 0.1
cooling_steps = 100
temp_cooling = np.linspace(temp_high, temp_low, cooling_steps)

# Initialize lattice
def init_lattice(size, random_spins=False):
    return (
        np.random.choice([-1, 1], size=(size, size))
        if random_spins
        else np.ones((size, size), dtype=int)
    )

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

# Compute magnetization
def calc_magnetization(lattice):
    return np.sum(lattice) / (grid_size ** 2)

# External field values
field_values = [0.0, 0.1, -0.1]
results = {}
energy_results = {}

total_simulation_steps = len(field_values) * cooling_steps
current_step = 0
start_time = time.time()

for field in field_values:
    magnetization_data = []
    energy_data = []

    lattice = init_lattice(grid_size, random_spins=True)

    for idx, temp in enumerate(temp_cooling):
        for _ in range(steps_per_temp):
            metropolis_update(lattice, temp, interaction_strength, field)

        magnetization_data.append(calc_magnetization(lattice))
        energy = (
            -interaction_strength * np.sum(
                lattice
                * (
                    np.roll(lattice, 1, axis=0)
                    + np.roll(lattice, -1, axis=0)
                    + np.roll(lattice, 1, axis=1)
                    + np.roll(lattice, -1, axis=1)
                )
            )
            / 2
            - field * np.sum(lattice)
        )
        energy_data.append(energy / (grid_size ** 2))

        current_step += 1
        percent_complete = (current_step / total_simulation_steps) * 100
        elapsed = time.time() - start_time
        print(f"\rSimulation Progress: {percent_complete:.1f}% | Elapsed: {elapsed:.2f}s", end="")

    results[field] = magnetization_data
    energy_results[field] = energy_data

total_time = time.time() - start_time
print(f"\nSimulation completed in {total_time:.2f} seconds.")

# Save results to a file
np.savez('cooling_simulation_results.npz', temp_cooling=temp_cooling, results=results, energy_results=energy_results, total_time=total_time)
print("Simulation results saved to 'cooling_simulation_results.npz'.")
