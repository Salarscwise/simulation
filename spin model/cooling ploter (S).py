
"""
@author: Salar
"""

import numpy as np
import matplotlib.pyplot as plt

# Load results from file
data = np.load('cooling_simulation_results.npz', allow_pickle=True)
temp_cooling = data['temp_cooling']
results = data['results'].item()
energy_results = data['energy_results'].item()
total_time = data['total_time']

# Display the timing
print(f"Total simulation time: {total_time:.2f} seconds.")

# Plot magnetization vs temperature
plt.figure(figsize=(8, 6))
for field, magnetization in results.items():
    plt.plot(temp_cooling, magnetization, label=f"H = {field:.2f}")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.gca().invert_xaxis()
plt.xlabel("Temperature (T / k_BJ)")
plt.ylabel("Magnetization per spin")
plt.title("Cooling: Magnetization vs Temperature")
plt.legend()

final_temp = temp_cooling[-1]
plt.annotate(
    "Fluctuates for H=0\nfrom negative to positive",
    xy=(final_temp, 0),
    xytext=(final_temp + 2, -0.3),
    arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="lightyellow")
)
plt.savefig("cooling M-T (S)")
plt.show()

# Plot energy vs temperature
plt.figure(figsize=(8, 6))
for field, energy in energy_results.items():
    plt.plot(temp_cooling, energy, label=f"H = {field:.2f}")
plt.gca().invert_xaxis()
plt.xlabel("Temperature (T / k_BJ)")
plt.ylabel("Energy per spin")
plt.title("Cooling: Energy vs Temperature")
plt.legend()
plt.savefig("cooling E-T (S)")
plt.show()
