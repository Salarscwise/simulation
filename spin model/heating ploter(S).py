
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load results
with open("ising_results.pkl", "rb") as f:
    results = pickle.load(f)

# Define constants
kB = 1  # Boltzmann constant (in units where k_B = 1)
J = 1   # Interaction energy (in units where J = 1)
Tc = 2.269185  # Critical temperature

# Analytical magnetization function
def analytical_magnetization(temp):
    z = np.exp(-2 * J / (kB * temp))
    M = np.zeros_like(temp)
    below_Tc = temp < Tc
    M[below_Tc] = (
        (1 + z[below_Tc]**2)**0.25 *
        (1 - 6*z[below_Tc]**2 + z[below_Tc]**4)**0.125 *
        np.sqrt(1 - z[below_Tc]**2)
    )
    return M

# Temperatures for analytical curve
temp_analytical = np.linspace(1.5, 3.5, 500)
mag_analytical = analytical_magnetization(temp_analytical)

# Magnetization plot with analytical comparison
plt.figure(figsize=(8, 6))
for field, data in results.items():
    plt.plot(data["temp"], data["magnetization"], label=f"Simulated: H = {field:.2f}")
plt.plot(temp_analytical, mag_analytical, 'k--', label="Analytical")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Temperature (T / k_BJ)")
plt.ylabel("Magnetization per spin")
plt.title("Magnetization vs Temperature for Various Fields")
plt.legend()
plt.savefig("heating_M-T_comparison.png")
plt.show()

# Energy plot
plt.figure(figsize=(8, 6))
for field, data in results.items():
    plt.plot(data["temp"], data["energy"], label=f"H = {field:.2f}")
plt.xlabel("Temperature (T / k_BJ)")
plt.ylabel("Energy per spin")
plt.title("Energy vs Temperature for Various Fields")
plt.legend()
plt.savefig("heating_E-T.png")
plt.show()

