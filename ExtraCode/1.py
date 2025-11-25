import numpy as np
import matplotlib.pyplot as plt

# Neuron Parameters
V_threshold = 1.0      # Spike threshold
V_reset = 0.0          # Reset potential
tau_m = 10.0           # Membrane time constant (for leak)
simulation_time = 50   # Total time to simulate (ms)
dt = 0.1               # Simulation time step (ms)

# Input Currents (representing different stimulus strengths)
input_currents = [1.5, 1.2, 0.9, 0.6]
colors = ['red', 'orange', 'green', 'blue']
ttfs = []  # To store the Time to First Spike for each current

# Create time array
time = np.arange(0, simulation_time, dt)

plt.figure(figsize=(10, 6))

for i, I in enumerate(input_currents):
    V = V_reset  # Start at reset potential
    spike_time = None
    membrane_potential = []

    for t in time:
        # Leaky Integrate-and-Fire dynamics
        dV = (-V + I) / tau_m * dt
        V += dV

        # Check for spike
        if V >= V_threshold and spike_time is None:
            spike_time = t  # Record the time of the first spike
            V = V_reset     # Reset membrane potential

        # After the first spike, clamp the voltage to reset
        if spike_time is not None:
            V = V_reset

        membrane_potential.append(V)

    # Plot the membrane potential trace
    plt.plot(time, membrane_potential, color=colors[i], label=f'I={I}, TTFS={spike_time if spike_time else "No Spike"}')
    ttfs.append(spike_time)

plt.axhline(y=V_threshold, color='k', linestyle='--', label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential')
plt.title('Time to First Spike for Different Input Currents')
plt.legend()
plt.grid(True)
plt.show()

print("Times to First Spike:", ttfs)