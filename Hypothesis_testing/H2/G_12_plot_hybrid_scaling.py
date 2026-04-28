import numpy as np
import matplotlib.pyplot as plt

# GPU counts
gpus = np.array([8, 16, 32, 64, 128, 256])
labels = ['2³ (8)', '2⁴ (16)', '2⁵ (32)', '2⁶ (64)', '2⁷ (128)', '2⁸ (256)']

# Model parameters
base_compute = 3200 * 8   # total work
comm_coeff = 120

# Compute runtime values
runtime = []
for n in gpus:
    compute = base_compute / n
    
    if n > 8:
        comm = comm_coeff * np.log2(n) * 1.8
    else:
        comm = comm_coeff * np.log2(n)
    
    runtime.append(compute + comm)

runtime = np.round(runtime)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(labels, runtime, marker='o', linewidth=2)

plt.xlabel("GPUs")
plt.ylabel("Runtime (µs)")
plt.title("Runtime Scaling with GPUs")

plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save the plot
plt.savefig("runtime_scaling.png", dpi=300)

plt.show()
