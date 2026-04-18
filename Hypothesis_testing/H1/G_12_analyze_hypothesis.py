import matplotlib.pyplot as plt
import os

# Make sure results folder exists
os.makedirs("results", exist_ok=True)

# Your data
sequence_lengths = [128, 256, 512, 768, 1024, 1536, 2048, 3072]

communication_time = [
    5378.13700,
    7739.72800,
    12145.27900,
    30461.67100,
    48886.95100,
    119476.53300,
    177929.79100,
    327110.04400
]

computation_time = [
    0.08400,
    0.18000,
    0.32400,
    0.51600,
    0.80400,
    1.18000,
    1.57200,
    2.34000
]

# ---------------------------
# Plot 1: Communication
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(sequence_lengths, communication_time, marker='o')
plt.xlabel("Sequence Length")
plt.ylabel("Communication Time")
plt.title("Communication vs Sequence Length")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/communication_vs_sequence.png")
plt.close()

# ---------------------------
# Plot 2: Computation
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(sequence_lengths, computation_time, marker='s')
plt.xlabel("Sequence Length")
plt.ylabel("Computation Time")
plt.title("Computation vs Sequence Length")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/computation_vs_sequence.png")
plt.close()

# ---------------------------
# Plot 3: Combined
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(sequence_lengths, communication_time, marker='o', label='Communication')
plt.plot(sequence_lengths, computation_time, marker='s', label='Computation')
plt.xlabel("Sequence Length")
plt.ylabel("Time")
plt.title("Communication vs Computation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/comm_vs_comp.png")
plt.close()

# ---------------------------
# Plot 4: Log scale (important)
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(sequence_lengths, communication_time, marker='o', label='Communication')
plt.plot(sequence_lengths, computation_time, marker='s', label='Computation')
plt.xlabel("Sequence Length")
plt.ylabel("Time (log scale)")
plt.title("Communication vs Computation (Log Scale)")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("results/comm_vs_comp_log.png")
plt.close()

print("Graphs saved inside 'results/' folder ✅")