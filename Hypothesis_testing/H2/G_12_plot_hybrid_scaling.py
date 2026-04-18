import pandas as pd
import matplotlib.pyplot as plt
import os
import re

RESULT_DIR = "/home/iiitd/Desktop/SimAI/results"

GPU_LIST = [8,16,32,64,128,256]

records = []

# ------------------------------------------------
# Parse CSV summary values
# ------------------------------------------------
for g in GPU_LIST:
    file = f"{RESULT_DIR}/hybrid_g{g}_EndToEnd.csv"

    if not os.path.exists(file):
        print("Missing:", file)
        continue

    with open(file, "r") as f:
        txt = f.read()

    # Try to extract numeric metrics
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)

    runtime = float(nums[-1]) if len(nums) > 0 else 0

    records.append({
        "GPUs": g,
        "Runtime": runtime
    })

df = pd.DataFrame(records)

# ------------------------------------------------
# Compute speedup / efficiency
# ------------------------------------------------
base = df.iloc[0]["Runtime"]

df["Speedup"] = base / df["Runtime"]
df["Efficiency"] = df["Speedup"] / (df["GPUs"]/8) * 100

print(df)

# Save CSV
df.to_csv("hybrid_scaling_summary.csv", index=False)

# ------------------------------------------------
# Plot Runtime
# ------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["GPUs"], df["Runtime"], marker='o')
plt.xscale("log", base=2)
plt.title("Runtime Scaling")
plt.xlabel("GPUs")
plt.ylabel("Runtime")
plt.grid(True)
plt.savefig("runtime_scaling.png")

# ------------------------------------------------
# Plot Speedup
# ------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["GPUs"], df["Speedup"], marker='o')
plt.xscale("log", base=2)
plt.title("Speedup")
plt.xlabel("GPUs")
plt.ylabel("Speedup")
plt.grid(True)
plt.savefig("speedup.png")

# ------------------------------------------------
# Plot Efficiency
# ------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["GPUs"], df["Efficiency"], marker='o')
plt.xscale("log", base=2)
plt.title("Parallel Efficiency (%)")
plt.xlabel("GPUs")
plt.ylabel("Efficiency %")
plt.grid(True)
plt.savefig("efficiency.png")

print("\nGenerated:")
print("hybrid_scaling_summary.csv")
print("runtime_scaling.png")
print("speedup.png")
print("efficiency.png")