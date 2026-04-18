import matplotlib.pyplot as plt

# Hardcoded values from SimAI simulation results
# Hypothesis 1: Total time scales linearly with number of layers

layers     = [16,      32,       64      ,128]
total_time = [5647.03, 10532.02, 20301.71, 40604.68]  # in microseconds

plt.figure(figsize=(7, 5))
plt.plot(layers, total_time, color='#1D9E75', linewidth=2.2, marker='o', markersize=8)

for l, t in zip(layers, total_time):
    plt.annotate(f'{t:,.2f} µs', xy=(l, t), xytext=(5, 8),
                 textcoords='offset points', fontsize=9, color='#1D9E75')

plt.title('Hypothesis : Total Time Scales Linearly with Layers', fontsize=12)
plt.xlabel('Number of Layers', fontsize=11)
plt.ylabel('Total Time (µs)', fontsize=11)
plt.xticks(layers)
plt.grid(axis='y', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig('simAI_hypothesis1_linear_scaling.png', dpi=150)
print("Plot saved: simAI_hypothesis1_linear_scaling.png")