import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Hardcoded values extracted from actual simulation logs ────────────────────
# nvls_result.log : AS_SEND_LAT=12 AS_NVLS_ENABLE=1
# ring_result.log : AS_SEND_LAT=2  AS_PXN_ENABLE=1
# 32x H100 | Rail-Optimized Single-ToR | 400 Gbps NIC | model_parallel_NPU_group=8

msg_labels   = ["16M", "32M", "64M", "128M", "256M", "512M"]
msg_sizes_mb = [16,     32,    64,    128,    256,    512   ]

# Cycles = "Total cycles spent on fwd pass comm" per layer (1 cycle ≈ 1 ns in SimAI)
nvls_cycles = [176678, 293182, 526190, 992252, 1924307, 3788434]
ring_cycles = [192767, 355847, 681016, 1335062, 2639862, 5249504]

# Reference values from SimAI Tutorial table
ref_nvls = [148.88, 178.04, 197.38, 208.70, 214.87, 218.09]
ref_ring  = [141.84, 153.68, 160.60, 163.85, 165.72, 166.68]

# Total end-to-end simulation time (cycles) from logs
nvls_total_cycles = 11_037_072
ring_total_cycles = 13_790_087

# ── Bus bandwidth calculation ─────────────────────────────────────────────────
# BusBW (GB/s) = msg_size_bytes × 2(N-1)/N  ÷  comm_cycles
# N = 8 (intra-node ring / model_parallel_NPU_group)
N = 8
alpha = 2 * (N - 1) / N   # = 1.75

nvls_bw = [(s * 1024**2 * alpha) / c for s, c in zip(msg_sizes_mb, nvls_cycles)]
ring_bw  = [(s * 1024**2 * alpha) / c for s, c in zip(msg_sizes_mb, ring_cycles)]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.facecolor":   "#f9f9f9",
    "axes.facecolor":     "#f9f9f9",
})

C = {
    "nvls_act": "#2166ac",
    "ring_act":  "#d6604d",
    "nvls_ref":  "#92c5de",
    "ring_ref":  "#f4a582",
}

x = np.arange(len(msg_labels))
w = 0.20

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
fig.suptitle(
    "Ring-AllReduce vs NVLS AllReduce  ·  Actual Log Results vs Tutorial Reference\n"
    "32× H100 | Rail-Optimized Single-ToR | 400 Gbps NIC"
    " | AS_SEND_LAT: NVLS=12 / Ring=2",
    fontsize=12, fontweight="bold", y=1.01,
)

# ── Panel 1: Grouped bar — Actual vs Reference ────────────────────────────────
ax = axes[0]
ax.bar(x - 1.5*w, ref_nvls, w, label="NVLS Reference", color=C["nvls_ref"], edgecolor="white")
ax.bar(x - 0.5*w, nvls_bw,  w, label="NVLS Actual",    color=C["nvls_act"], edgecolor="white")
ax.bar(x + 0.5*w, ref_ring,  w, label="Ring Reference",  color=C["ring_ref"], edgecolor="white")
ax.bar(x + 1.5*w, ring_bw,   w, label="Ring Actual",     color=C["ring_act"], edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(msg_labels)
ax.set_xlabel("Message Size")
ax.set_ylabel("Bus Bandwidth (GB/s)")
ax.set_title("Actual vs Reference BusBW", fontweight="bold")
ax.set_ylim(0, 290)
ax.legend(fontsize=8, ncol=2, loc="upper left")

# ── Panel 2: Line chart — actual vs reference ─────────────────────────────────
ax = axes[1]
ax.plot(msg_labels, nvls_bw,  "o-",  color=C["nvls_act"], lw=2.3, ms=8,
        markerfacecolor="white", markeredgewidth=2, label="NVLS (actual)")
ax.plot(msg_labels, ring_bw,  "s-",  color=C["ring_act"], lw=2.3, ms=8,
        markerfacecolor="white", markeredgewidth=2, label="Ring (actual)")
ax.plot(msg_labels, ref_nvls, "o--", color=C["nvls_ref"], lw=1.5, ms=5, label="NVLS (reference)")
ax.plot(msg_labels, ref_ring,  "s--", color=C["ring_ref"], lw=1.5, ms=5, label="Ring (reference)")

ax.fill_between(msg_labels, nvls_bw, ring_bw, alpha=0.10, color=C["nvls_act"])

gap = nvls_bw[-1] - ring_bw[-1]
ax.annotate(
    f"NVLS leads by\n{gap:.1f} GB/s at 512M",
    xy=(5, (nvls_bw[-1] + ring_bw[-1]) / 2),
    xytext=(3.4, 148), fontsize=8, color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.1),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
)

ax.set_xlabel("Message Size")
ax.set_ylabel("Bus Bandwidth (GB/s)")
ax.set_title("BusBW Scaling: NVLS Dominates Across All Sizes", fontweight="bold")
ax.set_ylim(100, 290)
ax.legend(fontsize=8, loc="lower right")

# ── Panel 3: Comm time (k-cycles) with Ring overhead % ───────────────────────
ax = axes[2]
ax.plot(msg_labels, [c / 1000 for c in nvls_cycles], "o-", color=C["nvls_act"],
        lw=2.3, ms=8, markerfacecolor="white", markeredgewidth=2, label="NVLS comm time")
ax.plot(msg_labels, [c / 1000 for c in ring_cycles], "s-", color=C["ring_act"],
        lw=2.3, ms=8, markerfacecolor="white", markeredgewidth=2, label="Ring comm time")

for i in range(len(msg_labels)):
    overhead = (ring_cycles[i] - nvls_cycles[i]) / nvls_cycles[i] * 100
    ax.annotate(
        f"+{overhead:.0f}%",
        xy=(i, ring_cycles[i] / 1000),
        xytext=(i - 0.25, ring_cycles[i] / 1000 + 120),
        fontsize=7, color=C["ring_act"], fontweight="bold",
    )

ax.set_xlabel("Message Size")
ax.set_ylabel("Comm Time (k cycles / ns)")
ax.set_title(
    "Communication Latency: Ring is Slower at Every Size\n"
    "(% = Ring overhead vs NVLS)",
    fontweight="bold",
)
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))

# ── Footer ────────────────────────────────────────────────────────────────────
speedup = (ring_total_cycles - nvls_total_cycles) / ring_total_cycles * 100
fig.text(
    0.5, -0.03,
    f"Source: nvls_result.log & ring_result.log  ·  "
    f"BusBW = msg × 2(N−1)/N ÷ comm_cycles  (N=8, 1 cycle≈1 ns)  ·  "
    f"NVLS total sim: {nvls_total_cycles:,} cycles  ·  "
    f"Ring total sim: {ring_total_cycles:,} cycles  ·  "
    f"NVLS is {speedup:.0f}% faster overall",
    ha="center", fontsize=7.5, color="gray", style="italic",
)

plt.tight_layout()
plt.savefig("ring_vs_nvls_actual.png", dpi=180, bbox_inches="tight")
plt.savefig("ring_vs_nvls_actual.pdf", bbox_inches="tight")
print("Saved: ring_vs_nvls_actual.png  and  ring_vs_nvls_actual.pdf")
plt.show()
