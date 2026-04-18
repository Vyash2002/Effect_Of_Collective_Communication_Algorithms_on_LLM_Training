import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Hardcoded from nvls/ring_ncclFlowModel_EndToEnd.csv ──────────────────────
# Workload: multiCollective.txt — 18 layers
# Rows 1-4:  AllReduce   16M, 64M, 256M, 512M
# Rows 5-8:  AllGather   16M, 64M, 256M, 512M
# Rows 9-12: ReduceScatter 16M, 64M, 256M, 512M
# N=8, 32 GPUs, H100, 400 Gbps NIC, AS_NVLS_ENABLE=1 / AS_PXN_ENABLE=1

msg_labels = ["16M", "64M", "256M", "512M"]

# ── BusBW (GB/s) ──────────────────────────────────────────────────────────────
nvls_ar_bw = [148.877, 197.375, 214.866, 218.089]
nvls_ag_bw = [ 79.807, 127.134, 148.986, 153.436]
nvls_rs_bw = [ 79.723, 127.081, 148.968, 153.426]

ring_ar_bw = [141.849, 160.606, 165.728, 166.682]
ring_ag_bw = [141.817, 160.576, 165.728, 166.682]
ring_rs_bw = [141.553, 160.491, 165.705, 166.670]

# ── AlgBW (GB/s) ──────────────────────────────────────────────────────────────
nvls_ar_algbw = [ 85.072, 112.786, 122.781, 124.622]
nvls_ag_algbw = [ 91.208, 145.296, 170.270, 175.355]
nvls_rs_algbw = [ 91.112, 145.235, 170.249, 175.344]

ring_ar_algbw = [ 81.056,  91.775,  94.702,  95.247]
ring_ag_algbw = [162.077, 183.515, 189.403, 190.493]
ring_rs_algbw = [161.775, 183.418, 189.377, 190.480]

# ── Absolute comm time (µs) ───────────────────────────────────────────────────
nvls_ar_time = [183.667,  554.147, 2036.150, 4012.126]
nvls_ag_time = [171.312,  430.156, 1468.258, 2851.360]
nvls_rs_time = [171.492,  430.336, 1468.438, 2851.540]

ring_ar_time = [192.767,  681.016, 2639.862, 5249.504]
ring_ag_time = [ 96.405,  340.571, 1319.935, 2624.763]
ring_rs_time = [ 96.585,  340.751, 1320.115, 2624.943]

# ── Derived ratios ────────────────────────────────────────────────────────────
nvls_ring_ar   = [n/r for n,r in zip(nvls_ar_bw, ring_ar_bw)]
nvls_ring_ag   = [n/r for n,r in zip(nvls_ag_bw, ring_ag_bw)]
nvls_ring_rs   = [n/r for n,r in zip(nvls_rs_bw, ring_rs_bw)]
nvls_ar_ag_ratio = [a/b for a,b in zip(nvls_ar_bw, nvls_ag_bw)]
ring_ar_ag_ratio  = [a/b for a,b in zip(ring_ar_bw,  ring_ag_bw)]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "figure.facecolor":  "#f9f9f9",
    "axes.facecolor":    "#f9f9f9",
})

C = {
    "nvls_ar": "#2166ac", "ring_ar": "#d6604d",
    "nvls_ag": "#1a9641", "ring_ag": "#a6d96a",
    "nvls_rs": "#7b3294", "ring_rs": "#c2a5cf",
    "ref":     "#888888",
}

x = np.arange(len(msg_labels))
w = 0.13

fig = plt.figure(figsize=(22, 14))
fig.suptitle(
    "AICB Figure 3 Replication — BusBW vs Collective Type vs Algorithm\n"
    "32x H100 | 400 Gbps NIC | N=8 | NVLS (AS_SEND_LAT=12) vs Ring (AS_SEND_LAT=2)",
    fontsize=13, fontweight="bold", y=0.98
)

# ── Panel 1: Grouped bar — all 6 series ──────────────────────────────────────
ax1 = fig.add_subplot(3, 3, (1, 2))
offsets = [-2.5*w, -1.5*w, -0.5*w, 0.5*w, 1.5*w, 2.5*w]
series = [
    (nvls_ar_bw, "NVLS AllReduce",     C["nvls_ar"]),
    (ring_ar_bw, "Ring AllReduce",     C["ring_ar"]),
    (nvls_ag_bw, "NVLS AllGather",     C["nvls_ag"]),
    (ring_ag_bw, "Ring AllGather",     C["ring_ag"]),
    (nvls_rs_bw, "NVLS ReduceScatter", C["nvls_rs"]),
    (ring_rs_bw, "Ring ReduceScatter", C["ring_rs"]),
]
for (data, label, color), off in zip(series, offsets):
    ax1.bar(x + off, data, w, label=label, color=color,
            edgecolor="white", linewidth=0.6)
ax1.set_xticks(x); ax1.set_xticklabels(msg_labels)
ax1.set_xlabel("Message Size"); ax1.set_ylabel("BusBW (GB/s)")
ax1.set_title("BusBW by Collective Type & Algorithm", fontweight="bold")
ax1.set_ylim(0, 260)
ax1.legend(fontsize=7.5, ncol=3, loc="upper left")

# ── Panel 2: AR/AG busbw ratio — hypothesis check ────────────────────────────
ax2 = fig.add_subplot(3, 3, 3)
ax2.plot(msg_labels, nvls_ar_ag_ratio, "o-", color=C["nvls_ar"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="NVLS  AR/AG ratio")
ax2.plot(msg_labels, ring_ar_ag_ratio,  "s-", color=C["ring_ar"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="Ring   AR/AG ratio")
ax2.axhline(2.0, color=C["ref"], linestyle="--", lw=1.5, label="Hypothesis = 2.0")
ax2.axhline(1.0, color="#aaa",   linestyle=":",  lw=1.0, label="Parity = 1.0")
for i, (nv, ri) in enumerate(zip(nvls_ar_ag_ratio, ring_ar_ag_ratio)):
    ax2.annotate(f"{nv:.2f}", (i, nv), textcoords="offset points",
                 xytext=(5, 4), fontsize=7.5, color=C["nvls_ar"], fontweight="bold")
    ax2.annotate(f"{ri:.3f}", (i, ri), textcoords="offset points",
                 xytext=(5,-12), fontsize=7.5, color=C["ring_ar"], fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(msg_labels)
ax2.set_ylim(0.8, 2.3)
ax2.set_xlabel("Message Size"); ax2.set_ylabel("AR busbw / AG busbw")
ax2.set_title("Hypothesis Check:\nAR/AG BusBW ratio (expected = 2.0)", fontweight="bold")
ax2.legend(fontsize=7.5)

# ── Panel 3: NVLS BusBW lines ─────────────────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 4)
ax3.plot(msg_labels, nvls_ar_bw, "o-", color=C["nvls_ar"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="AllReduce")
ax3.plot(msg_labels, nvls_ag_bw, "s-", color=C["nvls_ag"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="AllGather")
ax3.plot(msg_labels, nvls_rs_bw, "^-", color=C["nvls_rs"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="ReduceScatter")
ax3.fill_between(msg_labels, nvls_ar_bw, nvls_ag_bw,
                 alpha=0.10, color=C["nvls_ar"])
ax3.set_xticks(x); ax3.set_xticklabels(msg_labels)
ax3.set_ylabel("BusBW (GB/s)"); ax3.set_xlabel("Message Size")
ax3.set_title("NVLS: BusBW per Collective\n(AR >> AG = RS)", fontweight="bold")
ax3.set_ylim(0, 250); ax3.legend(fontsize=8)

# ── Panel 4: Ring BusBW lines ─────────────────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 5)
ax4.plot(msg_labels, ring_ar_bw, "o-", color=C["ring_ar"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="AllReduce")
ax4.plot(msg_labels, ring_ag_bw, "s-", color=C["ring_ag"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="AllGather")
ax4.plot(msg_labels, ring_rs_bw, "^-", color=C["ring_rs"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="ReduceScatter")
ax4.set_xticks(x); ax4.set_xticklabels(msg_labels)
ax4.set_ylabel("BusBW (GB/s)"); ax4.set_xlabel("Message Size")
ax4.set_title("Ring: BusBW per Collective\n(AR = AG = RS — symmetric)", fontweight="bold")
ax4.set_ylim(0, 250); ax4.legend(fontsize=8)

# ── Panel 5: NVLS/Ring ratio per collective ───────────────────────────────────
ax5 = fig.add_subplot(3, 3, 6)
ax5.plot(msg_labels, nvls_ring_ar, "o-", color=C["nvls_ar"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="AllReduce")
ax5.plot(msg_labels, nvls_ring_ag, "s-", color=C["nvls_ag"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="AllGather")
ax5.plot(msg_labels, nvls_ring_rs, "^-", color=C["nvls_rs"], lw=2.2, ms=8,
         markerfacecolor="white", markeredgewidth=2, label="ReduceScatter")
ax5.axhline(1.0, color=C["ref"], linestyle="--", lw=1.5, label="Parity")
for i in range(4):
    ax5.annotate(f"{nvls_ring_ar[i]:.2f}x", (i, nvls_ring_ar[i]),
                 textcoords="offset points", xytext=(5, 3),
                 fontsize=7, color=C["nvls_ar"])
    ax5.annotate(f"{nvls_ring_ag[i]:.2f}x", (i, nvls_ring_ag[i]),
                 textcoords="offset points", xytext=(5,-11),
                 fontsize=7, color=C["nvls_ag"])
ax5.set_xticks(x); ax5.set_xticklabels(msg_labels)
ax5.set_xlabel("Message Size"); ax5.set_ylabel("NVLS BusBW / Ring BusBW")
ax5.set_title("NVLS vs Ring Advantage Ratio\nper Collective Type", fontweight="bold")
ax5.set_ylim(0.4, 1.7); ax5.legend(fontsize=7.5)

# ── Panel 6: Absolute comm time — NVLS ───────────────────────────────────────
ax6 = fig.add_subplot(3, 3, 7)
ax6.plot(msg_labels, [t/1000 for t in nvls_ar_time], "o-", color=C["nvls_ar"],
         lw=2.2, ms=8, markerfacecolor="white", markeredgewidth=2, label="AllReduce")
ax6.plot(msg_labels, [t/1000 for t in nvls_ag_time], "s-", color=C["nvls_ag"],
         lw=2.2, ms=8, markerfacecolor="white", markeredgewidth=2, label="AllGather")
ax6.plot(msg_labels, [t/1000 for t in nvls_rs_time], "^-", color=C["nvls_rs"],
         lw=2.2, ms=8, markerfacecolor="white", markeredgewidth=2, label="ReduceScatter")
for i in range(4):
    s = (nvls_ar_time[i] - nvls_ag_time[i]) / nvls_ar_time[i] * 100
    ax6.annotate(f"-{s:.0f}%", (i, nvls_ag_time[i]/1000),
                 textcoords="offset points", xytext=(5, 3),
                 fontsize=7, color=C["nvls_ag"])
ax6.set_xticks(x); ax6.set_xticklabels(msg_labels)
ax6.set_xlabel("Message Size"); ax6.set_ylabel("Comm Time (ms)")
ax6.set_title("NVLS: Absolute Comm Time\n(% = AG/RS saving vs AR)", fontweight="bold")
ax6.legend(fontsize=8)

# ── Panel 7: Absolute comm time — Ring ───────────────────────────────────────
ax7 = fig.add_subplot(3, 3, 8)
ax7.plot(msg_labels, [t/1000 for t in ring_ar_time], "o-", color=C["ring_ar"],
         lw=2.2, ms=8, markerfacecolor="white", markeredgewidth=2, label="AllReduce")
ax7.plot(msg_labels, [t/1000 for t in ring_ag_time], "s-", color=C["ring_ag"],
         lw=2.2, ms=8, markerfacecolor="white", markeredgewidth=2, label="AllGather")
ax7.plot(msg_labels, [t/1000 for t in ring_rs_time], "^-", color=C["ring_rs"],
         lw=2.2, ms=8, markerfacecolor="white", markeredgewidth=2, label="ReduceScatter")
for i in range(4):
    s = (ring_ar_time[i] - ring_ag_time[i]) / ring_ar_time[i] * 100
    ax7.annotate(f"-{s:.0f}%", (i, ring_ag_time[i]/1000),
                 textcoords="offset points", xytext=(5, 3),
                 fontsize=7, color=C["ring_ag"])
ax7.set_xticks(x); ax7.set_xticklabels(msg_labels)
ax7.set_xlabel("Message Size"); ax7.set_ylabel("Comm Time (ms)")
ax7.set_title("Ring: Absolute Comm Time\n(% = AG/RS saving vs AR)", fontweight="bold")
ax7.legend(fontsize=8)

# ── Panel 8: Verdict ──────────────────────────────────────────────────────────
ax8 = fig.add_subplot(3, 3, 9)
ax8.axis("off")
verdict = (
    "HYPOTHESIS VERDICT\n"
    "--------------------------------------\n\n"
    "H: AG busbw ~= RS busbw ~= AR busbw/2\n\n"
    "[CONFIRMED] for Ring:\n"
    "  AR ~= AG ~= RS busbw (141-167 GB/s)\n"
    "  Ring is symmetric across collectives\n"
    "  BusBW normalization is consistent\n\n"
    "[REFUTED] for NVLS:\n"
    "  AR busbw >> AG ~= RS (ratio 1.42-1.87x)\n"
    "  NVLS tree HW accelerates reduction only\n"
    "  No HW advantage for AG/RS\n\n"
    "[REFUTED] RS faster than AG in abs time:\n"
    "  AG ~= RS comm time for both algos\n\n"
    "[NEW FINDING]:\n"
    "  Ring wins AG & RS vs NVLS (1.09-1.78x)\n"
    "  NVLS only wins AllReduce (1.05-1.31x)\n"
    "  NVLS specialization = double-edged sword"
)
ax8.text(0.03, 0.97, verdict, transform=ax8.transAxes,
         fontsize=8.5, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7",
                   edgecolor="#f9a825", linewidth=1.5))

fig.text(0.5, 0.01,
    "Source: nvls/ring_ncclFlowModel_EndToEnd.csv  |  "
    "BusBW_AR = algbw x 2(N-1)/N,  BusBW_AG/RS = algbw x (N-1)/N  |  N=8, 32 GPUs",
    ha="center", fontsize=7.5, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig("aicb_fig3_multicollective.png", dpi=180, bbox_inches="tight")
plt.savefig("aicb_fig3_multicollective.pdf", bbox_inches="tight")
print("Saved: aicb_fig3_multicollective.png and .pdf")
plt.show()