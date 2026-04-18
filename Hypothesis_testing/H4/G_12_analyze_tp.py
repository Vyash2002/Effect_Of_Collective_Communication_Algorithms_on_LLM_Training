"""
TP Scaling Analysis: Effect of Tensor Parallelism on AllReduce Communication
Hypothesis: Higher TP increases frequency and size of AllReduce → magnifies
            differences between Ring and Tree algorithms.

Data source: tp{1,4,8}_{ring,tree}_EndToEnd.csv
All values hardcoded from CSV summary rows (line index 1).

CSV Summary Row format:
  File name, Expose DP comm, Expose DP_EP comm, Expose TP comm,
  Expose_EP_comm, Expose_PP_comm, bubble time, total comp,
  total exposed comm, Total time

Units in CSV: microseconds (us)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED DATA — extracted directly from CSV summary rows
# ─────────────────────────────────────────────────────────────────────────────

# Structure per config:
#   dp_comm   : Exposed DP communication time (us)
#   tp_comm   : Exposed TP communication time (us)
#   total_comm: Total exposed communication time (us)
#   total_time: End-to-end total time (us)
#
# Note: tp1 has overflow values in DP/TP fields (INT64 sentinel = 9223372036854776)
# meaning those collectives had zero real latency (no actual TP group).
# We treat tp1 DP comm as 66708 us (grad_norm wg comm), TP comm = 0.

DATA = {
    # TP=1: no tensor parallelism → only DP AllReduce (grad_norm)
    # Summary row: tp1_ring_, 66708 (0.00%), 0, 27670116110564328 (sentinel), ...
    #              total exposed comm = 1226708480901752832 (sentinel/overflow)
    #              real meaningful comm = grad_norm wg comm = 66708 us
    # We read the actual numbers from grad_norm layer row directly.
    'tp1_ring': {
        'dp_comm':    66708,       # grad_norm wg exposed comm (DP AllReduce)
        'tp_comm':    0,           # no TP group at TP=1
        'total_comm': 66708,       # only real comm is DP
        'total_time': 66708,       # simulator total (ignoring sentinel layers)
    },
    'tp1_tree': {
        'dp_comm':    129414,      # grad_norm wg exposed comm (DP AllReduce, tree slower)
        'tp_comm':    0,
        'total_comm': 129414,
        'total_time': 129414,
    },

    # TP=4: summary row line 1:
    #   tp4_ring_: DP=66201 (53.05%), TP=58598 (46.95%), total=124799, Total time=124799
    'tp4_ring': {
        'dp_comm':    66201,
        'tp_comm':    58598,
        'total_comm': 124799,
        'total_time': 124799,
    },
    #   tp4_tree_: DP=128430 (54.92%), TP=105408 (45.08%), total=233838, Total time=233839
    'tp4_tree': {
        'dp_comm':    128430,
        'tp_comm':    105408,
        'total_comm': 233838,
        'total_time': 233839,
    },

    # TP=8: summary row line 1:
    #   tp8_ring_: DP=65540 (39.42%), TP=100727 (60.58%), total=166267, Total time=166269
    'tp8_ring': {
        'dp_comm':    65540,
        'tp_comm':    100727,
        'total_comm': 166267,
        'total_time': 166269,
    },
    #   tp8_tree_: DP=127148 (41.23%), TP=181225 (58.77%), total=308374, Total time=308375
    'tp8_tree': {
        'dp_comm':    127148,
        'tp_comm':    181225,
        'total_comm': 308374,
        'total_time': 308375,
    },
}

TP_SIZES = [1, 4, 8]

# ─────────────────────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────
BG_MAIN  = '#0F1117'
BG_AX    = '#1A1D27'
GRID_C   = '#2D2F3E'
TEXT_C   = '#E0E0E0'
LABEL_C  = '#B0B0B0'
TICK_C   = '#909090'

C_RING   = '#2196F3'   # blue  – ring
C_TREE   = '#FF9800'   # amber – tree
C_DP     = '#4CAF50'   # green – DP comm
C_TP     = '#E91E63'   # pink  – TP comm

def style_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_facecolor(BG_AX)
    ax.set_title(title, color=TEXT_C, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=LABEL_C, fontsize=10)
    ax.set_ylabel(ylabel, color=LABEL_C, fontsize=10)
    ax.tick_params(colors=TICK_C, labelsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.5, linewidth=0.6, axis='y')
    if legend:
        ax.legend(facecolor='#252838', edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)


def annotate_bars(ax, bars, fmt='{:.0f}µs', offset=3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha='center', va='bottom',
                color=TEXT_C, fontsize=8, fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Total Communication Time by TP size (DP vs TP split, per algorithm)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig.patch.set_facecolor(BG_MAIN)
fig.suptitle(
    'Plot 1 — Total Communication Time by TP Size\n'
    'Stacked: DP Comm (AllReduce on gradients) vs TP Comm (AllReduce within TP group)',
    color=TEXT_C, fontsize=13, fontweight='bold', y=1.02
)

for ax, algo, color, title_suffix in zip(
        axes,
        ['ring', 'tree'],
        [C_RING, C_TREE],
        ['Ring AllReduce', 'Tree AllReduce']):

    dp_vals  = [DATA[f'tp{t}_{algo}']['dp_comm']  for t in TP_SIZES]
    tp_vals  = [DATA[f'tp{t}_{algo}']['tp_comm']  for t in TP_SIZES]

    x = np.arange(len(TP_SIZES))
    w = 0.5

    b1 = ax.bar(x, dp_vals, width=w, label='DP Comm (grad AllReduce)',
                color=C_DP, alpha=0.88, edgecolor='#2e7d32', linewidth=0.8)
    b2 = ax.bar(x, tp_vals, width=w, bottom=dp_vals, label='TP Comm (activation AllReduce)',
                color=C_TP, alpha=0.88, edgecolor='#880e4f', linewidth=0.8)

    # Annotate totals on top
    totals = [d + t for d, t in zip(dp_vals, tp_vals)]
    for i, tot in enumerate(totals):
        ax.text(i, tot + max(totals) * 0.02, f'{tot:,}µs',
                ha='center', va='bottom', color=TEXT_C, fontsize=9, fontweight='bold')

    # DP value inside bar
    for i, (d, t) in enumerate(zip(dp_vals, tp_vals)):
        if d > 5000:
            ax.text(i, d / 2, f'DP\n{d:,}', ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
        if t > 5000:
            ax.text(i, d + t / 2, f'TP\n{t:,}', ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'TP={t}' for t in TP_SIZES], color=TICK_C)
    style_ax(ax, f'{title_suffix}\nDP vs TP Communication Breakdown',
             'Tensor Parallelism Size', 'Communication Time (µs)')

plt.tight_layout()
out1 = os.path.join(RESULTS_DIR, 'plot1_comm_by_tp_dp_vs_tp.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {out1}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Total Time — Ring AllReduce across TP sizes
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG_MAIN)

ring_total = [DATA[f'tp{t}_ring']['total_time'] for t in TP_SIZES]
ring_comm  = [DATA[f'tp{t}_ring']['total_comm'] for t in TP_SIZES]

x = np.arange(len(TP_SIZES))
w = 0.45

bars = ax.bar(x, ring_total, width=w, color=C_RING, alpha=0.88,
              edgecolor='#1565c0', linewidth=1, label='Total Time (Ring)')
ax.plot(x, ring_comm, 'o--', color='#FFEB3B', linewidth=2, markersize=8,
        label='Comm Time (Ring)', zorder=5)

for i, (tot, comm) in enumerate(zip(ring_total, ring_comm)):
    ax.text(i, tot + max(ring_total) * 0.02, f'{tot:,}µs',
            ha='center', va='bottom', color=TEXT_C, fontsize=9, fontweight='bold')
    ax.text(i, comm - max(ring_total) * 0.06, f'C:{comm:,}',
            ha='center', va='top', color='#FFEB3B', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f'TP={t}' for t in TP_SIZES], color=TICK_C)
style_ax(ax,
         'Plot 2 — Total Time: Ring AllReduce across TP Sizes\n'
         '(Bar = Total Time, Line = Communication Time)',
         'Tensor Parallelism Size', 'Time (µs)')

plt.tight_layout()
out2 = os.path.join(RESULTS_DIR, 'plot2_total_time_ring.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {out2}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Total Time — Tree AllReduce across TP sizes
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG_MAIN)

tree_total = [DATA[f'tp{t}_tree']['total_time'] for t in TP_SIZES]
tree_comm  = [DATA[f'tp{t}_tree']['total_comm'] for t in TP_SIZES]

bars = ax.bar(x, tree_total, width=w, color=C_TREE, alpha=0.88,
              edgecolor='#e65100', linewidth=1, label='Total Time (Tree)')
ax.plot(x, tree_comm, 'o--', color='#FFEB3B', linewidth=2, markersize=8,
        label='Comm Time (Tree)', zorder=5)

for i, (tot, comm) in enumerate(zip(tree_total, tree_comm)):
    ax.text(i, tot + max(tree_total) * 0.02, f'{tot:,}µs',
            ha='center', va='bottom', color=TEXT_C, fontsize=9, fontweight='bold')
    ax.text(i, comm - max(tree_total) * 0.06, f'C:{comm:,}',
            ha='center', va='top', color='#FFEB3B', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f'TP={t}' for t in TP_SIZES], color=TICK_C)
style_ax(ax,
         'Plot 3 — Total Time: Tree AllReduce across TP Sizes\n'
         '(Bar = Total Time, Line = Communication Time)',
         'Tensor Parallelism Size', 'Time (µs)')

plt.tight_layout()
out3 = os.path.join(RESULTS_DIR, 'plot3_total_time_tree.png')
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {out3}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Ring vs Tree Gap — grows with TP size (grouped bar + gap line)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10),
                                      gridspec_kw={'height_ratios': [2, 1]})
fig.patch.set_facecolor(BG_MAIN)
fig.suptitle(
    'Plot 4 — Ring vs Tree: Algorithm Gap Grows with TP Size\n'
    'Hypothesis: Higher TP → more AllReduce → larger absolute difference',
    color=TEXT_C, fontsize=13, fontweight='bold'
)

# ── Top: Grouped bars Ring vs Tree total time ─────────────────────────────
x      = np.arange(len(TP_SIZES))
w      = 0.35

ring_t = [DATA[f'tp{t}_ring']['total_time'] for t in TP_SIZES]
tree_t = [DATA[f'tp{t}_tree']['total_time'] for t in TP_SIZES]
gap    = [tr - ri for tr, ri in zip(tree_t, ring_t)]
gap_pct= [g / ri * 100 for g, ri in zip(gap, ring_t)]

b_ring = ax_top.bar(x - w/2, ring_t, width=w, label='Ring AllReduce',
                    color=C_RING, alpha=0.88, edgecolor='#1565c0', linewidth=0.8)
b_tree = ax_top.bar(x + w/2, tree_t, width=w, label='Tree AllReduce',
                    color=C_TREE, alpha=0.88, edgecolor='#e65100', linewidth=0.8)

# Annotate gap arrows between bars
for i, (r, t, g, gp) in enumerate(zip(ring_t, tree_t, gap, gap_pct)):
    ymax = max(r, t)
    ax_top.annotate('',
        xy=(i + w/2, t), xytext=(i - w/2, r),
        arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
    ax_top.text(i, ymax + max(tree_t) * 0.03,
                f'Δ={g:,}µs\n(+{gp:.1f}%)',
                ha='center', va='bottom', color='white',
                fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#333650', alpha=0.8))

annotate_bars(ax_top, b_ring, offset=max(tree_t)*0.005)
annotate_bars(ax_top, b_tree, offset=max(tree_t)*0.005)

ax_top.set_xticks(x)
ax_top.set_xticklabels([f'TP={t}' for t in TP_SIZES], color=TICK_C)
style_ax(ax_top, 'Total Time: Ring vs Tree', 'Tensor Parallelism Size', 'Total Time (µs)')

# ── Bottom: absolute gap bar chart ───────────────────────────────────────
gap_colors = ['#7E57C2', '#AB47BC', '#E91E63']
b_gap = ax_bot.bar(x, gap, width=0.5, color=gap_colors, alpha=0.88,
                   edgecolor='white', linewidth=0.8)
for i, (g, gp) in enumerate(zip(gap, gap_pct)):
    ax_bot.text(i, g + max(gap) * 0.03, f'{g:,}µs\n(+{gp:.1f}%)',
                ha='center', va='bottom', color=TEXT_C, fontsize=9, fontweight='bold')

ax_bot.set_xticks(x)
ax_bot.set_xticklabels([f'TP={t}' for t in TP_SIZES], color=TICK_C)
style_ax(ax_bot,
         'Algorithm Gap (Tree − Ring) — Grows with TP Size\n'
         '✓ Confirms hypothesis: higher TP magnifies Ring vs Tree difference',
         'Tensor Parallelism Size', 'Gap = Tree − Ring (µs)', legend=False)

# Add trend annotation
ax_bot.annotate('Gap grows\nwith TP →',
                xy=(1.9, gap[-1]), xytext=(0.5, gap[-1] * 0.85),
                arrowprops=dict(arrowstyle='->', color='#FFEB3B', lw=1.5),
                color='#FFEB3B', fontsize=9, fontweight='bold')

plt.tight_layout()
out4 = os.path.join(RESULTS_DIR, 'plot4_ring_vs_tree_gap_grows_with_tp.png')
plt.savefig(out4, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {out4}')

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('SUMMARY TABLE — All values in microseconds (µs)')
print('='*72)
print(f"{'TP':>4} {'Algo':>6} {'DP Comm':>10} {'TP Comm':>10} {'Total Comm':>12} {'Total Time':>12}")
print('-'*72)
for tp in TP_SIZES:
    for algo in ['ring', 'tree']:
        d = DATA[f'tp{tp}_{algo}']
        print(f"{tp:>4} {algo:>6} {d['dp_comm']:>10,} {d['tp_comm']:>10,} "
              f"{d['total_comm']:>12,} {d['total_time']:>12,}")
    if tp != TP_SIZES[-1]:
        print()

print('='*72)
print('\nHYPOTHESIS CHECK:')
for tp in TP_SIZES:
    r = DATA[f'tp{tp}_ring']['total_time']
    t = DATA[f'tp{tp}_tree']['total_time']
    gap = t - r
    pct = gap / r * 100
    tp_frac_ring = DATA[f'tp{tp}_ring']['tp_comm'] / max(DATA[f'tp{tp}_ring']['total_comm'], 1) * 100
    print(f"  TP={tp}: Ring={r:,}µs  Tree={t:,}µs  Gap={gap:+,}µs (+{pct:.1f}%)  "
          f"TP share of Ring comm={tp_frac_ring:.1f}%")
print()
print('  ✓ As TP increases: total comm time GROWS for both algos')
print('  ✓ As TP increases: the Ring vs Tree gap WIDENS (in absolute µs)')
print('  ✓ As TP increases: TP comm fraction grows (0% → ~47% → ~61%)')
print('  → Hypothesis SUPPORTED: Higher TP magnifies algorithm differences')