# =============================================================================
# plot_results.py
# -----------------------------------------------------------------------------
# Reads the CSV files produced by run_experiments.py and generates plots.
#
# WHAT IT PRODUCES (saved in results/ folder):
#   convergence_F1.png ... convergence_F7.png   one curve per function
#   convergence_all.png                         all 7 functions on a grid
#   summary_bars.png                            median final cost as bar chart
#
# WHAT THE PLOTS SHOW:
# Each convergence plot shows the best cost found at each iteration of the
# algorithm, median over all 30 independent runs (with a shaded band
# showing min/max range across those runs). The y-axis is logarithmic so we
# can see improvement across many orders of magnitude.
#
# HOW TO USE:
#   python plot_results.py
# Run AFTER run_experiments.py has produced the CSV files.
# =============================================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_DIR     = 'results'                            # where CSVs and plots live
SUMMARY_CSV     = 'summary_original.csv'               # statistics per function
CONVERGENCE_CSV = 'convergence_original.csv'           # full curves per run


# =============================================================================
# LOAD SUMMARY CSV
# -----------------------------------------------------------------------------
# Returns a list of dicts, one per function, with the statistics columns.
# =============================================================================
def load_summary(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)                           # keep as dict per function
    return rows


# =============================================================================
# LOAD CONVERGENCE CSV
# -----------------------------------------------------------------------------
# Returns a dict keyed by function name. Each value is a 2D numpy array of
# shape (n_runs, n_iterations) with the best cost at each iteration of each run.
# =============================================================================
def load_convergence(path):
    # temp structure: data[fname][run] = list of costs by iteration
    raw = defaultdict(lambda: defaultdict(list))

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['function']                    # e.g. 'F1'
            run   = int(row['run'])                    # 1..30
            cost  = float(row['best_cost'])            # cost at this iteration
            raw[fname][run].append(cost)

    # convert to 2D numpy arrays
    result = {}
    for fname, runs_dict in raw.items():
        runs = sorted(runs_dict.keys())
        result[fname] = np.array([runs_dict[r] for r in runs])
    return result


# =============================================================================
# PLOT A SINGLE FUNCTION'S CONVERGENCE
# -----------------------------------------------------------------------------
# Median curve in solid line, min/max range as a shaded band around it.
# =============================================================================
def plot_one(fname, name, curves, save_path):
    # curves shape: (n_runs, n_iterations)
    median_curve = np.median(curves, axis=0)           # middle across runs
    min_curve  = np.min(curves, axis=0)                # best run at each iter
    max_curve  = np.max(curves, axis=0)                # worst run at each iter
    iters      = np.arange(1, len(median_curve) + 1)   # x axis: iteration number

    fig, ax = plt.subplots(figsize=(8, 5))
    # shaded band: min to max across runs
    ax.fill_between(iters, min_curve, max_curve,
                    alpha=0.2, color='steelblue', label='min/max range')
    # median curve on top
    ax.plot(iters, median_curve, color='crimson',
            linewidth=1.5, label='median of 30 runs')

    ax.set_title(f"{fname}: {name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost (log scale)")
    ax.set_yscale('log')                               # log scale shows improvement
    ax.set_ylim(bottom=1e-16)                          # show values down to 10^-16 so the line stays in frame
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


# =============================================================================
# PLOT ALL 7 FUNCTIONS IN ONE GRID FIGURE
# -----------------------------------------------------------------------------
# 7 subplots arranged in a 3x3 grid (8th and 9th cells left empty).
# Good for the report: one figure shows the full picture at a glance.
# =============================================================================
def plot_grid(all_curves, summary, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # build a name lookup from summary
    name_of = {row['function']: row['name'] for row in summary}

    func_keys = sorted(all_curves.keys())              # F1 ... F7

    for i, fname in enumerate(func_keys):
        curves     = all_curves[fname]
        median_curve = np.median(curves, axis=0)
        min_curve  = np.min(curves, axis=0)
        max_curve  = np.max(curves, axis=0)
        iters      = np.arange(1, len(median_curve) + 1)

        ax = axes[i]
        ax.fill_between(iters, min_curve, max_curve,
                        alpha=0.2, color='steelblue')
        ax.plot(iters, median_curve, color='crimson', linewidth=1.2)
        ax.set_title(f"{fname}: {name_of[fname]}", fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Best cost (log)", fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for j in range(len(func_keys), len(axes)):
        axes[j].axis('off')

    fig.suptitle("ICA Convergence on 7 CEC Benchmark Functions "
                 "(median of 30 runs, shaded = min/max)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


# =============================================================================
# BAR CHART OF MEDIAN FINAL COSTS
# -----------------------------------------------------------------------------
# Quick visual comparison of how well the algorithm did on each function.
# Lower bars = better. Log scale because costs span many orders of magnitude.
# =============================================================================
def plot_summary_bars(summary, save_path):
    fnames  = [row['function']           for row in summary]
    medians = [float(row['median'])      for row in summary]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fnames))
    ax.bar(x, medians, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(fnames)
    ax.set_ylabel("Median final cost (log scale)")
    ax.set_title("ICA Median Final Cost Across 7 Benchmark Functions ")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    summary_path     = os.path.join(RESULTS_DIR, SUMMARY_CSV)
    convergence_path = os.path.join(RESULTS_DIR, CONVERGENCE_CSV)

    # check that required files exist
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found.")
        print("Run run_experiments.py first.")
        exit(1)
    if not os.path.exists(convergence_path):
        print(f"ERROR: {convergence_path} not found.")
        print("Run run_experiments.py first.")
        exit(1)

    print("Loading CSV files ...")
    summary    = load_summary(summary_path)
    all_curves = load_convergence(convergence_path)

    print("\nGenerating per-function plots:")
    name_of = {row['function']: row['name'] for row in summary}
    for fname, curves in sorted(all_curves.items()):
        out = os.path.join(RESULTS_DIR, f"convergence_{fname}.png")
        plot_one(fname, name_of[fname], curves, out)

    print("\nGenerating grid plot:")
    plot_grid(all_curves, summary,
              os.path.join(RESULTS_DIR, 'convergence_all.png'))

    print("\nGenerating summary bar chart:")
    plot_summary_bars(summary,
                      os.path.join(RESULTS_DIR, 'summary_bars.png'))

    print("\nAll plots generated in", RESULTS_DIR)
