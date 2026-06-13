# =============================================================================
# _plot_results_nam.py
# -----------------------------------------------------------------------------
# Draws NAM-ICA's own convergence plots from its CSV files,
# (median as a solid line, the min-to-max range as ashaded band, log y-axis). 
# Run this after the NAM-ICA runs have produced their CSVs.
#
# WHICH RUN IT PLOTS: set TAG below. 'namica_03' plots the rate-0.3 NAM-ICA run,
# 'namica_05' plots the rate-0.5 one. The TAG also decides the output names, so
# the two runs never overwrite each other's plots.
# =============================================================================

import os                                              # to build file paths and check files exist
import csv                                             # to read the CSV files
import numpy as np                                     # arrays plus median, min and max
import matplotlib.pyplot as plt                        # the plotting library
from collections import defaultdict                    # a dict that auto-creates entries, handy while grouping rows


# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_DIR = 'results'                                # folder where the CSVs live and the plots are written
TAG         = 'namica_03'                              # which result set to plot: 'ica_03', 'ica_05', 'namica_03' or 'namica_05'. change this and rerun for another set.
LABEL       = 'NAM-ICA' if TAG.startswith('namica') else 'ICA'  # title label taken from the tag, so vanilla sets read ICA and modified sets read NAM-ICA
SUMMARY_CSV     = f"summary_{TAG}.csv"                 # the per-function statistics file for that run
CONVERGENCE_CSV = f"convergence_{TAG}.csv"             # the full per-run curves file for that run


# =============================================================================
# LOAD THE SUMMARY CSV
# -----------------------------------------------------------------------------
# Returns a list of dicts, one per function, used only to look up names here.
# =============================================================================
def load_summary(path):
    rows = []                                          # one dict per function row
    with open(path, 'r') as f:                         # open the summary file
        reader = csv.DictReader(f)                     # read rows as dicts keyed by the header
        for row in reader:                             # walk every function row
            rows.append(row)                           # keep it as-is
    return rows                                        # the full list of function rows


# =============================================================================
# LOAD THE CONVERGENCE CSV
# -----------------------------------------------------------------------------
# Returns a dict keyed by function name, each value a 2D array shaped
# (runs, iterations) holding the best cost at each iteration of each run.
# =============================================================================
def load_convergence(path):
    raw = defaultdict(lambda: defaultdict(list))       # raw[function][run] = list of costs in iteration order

    with open(path, 'r') as f:                         # open the curves file
        reader = csv.DictReader(f)                     # read rows as dicts
        for row in reader:                             # for every (function, run, iteration) row
            fname = row['function']                    # which function
            run   = int(row['run'])                    # which run, as an integer for sorting
            cost  = float(row['best_cost'])            # the cost at this iteration
            raw[fname][run].append(cost)               # append in file order, which is iteration order

    result = {}                                        # the cleaned-up output
    for fname, runs_dict in raw.items():               # for each function
        runs = sorted(runs_dict.keys())                # sort run numbers so rows line up
        result[fname] = np.array([runs_dict[r] for r in runs])  # stack the runs into one 2D array
    return result                                      # function -> (runs, iterations) array


# =============================================================================
# PLOT ONE FUNCTION
# -----------------------------------------------------------------------------
# Median curve as a solid line, the min-to-max spread as a shaded band.
# =============================================================================
def plot_one(fname, name, curves, save_path):
    median_curve = np.median(curves, axis=0)           # the middle run at each iteration, robust to a lucky or unlucky run
    min_curve    = np.min(curves, axis=0)              # the best run at each iteration
    max_curve    = np.max(curves, axis=0)              # the worst run at each iteration
    iters        = np.arange(1, len(median_curve) + 1) # x-axis: iteration numbers starting at 1

    fig, ax = plt.subplots(figsize=(8, 5))             # a single plot
    ax.fill_between(iters, np.maximum(min_curve, 1e-16), np.maximum(max_curve, 1e-16),  # floor at 1e-16 so a run that hits exactly 0 (e.g. F2) still draws on the log axis instead of vanishing
                    alpha=0.2, color='steelblue', label='min/max range')
    ax.plot(iters, np.maximum(median_curve, 1e-16), color='crimson',  # same 1e-16 floor for the median line, so it stays continuous when the cost reaches 0
            linewidth=1.5, label='median of 30 runs')

    ax.set_title(f"{LABEL} [{TAG}] {fname}: {name}")   # title names the version and the function
    ax.set_xlabel("Iteration")                         # x-axis label
    ax.set_ylabel("Best cost (log scale)")             # y-axis label
    ax.set_yscale('log')                               # log scale, because the cost improves across many orders of magnitude
    ax.set_ylim(bottom=1e-16)                          # show values down to 1e-16 so the line stays inside the frame
    ax.grid(True, alpha=0.3)                           # faint grid for readability
    ax.legend()                                        # show the legend

    plt.tight_layout()                                 # tidy spacing
    plt.savefig(save_path, dpi=120, bbox_inches='tight')  # write the image
    plt.close()                                        # free the figure
    print(f"  saved: {save_path}")                     # confirm


# =============================================================================
# PLOT ALL 7 FUNCTIONS IN A 3x3 GRID
# =============================================================================
def plot_grid(all_curves, summary, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))   # a 3 by 3 grid, 7 used and 2 left empty
    axes = axes.flatten()                              # flatten so we can index cells 0..8

    name_of = {row['function']: row['name'] for row in summary}  # quick lookup from F-code to readable name

    func_keys = sorted(all_curves.keys())              # F1 through F7 in order

    for i, fname in enumerate(func_keys):              # one subplot per function
        curves       = all_curves[fname]               # this function's (runs, iterations) array
        median_curve = np.median(curves, axis=0)       # median curve
        min_curve    = np.min(curves, axis=0)          # best-run curve
        max_curve    = np.max(curves, axis=0)          # worst-run curve
        iters        = np.arange(1, len(median_curve) + 1)  # iteration numbers

        ax = axes[i]                                   # the cell for this function
        ax.fill_between(iters, np.maximum(min_curve, 1e-16), np.maximum(max_curve, 1e-16),  # floor at 1e-16 so exact-zero runs still draw in the grid cells
                        alpha=0.2, color='steelblue')
        ax.plot(iters, np.maximum(median_curve, 1e-16), color='crimson', linewidth=1.2)  # same floor for the median line in the grid
        ax.set_title(f"{fname}: {name_of[fname]}", fontsize=10)  # small per-cell title
        ax.set_xlabel("Iteration", fontsize=8)         # small axis labels to fit the grid
        ax.set_ylabel("Best cost (log)", fontsize=8)
        ax.set_yscale('log')                           # log y in every cell
        ax.grid(True, alpha=0.3)                       # faint grid
        ax.tick_params(labelsize=7)                    # small tick labels

    for j in range(len(func_keys), len(axes)):         # the leftover cells (8th and 9th)
        axes[j].axis('off')                            # hide them

    fig.suptitle(f"{LABEL} [{TAG}] convergence on 7 CEC functions "  # one overall title
                 "(median of 30 runs, shaded = min/max)", fontsize=12)
    plt.tight_layout()                                 # tidy spacing
    plt.savefig(save_path, dpi=120, bbox_inches='tight')  # write the grid image
    plt.close()                                        # free the figure
    print(f"  saved: {save_path}")                     # confirm


# =============================================================================
# BAR CHART OF MEDIAN FINAL COSTS
# =============================================================================
def plot_summary_bars(summary, save_path):
    fnames  = [row['function']      for row in summary]  # F-codes for the x-axis
    medians = [float(row['median']) for row in summary]  # median final cost per function

    fig, ax = plt.subplots(figsize=(9, 5))             # a single bar chart
    x = np.arange(len(fnames))                         # one x position per function
    ax.bar(x, np.maximum(medians, 1e-16), color='steelblue', edgecolor='black')  # the bars, lower is better. floor at 1e-16 so a zero-median function (e.g. F2) still shows a bar on the log axis
    ax.set_xticks(x)                                   # tick at each bar
    ax.set_xticklabels(fnames)                         # label them F1..F7
    ax.set_ylabel("Median final cost (log scale)")     # y-axis label
    ax.set_title(f"{LABEL} [{TAG}] median final cost across 7 functions")  # title names the version
    ax.set_yscale('log')                               # log y, costs span many orders of magnitude
    ax.grid(True, alpha=0.3, axis='y')                 # horizontal grid only

    plt.tight_layout()                                 # tidy spacing
    plt.savefig(save_path, dpi=120, bbox_inches='tight')  # write the image
    plt.close()                                        # free the figure
    print(f"  saved: {save_path}")                     # confirm


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    summary_path     = os.path.join(RESULTS_DIR, SUMMARY_CSV)      # full path to the summary file
    convergence_path = os.path.join(RESULTS_DIR, CONVERGENCE_CSV)  # full path to the curves file

    if not os.path.exists(summary_path):               # guard: the summary must exist
        print(f"ERROR: {summary_path} not found. Run _run_experiments_nam.py for this configuration first.")
        exit(1)
    if not os.path.exists(convergence_path):           # guard: the curves must exist
        print(f"ERROR: {convergence_path} not found. Run _run_experiments_nam.py for this configuration first.")
        exit(1)

    print(f"Loading {TAG} CSV files ...")
    summary    = load_summary(summary_path)            # load names
    all_curves = load_convergence(convergence_path)    # load curves

    print("\nGenerating per-function plots:")
    name_of = {row['function']: row['name'] for row in summary}  # name lookup
    for fname, curves in sorted(all_curves.items()):   # one plot per function
        out = os.path.join(RESULTS_DIR, f"convergence_{TAG}_{fname}.png")  # tagged output name
        plot_one(fname, name_of[fname], curves, out)

    print("\nGenerating grid plot:")
    plot_grid(all_curves, summary,
              os.path.join(RESULTS_DIR, f"convergence_{TAG}_all.png"))  # tagged grid

    print("\nGenerating summary bar chart:")
    plot_summary_bars(summary,
                      os.path.join(RESULTS_DIR, f"summary_{TAG}_bars.png"))  # tagged bars

    print(f"\nAll {TAG} plots generated in", RESULTS_DIR)
