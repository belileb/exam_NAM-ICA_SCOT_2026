# =============================================================================
# _compare_and_stats.py
# -----------------------------------------------------------------------------
# The final analysis, run LAST, after all four result sets exist. This is the
# ONLY file that imports scipy, and it uses scipy for one thing only: the
# Wilcoxon signed-rank test the assignment asks for.
#
# IT LOADS THESE FOUR RESULT SETS (summary + convergence for each):
#   ica_03      plain ICA, revolution 0.3
#   ica_05      plain ICA, revolution 0.5
#   namica_03   NAM-ICA,  revolution 0.3
#   namica_05   NAM-ICA,  revolution 0.5
#
# IT PRODUCES (all into the results folder):
#   1. One overlaid convergence figure per function, all four median curves on
#      one axis, grouped into single-valley (F1, F3, F5) and many-valley
#      (F2, F4, F6, F7). This shows convergence speed and final accuracy together.
#   2. comparison_table.csv, the combined accuracy and robustness table:
#      optimum, min, max, mean, median, std per function for all four versions.
#   3. wilcoxon_results.csv, the significance tests: NAM-ICA 0.3 vs ICA 0.3 (the
#      headline) and NAM-ICA 0.5 vs ICA 0.5 (the robustness check), with the
#      p-value and which version won (by median final cost).
#
# PAIRING: run i of every version used seed i, so the Wilcoxon test pairs run i
# of NAM-ICA with run i of ICA. That paired design is exactly what the signed-
# rank test needs.
# =============================================================================

import os                                              # to build file paths and check files exist
import csv                                             # to read the result CSVs and write the output tables
import numpy as np                                     # arrays plus median, mean and std
import matplotlib.pyplot as plt                        # the plotting library
from collections import defaultdict                    # auto-creating dict, handy while grouping rows
from scipy.stats import wilcoxon                       # the ONE scipy use: the signed-rank significance test the assignment requires


# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_DIR = 'results'                                # folder holding the four result sets and where outputs go

# Each version: its file tag, the label shown on plots and tables, and a fixed
# colour and line style. Same colour means same algorithm, same line style means
# same revolution rate, so the eye can separate "which algorithm" from "which rate".
VERSIONS = [
    ('ica_03',    'ICA r=0.3',     'tab:blue', '-'),   # plain ICA at the normal rate, the standard baseline
    ('ica_05',    'ICA r=0.5',     'tab:blue', '--'),  # plain ICA at high randomness, the control
    ('namica_03', 'NAM-ICA r=0.3', 'tab:red',  '-'),   # NAM-ICA at the normal rate, the main comparison
    ('namica_05', 'NAM-ICA r=0.5', 'tab:red',  '--'),  # NAM-ICA at high randomness, the robustness check
]

# Functions split into the two shape groups the brief asks for.
GROUPS = [
    ('singlevalley', ['F1', 'F3', 'F5']),              # one broad valley, NAM is not expected to change these
    ('manyvalley',   ['F2', 'F4', 'F6', 'F7']),        # many local minima, the targets the modification aims at
]

# The two paired comparisons for the significance test: (NAM version, ICA version).
COMPARISONS = [
    ('namica_03', 'ica_03'),                           # headline: same rate 0.3, only the modification differs
    ('namica_05', 'ica_05'),                           # robustness: same rate 0.5, does the benefit survive high randomness
]


# =============================================================================
# LOAD A SUMMARY CSV INTO function -> {stat: value}
# =============================================================================
def load_summary(path):
    out = {}                                           # function code -> dict of its statistics
    with open(path, 'r') as f:                         # open the summary file
        reader = csv.DictReader(f)                     # read rows as dicts keyed by the header
        for row in reader:                             # one row per function
            out[row['function']] = row                 # store the whole row, keyed by F-code
    return out                                         # function -> stats dict


# =============================================================================
# LOAD A CONVERGENCE CSV INTO function -> (runs, iterations) array
# =============================================================================
def load_convergence(path):
    raw = defaultdict(lambda: defaultdict(list))       # raw[function][run] = costs in iteration order
    with open(path, 'r') as f:                         # open the curves file
        reader = csv.DictReader(f)                     # read rows as dicts
        for row in reader:                             # for every (function, run, iteration) row
            raw[row['function']][int(row['run'])].append(float(row['best_cost']))  # collect costs in file (iteration) order
    result = {}                                        # cleaned output
    for fname, runs_dict in raw.items():               # for each function
        runs = sorted(runs_dict.keys())                # sort run numbers so seed i lands at index i-1
        result[fname] = np.array([runs_dict[r] for r in runs])  # stack into a 2D array
    return result                                      # function -> (runs, iterations)


# =============================================================================
# LOAD ALL FOUR VERSIONS, OR EXPLAIN WHAT IS MISSING
# =============================================================================
def load_all():
    summaries = {}                                     # tag -> summary dict
    curves    = {}                                     # tag -> convergence dict
    missing   = []                                     # any files we could not find
    for tag, _, _, _ in VERSIONS:                      # for each of the four versions
        s = os.path.join(RESULTS_DIR, f"summary_{tag}.csv")       # expected summary path
        c = os.path.join(RESULTS_DIR, f"convergence_{tag}.csv")   # expected curves path
        if not os.path.exists(s): missing.append(s)    # note a missing summary
        if not os.path.exists(c): missing.append(c)    # note a missing curves file
        if os.path.exists(s): summaries[tag] = load_summary(s)    # load if present
        if os.path.exists(c): curves[tag]    = load_convergence(c)  # load if present
    if missing:                                        # if anything is missing, stop with a clear message
        print("ERROR: these result files are missing. Generate them with _run_experiments_nam.py:")
        for m in missing:                              # list each missing file
            print("  ", m)
        print("\nReminder of the four runs you need:")
        print("  USE_NAM=False REVOLUTION_RATE=0.3  -> ica_03")
        print("  USE_NAM=False REVOLUTION_RATE=0.5  -> ica_05")
        print("  USE_NAM=True  REVOLUTION_RATE=0.3  -> namica_03")
        print("  USE_NAM=True  REVOLUTION_RATE=0.5  -> namica_05")
        raise SystemExit(1)                            # exit cleanly
    return summaries, curves                           # all four sets, ready to analyse


# =============================================================================
# 1. OVERLAID CONVERGENCE FIGURE PER FUNCTION
# -----------------------------------------------------------------------------
# All four median curves on one log-scale axis, so for each function you see
# both how fast each version drops (speed) and where it ends up (accuracy).
# =============================================================================
def plot_overlay(fname, fname_label, curves, group, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))             # a single plot per function
    for tag, label, color, style in VERSIONS:          # draw one median line per version
        med  = np.median(curves[tag][fname], axis=0)   # median over the 30 runs at each iteration
        iters = np.arange(1, len(med) + 1)             # iteration numbers on the x-axis
        ax.plot(iters, np.maximum(med, 1e-16), color=color, linestyle=style, linewidth=1.5, label=label)  # the median curve, colour=algorithm, style=rate. floor at 1e-16 so a zero-cost median (e.g. F2) stays on the log axis

    ax.set_title(f"{fname}: {fname_label}  (median of 30 runs)")  # title with function code and name
    ax.set_xlabel("Iteration")                         # x-axis label
    ax.set_ylabel("Best cost (log scale)")             # y-axis label
    ax.set_yscale('log')                               # log scale so many orders of magnitude are visible
    ax.grid(True, alpha=0.3)                           # faint grid
    ax.legend(fontsize=8)                              # legend naming all four versions

    plt.tight_layout()                                 # tidy spacing
    plt.savefig(save_path, dpi=120, bbox_inches='tight')  # write the figure
    plt.close()                                        # free it
    print(f"  saved: {save_path}")                     # confirm


# =============================================================================
# 2. COMBINED ACCURACY AND ROBUSTNESS TABLE
# -----------------------------------------------------------------------------
# One block per function (single-valley first, then many-valley), one row per
# version, columns optimum/min/max/mean/median/std straight from each summary.
# Printed to the screen and saved as comparison_table.csv.
# =============================================================================
def build_table(summaries):
    cols = ['optimum', 'min', 'max', 'mean', 'median', 'std']  # the numeric columns we report, in this order
    out_path = os.path.join(RESULTS_DIR, 'comparison_table.csv')  # where the table CSV goes

    with open(out_path, 'w', newline='') as f:         # open the output table file
        w = csv.writer(f)                              # plain csv writer
        w.writerow(['function', 'name', 'version'] + cols)  # header row

        for group_name, fkeys in GROUPS:               # single-valley group first, then many-valley
            for fk in fkeys:                           # each function in the group
                name = summaries['ica_03'][fk]['name']  # readable name (identical across versions, take it from any)
                print(f"\n=== {fk}: {name}  ({group_name}) ===")  # screen header for this function block
                header = f"{'version':14s} " + " ".join(f"{c:>13s}" for c in cols)  # aligned screen header
                print(header)                          # print it
                print("-" * len(header))               # underline
                for tag, label, _, _ in VERSIONS:      # one row per version
                    row = summaries[tag][fk]           # this version's stats for this function
                    vals = [row[c] for c in cols]      # pull the six numbers as strings
                    print(f"{label:14s} " + " ".join(f"{v:>13s}" for v in vals))  # aligned screen row
                    w.writerow([fk, name, label] + vals)  # the same row into the CSV

    print(f"\nCombined table saved: {out_path}")       # confirm where the CSV went


# =============================================================================
# 3. WILCOXON SIGNED-RANK TESTS
# -----------------------------------------------------------------------------
# For each function and each paired comparison, take the 30 final costs from
# each version (paired by seed), run the signed-rank test, and report the
# p-value plus which version won by median. Saved as wilcoxon_results.csv.
# =============================================================================
def run_wilcoxon(curves):
    out_path = os.path.join(RESULTS_DIR, 'wilcoxon_results.csv')  # where the stats CSV goes
    print("\n\n================= WILCOXON SIGNED-RANK TESTS =================")
    print("Final cost per run, paired by seed. Lower median is better.")
    print("p < 0.05 means the difference is statistically significant.\n")

    header = f"{'fn':3s} {'comparison':26s} {'median NAM':>13s} {'median ICA':>13s} {'p-value':>11s} {'winner':>14s} {'significant':>12s}"
    print(header)                                      # aligned screen header
    print("-" * len(header))                           # underline

    with open(out_path, 'w', newline='') as f:         # open the output stats file
        w = csv.writer(f)                              # plain csv writer
        w.writerow(['function', 'comparison', 'median_nam', 'median_ica',  # header row
                    'p_value', 'winner', 'significant'])

        for group_name, fkeys in GROUPS:               # group order: single-valley then many-valley
            for fk in fkeys:                           # each function
                for nam_tag, ica_tag in COMPARISONS:   # the two paired comparisons
                    a = curves[nam_tag][fk][:, -1]     # NAM-ICA final cost of each run (last iteration), ordered by seed
                    b = curves[ica_tag][fk][:, -1]     # ICA final cost of each run, same seed order, so a[i] pairs with b[i]
                    med_a = float(np.median(a))        # NAM-ICA median final cost
                    med_b = float(np.median(b))        # ICA median final cost

                    try:                               # the test can fail if every paired difference is exactly zero
                        _, p = wilcoxon(a, b)          # signed-rank test on the paired differences, two-sided
                        p_str = f"{p:.3e}"             # format the p-value
                    except ValueError:                 # happens when a and b are identical on every run
                        p   = float('nan')             # no meaningful p-value
                        p_str = "n/a (equal)"          # say so plainly

                    if med_a < med_b:                  # lower median cost is the better, winning version
                        winner = 'NAM-ICA'
                    elif med_b < med_a:
                        winner = 'ICA'
                    else:
                        winner = 'tie'                 # identical medians

                    sig = 'yes' if (p == p and p < 0.05) else 'no'  # 'p == p' is False only for NaN, so NaN counts as not significant
                    comp_label = f"{nam_tag} vs {ica_tag}"  # human-readable comparison name

                    print(f"{fk:3s} {comp_label:26s} {med_a:13.3e} {med_b:13.3e} {p_str:>11s} {winner:>14s} {sig:>12s}")  # screen row
                    w.writerow([fk, comp_label, f"{med_a:.6e}", f"{med_b:.6e}", p_str, winner, sig])  # CSV row

    print(f"\nWilcoxon results saved: {out_path}")     # confirm where the CSV went


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    summaries, curves = load_all()                     # load all four sets, or stop with a clear message

    print("Generating overlaid convergence figures (4 versions per function):")
    for group_name, fkeys in GROUPS:                   # single-valley then many-valley
        for fk in fkeys:                               # one figure per function
            label = summaries['ica_03'][fk]['name']    # readable function name
            out   = os.path.join(RESULTS_DIR, f"compare_{group_name}_{fk}.png")  # grouped, function-tagged name
            plot_overlay(fk, label, curves, group_name, out)

    build_table(summaries)                             # 2. the combined accuracy and robustness table
    run_wilcoxon(curves)                               # 3. the significance tests

    print("\nComparison complete. See the results folder for figures, "
          "comparison_table.csv and wilcoxon_results.csv.")
