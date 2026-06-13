# =============================================================================
# run_experiments.py
# -----------------------------------------------------------------------------
# Runs the ICA algorithm on all 7 benchmark functions, multiple times each,
# and saves the results to CSV files for later analysis and plotting.
#
# WHY MULTIPLE RUNS?
# Metaheuristic algorithms like ICA use randomness. A single run might get
# lucky or unlucky. To measure real performance we run the algorithm many
# times with different random seeds and compute statistics: best, worst,
# mean, median, standard deviation.
#
# The project assignment requires 30 independent runs per function.
#
# OUTPUT FILES (saved in results/ folder):
#   summary_original.csv     final cost statistics per function
#   convergence_original.csv full convergence curves for every run
#
# HOW TO USE:
#   python run_experiments.py
# Then look in the results/ folder for the CSV files.
# =============================================================================

import os
import csv
import time
import numpy as np
from _benchmarks import FUNCTIONS
from _ica_original import ica


# =============================================================================
# EXPERIMENT CONFIGURATION
# -----------------------------------------------------------------------------
# Change these values to adjust the experiment scale.
# =============================================================================
DIMS         = 10                                      # search space dimensions
N_RUNS       = 30                                      # runs per function (assignment req)
N_ITERATIONS = 2000                                    # iterations per run
RESULTS_DIR  = 'results'                               # where CSVs land


# =============================================================================
# RUN ICA ONCE AND RETURN BEST COST + CONVERGENCE HISTORY
# -----------------------------------------------------------------------------
# Wrapper around ica() that uses a specific random seed so results are
# reproducible (running the script twice gives the same numbers).
# =============================================================================
def single_run(func_info, seed):
    np.random.seed(seed)                               # fix randomness for this run
    _, best_cost, history = ica(
        cost_func    = func_info['func'],
        dims         = DIMS,
        lb           = func_info['lb'],
        ub           = func_info['ub'],
        n_iterations = N_ITERATIONS)
    return best_cost, history


# =============================================================================
# RUN ALL EXPERIMENTS
# -----------------------------------------------------------------------------
# For each of the 7 functions:
#   - run ICA N_RUNS times with different seeds
#   - collect best cost and convergence history from each run
# =============================================================================
def run_all():
    # create results folder if it does not exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}                                   # holds all numeric data

    for fname, info in FUNCTIONS.items():
        print(f"Running ICA on {fname} ({info['name']}) ...")
        start = time.time()

        best_costs    = []                             # final best cost per run
        all_histories = []                             # full convergence curves

        for run_idx in range(N_RUNS):
            seed = run_idx + 1                         # seed 1, 2, ..., 30
            best_cost, history = single_run(info, seed)
            best_costs.append(best_cost)
            all_histories.append(history)

        elapsed = time.time() - start
        print(f"  done in {elapsed:.1f}s "
              f"| mean={np.mean(best_costs):.3e} "
              f"| best={np.min(best_costs):.3e}")

        all_results[fname] = {
            'best_costs'   : best_costs,
            'histories'    : all_histories,
            'name'         : info['name'],
            'optimum'      : info['optimum'],
        }

    return all_results


# =============================================================================
# SAVE SUMMARY STATISTICS
# -----------------------------------------------------------------------------
# One row per function with min / max / mean / median / std of best costs
# across all 30 runs. This is the headline table for the report.
# =============================================================================
def save_summary(results, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['function', 'name', 'optimum',
                         'min', 'max', 'mean', 'median', 'std'])
        for fname, data in results.items():
            bc = np.array(data['best_costs'])
            writer.writerow([fname, data['name'], data['optimum'],
                             f"{bc.min():.6e}",
                             f"{bc.max():.6e}",
                             f"{bc.mean():.6e}",
                             f"{np.median(bc):.6e}",
                             f"{bc.std():.6e}"])
    print(f"Summary saved: {path}")


# =============================================================================
# SAVE FULL CONVERGENCE CURVES
# -----------------------------------------------------------------------------
# One row per (function, run, iteration). This file is bigger but lets us
# plot any subset of runs later (mean curve, individual runs, etc).
# =============================================================================
def save_convergence(results, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['function', 'run', 'iteration', 'best_cost'])
        for fname, data in results.items():
            for run_idx, history in enumerate(data['histories']):
                for it_idx, cost in enumerate(history):
                    writer.writerow([fname, run_idx + 1, it_idx + 1,
                                     f"{cost:.6e}"])
    print(f"Convergence curves saved: {path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print(f"ICA experiments | {DIMS}D | {N_RUNS} runs | "
          f"{N_ITERATIONS} iterations\n")

    results = run_all()

    print("\nSaving CSV files ...")
    save_summary(results,
                 os.path.join(RESULTS_DIR, 'summary_original.csv'))
    save_convergence(results,
                     os.path.join(RESULTS_DIR, 'convergence_original.csv'))

    print("\nAll experiments complete.")
