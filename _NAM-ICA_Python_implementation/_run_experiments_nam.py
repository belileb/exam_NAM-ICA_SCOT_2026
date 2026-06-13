# =============================================================================
# _run_experiments_nam.py
# -----------------------------------------------------------------------------
# Runs ONE configuration of the algorithm over all 7 benchmark functions, 30
# times each, and saves the results to two CSV files whose names are built
# automatically from the two switches below.
# Run this by hand four times,
# changing only the two switches each time, and it never overwrites a previous
# run because each run names its own files.
#
# THE FOUR RUNS (set the two switches, run, repeat):
#   USE_NAM=False REVOLUTION_RATE=0.3  ->  summary_ica_03.csv     convergence_ica_03.csv
#   USE_NAM=False REVOLUTION_RATE=0.5  ->  summary_ica_05.csv     convergence_ica_05.csv
#   USE_NAM=True  REVOLUTION_RATE=0.3  ->  summary_namica_03.csv  convergence_namica_03.csv
#   USE_NAM=True  REVOLUTION_RATE=0.5  ->  summary_namica_05.csv  convergence_namica_05.csv
#
# NO-LEAK CHECK: running USE_NAM=False at REVOLUTION_RATE=0.5 must reproduce
# refferenced existing summary_original.csv exactly (those uploaded files are the r=0.5
# run). If the numbers match, the modification has not leaked into the base.
#
# FAIRNESS: every run uses the same locked settings and the same paired seeds
# (run i uses seed i in every version), and the modification adds zero extra
# cost-function calls, so matching the iteration count matches the budget.
# =============================================================================

import os                                              # to build file paths and make the results folder
import csv                                             # to write the two output CSV files
import time                                            # to time each function so you get progress feedback
import numpy as np                                     # vector maths plus the seeded random generator
from _benchmarks import FUNCTIONS                      # the 7 CEC functions with their bounds and known optima
from _nam_ica import ica                               # the single algorithm: plain ICA or NAM-ICA depending on the flag below


# =============================================================================
# THE TWO SWITCHES (the only things you change between runs)
# =============================================================================
REVOLUTION_RATE = 0.3                                  # set 0.3 for the normal runs or 0.5 for the high-randomness runs. this is the only numeric setting that changes between runs.
USE_NAM         = True                                 # NAM: False = plain ICA, True = NAM-ICA. this is the modification switch and it is passed straight into ica() below.


# =============================================================================
# LOCKED SETTINGS (identical for every run, so the comparison is fair)
# =============================================================================
DIMS         = 10                                      # search space dimensions
N_RUNS       = 30                                      # independent runs per function (assignment requirement)
N_ITERATIONS = 2000                                    # iterations per run
RESULTS_DIR  = 'results'                               # folder where the CSVs are written
# The locked ICA parameters, surfaced here so every fixed setting is visible in one place and passed in explicitly (these equal the algorithm defaults, so the numbers are unchanged).
N_COUNTRIES       = 300                                # total population size
N_IMPERIALISTS    = 20                                 # number of starting empires
BETA              = 2.0                                # assimilation pull coefficient
DAMP_RATIO        = 0.99                               # revolution-rate decay per iteration
ZETA              = 0.02                               # weight of colony average in an empire's total cost
UNITING_THRESHOLD = 0.02                               # empires closer than this fraction of the box merge
COMPETITION_PROB  = 0.11                               # chance the competition phase fires per iteration


# =============================================================================
# SELF-NAMING: build the output names from the two switches
# -----------------------------------------------------------------------------
# This is what stops you ever overwriting a previous run by forgetting to
# rename: the filename is decided by the switches, not typed by hand.
# =============================================================================
algo_tag = 'namica' if USE_NAM else 'ica'              # NAM: the algorithm part of the name follows the switch: 'namica' when on, 'ica' when off
rate_tag = f"{int(round(REVOLUTION_RATE * 10)):02d}"   # the rate part: 0.3 becomes '03' and 0.5 becomes '05' (multiply by 10, round, pad to two digits)
VERSION  = f"{algo_tag}_{rate_tag}"                    # the full tag, e.g. 'ica_03' or 'namica_05', reused in both filenames and the console output
SUMMARY_CSV     = f"summary_{VERSION}.csv"             # per-function statistics file for this run
CONVERGENCE_CSV = f"convergence_{VERSION}.csv"         # full per-run, per-iteration curves file for this run


# =============================================================================
# RUN THE ALGORITHM ONCE WITH A FIXED SEED
# -----------------------------------------------------------------------------
# Seeding before each run makes the whole run repeatable and, because the seed
# depends only on the run index, run i of any version pairs with run i of any
# other version (needed for the paired Wilcoxon test later).
# =============================================================================
def single_run(func_info, seed):
    np.random.seed(seed)                               # fix randomness for this single run, so it is reproducible and paired across versions
    _, best_cost, history = ica(                       # call the one algorithm; the two switches decide which version actually runs
        cost_func    = func_info['func'],              # the function to minimize
        dims         = DIMS,                           # locked dimensions
        lb           = func_info['lb'],                # this function's lower bound
        ub           = func_info['ub'],                # this function's upper bound
        n_iterations = N_ITERATIONS,                   # locked iteration count
        n_countries       = N_COUNTRIES,               # locked population size, passed explicitly so it is auditable here
        n_imperialists    = N_IMPERIALISTS,            # locked starting empire count
        beta              = BETA,                      # locked assimilation coefficient
        damp_ratio        = DAMP_RATIO,                # locked revolution decay
        zeta              = ZETA,                      # locked colony weight
        uniting_threshold = UNITING_THRESHOLD,         # locked merge threshold
        competition_prob  = COMPETITION_PROB,          # locked competition probability
        rev_rate     = REVOLUTION_RATE,                # the revolution-rate switch feeds straight in here
        nonaligned   = USE_NAM)                        # NAM: the modification switch feeds straight in here. False reproduces ICA, True runs NAM-ICA.
    return best_cost, history                          # the final best cost and the full convergence curve for this run


# =============================================================================
# RUN ALL 7 FUNCTIONS, N_RUNS TIMES EACH
# =============================================================================
def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)            # create the results folder if it is not there yet

    all_results = {}                                   # collects every function's numbers, keyed by function name

    for fname, info in FUNCTIONS.items():              # loop over F1 through F7 in order
        print(f"Running {VERSION} on {fname} ({info['name']}) ...")  # progress line that names the exact version so the console is self-documenting
        start = time.time()                            # start the timer for this function

        best_costs    = []                             # the final best cost from each of the 30 runs
        all_histories = []                             # the full convergence curve from each of the 30 runs

        for run_idx in range(N_RUNS):                  # do 30 independent runs
            seed = run_idx + 1                         # seeds 1, 2, ..., 30, the same set every version uses, which is what pairs the runs
            best_cost, history = single_run(info, seed)  # run once with this seed
            best_costs.append(best_cost)               # store its final cost
            all_histories.append(history)              # store its full curve

        elapsed = time.time() - start                  # how long this function took
        print(f"  done in {elapsed:.1f}s "             # report timing plus a quick quality readout
              f"| mean={np.mean(best_costs):.3e} "
              f"| best={np.min(best_costs):.3e}")

        all_results[fname] = {                         # bundle this function's results
            'best_costs'   : best_costs,               # the 30 final costs
            'histories'    : all_histories,            # the 30 curves
            'name'         : info['name'],             # the human-readable name for the CSV
            'optimum'      : info['optimum'],          # the known global minimum for reference
        }

    return all_results                                 # all functions, ready to be written out


# =============================================================================
# SAVE PER-FUNCTION SUMMARY STATISTICS
# -----------------------------------------------------------------------------
# One row per function. Same columns as the baseline runner, so every later
# script can read ICA and NAM-ICA files with identical code.
# =============================================================================
def save_summary(results, path):
    with open(path, 'w', newline='') as f:             # open the file, newline='' so csv writes clean line endings
        writer = csv.writer(f)                         # a plain csv writer
        writer.writerow(['function', 'name', 'optimum',  # header row, matching the baseline format exactly
                         'min', 'max', 'mean', 'median', 'std'])
        for fname, data in results.items():            # one row per function
            bc = np.array(data['best_costs'])          # the 30 final costs as an array, so the statistics are easy
            writer.writerow([fname, data['name'], data['optimum'],  # identity columns
                             f"{bc.min():.6e}",        # best single run (accuracy ceiling)
                             f"{bc.max():.6e}",        # worst single run (accuracy floor)
                             f"{bc.mean():.6e}",       # average across runs
                             f"{np.median(bc):.6e}",   # median across runs, the headline robust number
                             f"{bc.std():.6e}"])       # spread across runs (robustness)
    print(f"Summary saved: {path}")                    # confirm where it went


# =============================================================================
# SAVE THE FULL CONVERGENCE CURVES
# -----------------------------------------------------------------------------
# One row per (function, run, iteration). Bigger file, but it lets the plotting
# and the Wilcoxon test pull any run or any iteration later.
# =============================================================================
def save_convergence(results, path):
    with open(path, 'w', newline='') as f:             # open the curves file
        writer = csv.writer(f)                         # plain csv writer
        writer.writerow(['function', 'run', 'iteration', 'best_cost'])  # header row
        for fname, data in results.items():            # for every function
            for run_idx, history in enumerate(data['histories']):  # for every run of that function
                for it_idx, cost in enumerate(history):  # for every iteration of that run
                    writer.writerow([fname, run_idx + 1, it_idx + 1,  # 1-based run and iteration numbers for readability
                                     f"{cost:.6e}"])   # the best cost recorded at that iteration
    print(f"Convergence curves saved: {path}")         # confirm where it went


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print(f"Experiment | version={VERSION} | USE_NAM={USE_NAM} | "  # echo the full configuration so the run is unambiguous
          f"rev_rate={REVOLUTION_RATE} | {DIMS}D | {N_RUNS} runs | "
          f"{N_ITERATIONS} iterations\n")

    results = run_all()                                # do all the work

    print("\nSaving CSV files ...")
    save_summary(results, os.path.join(RESULTS_DIR, SUMMARY_CSV))          # write the self-named summary
    save_convergence(results, os.path.join(RESULTS_DIR, CONVERGENCE_CSV))  # write the self-named curves

    print(f"\nDone. This run wrote {SUMMARY_CSV} and {CONVERGENCE_CSV}.")  # remind you exactly which files this run produced