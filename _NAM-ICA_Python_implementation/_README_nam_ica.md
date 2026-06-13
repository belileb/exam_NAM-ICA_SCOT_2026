# NAM-ICA: Non-Aligned Movement modification of the Imperialist Competitive Algorithm

This adds ONE modification (NAM-ICA) on top of plain ICA and compares
the two on the seven CEC functions (F1 to F7). 
The NAM modification lives behind a single on/off flag in _run_experiments_nam.py file, 
so ICA and NAM-ICA are identical everywhere except the one rule that changes.

## What the modification is

In plain ICA, during competition the colony taken from the weakest empire can
be handed to ANY other empire from round one, so the strongest empire swallows
far-away rivals early and the search collapses too soon (premature convergence).

NAM-ICA restricts the receiver to the weakest empire's `k` nearest empires,
where `k` starts at 2 and grows in a straight line to global by the final iteration. 
Early world is local and multipolar, late world is global, so the
run still ends with a single winner exactly like ICA. 
The only constant is the starting neighbourhood size of 2, 
which is a fixed design choice, not a tuned parameter.

## Deliverables

**IMPORTANT: Run files exactly in order:**

| Run | File                      | Job                                                                                                              |
|-----|---------------------------|------------------------------------------------------------------------------------------------------------------|
| 1.  | `_README_nam_ica.md`      | This file. |
| 2.  | `_benchmarks.py`          | 7 Benchmark functions. |
| 3.  | `_nam_ica.py`             | The algorithm. Plain ICA when `nonaligned=False`, NAM-ICA when `True`. Holds the two `nam_*` helpers. |
| 4.  | `_run_experiments_nam.py` | The experiment runner. Two switches at the top, self-naming output. Must be ran four times. |
| 5.  | `_compare_and_stats.py`   | Final analysis of all four runs. Overlaid curves, combined table, Wilcoxon tests. Only file that uses scipy. |
| 6.  | `_plot_results_nam.py`    | Optional: Per-function convergence plots for one NAM-ICA run. |

Original `_ica_original.py` is not needed for this workflow (as the unified `_nam_ica.py` file reproduces it), but is kept for reference.

## Requirements

```
pip install numpy matplotlib scipy
```

`numpy` and `matplotlib` are used everywhere. `scipy` is used in one place only,
in the Wilcoxon signed-rank test inside `_compare_and_stats.py`. It is not imported
by the algorithm.

## The two switches (in `_run_experiments_nam.py`)

```
REVOLUTION_RATE = 0.3     # 0.3 for the normal runs, 0.5 for the high-randomness runs
USE_NAM         = False   # False = plain ICA, True = NAM-ICA
```

The output filenames are built from these two switches.

## Run order

All commands are run from the project folder. 

**The four benchmark runs (about 2 to 3 minutes each)**
Set the two switches, run, then change the switches and run again. Four times.

| Set switches to                        | Command                          | Produces                                             |
|----------------------------------------|----------------------------------|------------------------------------------------------|
| `USE_NAM=False`, `REVOLUTION_RATE=0.3` | `python _run_experiments_nam.py` | `summary_ica_03.csv`, `convergence_ica_03.csv` |
| `USE_NAM=False`, `REVOLUTION_RATE=0.5` | `python _run_experiments_nam.py` | `summary_ica_05.csv`, `convergence_ica_05.csv` |
| `USE_NAM=True`,  `REVOLUTION_RATE=0.3` | `python _run_experiments_nam.py` | `summary_namica_03.csv`, `convergence_namica_03.csv` |
| `USE_NAM=True`,  `REVOLUTION_RATE=0.5` | `python _run_experiments_nam.py` | `summary_namica_05.csv`, `convergence_namica_05.csv` |

All eight CSVs land in the `results/` folder.

**No-leak check:** the `USE_NAM=False, REVOLUTION_RATE=0.5` run reproduce original `_ica_original.py` runs. 
Compare with reference and if the numbers match, the modification has not leaked into the base.

**Plot each NAM-ICA run (optional)**
```
python _plot_results_nam.py            # set TAG = 'namica_03' inside, then rerun with TAG = 'namica_05'
```
Produces per-function convergence plots, a 3x3 grid, and a bar chart for the chosen NAM-ICA run.

**The comparison and statistics (must be run last)**
```
python _compare_and_stats.py
```
Loads all four runs and produces, into `results/`:
- `compare_singlevalley_F1.png`, `_F3`, `_F5` and `compare_manyvalley_F2.png`,
  `_F4`, `_F6`, `_F7`: one overlaid figure per function with all four median
  curves (ICA and NAM-ICA at both rates), showing speed and accuracy together.
- `comparison_table.csv`: optimum, min, max, mean, median and std per function
  for all four versions, the accuracy and robustness table.
- `wilcoxon_results.csv`: the signed-rank tests, NAM-ICA 0.3 vs ICA 0.3 (the
  headline) and NAM-ICA 0.5 vs ICA 0.5 (the robustness check), with the p-value
  and which version won by median.

## What to look for in the results

The success bar from plain ICA runs (median of 30):
- ICA 0.3 leaves F6 Griewank at 0.655 and F7 Ackley at 1.231.
- ICA 0.5 pulled those to 0.552 and 0.903 but damaged F4 HGBat (0.14 to 1.25)
  and F5 Rosenbrock (0.067 to 0.14).

For NAM-ICA to count it should improve F6 and F7 (the multimodal targets) while
avoiding the F4 and F5 damage that raw randomness caused. F1 and F2 are already
solved (no room). F3 is ill-conditioned and outside this modification's scope.

## Fairness of the comparison

Every version uses the same locked settings and the same paired seeds (run i
uses seed i in every version). 
**The modification only measures distances between imperialist positions and never calls the cost function, so both versions spend the exact same number of cost-function evaluations per iteration.** 
Matching the iteration count therefore matches the evaluation budget.

## Locked settings (identical for every run)

10 dimensions, 30 runs per function, 2000 iterations, 300 countries, 20 starting
imperialists, assimilation coefficient 2.0, revolution damp ratio 0.99, zeta
0.02, uniting threshold 0.02, competition probability 0.11. 
**The only setting that changes between runs is the revolution rate (0.3 or 0.5).**