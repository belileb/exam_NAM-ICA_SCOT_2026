# ICA Project: Imperialist Competitive Algorithm

Faithful Python implementation of the Imperialist Competitive Algorithm (ICA)
by Atashpaz-Gargari and Lucas (IEEE CEC 2007), benchmarked on 7 CEC test
functions.

This is the **baseline** implementation. A modified version will be added
later for comparative analysis.

## Files

| File | Purpose |
|---|---|
| `benchmarks.py` | 7 CEC benchmark functions (F1-F7) with bounds and known optima |
| `ica_original.py` | Faithful Python implementation of the original ICA algorithm |
| `run_experiments.py` | Runs ICA on all 7 functions, 30 runs each, saves CSV |
| `plot_results.py` | Generates convergence plots from the CSV files |
| `results/` | Output folder for CSVs and PNGs |

## Requirements

```
numpy
matplotlib
```

Install with:
```
pip install numpy matplotlib
```

## How to Run

Run the scripts in order:

```
python _benchmarks.py        # sanity check: verifies formulas (instant)
python _ica_original.py      # self-test on Rastrigin (a few seconds)
python _run_experiments.py   # full benchmark: 30 runs x 7 functions (~2-3 min)
python _plot_results.py      # generates plots from CSV (instant)
```

After running, check the `results/` folder for:
- `summary_original.csv` final cost statistics per function
- `convergence_original.csv` full convergence curves
- `convergence_F1.png` ... `convergence_F7.png` per-function plots
- `convergence_all.png` 3x3 grid of all functions
- `summary_bars.png` bar chart of mean final costs

## Configuration

Default experiment settings (in `run_experiments.py`):
- 10 dimensions
- 30 independent runs per function
- 2000 iterations per run

Default ICA parameters (in `ica_original.py`, from original MATLAB):
- 300 countries
- 20 starting imperialists
- assimilation coefficient = 2.0
- initial revolution rate = 0.3
- revolution damp ratio = 0.99 (decay per iteration)
- zeta (colony weight) = 0.02
- uniting threshold = 0.02
- competition probability = 0.11

Change these by editing the constants at the top of each file.

## References

- Atashpaz-Gargari E, Lucas C (2007). "Imperialist competitive algorithm:
  an algorithm for optimization inspired by imperialistic competition."
  IEEE Congress on Evolutionary Computation, pp. 4661-4667.
  DOI: 10.1109/CEC.2007.4425083
- Original MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/22046
