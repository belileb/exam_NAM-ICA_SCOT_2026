# =============================================================================
# benchmarks.py
# -----------------------------------------------------------------------------
# The 7 benchmark (test) functions used to evaluate the ICA algorithm.
# These are taken from the project assignment (CEC benchmark suite).
#
# WHY DO WE NEED BENCHMARK FUNCTIONS?
# We need a way to test if our optimization algorithm is working correctly.
# Benchmark functions are mathematical functions where we KNOW the answer
# (the global minimum), so we can check how close the algorithm gets.
#
# HOW THEY WORK:
# Each function takes a position vector x (a list of numbers, one per dimension)
# and returns a single number (the "cost" or "fitness").
# The algorithm tries to find the x that gives the LOWEST cost.
# All 7 functions have their global minimum (lowest possible value) at or
# near the origin (all zeros), unless stated otherwise.
#
# DIMENSIONS:
# All functions are generalized to n dimensions (n = length of x).
# Default is 10 dimensions. This can be changed in run_experiments.py.
#
# BOUNDS (search space limits):
# Each function has a recommended range for x values, listed below.
# The algorithm will only search within these bounds.
# =============================================================================

import numpy as np


# -----------------------------------------------------------------------------
# F1: Bent Cigar Function
# -----------------------------------------------------------------------------
# SHAPE: A long, narrow valley (like a cigar) pointing toward the origin.
#        Very easy to find the direction, very hard to find the exact minimum.
# FORMULA: f(x) = x_1^2 + 10^6 * sum( x_i^2 )   for i = 2..n
# FORMULA: f(x) = x[0]^2 + 10^6 * sum(x[1:]^2)
# MINIMUM: f(0,...,0) = 0
# BOUNDS:  -100 <= xi <= 100
# WHY IT'S HARD: The first dimension is easy, but all others are scaled up
#                by 10^6, creating extreme sensitivity in those directions.
# -----------------------------------------------------------------------------
def f1_bent_cigar(x):
    x = np.array(x)                                    # ensure numpy array
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)            # first dimension cheap, rest expensive


# -----------------------------------------------------------------------------
# F2: Rastrigin Function
# -----------------------------------------------------------------------------
# SHAPE: Many, many local minima arranged in a regular grid (very bumpy).
#        The global minimum is at the center, but hard to find among the bumps.
# FORMULA: f(x, y) = 2A + x^2 - A*cos(2*pi*x) + y^2 - A*cos(2*pi*y)
# FORMULA: f(x) = 10n + sum(xi^2 - 10*cos(2*pi*xi))
# MINIMUM: f(0,...,0) = 0
# BOUNDS:  -5.12 <= xi <= 5.12
# WHY IT'S HARD: The cosine term creates ~10^n local minima. Easy to get
#                trapped in a nearby bump instead of the true minimum.
# -----------------------------------------------------------------------------
def f2_rastrigin(x):
    A = 10
    x = np.array(x)                                   # ensure numpy array
    n = len(x)                                        # number of dimensions
    return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x)) # standard nD Rastrigin


# -----------------------------------------------------------------------------
# F3: High Conditioned Elliptic Function
# -----------------------------------------------------------------------------
# SHAPE: An ellipsoid stretched enormously along later dimensions.
#        Like a very flat, elongated bowl.
# FORMULA: f(x) = sum( a^((i-1)/(d-1)) * x_i^2 )   for i = 1..n, a = 10^6
# MINIMUM: f(0,...,0) = 0
# BOUNDS:  -100 <= xi <= 100
# WHY IT'S HARD: The condition number (ratio of largest to smallest scaling)
#                is 10^6, making gradient-based methods very slow.
#                Good test of how well the algo handles ill-conditioned problems.
# -----------------------------------------------------------------------------
def f3_elliptic(x):
    x = np.array(x)                                    # ensure numpy array
    n = len(x)                                         # number of dimensions
    # exponent grows from 0 to 6 across dimensions
    exponents = 6 * np.arange(n) / max(n - 1, 1)      # avoid division by zero if n=1
    scales    = 10 ** exponents                        # scaling per dimension
    return np.sum(scales * x**2)                       # weighted sum of squares


# -----------------------------------------------------------------------------
# F4: HGBat Function (assignment-defined HGBat variant)
# -----------------------------------------------------------------------------
# SHAPE: Non-convex, asymmetric. Shifted minimum not at origin.
# ASSIGNMENT FORMULA: f(x_1, x_2) = ( sum_{i=1..2} (x_i - a_i)^2 )^2 + ( sum_{i=1..2} (x_i - a_i) )^2 / 10^6
# FORMULA: f(x) = (sum((xi-ai)^2))^2 + (sum((xi-ai)^2)) / 10^6
#          where a = [-10, -5, -10, -5, ...] alternating
# MINIMUM: near x = a (all variables at their shift values)
# BOUNDS:  -100 <= xi <= 100
# NOTE: Assignment gives a1=-10, a2=-5 for 2D. We generalize by alternating
#       these two shift values across all dimensions.
# WHY IT'S HARD: The squared-sum term creates a flat region near the minimum,
#                making it hard to pinpoint the exact location.
# -----------------------------------------------------------------------------
def f4_hgbat(x):
    x      = np.array(x)                               # ensure numpy array
    n      = len(x)                                    # number of dimensions
    # build shift vector: alternates -10 and -5 across dimensions
    shifts = np.array([-10 if i % 2 == 0 else -5       # Build shift vector: [-10, -5, -10, -5, ...] 
                       for i in range(n)])             # Generalizing the assignment's a1=-10, a2=-5
    d      = x - shifts                                # deviation from shift
    s      = np.sum(d**2)                              # sum of squared deviations
    return s**2 + s / 1e6                              # HGBat formula


# -----------------------------------------------------------------------------
# F5: Rosenbrock Function
# -----------------------------------------------------------------------------
# SHAPE: A long, curved, banana-shaped valley. The minimum is inside the valley
#        but the valley itself is very flat, making it hard to follow.
# FORMULA: f(x, y) = 100*(y - x^2)^2 + (1 - x)^2
# FORMULA: f(x) = sum( 100*(x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 )
# MINIMUM: f(1,...,1) = 0   (NOT at origin!)
# BOUNDS:  -5 <= xi <= 10
# WHY IT'S HARD: Easy to find the valley, very hard to roll to the bottom.
#                Classic test for algorithms that exploit gradient information.
# -----------------------------------------------------------------------------
def f5_rosenbrock(x):
    x = np.array(x)                                    # ensure numpy array
    # sum pairs of consecutive dimensions
    return np.sum(100*(x[1:] - x[:-1]**2)**2          # banana curve term
                  + (x[:-1] - 1)**2)                  # distance from 1 term


# -----------------------------------------------------------------------------
# F6: Griewank Function
# -----------------------------------------------------------------------------
# SHAPE: Many local minima on top of a large parabolic bowl.
#        At large scale looks like a bowl; zoom in and it's very bumpy.
# FORMULA: f(x) = 1 + (1/4000) * sum_{i=1..D-1}( x_i^2 ) - prod_{i=1..D}( cos( x_i / sqrt(i) ) )
# FORMULA: f(x) = 1 + (1/4000)*sum(xi^2) - prod(cos(xi/sqrt(i)))
# MINIMUM: f(0,...,0) = 0
# BOUNDS:  -600 <= xi <= 600
# WHY IT'S HARD: The product term creates many local minima. However, they
#                become less pronounced in high dimensions (self-regularizing).
# -----------------------------------------------------------------------------
def f6_griewank(x):
    x    = np.array(x)                                 # ensure numpy array
    n    = len(x)                                      # number of dimensions
    i    = np.arange(1, n+1)                           # indices 1..n for formula
    sum_term  = np.sum(x**2) / 4000                    # bowl component
    prod_term = np.prod(np.cos(x / np.sqrt(i)))        # oscillation component
    return 1 + sum_term - prod_term                    # Griewank formula


# -----------------------------------------------------------------------------
# F7: Ackley Function
# -----------------------------------------------------------------------------
# SHAPE: Nearly flat outer region with a deep hole at the center.
#        Many local minima on the flat region, one sharp global minimum.
# FORMULA: f(x, y) = -20 * exp( -0.2 * sqrt( 0.5*(x^2 + y^2) ) ) - exp( 0.5*(cos(2*pi*x) + cos(2*pi*y)) ) + e + 20
# FORMULA: f(x) = -20*exp(-0.2*sqrt(mean(xi^2))) - exp(mean(cos(2*pi*xi))) + e + 20
# MINIMUM: f(0,...,0) = 0
# BOUNDS:  -32.768 <= xi <= 32.768
# WHY IT'S HARD: The exponential decay makes the landscape look flat from far
#                away, luring algorithms away from the global minimum.
# -----------------------------------------------------------------------------
def f7_ackley(x):
    x   = np.array(x)                                  # ensure numpy array
    n   = len(x)                                       # number of dimensions
    # first exponential term: penalizes distance from origin
    t1  = -0.2 * np.sqrt(np.sum(x**2) / n)            # mean squared distance
    # second exponential term: rewards alignment with cosine peaks
    t2  = np.sum(np.cos(2 * np.pi * x)) / n           # mean cosine
    return -20*np.exp(t1) - np.exp(t2) + np.e + 20    # Ackley formula


# =============================================================================
# FUNCTION REGISTRY
# -----------------------------------------------------------------------------
# A single dictionary that maps function number to:
#   - the function itself
#   - its name (for plots and reports)
#   - its recommended search bounds
#   - its known global minimum value
#
# WHY A REGISTRY?
# Instead of writing if/else chains everywhere, the dictionary keeps things up in one place.
# run_experiments.py loops over this dict to test all 7 functions automatically.
# =============================================================================
FUNCTIONS = {
    'F1': {
        'func'    : f1_bent_cigar,                     # the actual function
        'name'    : 'Bent Cigar',                      # human-readable name
        'lb'      : -100,                              # lower bound per dimension
        'ub'      :  100,                              # upper bound per dimension
        'optimum' : 0.0,                               # known global minimum
    },
    'F2': {
        'func'    : f2_rastrigin,
        'name'    : 'Rastrigin',
        'lb'      : -5.12,
        'ub'      :  5.12,
        'optimum' : 0.0,
    },
    'F3': {
        'func'    : f3_elliptic,
        'name'    : 'High Conditioned Elliptic',
        'lb'      : -100,
        'ub'      :  100,
        'optimum' : 0.0,
    },
    'F4': {
        'func'    : f4_hgbat,
        'name'    : 'HGBat',
        'lb'      : -100,
        'ub'      :  100,
        'optimum' : 0.0,
    },
    'F5': {
        'func'    : f5_rosenbrock,
        'name'    : 'Rosenbrock',
        'lb'      : -5,
        'ub'      :  10,
        'optimum' : 0.0,
    },
    'F6': {
        'func'    : f6_griewank,
        'name'    : 'Griewank',
        'lb'      : -600,
        'ub'      :  600,
        'optimum' : 0.0,
    },
    'F7': {
        'func'    : f7_ackley,
        'name'    : 'Ackley',
        'lb'      : -32.768,
        'ub'      :  32.768,
        'optimum' : 0.0,
    },
}


# =============================================================================
# QUICK SELF-TEST
# -----------------------------------------------------------------------------
# Running this file directly (python benchmarks.py) verifies all 7 functions
# return 0 (or very close to 0) at the known global minimum position.
# This is a sanity check -- if any function fails, the formula is wrong.
# =============================================================================
if __name__ == '__main__':
    print("Sanity check: evaluating each function at its known minimum.")
    print("Expected result: 0.0 (or very close) for all functions.\n")

    # test points at the known minimum for each function
    test_points = {
        'F1': np.zeros(10),                            # minimum at origin
        'F2': np.zeros(10),                            # minimum at origin
        'F3': np.zeros(10),                            # minimum at origin
        'F4': np.array([-10,-5,-10,-5,-10,             # minimum at shift vector
                        -5,-10,-5,-10,-5]),
        'F5': np.ones(10),                             # minimum at all-ones
        'F6': np.zeros(10),                            # minimum at origin
        'F7': np.zeros(10),                            # minimum at origin
    }

    for fname, info in FUNCTIONS.items():
        val = info['func'](test_points[fname])         # evaluate at known minimum
        status = 'OK' if abs(val) < 1e-6 else 'FAIL'   # pass if close enough to 0
        print(f"  {fname} {info['name']:35s} f(x*) = {val:.6e}  [{status}]")
