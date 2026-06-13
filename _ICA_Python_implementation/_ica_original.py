# =============================================================================
# _ica_original.py
# -----------------------------------------------------------------------------
# Python implementation of the Imperialist Competitive Algorithm (ICA)
# from the original paper and MATLAB code by Esmaeil Atashpaz Gargari (2007).
#
# PAPER : Atashpaz-Gargari E, Lucas C (2007)
#         "Imperialist competitive algorithm: an algorithm for optimization
#          inspired by imperialistic competition"
#         IEEE Congress on Evolutionary Computation, pp. 4661-4667
# DOI   : https://doi.org/10.1109/CEC.2007.4425083
#
# WHAT IS ICA?
# ICA is a population-based optimization algorithm inspired by political
# imperialism. The "population" is a set of countries (candidate solutions).
# The best countries become "imperialists" and the rest become "colonies".
# Each imperialist plus its colonies forms an "empire". Empires compete:
# strong ones grow, weak ones shrink and eventually collapse. Over time the
# population converges to the best solution (one empire ruling all colonies).
#
# THE FIVE PHASES (one iteration = one "decade"):
#   1. ASSIMILATION       colonies move toward their imperialist
#   2. REVOLUTION         some colonies teleport to random locations
#   3. POSSESSION         if a colony beats its imperialist, they swap roles
#   4. UNITING            empires too close in space merge into one
#   5. COMPETITION        weakest empire loses a colony to a stronger one
#                         empires with zero colonies collapse and disappear
#
# WHY THESE PHASES?
#   ASSIMILATION  drives exploitation (refining around good solutions)
#   REVOLUTION    drives exploration (escaping local optima)
#   POSSESSION    promotes the best solution in each empire to leadership
#   UNITING       prevents redundant empires clustering on the same optimum
#   COMPETITION   gradually eliminates weak solutions, focuses search
#
# FAITHFULNESS NOTES:
# This implementation matches the original MATLAB code, including:
#   - UniteSimilarEmpires phase (often omitted in simplified versions)
#   - Revolution rate decays each iteration (DampRatio = 0.99)
#   - Competition fires probabilistically (~11% chance per iteration)
#   - Linear power distribution for empire formation (not exponential)
#   - Revolution = random teleport (not Gaussian nudge)
#   - Revolution applies to colonies only, NOT to imperialists
# =============================================================================

import numpy as np


# =============================================================================
# PHASE 1: ASSIMILATION
# -----------------------------------------------------------------------------
# Each colony moves toward its imperialist by a random fraction of the
# distance between them. This is the "exploitation" phase: good solutions
# (imperialists) attract nearby candidates (colonies) to refine the search.
#
# FORMULA (from original MATLAB):
#   new_position = old_position + 2 * beta * rand() * (imperialist - colony)
#
# The factor of 2 ensures colonies can sometimes overshoot the imperialist,
# which helps explore the area BEYOND it (not just the segment between them).
# =============================================================================
def assimilate(colonies, imperialist, beta, lb, ub):
    n_col   = len(colonies)                            # number of colonies in empire
    vector  = imperialist - colonies                   # direction from each colony to imp
    rand    = np.random.random(colonies.shape)         # random factor per dimension per colony
    new_pos = colonies + 2 * beta * rand * vector      # move colonies toward imperialist # ASSIMILATION FORMULA
    return np.clip(new_pos, lb, ub)                    # keep within search bounds


# =============================================================================
# PHASE 2: REVOLUTION
# -----------------------------------------------------------------------------
# A fraction of colonies are randomly replaced with brand new countries.
# This is the "exploration" phase: it prevents the population from getting
# stuck in a local minimum by injecting fresh random candidates.
#
# In the original, the revolution rate DECAYS over time (multiplied by 0.99
# each iteration). Early on: many colonies revolt (lots of exploration).
# Later on: few revolt (focus on exploitation).
#
# Revolution affects ONLY colonies, never the imperialist (original behavior).
# =============================================================================
def revolve(colonies, rev_rate, lb, ub):
    n_col       = len(colonies)                        # number of colonies
    n_revolve   = int(round(rev_rate * n_col))         # how many colonies will revolt
    if n_revolve == 0:                                 # no revolution this round
        return colonies
    idx         = np.random.permutation(n_col)[:n_revolve]  # pick random colonies
    dims        = colonies.shape[1]                    # number of dimensions
    new_random  = np.random.uniform(lb, ub,            # generate fresh random countries
                                    (n_revolve, dims))
    colonies[idx] = new_random                         # replace selected colonies
    return colonies


# =============================================================================
# PHASE 3: POSSESSION (intra-empire competition)
# -----------------------------------------------------------------------------
# After assimilation and revolution, a colony may have become better than its
# imperialist. If so, they swap roles: the colony becomes the new imperialist
# and the old imperialist becomes a colony. This ensures the best solution
# in each empire always leads it.
# =============================================================================
def possess(colonies, colony_costs, imperialist, imp_cost):
    best_col_idx  = np.argmin(colony_costs)            # find best colony in empire
    best_col_cost = colony_costs[best_col_idx]

    if best_col_cost < imp_cost:                       # colony beats imperialist
        new_imp        = colonies[best_col_idx].copy() # promote colony
        new_imp_cost   = best_col_cost
        colonies[best_col_idx]     = imperialist       # demote imperialist
        colony_costs[best_col_idx] = imp_cost
        return colonies, colony_costs, new_imp, new_imp_cost

    return colonies, colony_costs, imperialist, imp_cost


# =============================================================================
# PHASE 4: UNITING SIMILAR EMPIRES
# -----------------------------------------------------------------------------
# If two imperialists end up very close in search space, they are essentially
# converging on the same solution. The weaker empire is absorbed by the
# stronger one (its imperialist becomes a colony of the winner).
#
# "Very close" means within UnitingThreshold fraction of the search space.
# Original default: 0.02 (2% of search space size).
#
# This prevents redundant empires from wasting effort on the same optimum.
# =============================================================================
def unite_similar(empires, uniting_threshold, lb, ub):
    search_size = np.linalg.norm(np.array(ub) - np.array(lb))  # diagonal of search box
    threshold   = uniting_threshold * search_size      # distance considered "too close"

    n_emp = len(empires)
    for i in range(n_emp - 1):                         # check every pair of empires
        for j in range(i + 1, n_emp):
            if empires[i] is None or empires[j] is None:  # already merged this round
                continue
            distance = np.linalg.norm(                 # euclidean distance between imps
                empires[i]['imp'] - empires[j]['imp'])
            if distance <= threshold:                  # too close: merge them
                # decide which empire is stronger (lower cost wins)
                if empires[i]['imp_cost'] < empires[j]['imp_cost']:
                    strong, weak = i, j
                else:
                    strong, weak = j, i
                # weak imperialist becomes a colony of the strong empire
                empires[strong]['colonies']     = np.vstack([
                    empires[strong]['colonies'],
                    empires[weak]['imp'].reshape(1, -1),
                    empires[weak]['colonies']])
                empires[strong]['colony_costs'] = np.concatenate([
                    empires[strong]['colony_costs'],
                    [empires[weak]['imp_cost']],
                    empires[weak]['colony_costs']])
                empires[weak] = None                   # mark weak empire for removal

    return [e for e in empires if e is not None]       # drop merged empires


# =============================================================================
# PHASE 5: IMPERIALISTIC COMPETITION
# -----------------------------------------------------------------------------
# The weakest empire loses its weakest colony to a probabilistically chosen
# stronger empire. If an empire ends up with zero colonies, it collapses
# (its imperialist becomes a colony of the winner empire).
#
# In the original MATLAB, this phase only fires with probability 0.11 per
# iteration, NOT every iteration. This slows down the competitive pressure
# and gives empires more time to develop their colonies before competing.
#
# TOTAL COST of an empire = imperialist cost + zeta * mean(colony costs)
# Empires are ranked by total cost. Highest total cost = weakest empire.
# =============================================================================
def compete(empires, zeta, competition_prob):
    if np.random.random() > competition_prob:          # competition does not fire
        return empires
    if len(empires) <= 1:                              # only one empire left
        return empires

    # compute total cost for each empire
    total_costs = []
    for emp in empires:
        if len(emp['colony_costs']) > 0:               # empire has colonies
            tc = emp['imp_cost'] + zeta * np.mean(emp['colony_costs'])
        else:                                          # empire has no colonies
            tc = emp['imp_cost']
        total_costs.append(tc)
    total_costs = np.array(total_costs)

    weakest_idx  = int(np.argmax(total_costs))         # highest total cost = weakest
    weakest      = empires[weakest_idx]

    if len(weakest['colony_costs']) == 0:              # already empty: nothing to steal
        return empires

    # take the worst colony from the weakest empire
    worst_col_idx = int(np.argmax(weakest['colony_costs']))
    stolen_pos    = weakest['colonies'][worst_col_idx].copy()
    stolen_cost   = weakest['colony_costs'][worst_col_idx]

    # remove the stolen colony from the weakest empire
    weakest['colonies']     = np.delete(weakest['colonies'],
                                        worst_col_idx, axis=0)
    weakest['colony_costs'] = np.delete(weakest['colony_costs'],
                                        worst_col_idx)

    # pick a winner empire probabilistically (stronger = higher chance)
    powers = np.max(total_costs) - total_costs         # invert: best empire = highest
    powers[weakest_idx] = 0                            # weakest cannot steal from itself
    if powers.sum() > 0:
        probs  = powers / powers.sum()                 # normalize to probabilities
        winner_idx = int(np.random.choice(len(empires), p=probs))
    else:                                              # edge case: all equal cost
        winner_idx = (weakest_idx + 1) % len(empires)

    # winner absorbs the stolen colony
    empires[winner_idx]['colonies']     = np.vstack([
        empires[winner_idx]['colonies'],
        stolen_pos.reshape(1, -1)])
    empires[winner_idx]['colony_costs'] = np.concatenate([
        empires[winner_idx]['colony_costs'],
        [stolen_cost]])

    # collapse: if weakest empire has no colonies left, it disappears
    if len(weakest['colony_costs']) == 0:
        empires[winner_idx]['colonies']     = np.vstack([
            empires[winner_idx]['colonies'],
            weakest['imp'].reshape(1, -1)])
        empires[winner_idx]['colony_costs'] = np.concatenate([
            empires[winner_idx]['colony_costs'],
            [weakest['imp_cost']]])
        empires.pop(weakest_idx)                       # remove the dead empire

    return empires


# =============================================================================
# INITIALIZATION: CREATE INITIAL EMPIRES
# -----------------------------------------------------------------------------
# 1. Generate random countries (candidate solutions).
# 2. Evaluate cost of each country.
# 3. Sort by cost (lowest = best).
# 4. The top n_imp countries become imperialists, rest become colonies.
# 5. Each imperialist receives a number of colonies proportional to its power
#    (more powerful = lower cost = more colonies).
# =============================================================================
def create_initial_empires(cost_func, n_countries, n_imp, dims, lb, ub):
    # generate random initial population
    countries = np.random.uniform(lb, ub, (n_countries, dims))
    costs     = np.array([cost_func(c) for c in countries])

    # sort by cost: best countries first
    order     = np.argsort(costs)
    countries = countries[order]
    costs     = costs[order]

    # first n_imp are imperialists, the rest are colonies-to-be-assigned
    imp_pos  = countries[:n_imp]
    imp_cost = costs[:n_imp]
    col_pos  = countries[n_imp:]
    col_cost = costs[n_imp:]

    # compute imperialist power (linear formula from original MATLAB)
    # if max cost > 0: use 1.3 * max - cost  (gives best imp the highest power)
    # else:            use 0.7 * max - cost  (sign-flipped case)
    max_cost = np.max(imp_cost)
    if max_cost > 0:
        powers = 1.3 * max_cost - imp_cost
    else:
        powers = 0.7 * max_cost - imp_cost

    # normalize and compute colony allocation per empire
    powers     = powers / powers.sum()                        # divide by sum → proportions that add to 1 (probabilities)
    n_colonies = np.round(powers * len(col_pos)).astype(int)  # multiply by total colonies, round to integers
    # adjust last empire to absorb rounding error
    n_colonies[-1] = len(col_pos) - n_colonies[:-1].sum()     # give leftover colonies (from rounding) to the last empire

    # randomly distribute colonies to empires
    rand_idx = np.random.permutation(len(col_pos))     # shuffle colony indices
    empires  = []
    cursor   = 0
    for i in range(n_imp):
        take       = n_colonies[i]                     # how many colonies for this empire
        chosen     = rand_idx[cursor:cursor + take]    # which colonies
        cursor    += take
        empires.append({
            'imp'          : imp_pos[i].copy(),        # imperialist position
            'imp_cost'     : imp_cost[i],              # imperialist cost
            'colonies'     : col_pos[chosen].copy(),   # colony positions
            'colony_costs' : col_cost[chosen].copy(),  # colony costs
        })

    return empires


# =============================================================================
# MAIN ICA LOOP
# -----------------------------------------------------------------------------
# Runs the full algorithm and returns the best solution found, the best
# cost, and the convergence history (best cost per iteration).
#
# PARAMETERS (defaults from original MATLAB):
#   cost_func          function to minimize
#   dims               number of dimensions of the search space
#   lb, ub             lower and upper bounds of each dimension
#   n_countries        total population size                       (300)
#   n_imperialists     number of starting empires                  (20)
#   n_iterations       max iterations (decades)                    (2000)
#   beta               assimilation coefficient                    (2.0)
#   rev_rate           initial revolution rate                     (0.3)
#   damp_ratio         multiply rev_rate by this each iteration    (0.99)
#   zeta               weight of colony mean cost in empire power  (0.02)
#   uniting_threshold  empires closer than this fraction merge     (0.02)
#   competition_prob   probability competition phase fires         (0.11)
# =============================================================================
def ica(cost_func, dims, lb, ub,
        n_countries       = 300,
        n_imperialists    = 20,
        n_iterations      = 2000,
        beta              = 2.0,
        rev_rate          = 0.5,
        damp_ratio        = 0.99,
        zeta              = 0.02,
        uniting_threshold = 0.02,
        competition_prob  = 0.11):

    # turn lb and ub into arrays so they work with vector operations
    lb = np.full(dims, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(dims, ub) if np.isscalar(ub) else np.array(ub)

    # set up initial empires
    empires = create_initial_empires(cost_func, n_countries,
                                     n_imperialists, dims, lb, ub)

    history = []                                       # best cost per iteration

    # main loop: each iteration is one "decade"
    for iteration in range(n_iterations):

        # decay revolution rate over time (more exploration early, less later)
        rev_rate = rev_rate * damp_ratio

        # PHASE 1-3: per-empire operations
        for emp in empires:
            if len(emp['colonies']) == 0:              # empire has no colonies
                continue

            # phase 1: assimilate
            emp['colonies'] = assimilate(emp['colonies'], emp['imp'],
                                         beta, lb, ub)

            # phase 2: revolve
            emp['colonies'] = revolve(emp['colonies'], rev_rate, lb, ub)

            # re-evaluate colony costs after movement
            emp['colony_costs'] = np.array([cost_func(c)
                                            for c in emp['colonies']])

            # phase 3: possession (intra-empire swap if colony beats imp)
            (emp['colonies'], emp['colony_costs'],
             emp['imp'],      emp['imp_cost']) = possess(
                emp['colonies'], emp['colony_costs'],
                emp['imp'],      emp['imp_cost'])

        # PHASE 4: unite empires that are too close in search space
        empires = unite_similar(empires, uniting_threshold, lb, ub)

        # PHASE 5: imperialistic competition (fires probabilistically)
        empires = compete(empires, zeta, competition_prob)

        # record best cost this iteration
        best_cost = min(emp['imp_cost'] for emp in empires)
        history.append(best_cost)

        # early stop if only one empire remains
        if len(empires) == 1:
            # pad history so all runs have same length (for averaging later)
            while len(history) < n_iterations:
                history.append(best_cost)
            break

    # find the overall best solution
    best_emp  = min(empires, key=lambda e: e['imp_cost'])
    best_pos  = best_emp['imp']
    best_cost = best_emp['imp_cost']

    return best_pos, best_cost, history


# =============================================================================
# QUICK SELF-TEST
# -----------------------------------------------------------------------------
# Run this file directly to verify the algorithm works on Rastrigin.
# Expected: best cost should be very close to 0.
# =============================================================================
if __name__ == '__main__':
    from _benchmarks import FUNCTIONS

    np.random.seed(42)                                 # reproducible test
    f2 = FUNCTIONS['F2']                               # Rastrigin

    print(f"Self-test: running ICA on {f2['name']} (10D, 500 iterations)...")
    best_pos, best_cost, history = ica(
        cost_func    = f2['func'],
        dims         = 10,
        lb           = f2['lb'],
        ub           = f2['ub'],
        n_iterations = 500)

    print(f"Best cost found    : {best_cost:.6f}")
    print(f"Known global min   : {f2['optimum']}")
    print(f"Best position      : {np.round(best_pos, 4)}")
    print(f"Iterations recorded: {len(history)}")
