# =============================================================================
# _nam_ica.py
# -----------------------------------------------------------------------------
# Imperialist Competitive Algorithm (ICA) with an optional Nonaligned (NAM)
# modification. Both live in ONE file behind a single on/off switch.
#
# WHY ONE FILE WITH A FLAG (not two copies):
# Keeping ICA and NAM-ICA in the same code guarantees they are byte for byte
# identical everywhere EXCEPT the one rule we change. That is what makes the
# later comparison fair: any difference in the results can only come from the
# modification, never from some stray edit that drifted between two files.
#
# THE SWITCH:
#   nonaligned = False  ->  plain ICA, exactly the original algorithm
#   nonaligned = True   ->  NAM-ICA, the modification is active
#
# WHAT THE MODIFICATION CHANGES (one sentence):
# During imperialistic competition the colony taken from the weakest empire
# may, early in the run, only be handed to that empire's nearest rivals, and
# the circle of allowed rivals widens to the whole world by the end.
#
# WHY THAT HELPS (the Non-Aligned Movement metaphor):
# In plain ICA the strongest empire can swallow far-away rivals from round one,
# so the whole search collapses onto one region too fast (premature
# convergence). Forbidding that early long reach keeps several distant regions
# alive and exploring, then lets them compete globally later, so the run still
# ends with a single winner just like ICA.
#
# PAPER : Atashpaz-Gargari E, Lucas C (2007)
#         "Imperialist competitive algorithm: an algorithm for optimization
#          inspired by imperialistic competition"
#         IEEE Congress on Evolutionary Computation, pp. 4661-4667
# DOI   : https://doi.org/10.1109/CEC.2007.4425083
#
# FAIRNESS OF COST EVALUATIONS:
# The modification only measures distances between imperialist positions, which
# are numbers we already hold. The modification never calls the cost function. 
# So both versions spend the exact same number of cost-function evaluations per
# iteration, and matching the iteration count already matches the budget.
# =============================================================================

import numpy as np     # the only dependency: vector maths plus the random generator. scipy is NOT imported here, it is only used later in the stats file.


# =============================================================================
# NAM HELPER 1: SIZE OF THE ALLOWED NEIGHBOURHOOD AT ITERATION t
# -----------------------------------------------------------------------------
# NAM: this single growing number IS the whole modification.
# NAM: it answers "right now, how many rival empires is an empire allowed to fight?"
# NAM: Small early (stay local, shield far-apart empires so several
# NAM: regions keep exploring) and large late (go global so the survivors
# NAM: finally compete and the run ends with one winner, exactly like ICA).
# NAM: the start value 2 is a fixed design choice (the smallest sensible
# NAM: neighbourhood, you plus one rival), NOT a tuned parameter.
# =============================================================================
def nam_neighbourhood_size(t, max_iter, n_emp):        # NAM: t = current iteration, max_iter = total iterations, n_emp = empires alive right now
    k = 2 + ((n_emp - 3) * t) // max(max_iter - 1, 1)  # NAM: straight-line ramp. at t=0 this gives 2, at the LAST index t=max_iter-1 it gives exactly n_emp-1 (fully global, like ICA).
                                                       # NAM: dividing by max_iter-1 (not max_iter) is what makes the final iteration reach global; max(...,1) guards the silly case of a 1-iteration run.
                                                       # NAM: the inner parens force (n_emp-3)*t to happen first, then // integer-divides so k stays whole (you cannot fight half an empire).
    return max(1, min(k, n_emp - 1))                   # NAM: clamp into the legal range. empires collapse over the run so n_emp shrinks, this keeps k valid: never above the rivals that exist, never below 1.


# =============================================================================
# NAM HELPER 2: THE k NEAREST OTHER EMPIRES TO A GIVEN EMPIRE
# -----------------------------------------------------------------------------
# NAM: distance is measured imperialist to imperialist, so "near" means the two
# NAM: empires are searching the same part of the map. rebuilt fresh on every
# NAM: call, so when imperialists move (assimilation) or swap with a colony
# NAM: (possession) the neighbourhood updates by itself and is never stale.
# NAM: cheap, because there are only about 20 imperialists, not the whole 300.
# =============================================================================
def nam_nearest_empires(imps, idx, k):                 # NAM: imps = list of imperialist positions, idx = the empire asking, k = how many neighbours to return
    here  = imps[idx]                                  # NAM: the point we measure every distance from (the asking empire's imperialist)
    dist  = [np.linalg.norm(here - other)              # NAM: ordinary straight-line (Euclidean) distance from "here" to every imperialist
             for other in imps]                        # NAM: this list includes the distance to itself, which is 0 and is dropped below
    order = np.argsort(dist)                           # NAM: empire indices sorted nearest first; position 0 is the empire itself (distance 0)
    return [j for j in order if j != idx][:k]          # NAM: drop itself, then keep only the k closest rivals as the allowed receivers


# =============================================================================
# PHASE 1: ASSIMILATION (the exploitation step)
# -----------------------------------------------------------------------------
# Each colony is pulled toward its own imperialist by a random fraction of the
# gap between them. This refines the search around solutions that already work.
# The factor 2 lets a colony sometimes overshoot past the imperialist, so the
# area BEYOND the imperialist gets explored too, not only the line between them.
# =============================================================================
def assimilate(colonies, imperialist, beta, lb, ub):
    n_col   = len(colonies)                            # how many colonies are in this empire. not used below, kept so the line count mirrors ICA exactly and nothing drifts.
    vector  = imperialist - colonies                   # arrow from each colony toward the imperialist: the direction we want every colony to move in
    rand    = np.random.random(colonies.shape)         # one fresh random fraction per coordinate per colony, so each colony moves a different, irregular amount (this is the exploration jitter)
    new_pos = colonies + 2 * beta * rand * vector      # take the step. 2*beta scales the pull so a colony can land short of, on, or just past the imperialist, covering both sides
    return np.clip(new_pos, lb, ub)                    # snap any colony that left the legal box back onto the boundary, because a position outside the bounds is not a valid solution


# =============================================================================
# PHASE 2: REVOLUTION (the exploration step)
# -----------------------------------------------------------------------------
# A fraction of colonies are thrown out and replaced by brand new random
# countries. This is how the search escapes a local minimum: it keeps injecting
# fresh candidates instead of letting everyone crowd around one spot.
# The fraction decays each iteration (handled in the main loop), so there is a
# lot of shaking early and very little late. Revolution touches colonies only,
# never the imperialist, which matches the original behaviour.
# =============================================================================
def revolve(colonies, rev_rate, lb, ub):
    n_col       = len(colonies)                        # how many colonies this empire has, the pool we might shake up
    n_revolve   = int(round(rev_rate * n_col))         # how many of them actually revolt this round: the rate times the count, rounded to a whole number
    if n_revolve == 0:                                 # if the rounded count is zero there is nothing to do
        return colonies                                # so return the colonies untouched and skip the random work below
    idx         = np.random.permutation(n_col)[:n_revolve]  # shuffle all colony slots and take the first n_revolve, an unbiased random pick of who revolts
    dims        = colonies.shape[1]                    # number of coordinates per colony, needed to size the replacement block
    new_random  = np.random.uniform(lb, ub,            # draw brand new random countries uniformly across the whole search box
                                    (n_revolve, dims))
    colonies[idx] = new_random                         # drop the new countries into the chosen slots, replacing the old colonies in place
    return colonies                                    # hand back the colony array with the revolted members swapped out


# =============================================================================
# PHASE 3: POSSESSION (intra-empire role swap)
# -----------------------------------------------------------------------------
# After moving and revolting, a colony may now be better than its own
# imperialist. If so the two swap jobs: the strong colony becomes the new
# imperialist and the old imperialist becomes a colony. This keeps the best
# solution in each empire as its leader.
# =============================================================================
def possess(colonies, colony_costs, imperialist, imp_cost):
    best_col_idx  = np.argmin(colony_costs)            # find the empire's best colony: the one with the lowest (best) cost
    best_col_cost = colony_costs[best_col_idx]         # remember that best colony's cost so we can compare it to the imperialist

    if best_col_cost < imp_cost:                       # the best colony has beaten the imperialist, so a swap is due
        new_imp        = colonies[best_col_idx].copy() # copy the winning colony out to become the new imperialist (copy so later edits do not alias it)
        new_imp_cost   = best_col_cost                 # its cost is the new imperialist cost
        colonies[best_col_idx]     = imperialist       # the old imperialist drops down into the freed colony slot
        colony_costs[best_col_idx] = imp_cost          # and its cost goes with it, keeping positions and costs aligned
        return colonies, colony_costs, new_imp, new_imp_cost  # report the swapped empire back to the caller

    return colonies, colony_costs, imperialist, imp_cost  # no colony beat the imperialist, so return everything unchanged


# =============================================================================
# PHASE 4: UNITING SIMILAR EMPIRES
# -----------------------------------------------------------------------------
# If two imperialists sit very close together they are really chasing the same
# solution, so the weaker of the two is absorbed by the stronger (its
# imperialist becomes a colony of the winner). "Very close" means within a small
# fraction of the search-box diagonal. This stops two empires from wasting
# effort on the same optimum.
# =============================================================================
def unite_similar(empires, uniting_threshold, lb, ub):
    search_size = np.linalg.norm(np.array(ub) - np.array(lb))  # length of the search-box diagonal, our yardstick for "how big is the whole space"
    threshold   = uniting_threshold * search_size      # turn the fraction into an actual distance: empires closer than this count as the same

    n_emp = len(empires)                               # how many empires we start the pass with
    for i in range(n_emp - 1):                         # compare every unordered pair of empires (i, j)
        for j in range(i + 1, n_emp):                  # j always after i, so each pair is checked once
            if empires[i] is None or empires[j] is None:  # one of this pair was already merged away earlier in the pass
                continue                               # so skip it, there is nothing left to compare
            distance = np.linalg.norm(                 # straight-line distance between the two imperialists
                empires[i]['imp'] - empires[j]['imp'])
            if distance <= threshold:                  # the two empires are close enough to be treated as one
                if empires[i]['imp_cost'] < empires[j]['imp_cost']:  # lower cost is stronger, decide who absorbs whom
                    strong, weak = i, j                # i is stronger, it keeps its identity
                else:
                    strong, weak = j, i                # otherwise j is stronger
                empires[strong]['colonies']     = np.vstack([  # the strong empire absorbs the weak one's people
                    empires[strong]['colonies'],
                    empires[weak]['imp'].reshape(1, -1),       # the weak imperialist joins as a colony
                    empires[weak]['colonies']])                # along with all of the weak empire's colonies
                empires[strong]['colony_costs'] = np.concatenate([  # and the matching costs, kept aligned with the positions above
                    empires[strong]['colony_costs'],
                    [empires[weak]['imp_cost']],
                    empires[weak]['colony_costs']])
                empires[weak] = None                   # mark the weak empire dead so the pair-checks above skip it and it is dropped below

    return [e for e in empires if e is not None]       # rebuild the list without the merged-away empires


# =============================================================================
# PHASE 5: IMPERIALISTIC COMPETITION (with the optional NAM receiver rule)
# -----------------------------------------------------------------------------
# The weakest empire loses its single worst colony to some stronger empire,
# chosen with a probability tilted toward the strong. If the weakest empire runs
# out of colonies it collapses and its imperialist is absorbed too.
#
# Plain ICA: every other empire is a candidate to receive that colony.
# NAM-ICA  : only the weakest empire's k nearest empires are candidates, where
#            k grows from 2 to global over the run (see the two NAM helpers).
# Everything else here, including the strength-weighted draw, is identical.
#
# This phase fires only with probability competition_prob per iteration (not
# every iteration), which slows the competitive pressure so empires get time to
# develop their colonies before they fight.
# =============================================================================
def compete(empires, zeta, competition_prob,
            nonaligned, iteration, n_iterations):      # NAM: three extra inputs threaded in: the switch, plus iteration and the total, so the neighbourhood size can be worked out
    if np.random.random() > competition_prob:          # roll one die per call. most rounds the competition does not fire at all, which keeps pressure gentle
        return empires                                 # not this round: hand every empire back untouched
    if len(empires) <= 1:                              # with a single empire there is no rival to take from or give to
        return empires                                 # so there is nothing to do

    total_costs = []                                   # we will score each empire by a blended "total cost" used only for this ranking
    for emp in empires:                                # walk every surviving empire
        if len(emp['colony_costs']) > 0:               # an empire that still owns colonies
            tc = emp['imp_cost'] + zeta * np.mean(emp['colony_costs'])  # mostly the imperialist cost, plus a small zeta share of the colonies' average, so colonies matter a little
        else:                                          # an empire with no colonies left
            tc = emp['imp_cost']                        # its total cost is just the imperialist's own cost
        total_costs.append(tc)                          # store this empire's score
    total_costs = np.array(total_costs)                 # to a numpy array so max, argmax and arithmetic work elementwise

    weakest_idx  = int(np.argmax(total_costs))          # the weakest empire is the one with the HIGHEST total cost
    weakest      = empires[weakest_idx]                 # grab it: this is the empire that will give up a colony

    if len(weakest['colony_costs']) == 0:               # the weakest has no colony to hand over
        return empires                                  # so nothing can be transferred this round

    worst_col_idx = int(np.argmax(weakest['colony_costs']))  # inside the weakest empire, locate its worst (highest-cost) colony
    stolen_pos    = weakest['colonies'][worst_col_idx].copy()  # copy that colony's position out so we can safely move it
    stolen_cost   = weakest['colony_costs'][worst_col_idx]     # and keep its cost alongside the position

    weakest['colonies']     = np.delete(weakest['colonies'],
                                        worst_col_idx, axis=0)  # remove the taken colony's position from the weakest empire
    weakest['colony_costs'] = np.delete(weakest['colony_costs'],
                                        worst_col_idx)          # and remove its matching cost, so the two arrays stay aligned

    powers = np.max(total_costs) - total_costs          # flip cost into strength: subtract each empire's cost from the worst cost, so the lowest-cost (best) empire ends up with the biggest number. the winner is drawn by probability and probabilities need big-equals-likely, so strong empires must carry the large values.

    if nonaligned:                                      # NAM: modification ON, the receiver must come from a limited circle of nearby empires
        imps          = [e['imp'] for e in empires]     # NAM: every imperialist position in empire order, so distances line up with empire indices
        k             = nam_neighbourhood_size(iteration, n_iterations, len(empires))  # NAM: how many neighbours are allowed this round (2 early, all-others late)
        allowed       = nam_nearest_empires(imps, weakest_idx, k)  # NAM: the weakest empire's k nearest rivals, the only empires permitted to receive the colony
        mask          = np.zeros(len(empires), dtype=bool)  # NAM: a yes/no flag per empire, every entry starts as no
        mask[allowed] = True                            # NAM: flip only the k nearest rivals to yes
        powers[~mask] = 0                               # NAM: force the strength of every empire outside the circle to zero so the draw below can never pick it. zeroing is the same as deleting them from the lottery, and it keeps this line shaped exactly like ICA, which is what makes nonaligned=False reproduce ICA bit for bit.
    else:                                               # plain ICA path: every empire except the weakest is a candidate
        powers[weakest_idx] = 0                         # the weakest cannot win its own colony back, so its strength is set to zero

    if powers.sum() > 0:                                # normal case: at least one allowed empire still has real strength
        probs      = powers / powers.sum()              # turn the strengths into probabilities that sum to 1, ready for a weighted draw
        winner_idx = int(np.random.choice(len(empires), p=probs))  # draw the receiving empire: strong empires are likelier, any zeroed empire is impossible
    else:                                               # rare tie: every allowed strength is zero (all candidates share the same cost)
        winner_idx = (weakest_idx + 1) % len(empires)   # deterministic fallback with no random draw, copied straight from ICA so reproduction is never broken

    empires[winner_idx]['colonies']     = np.vstack([
        empires[winner_idx]['colonies'],
        stolen_pos.reshape(1, -1)])                     # give the stolen colony's position to the winner by stacking it on
    empires[winner_idx]['colony_costs'] = np.concatenate([
        empires[winner_idx]['colony_costs'],
        [stolen_cost]])                                 # and append its cost so the winner's two arrays stay aligned

    if len(weakest['colony_costs']) == 0:               # the weakest just lost its last colony, so the empire now collapses
        empires[winner_idx]['colonies']     = np.vstack([
            empires[winner_idx]['colonies'],
            weakest['imp'].reshape(1, -1)])             # its fallen imperialist becomes one more colony of the winner
        empires[winner_idx]['colony_costs'] = np.concatenate([
            empires[winner_idx]['colony_costs'],
            [weakest['imp_cost']]])                     # carry the fallen imperialist's cost across as well
        empires.pop(weakest_idx)                        # delete the dead empire from the world

    return empires                                      # hand back the updated list of empires


# =============================================================================
# INITIALIZATION: CREATE THE STARTING EMPIRES
# -----------------------------------------------------------------------------
# 1. Scatter random countries across the search box.
# 2. Score each one with the cost function.
# 3. Sort so the best are first.
# 4. The best n_imp become imperialists, the rest are colonies to share out.
# 5. Each imperialist gets a share of colonies proportional to its power, so
#    stronger imperialists start with bigger empires.
# =============================================================================
def create_initial_empires(cost_func, n_countries, n_imp, dims, lb, ub):
    countries = np.random.uniform(lb, ub, (n_countries, dims))  # scatter the whole starting population uniformly inside the legal box
    costs     = np.array([cost_func(c) for c in countries])     # score every country once; this is the only place init spends evaluations

    order     = np.argsort(costs)                       # indices that would sort the population from best (lowest cost) to worst
    countries = countries[order]                        # reorder positions so the best country is first
    costs     = costs[order]                            # reorder costs the same way to stay aligned

    imp_pos  = countries[:n_imp]                        # the first n_imp are the strongest, they become imperialists
    imp_cost = costs[:n_imp]                            # their costs
    col_pos  = countries[n_imp:]                        # everyone else becomes a colony waiting to be assigned
    col_cost = costs[n_imp:]                            # their costs

    max_cost = np.max(imp_cost)                         # the worst imperialist cost, used as the reference for the power formula
    if max_cost > 0:                                    # normal case where costs are positive
        powers = 1.3 * max_cost - imp_cost              # linear power from the original MATLAB: best imperialist gets the most, the 1.3 keeps even the worst slightly positive
    else:                                               # sign-flipped case when the worst cost is zero or negative
        powers = 0.7 * max_cost - imp_cost              # mirror formula so powers stay sensible when costs are not positive

    powers     = powers / powers.sum()                  # normalize into proportions that add up to 1, i.e. each imperialist's share of the colonies
    n_colonies = np.round(powers * len(col_pos)).astype(int)  # turn each share into an integer count of colonies
    n_colonies[-1] = len(col_pos) - n_colonies[:-1].sum()     # hand any leftover from rounding to the last empire so every colony is placed and none is lost

    rand_idx = np.random.permutation(len(col_pos))      # shuffle colony indices so the split is random, not best-to-first
    empires  = []                                       # the list of empires we are about to build
    cursor   = 0                                        # a moving pointer into the shuffled colony indices
    for i in range(n_imp):                              # build one empire per imperialist
        take       = n_colonies[i]                      # how many colonies this empire should receive
        chosen     = rand_idx[cursor:cursor + take]     # slice off that many shuffled colony indices
        cursor    += take                               # advance the pointer past the ones we just took
        empires.append({                                # store the empire as a small dictionary
            'imp'          : imp_pos[i].copy(),         # imperialist position (copy so later moves do not alias the population array)
            'imp_cost'     : imp_cost[i],               # imperialist cost
            'colonies'     : col_pos[chosen].copy(),    # this empire's colony positions
            'colony_costs' : col_cost[chosen].copy(),   # and their costs
        })

    return empires                                      # the fully formed starting empires


# =============================================================================
# MAIN LOOP: RUN THE ALGORITHM
# -----------------------------------------------------------------------------
# Returns the best solution found, its cost, and the convergence history (the
# best cost at every iteration). The defaults below match the original MATLAB.
# The only added parameter is `nonaligned`, the modification switch.
# =============================================================================
def ica(cost_func, dims, lb, ub,
        n_countries       = 300,                       # total population size
        n_imperialists    = 20,                        # number of starting empires
        n_iterations      = 2000,                      # how many iterations (decades) to run
        beta              = 2.0,                       # assimilation pull coefficient
        rev_rate          = 0.5,                       # starting revolution rate. matches your on-disk default; the runner overrides this per run with the REVOLUTION_RATE switch.
        damp_ratio        = 0.99,                      # multiply rev_rate by this each iteration
        zeta              = 0.02,                      # weight of colony average in an empire's total cost
        uniting_threshold = 0.02,                      # empires closer than this fraction of the box merge
        competition_prob  = 0.11,                      # chance the competition phase fires in a given iteration
        nonaligned        = False):                    # NAM: the one on/off switch. False keeps this a plain ICA (default, so older call sites are unaffected), True activates NAM-ICA.

    lb = np.full(dims, lb) if np.isscalar(lb) else np.array(lb)  # turn a single lower bound into a per-dimension array so vector maths works
    ub = np.full(dims, ub) if np.isscalar(ub) else np.array(ub)  # same for the upper bound

    empires = create_initial_empires(cost_func, n_countries,
                                     n_imperialists, dims, lb, ub)  # build the starting world of empires

    history = []                                       # best cost recorded at each iteration, for the convergence plots later

    for iteration in range(n_iterations):              # one pass through this loop is one decade

        rev_rate = rev_rate * damp_ratio               # shrink the revolution rate a little each round (0.99 of the previous). early we want many random shake-ups to explore, later we want calm so the search can settle, so the rate fades over time.

        for emp in empires:                            # phases 1 to 3 happen inside each empire on its own
            if len(emp['colonies']) == 0:              # an empire with no colonies has nothing to assimilate or revolt
                continue                               # so skip straight to the next empire

            emp['colonies'] = assimilate(emp['colonies'], emp['imp'],
                                         beta, lb, ub)  # phase 1: pull this empire's colonies toward its imperialist

            emp['colonies'] = revolve(emp['colonies'], rev_rate, lb, ub)  # phase 2: randomly replace a shrinking fraction of them to keep exploring

            emp['colony_costs'] = np.array([cost_func(c)             # re-score every colony after it moved or revolted, because their positions changed
                                            for c in emp['colonies']])  # this re-evaluation is identical in ICA and NAM-ICA, which is why the evaluation budget matches

            (emp['colonies'], emp['colony_costs'],
             emp['imp'],      emp['imp_cost']) = possess(            # phase 3: if a colony now beats the imperialist, swap their roles
                emp['colonies'], emp['colony_costs'],
                emp['imp'],      emp['imp_cost'])

        empires = unite_similar(empires, uniting_threshold, lb, ub)  # phase 4: merge any two empires that have drifted onto the same spot

        empires = compete(empires, zeta, competition_prob,
                          nonaligned, iteration, n_iterations)       # NAM: phase 5. pass the switch plus iteration and total in, so competition can size the neighbourhood for this round (ignored when nonaligned is False)

        best_cost = min(emp['imp_cost'] for emp in empires)  # the best imperialist cost across all empires this iteration
        history.append(best_cost)                      # log it for the convergence curve

        if len(empires) == 1:                          # only one empire is left, the world has converged
            while len(history) < n_iterations:         # pad the history out to full length so every run is the same size for averaging later
                history.append(best_cost)              # repeat the final value, since nothing changes after convergence
            break                                      # stop early, there is no more competition to run

    best_emp  = min(empires, key=lambda e: e['imp_cost'])  # of the empires that survived, the one with the lowest imperialist cost
    best_pos  = best_emp['imp']                        # its imperialist position is our best solution
    best_cost = best_emp['imp_cost']                   # and its cost is our best cost

    return best_pos, best_cost, history                # give the caller the answer plus the full convergence history


# =============================================================================
# QUICK SELF-TEST
# -----------------------------------------------------------------------------
# Run this file directly (python _nam_ica.py) to confirm it works on Rastrigin
# in both modes. The full benchmarking and the real comparison are done by the
# other scripts. Expected: both best costs land close to 0.
# =============================================================================
if __name__ == '__main__':
    from _benchmarks import FUNCTIONS                  # the 7 CEC functions

    f2 = FUNCTIONS['F2']                               # Rastrigin, a bumpy multimodal function, a good quick test

    np.random.seed(42)                                 # fix the seed so plain ICA gives a repeatable answer
    _, cost_ica, hist_ica = ica(                       # run with the default nonaligned=False, i.e. plain ICA
        cost_func    = f2['func'],
        dims         = 10,
        lb           = f2['lb'],
        ub           = f2['ub'],
        n_iterations = 500)

    np.random.seed(42)                                 # reset to the SAME seed so the only difference is the switch
    _, cost_nam, hist_nam = ica(                       # NAM: run again with the modification active
        cost_func    = f2['func'],
        dims         = 10,
        lb           = f2['lb'],
        ub           = f2['ub'],
        n_iterations = 500,
        nonaligned   = True)                           # NAM: turn the modification on

    print("Self-test on Rastrigin (10D, 500 iterations):")
    print(f"  plain ICA  best cost : {cost_ica:.6e}")  # should be close to 0
    print(f"  NAM-ICA    best cost : {cost_nam:.6e}")  # NAM: should also be close to 0, and the two histories should differ since the switch changed the search
    print(f"  histories differ     : {hist_ica != hist_nam}")  # NAM: a quick eyeball that the flag actually changes behaviour