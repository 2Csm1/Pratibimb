"""
costs.py — Cost model for attrition intervention decisions.

All dollar figures and rates carry source comments so reviewers can
challenge the assumptions without reading the code line-by-line.
"""

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

# Source: SHRM (2022) and Gallup research estimate total replacement cost
# at 50–200% of annual salary depending on role.  We use the conservative
# low end (50%) so the model does NOT over-recommend intervention.
REPLACEMENT_COST_FRACTION = 0.50

# Estimated direct cost of one retention intervention (manager time for a
# stay interview, compensation review, HR process).  Order-of-magnitude
# estimate; site-specific data should replace this.
INTERVENTION_COST = 1000.0   # USD

# Probability that an intervention prevents a departure that would
# otherwise have occurred.  NOT estimable from this dataset — it is a
# policy knob.  Literature ranges from 0.10 to 0.50; we use a moderate
# 0.30 as the default.
INTERVENTION_SUCCESS_RATE = 0.30


# ── Functions ─────────────────────────────────────────────────────────────────

def replacement_cost(monthly_income):
    """Annualised replacement cost for one employee.

    Returns REPLACEMENT_COST_FRACTION × 12 × monthly_income.
    """
    return REPLACEMENT_COST_FRACTION * 12 * monthly_income


def optimal_threshold(monthly_income):
    """Cost-optimal probability threshold for a single employee.

    Derivation
    ----------
    Intervene when the expected cost of intervention is lower than doing
    nothing:

        C + p·(1 − s)·R  <  p·R

    where C = intervention cost, p = attrition probability,
    s = intervention success rate, R = replacement cost.

    Solving for p:

        p*  =  C / (R · s)

    If the model's predicted probability exceeds p*, the expected cost of
    *not* intervening is higher than intervening.

    Returns min(1.0, C / (R · s)) so the threshold is capped at 1.
    """
    R = replacement_cost(monthly_income)
    if R * INTERVENTION_SUCCESS_RATE == 0:
        return 1.0
    return min(1.0, INTERVENTION_COST / (R * INTERVENTION_SUCCESS_RATE))


def expected_cost(y_true, y_prob, threshold, monthly_incomes):
    """Total expected cost under a given threshold decision rule.

    For each employee:
    - If y_prob >= threshold  →  we intervene.
      Cost = C  +  p(1-s)·R   (intervention cost + residual attrition risk)
    - If y_prob <  threshold  →  no action.
      Cost = p·R              (expected loss from unmitigated attrition)

    We use the *true labels* (y_true) as the realised outcome, not p,
    so the cost reflects what actually happened under counterfactual
    intervention.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob : array-like of floats in [0, 1]
    threshold : float
    monthly_incomes : array-like of floats

    Returns
    -------
    float — total expected cost across all employees
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    monthly_incomes = np.asarray(monthly_incomes, dtype=float)

    intervene = y_prob >= threshold
    R = REPLACEMENT_COST_FRACTION * 12 * monthly_incomes

    # Cost when we intervene
    cost_intervene = (
        INTERVENTION_COST
        + y_true * (1 - INTERVENTION_SUCCESS_RATE) * R
    )
    # Cost when we do nothing
    cost_nothing = y_true * R

    total = np.where(intervene, cost_intervene, cost_nothing)
    return float(total.sum())


def sweep_thresholds(y_true, y_prob, monthly_incomes,
                     grid=None):
    """Sweep thresholds and return the expected-cost curve.

    Parameters
    ----------
    y_true, y_prob, monthly_incomes : array-like
    grid : array-like of thresholds to evaluate, default 0.01–0.99 (99 pts)

    Returns
    -------
    dict with keys:
        thresholds : list[float]
        costs      : list[float]   — total expected cost at each threshold
        per_employee_costs : list[float] — cost / n
        optimal_threshold  : float — threshold with minimum total cost
        optimal_cost       : float
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    grid = np.asarray(grid)
    n = len(y_true)
    costs = []
    for t in grid:
        c = expected_cost(y_true, y_prob, t, monthly_incomes)
        costs.append(c)

    costs = np.array(costs)
    best_idx = int(np.argmin(costs))

    return {
        'thresholds': grid.tolist(),
        'costs': costs.tolist(),
        'per_employee_costs': (costs / n).tolist(),
        'optimal_threshold': float(grid[best_idx]),
        'optimal_cost': float(costs[best_idx]),
    }
