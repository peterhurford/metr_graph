"""Conditional AI takeoff scenarios; assumptions, not fitted forecasts.

Times are years after full coding automation. Natural logs represent capability
and software efficiency relative to the model available at that milestone.
All randomness is local and seeded; every setting uses the same base draws.
"""

import numpy as np


MODEL_VERSION = "takeoff-v2-rsi-onset"
MILESTONES = (
    "Sustained research feedback", "Full R&D automation",
    "Superhuman AI research", "Broad superintelligence",
)
DEFAULTS = {
    "onset_low": 6, "onset_high": 24, "years": 10, "seed": 20260906,
    "software_months": 9.0, "compute_growth": 2.0,
    "training_months": 3.0, "validation_months": 1.0,
    "training_share": 40, "experiment_share": 40,
    "taste_at_onset": 0.6, "taste_slope": 0.8, "coding_slope": 0.8,
    "transfer": 60, "difficulty": 0.5, "parallelization": 0.5,
    "general_gap": 3.0, "superiority_gap": 2.0,
    "uncertainty": 40, "feedback_cycles": 2,
}
PRESETS = {
    "Central": {},
    "Fast": {"software_months": 6.0, "training_months": 1.0,
             "validation_months": 0.5, "taste_slope": 1.2,
             "transfer": 90, "difficulty": 0.2, "general_gap": 2.0},
    "Bottlenecked": {"software_months": 18.0, "compute_growth": 1.3,
                     "training_months": 6.0, "validation_months": 2.0,
                     "taste_at_onset": 0.3, "taste_slope": 0.4,
                     "transfer": 20, "difficulty": 1.0,
                     "general_gap": 5.0},
}


def validate(params):
    """Reject invalid model inputs, including values from shared URLs."""
    p = dict(DEFAULTS, **params)
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("All takeoff settings must be finite numbers.")
    if not 0 <= p["onset_low"] <= p["onset_high"] <= 120:
        raise ValueError("Coding automation dates must be ordered within 0–120 months.")
    if not 0 < p["years"] <= 30:
        raise ValueError("The simulation horizon must be between 0 and 30 years.")
    if p["training_share"] <= 0 or p["experiment_share"] <= 0 \
            or p["training_share"] + p["experiment_share"] >= 100:
        raise ValueError("Reserve positive compute shares for training, experiments, and agents.")
    if not 0 <= p["transfer"] <= 100 or not 0 <= p["uncertainty"] <= 100:
        raise ValueError("Transfer and uncertainty must be between 0 and 100%.")
    if p["software_months"] <= 0 or p["training_months"] <= 0 \
            or p["validation_months"] < 0 or p["compute_growth"] < 1:
        raise ValueError("Use positive doubling/run times and compute growth of at least 1×.")
    if not 0 < p["taste_at_onset"] <= 1 or p["taste_slope"] <= 0 \
            or p["coding_slope"] <= 0:
        raise ValueError("Research judgment and capability slopes must be positive.")
    if p["difficulty"] < 0 or not 0 < p["parallelization"] <= 1 \
            or p["general_gap"] < 0 or p["superiority_gap"] <= 0:
        raise ValueError("Use nonnegative gaps/difficulty and parallelization in (0, 1].")
    if int(p["feedback_cycles"]) != p["feedback_cycles"] or p["feedback_cycles"] < 1:
        raise ValueError("Feedback must persist for at least one whole cycle.")
    if int(p["seed"]) != p["seed"] or not 0 <= p["seed"] < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2³²−1.")
    return p


def _harmonic(labor, compute):
    return 2.0 / (1.0 / labor + 1.0 / compute)


def simulate(params=None, n=1000, step=1 / 52, onset_years=None):
    """Weekly research and successor cycles, with censored milestone draws.

    A run freezes its algorithm snapshot at its start, integrates allocated
    training compute during the run, then waits for validation before deployment.
    Improvements discovered during either phase enter the next run. A fixed
    effective-compute reference prevents longer runs making the target easier.

    40/40/20 compute shares normalize training/experiments/agents to one at onset.
    The software doubling input is the initial rate under this reference split.
    Transfer discounts *additional* coding/judgment beyond onset, so observed
    assistance is not multiplied into the starting rate a second time.
    """
    p = validate(params or {})
    if n < 1 or not 0 < step <= 0.25:
        raise ValueError("Use at least one draw and a time step in (0, 0.25] years.")
    rng = np.random.default_rng(int(p["seed"]))
    # Common quantiles preserve paired scenario comparisons. Bounded log-space
    # variation: ±uncertainty is a multiplicative factor exp(±uncertainty/100).
    onset = rng.uniform(p["onset_low"], p["onset_high"], n) / 12
    if onset_years is not None:
        onset = np.asarray(onset_years, dtype=float).copy()
        if onset.shape != (n,) or np.isnan(onset).any() or np.isneginf(onset).any():
            raise ValueError("Provide one onset date per draw; dates must be finite or +inf.")
    z = rng.uniform(-1, 1, (8, n))
    spread = p["uncertainty"] / 100
    factors = np.exp(spread * z)
    software_rate = np.log(2) * 12 / p["software_months"] * factors[0]
    compute_rate = np.log(p["compute_growth"]) * factors[1]
    train_time = p["training_months"] / 12 * factors[2]
    eval_time = p["validation_months"] / 12 * factors[3]
    taste0 = np.minimum(p["taste_at_onset"] * factors[4], 1.0)
    taste_slope = p["taste_slope"] * factors[5]
    general_gap = p["general_gap"] * factors[6] * np.log(2)
    superiority_gap = p["superiority_gap"] * factors[7] * np.log(2)
    full_target = np.log(1 / taste0) / taste_slope
    research_target = np.log(3 / taste0) / taste_slope
    asi_target = research_target + general_gap + superiority_gap
    targets = (full_target, research_target, asi_target)
    events = {name: np.full(n, np.inf) for name in MILESTONES}
    events[MILESTONES[1]][full_target <= 0] = 0
    software = np.zeros(n)
    deployed = np.zeros(n)
    snapshot = np.zeros(n)
    run_compute = np.zeros(n)
    training_used = np.zeros(n)
    run_start = np.zeros(n)
    pending = np.zeros(n)
    finished = np.zeros(n, dtype=bool)
    cycles = np.zeros(n, dtype=int)
    last_labor = np.ones(n)
    last_experiments = np.ones(n)
    previous_ai_output = np.zeros(n)
    shares = np.array([p["training_share"], p["experiment_share"],
                       100 - p["training_share"] - p["experiment_share"]]) / 100
    transfer = p["transfer"] / 100
    beta = p["difficulty"]
    parallel = p["parallelization"]
    # The onset model's training compute: a three-month run at the reference
    # 40% allocation. This constant must not depend on the chosen run duration.
    reference_train_compute = 0.4 * 3 / 12
    # Each draw advances at most one week, stopping exactly at its own run
    # completion/deployment events. This avoids adding a rounding delay every
    # cycle (which could move an entire milestone to the next successor).
    t = np.zeros(n)
    limited = np.zeros(n, dtype=bool)
    while np.any(t < p["years"] - 1e-12):
        train_end = run_start + train_time
        deploy_end = train_end + eval_time
        next_event = np.where(finished, deploy_end, train_end)
        end = np.minimum(np.minimum(t + step, next_event), p["years"])
        active = t < p["years"] - 1e-12
        end = np.where(active, end, t)
        dt = np.maximum(end - t, 0)
        compute_log = compute_rate * (t + dt / 2)
        if transfer:
            coding_log = np.logaddexp(np.log1p(-transfer) if transfer < 1 else -np.inf,
                                     np.log(transfer) + p["coding_slope"] * deployed)
            quality_log = np.logaddexp(np.log1p(-transfer) if transfer < 1 else -np.inf,
                                      np.log(transfer) + taste_slope * deployed)
        else:
            coding_log = np.zeros(n)
            quality_log = np.zeros(n)
        labor_log = parallel * (compute_log + np.log(shares[2] / 0.2) + coding_log)
        experiment_log = parallel * (compute_log + np.log(shares[1] / 0.4))
        research_log = np.log(2) - np.logaddexp(-labor_log, -experiment_log) + quality_log
        log_amount = np.log(software_rate) + research_log + np.log(
            dt, out=np.full(n, -np.inf), where=dt > 0)
        if beta > 0:
            updated = np.logaddexp(beta * software, np.log(beta) + log_amount) / beta
        else:
            updated = software + np.exp(np.minimum(log_amount, 100))
        # Stop numerically divergent draws explicitly, rather than silently
        # treating a numerical ceiling as a capability plateau. In ordinary
        # scenarios all milestones complete long before this guard is relevant.
        diverged = active & (updated > 1e6)
        limited |= diverged
        software = np.minimum(updated, 1e6)
        train_budget = np.exp(compute_rate * t) * np.divide(
            np.expm1(compute_rate * dt), compute_rate,
            out=dt.copy(), where=compute_rate != 0)
        used = np.where(finished, 0, shares[0] * train_budget)
        run_compute += used
        training_used += used
        done = active & (~finished) & (end >= train_end - 1e-12)
        pending[done] = np.maximum(
            deployed[done], snapshot[done]
            + np.log(np.maximum(run_compute[done] / reference_train_compute, 1e-12)))
        finished |= done
        ready = active & finished & (end >= deploy_end - 1e-12)
        deployed[ready] = pending[ready]
        if transfer:
            ai_output_log = parallel * np.logaddexp(
                np.log1p(-transfer) if transfer < 1 else -np.inf,
                np.log(transfer) + p["coding_slope"] * deployed) + np.logaddexp(
                np.log1p(-transfer) if transfer < 1 else -np.inf,
                np.log(transfer) + taste_slope * deployed)
        else:
            ai_output_log = np.zeros(n)
        improved = ready & (snapshot > 0) & (ai_output_log >= previous_ai_output + np.log(1.1))
        cycles[ready] = np.where(improved[ready], cycles[ready] + 1, 0)
        previous_ai_output[ready] = ai_output_log[ready]
        feedback = cycles >= p["feedback_cycles"]
        ev = events[MILESTONES[0]]
        hit = feedback & ~np.isfinite(ev)
        ev[hit] = end[hit]
        for name, threshold in zip(MILESTONES[1:], targets):
            ev = events[name]
            hit = (deployed >= threshold) & ~np.isfinite(ev)
            ev[hit] = end[hit]
        tracking = events[MILESTONES[-1]] >= end
        last_labor[tracking] = labor_log[tracking]
        last_experiments[tracking] = experiment_log[tracking]
        snapshot[ready] = software[ready]
        run_start[ready] = end[ready]
        run_compute[ready] = 0
        finished[ready] = False
        all_reached = np.all(np.isfinite(np.array(list(events.values()))), axis=0)
        t = np.where(all_reached | diverged, p["years"], end)
    total_budget = np.divide(np.expm1(compute_rate * p["years"]), compute_rate,
                             out=np.full(n, float(p["years"])), where=compute_rate != 0)
    return {
        "params": p, "version": MODEL_VERSION, "onset": onset,
        "events": events,
        "compute_limited": last_experiments <= last_labor,
        "budget": total_budget, "allocated": shares[:, None] * total_budget,
        "training_used": training_used,
        "software_log": software, "deployed_log": deployed,
        "numerically_limited": limited,
    }


def cdf(samples, grid):
    """CDF over all draws, including +inf (not reached) in the denominator."""
    return np.searchsorted(np.sort(samples), grid, side="right") / len(samples)


def quantile(samples, q, horizon=np.inf):
    """Empirical inverse CDF, preserving censored probability mass."""
    value = np.sort(samples)[max(0, int(np.ceil(q * len(samples))) - 1)]
    return float(value) if value <= horizon else np.inf
