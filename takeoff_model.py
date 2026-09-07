"""AI research from today through takeoff; assumptions and explicit calibration.

Times are years after today; coding automation is an inherited milestone.
Natural logs represent capability and efficiency relative to today's system.
All randomness is local and seeded; every setting uses the same base draws.
"""

import hashlib
from pathlib import Path

import numpy as np


MODEL_VERSION = "takeoff-v4-present-feedback"
SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WORKFLOW_STAGES = ("Research direction", "Experiment design and interpretation",
                   "Verification and integration")
WORKFLOW_KEYS = ("direction", "experiments", "verification")
WORKFLOW_WEIGHTS = np.array([0.25, 0.5, 0.25])
MILESTONES = (
    "Sustained research feedback", "Full R&D automation",
    "Superhuman AI research", "Broad superintelligence",
)
DEFAULTS = {
    "onset_low": 6, "onset_high": 24, "years": 10, "seed": 20260906,
    "software_months": 9.0, "compute_growth": 2.0,
    "training_months": 3.0, "validation_months": 1.0,
    "training_share": 40, "experiment_share": 40,
    "taste_slope": 0.8, "coding_slope": 0.8,
    "transfer": 60, "difficulty": 0.5, "parallelization": 0.5,
    "general_gap": 3.0, "superiority_gap": 2.0,
    "uncertainty": 40, "feedback_cycles": 2,
    "human_direction": 40.0, "human_experiments": 30.0, "human_verification": 50.0,
    "half_direction": 6.0, "half_experiments": 9.0, "half_verification": 12.0,
    "human_floor": 1.0, "full_human": 5.0,
    "project_success": 60.0, "success_half": 12.0, "full_success": 90.0,
    "progress_target": 12.0, "advantage_target": 2.0, "progress_cycles": 2,
    "fast_share": 25.0, "fast_months": 1.0,
    "coding_today": 70.0,
    "pipeline_months": 2.0,
    "rsi_trend_weight": 50.0, "rsi_trend_months": 12.0, "rsi_useful_growth": 100.0,
}
PRESETS = {
    "Central": {},
    "Fast": {"software_months": 6.0, "training_months": 1.0,
             "validation_months": 0.5, "taste_slope": 1.2,
             "transfer": 90, "difficulty": 0.2, "general_gap": 2.0},
    "Bottlenecked": {"software_months": 18.0, "compute_growth": 1.3,
                     "training_months": 6.0, "validation_months": 2.0,
                     "taste_slope": 0.4,
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
    if p["taste_slope"] < 0 or p["coding_slope"] <= 0:
        raise ValueError("Research quality slope must be nonnegative and coding slope positive.")
    if p["difficulty"] < 0 or not 0 < p["parallelization"] <= 1 \
            or p["general_gap"] < 0 or p["superiority_gap"] <= 0:
        raise ValueError("Use nonnegative gaps/difficulty and parallelization in (0, 1].")
    if int(p["feedback_cycles"]) != p["feedback_cycles"] or p["feedback_cycles"] < 1:
        raise ValueError("Feedback must persist for at least one whole cycle.")
    if int(p["seed"]) != p["seed"] or not 0 <= p["seed"] < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2³²−1.")
    if not 0 <= p["human_floor"] <= 100 or not 0 <= p["full_human"] <= 100:
        raise ValueError("Human-work floor and automation threshold must be between 0 and 100%.")
    for stage in WORKFLOW_KEYS:
        if not p["human_floor"] <= p["human_" + stage] <= 100 or p["half_" + stage] <= 0:
            raise ValueError("Each workflow's starting human work must be at least its floor, "
                             "and its halving time must be positive.")
    if not 0 < p["project_success"] <= 100 or not 0 < p["full_success"] <= 100 \
            or p["success_half"] <= 0:
        raise ValueError("Project success rates must be in (0, 100] and halving time positive.")
    if min(p["progress_target"], p["advantage_target"]) <= 1 \
            or int(p["progress_cycles"]) != p["progress_cycles"] \
            or p["progress_cycles"] < 1:
        raise ValueError("Accelerated progress must exceed 1× and persist for whole cycles.")
    if not 0 <= p["fast_share"] <= 100 or p["fast_months"] <= 0:
        raise ValueError("Fast feedback needs a share in [0, 100] and a positive validation cycle.")
    if not 0 <= p["coding_today"] < 95:
        raise ValueError("Today's coding automation must be below the 95% coding milestone.")
    if not 0 <= p["pipeline_months"] <= 12:
        raise ValueError("Initial research pipeline must be between 0 and 12 months of progress.")
    if not 0 <= p["rsi_trend_weight"] <= 100 or not 0 <= p["rsi_useful_growth"] <= 100 \
            or p["rsi_trend_months"] <= 0:
        raise ValueError("RSI trend weights must be percentages and their half-life positive.")
    return p


def calibrate_feedback(experiment_growth, compute_growth, capability_growth,
                       yield_change=1.0, validated_growth=None, transfer=.6,
                       coding_slope=.8, parallel=.5):
    """Effective quality elasticity after accounting for compute and coding.

    Aggregate experiments are a proxy, not causal identification. A supplied
    validated-progress ratio supersedes that proxy. Ratios compare matched
    periods; current assistance is already in today's starting progress rate.
    """
    values = [experiment_growth, compute_growth, capability_growth, yield_change]
    if validated_growth is not None:
        values.append(validated_growth)
    if not all(np.isfinite(v) and v > 0 for v in values) or capability_growth <= 1:
        raise ValueError("Calibration needs positive ratios and capability growth above 1×.")
    if not 0 < transfer <= 1 or not 0 < parallel <= 1 or coding_slope < 0:
        raise ValueError("Calibration requires positive transfer and valid elasticities.")
    useful_growth = experiment_growth * yield_change if validated_growth is None else validated_growth
    residual = useful_growth / compute_growth ** parallel
    coding = (1 - transfer + transfer * capability_growth ** coding_slope) ** parallel
    quality = residual / _harmonic(coding, 1.0)
    slope = np.log(max(1.0, (quality - 1 + transfer) / transfer)) / np.log(capability_growth)
    return dict(quality_slope=float(slope), resource_adjusted_growth=float(residual),
                useful_growth=float(useful_growth), boundary=bool(quality < 1),
                basis="validated progress" if validated_growth is not None else "experiment proxy")


def calibrate_workflow(baseline_hours, before_hours, now_hours, capability_growth,
                       baseline_rate, floor):
    """Fit one matched stage, including intervention/rescue hours in its totals."""
    values = [baseline_hours, before_hours, now_hours, capability_growth, baseline_rate]
    if not all(np.isfinite(v) and v > 0 for v in values) or capability_growth <= 1:
        raise ValueError("Enter positive matched-project hours and capability growth above 1×.")
    before, now = 100 * before_hours / baseline_hours, 100 * now_hours / baseline_hours
    if not 0 <= floor < now < before <= 100:
        raise ValueError("Matched human hours must decline, remain above the floor, and not exceed baseline.")
    equivalent_months = 12 * np.log(capability_growth) / baseline_rate
    half_months = equivalent_months / np.log2((before - floor) / (now - floor))
    return dict(human_now=now, half_months=half_months)


def after_coding(samples, onset):
    """Durations only where both calendar events are modeled; retain censoring."""
    return np.subtract(samples, onset, out=np.full_like(samples, np.inf),
                       where=np.isfinite(samples) & np.isfinite(onset))


def _harmonic(labor, compute):
    return 2.0 / (1.0 / labor + 1.0 / compute)


def workflow_state(capability_log, baseline_rate, starts, half_months, floor,
                   success_start, success_half_months):
    """Human hours relative to a fixed baseline project, and useful completion.

    Capability progress is measured in years at today's effective-compute
    growth rate. Halving times are elicitable from repeated matched-project
    measurements; defaults are assumptions, not fitted observations.
    """
    equivalent_years = np.maximum(capability_log, 0) / baseline_rate
    human = floor + (starts - floor) * np.exp(
        -np.log(2) * equivalent_years[:, None] * 12 / half_months)
    success = 1 - (1 - success_start) * np.exp(
        -np.log(2) * equivalent_years * 12 / success_half_months)
    return human, success


def progress_multiple(previous_software, new_software, cycle_years, baseline_rate):
    """Delivered fixed-quality software-efficiency progress per calendar year.

    Log efficiency is proportional to training-compute savings at fixed quality.
    Physical compute growth is excluded. Baseline is the already AI-assisted
    validated software progress rate today.
    """
    return np.maximum(new_software - previous_software, 0) / (cycle_years * baseline_rate)


def reference_software(years, rate, compute_rate, parallel, initial_throughput, difficulty):
    """Exact software progress with the same resources and frozen AI skills."""
    growth = parallel * compute_rate
    effort = np.divide(np.expm1(growth * years), growth,
                       out=np.array(years, dtype=float).copy(), where=growth != 0)
    amount = rate * initial_throughput * effort
    return np.log1p(difficulty * amount) / difficulty if difficulty > 0 else amount


def simulate(params=None, n=1000, step=1 / 52, onset_years=None, experiment_slopes=None):
    """Weekly research and successor cycles, with censored milestone draws.

    A run freezes its algorithm snapshot at its start, integrates allocated
    training compute during the run, then waits for validation before deployment.
    Improvements discovered during either phase enter the next run. A fixed
    effective-compute reference prevents longer runs making the target easier.

    40/40/20 compute shares normalize training/experiments/agents to one today.
    The software doubling input is the initial rate under this reference split.
    Transfer discounts *additional* coding/judgment beyond today, so observed
    assistance is not multiplied into the starting rate a second time.
    """
    p = validate(params or {})
    if n < 1 or not 0 < step <= 0.25:
        raise ValueError("Use at least one draw and a time step in (0, 0.25] years.")
    rng = np.random.default_rng(int(p["seed"]))
    if experiment_slopes is not None:
        experiment_slopes = np.asarray(experiment_slopes, dtype=float)
        if experiment_slopes.shape != (n,) or not np.isfinite(experiment_slopes).all():
            raise ValueError("Provide one finite log experiment-growth rate per year per draw.")
    # Common quantiles preserve paired scenario comparisons. Bounded log-space
    # variation: ±uncertainty is a multiplicative factor exp(±uncertainty/100).
    onset = rng.uniform(p["onset_low"], p["onset_high"], n) / 12
    if onset_years is not None:
        onset = np.asarray(onset_years, dtype=float).copy()
        if onset.shape != (n,) or np.isnan(onset).any() or np.isneginf(onset).any():
            raise ValueError("Provide one onset date per draw; dates must be finite or +inf.")
    z = rng.uniform(-1, 1, (15, n))
    spread = p["uncertainty"] / 100
    factors = np.exp(spread * z)
    software_rate = np.log(2) * 12 / p["software_months"] * factors[0]
    compute_rate = np.log(p["compute_growth"]) * factors[1]
    train_time = p["training_months"] / 12 * factors[2]
    eval_time = p["validation_months"] / 12 * factors[3]
    taste_slope = p["taste_slope"] * factors[5]
    general_gap = p["general_gap"] * factors[6] * np.log(2)
    superiority_gap = p["superiority_gap"] * factors[7] * np.log(2)
    # Human-work and completion priors share the realized capability path,
    # while retaining separate bottlenecks for each critical workflow.
    starts = np.minimum(np.array([p["human_" + s] for s in WORKFLOW_KEYS])[None, :]
                        / 100 * factors[12:15].T, 1)
    floor = p["human_floor"] / 100
    starts = np.maximum(starts, floor)
    halves = np.array([p["half_" + s] for s in WORKFLOW_KEYS])[None, :] * factors[8:11].T
    success0 = np.minimum(p["project_success"] / 100 * factors[4], 1)
    success_half = p["success_half"] * factors[11]
    baseline_capability_rate = software_rate + compute_rate
    asi_target = np.full(n, np.inf)
    events = {name: np.full(n, np.inf) for name in MILESTONES}
    outcomes = {name: {key: np.full(n, np.nan) for key in
                      ("human_mean", "human_worst", "project_success", "progress_multiple",
                       "compute_matched_advantage")}
                for name in ("Today", "Full coding automation") + MILESTONES[1:]}

    def record(name, mask, human, success, progress, advantage):
        for key, values in (("human_mean", human @ WORKFLOW_WEIGHTS),
                            ("human_worst", human.max(axis=1)),
                            ("project_success", success), ("progress_multiple", progress),
                            ("compute_matched_advantage", advantage)):
            outcomes[name][key][mask] = values[mask]

    record("Today", np.ones(n, bool), starts, success0, np.ones(n), np.ones(n))
    coding_recorded = onset <= 0
    record("Full coding automation", onset == 0, starts, success0, np.ones(n), np.ones(n))
    full0 = (onset <= 0) & (starts.max(axis=1) <= p["full_human"] / 100) & (success0 >= p["full_success"] / 100)
    events[MILESTONES[1]][full0] = 0
    record(MILESTONES[1], full0, starts, success0, np.ones(n), np.ones(n))
    # Undeployed discoveries already in progress today. The matched reference
    # receives the identical backlog; it is never a second starting multiplier.
    initial_stock = software_rate * p["pipeline_months"] / 12
    software = initial_stock.copy()
    deployed = np.zeros(n)
    # Monthly observations of available models, never unfinished training runs.
    workflow_grid = np.unique(np.r_[np.arange(0, p["years"], 1 / 12), p["years"]])
    capability_history = np.zeros((n, len(workflow_grid)))
    validated_history = np.zeros_like(capability_history)
    reference_history = np.zeros_like(capability_history)
    fast_fraction = p["fast_share"] / 100
    fast_time = p["fast_months"] / 12
    fast_next = np.full(n, fast_time)
    fast_snapshot = initial_stock.copy()
    fast_deployed = np.zeros(n)
    reference_fast = np.zeros(n)
    last_validated = np.zeros(n)
    last_reference = np.zeros(n)
    snapshot = initial_stock.copy()
    run_compute = np.zeros(n)
    training_used = np.zeros(n)
    run_start = np.zeros(n)
    pending = np.zeros(n)
    pending_software = np.zeros(n)
    deployed_software = np.zeros(n)
    last_deploy = np.zeros(n)
    delivered_rate = np.zeros(n)
    reference_delivered = np.zeros(n)
    reference_deployed = np.zeros(n)
    reference_pending_software = np.zeros(n)
    reference_pending_capability = np.zeros(n)
    advantage = np.zeros(n)
    progress_cycles = np.zeros(n, dtype=int)
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
    initial_throughput = _harmonic((shares[2] / 0.2) ** parallel,
                                   (shares[1] / 0.4) ** parallel)
    # Today's model's training compute: a three-month run at the reference
    # 40% allocation. This constant must not depend on the chosen run duration.
    reference_train_compute = 0.4 * 3 / 12
    # Each draw advances at most one week, stopping exactly at its own run
    # completion/deployment events. This avoids adding a rounding delay every
    # cycle (which could move an entire milestone to the next successor).
    t = np.zeros(n)
    stopped_at = np.full(n, np.inf)
    limited = np.zeros(n, dtype=bool)
    while np.any(t < p["years"] - 1e-12):
        train_end = run_start + train_time
        deploy_end = train_end + eval_time
        next_event = np.minimum(np.where(finished, deploy_end, train_end), fast_next)
        next_event = np.minimum(next_event, np.where(onset > t + 1e-12, onset, np.inf))
        end = np.minimum(np.minimum(t + step, next_event), p["years"])
        active = t < p["years"] - 1e-12
        end = np.where(active, end, t)
        dt = np.maximum(end - t, 0)
        compute_log = compute_rate * (t + dt / 2)
        effective = deployed + fast_deployed
        # The RSI anchor supplies a smooth reduction in coding human hours.
        # Its productivity gain and endogenous coding gains overlap: use the
        # larger, not their product. Existing assistance is normalized to 1.
        coding_fraction = np.clip(np.divide(t + dt / 2, onset,
            out=np.ones(n), where=onset > 0), 0, 1)
        coding_fraction = np.where(onset <= 0, 0, coding_fraction)
        rsi_coding_log = np.log((1 - p["coding_today"] / 100) / .05) * coding_fraction
        coding_gain = np.maximum(rsi_coding_log, p["coding_slope"] * effective)
        if transfer:
            coding_log = np.logaddexp(np.log1p(-transfer) if transfer < 1 else -np.inf,
                                     np.log(transfer) + coding_gain)
            quality_log = np.logaddexp(np.log1p(-transfer) if transfer < 1 else -np.inf,
                                      np.log(transfer) + taste_slope * effective)
        else:
            coding_log = np.zeros(n)
            quality_log = np.zeros(n)
        labor_log = parallel * (compute_log + np.log(shares[2] / 0.2) + coding_log)
        experiment_log = parallel * (compute_log + np.log(shares[1] / 0.4))
        research_log = np.log(2) - np.logaddexp(-labor_log, -experiment_log) + quality_log
        if experiment_slopes is not None and transfer:
            mid = t + dt / 2
            resource_log = np.log(initial_throughput) + parallel * compute_rate * mid
            residual = (experiment_slopes - parallel * compute_rate) * mid
            proxy_log = resource_log + p["rsi_useful_growth"] / 100 * residual
            weight = p["rsi_trend_weight"] / 100 * np.exp(
                -np.log(2) * mid * 12 / p["rsi_trend_months"])
            # Substitute between two estimates of total research effort; do
            # not multiply the trend into feedback already calibrated from it.
            research_log = (1 - weight) * research_log + weight * proxy_log
        log_amount = np.log(software_rate) + research_log + np.log(
            dt, out=np.full(n, -np.inf), where=dt > 0)
        if beta > 0:
            updated = initial_stock + np.logaddexp(
                beta * (software - initial_stock), np.log(beta) + log_amount) / beta
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
        candidate = (1 - fast_fraction) * snapshot + np.log(np.maximum(run_compute / reference_train_compute, 1e-12))
        adopted = done & (candidate > deployed)
        pending[done] = np.maximum(deployed[done], candidate[done])
        pending_software[done] = np.where(adopted[done], (1 - fast_fraction) * snapshot[done], deployed_software[done])
        reference_snapshot = initial_stock + reference_software(run_start, software_rate, compute_rate,
                                                 parallel, initial_throughput, beta)
        reference_candidate = (1 - fast_fraction) * reference_snapshot + np.log(
            np.maximum(run_compute / reference_train_compute, 1e-12))
        reference_pending_capability[done] = np.maximum(reference_deployed[done], reference_candidate[done])
        reference_pending_software[done] = np.where(
            reference_candidate[done] > reference_deployed[done],
            (1 - fast_fraction) * reference_snapshot[done], reference_delivered[done])
        finished |= done
        ready = active & finished & (end >= deploy_end - 1e-12)
        deployed[ready] = pending[ready]
        fast_ready = active & (end >= fast_next - 1e-12)
        fast_deployed[fast_ready] = fast_fraction * fast_snapshot[fast_ready]
        reference_fast[fast_ready] = fast_fraction * (initial_stock[fast_ready] + reference_software(
            np.maximum(0, fast_next[fast_ready] - fast_time), software_rate[fast_ready],
            compute_rate[fast_ready], parallel, initial_throughput, beta))
        fast_snapshot[fast_ready] = software[fast_ready]
        fast_next[fast_ready] += fast_time
        # For non-ready slow runs, only their already deployed algorithms count.
        validated = np.where(ready, pending_software, deployed_software) + fast_deployed
        reference_validated = np.where(ready, reference_pending_software,
                                        reference_delivered) + reference_fast
        delivered_rate[ready] = progress_multiple(last_validated[ready], validated[ready],
            end[ready] - last_deploy[ready], software_rate[ready])
        reference_rate = progress_multiple(last_reference[ready], reference_validated[ready],
            end[ready] - last_deploy[ready], software_rate[ready])
        advantage[ready] = np.divide(delivered_rate[ready], reference_rate,
            out=np.zeros(ready.sum()), where=reference_rate > 0)
        last_validated[ready] = validated[ready]
        last_reference[ready] = reference_validated[ready]
        reference_delivered[ready] = reference_pending_software[ready]
        reference_deployed[ready] = reference_pending_capability[ready]
        deployed_software[ready] = pending_software[ready]
        last_deploy[ready] = end[ready]
        effective = deployed + fast_deployed
        rows = np.flatnonzero(ready | fast_ready)
        future = workflow_grid[None, :] >= end[rows, None] - 1e-12
        for history, values in ((capability_history, effective),
                                (validated_history, validated),
                                (reference_history, reference_validated)):
            history[rows] = np.where(future, values[rows, None], history[rows])
        coding_gain = np.maximum(rsi_coding_log, p["coding_slope"] * effective)
        if transfer:
            ai_output_log = parallel * np.logaddexp(
                np.log1p(-transfer) if transfer < 1 else -np.inf,
                np.log(transfer) + coding_gain) + np.logaddexp(
                np.log1p(-transfer) if transfer < 1 else -np.inf,
                np.log(transfer) + taste_slope * effective)
        else:
            ai_output_log = np.zeros(n)
        improved = ready & (snapshot > 0) & (ai_output_log >= previous_ai_output + np.log(1.1))
        cycles[ready] = np.where(improved[ready], cycles[ready] + 1, 0)
        previous_ai_output[ready] = ai_output_log[ready]
        feedback = cycles >= p["feedback_cycles"]
        ev = events[MILESTONES[0]]
        hit = feedback & ~np.isfinite(ev)
        ev[hit] = end[hit]
        human, success = workflow_state(effective, baseline_capability_rate, starts,
                                        halves, floor, success0, success_half)
        coding_hit = active & (end >= onset) & ~coding_recorded
        record("Full coding automation", coding_hit, human, success, delivered_rate, advantage)
        coding_recorded |= coding_hit
        full = (end >= onset) & (human.max(axis=1) <= p["full_human"] / 100) & (success >= p["full_success"] / 100)
        hit = full & ~np.isfinite(events[MILESTONES[1]])
        events[MILESTONES[1]][hit] = end[hit]
        record(MILESTONES[1], hit, human, success, delivered_rate, advantage)
        fast = full & (delivered_rate >= p["progress_target"]) \
            & (advantage >= p["advantage_target"])
        progress_cycles[ready] = np.where(fast[ready], progress_cycles[ready] + 1, 0)
        hit = (progress_cycles >= p["progress_cycles"]) & ~np.isfinite(events[MILESTONES[2]])
        events[MILESTONES[2]][hit] = end[hit]
        record(MILESTONES[2], hit, human, success, delivered_rate, advantage)
        # The breadth bridge remains speculative, now anchored to the actual
        # autonomous progress milestone instead of a 3× taste threshold.
        asi_target[hit] = deployed[hit] + general_gap[hit] + superiority_gap[hit]
        hit = np.isfinite(asi_target) & (deployed >= asi_target) & ~np.isfinite(events[MILESTONES[3]])
        events[MILESTONES[3]][hit] = end[hit]
        record(MILESTONES[3], hit, human, success, delivered_rate, advantage)
        tracking = events[MILESTONES[-1]] >= end
        last_labor[tracking] = labor_log[tracking]
        last_experiments[tracking] = experiment_log[tracking]
        snapshot[ready] = software[ready]
        run_start[ready] = end[ready]
        run_compute[ready] = 0
        finished[ready] = False
        all_reached = np.all(np.isfinite(np.array(list(events.values()))), axis=0)
        stopped_at[active & (all_reached | diverged)] = end[active & (all_reached | diverged)]
        t = np.where(all_reached | diverged, p["years"], end)
    total_budget = np.divide(np.expm1(compute_rate * p["years"]), compute_rate,
                             out=np.full(n, float(p["years"])), where=compute_rate != 0)
    # Retain all draws in each percentile, including slow/non-arriving paths.
    # Early-stopped paths retain their last deployed state; no post-ASI
    # capability growth is invented for the chart.
    human_quantiles, success_quantiles = [], []
    for capability in capability_history.T:
        human, success = workflow_state(capability, baseline_capability_rate, starts,
                                        halves, floor, success0, success_half)
        human_quantiles.append(np.quantile(human, [.1, .5, .9], axis=0))
        success_quantiles.append(np.quantile(success, [.1, .5, .9]))
    # A trailing rate needs a complete measurement window. In particular,
    # there is no observed 1x rate at t=0 to connect to early batch deliveries.
    rate_history = np.full_like(validated_history, np.nan)
    matched_history = np.full_like(validated_history, np.nan)
    for j, year in enumerate(workflow_grid[1:], 1):
        if year < .25 - 1e-12:
            continue
        left = max(0, year - .25)
        k = np.searchsorted(workflow_grid, left + 1e-12, side="right") - 1
        elapsed = year - workflow_grid[k]
        gains = validated_history[:, j] - validated_history[:, k]
        ref_gains = reference_history[:, j] - reference_history[:, k]
        rate_history[:, j] = gains / elapsed / software_rate
        matched_history[:, j] = np.divide(gains, ref_gains,
            out=np.full(n, np.nan), where=ref_gains > 1e-12)
        stopped = workflow_grid[j - 1] >= stopped_at
        rate_history[stopped, j] = rate_history[stopped, j - 1]
        matched_history[stopped, j] = matched_history[stopped, j - 1]
    acceleration_events = {}
    for target in (2, 5):
        crossed = rate_history >= target
        acceleration_events[f"{target}× research progress"] = np.where(
            crossed.any(axis=1), workflow_grid[np.argmax(crossed, axis=1)], np.inf)
    def finite_quantiles(history):
        return np.array([np.quantile(v[np.isfinite(v)], [.1, .5, .9])
                         if np.isfinite(v).any() else [np.nan] * 3 for v in history.T])

    return {
        "params": p, "version": MODEL_VERSION, "onset": onset,
        "events": events, "acceleration_events": acceleration_events,
        "time_origin": "today",
        "compute_limited": last_experiments <= last_labor,
        "budget": total_budget, "allocated": shares[:, None] * total_budget,
        "training_used": training_used,
        "software_log": software, "deployed_log": deployed,
        "deployed_software_log": deployed_software, "fast_software_log": fast_deployed,
        "outcomes": outcomes,
        "workflow_progress": {
            "years": workflow_grid,
            "human_quantiles": np.array(human_quantiles),
            "success_quantiles": np.array(success_quantiles),
            "rate_quantiles": finite_quantiles(rate_history),
            "cumulative_quantiles": np.quantile(
                validated_history / software_rate[:, None] * 12, [.1, .5, .9], axis=0).T,
            "matched_quantiles": finite_quantiles(matched_history),
            "matched_coverage": np.mean(np.isfinite(matched_history), axis=0),
            "validated_log": validated_history,
            "reference_log": reference_history,
            "rate": rate_history,
            "matched": matched_history,
        },
        "progress_multiple": delivered_rate,
        "compute_matched_advantage": advantage,
        "numerically_limited": limited,
    }


def cdf(samples, grid):
    """CDF over all draws, including +inf (not reached) in the denominator."""
    return np.searchsorted(np.sort(samples), grid, side="right") / len(samples)


def quantile(samples, q, horizon=np.inf):
    """Empirical inverse CDF, preserving censored probability mass."""
    value = np.sort(samples)[max(0, int(np.ceil(q * len(samples))) - 1)]
    return float(value) if value <= horizon else np.inf
