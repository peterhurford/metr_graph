"""AI research from today through takeoff; assumptions and explicit calibration.

Times are years after today; coding automation is an inherited milestone.
Natural logs represent capability and efficiency relative to today's system.
All randomness is local and seeded; every setting uses the same base draws.
"""

import hashlib
from statistics import NormalDist
from pathlib import Path

import numpy as np


MODEL_VERSION = "takeoff-v22-ladder-human-hours"
SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WORKFLOW_STAGES = ("Research direction", "Experiment design and interpretation",
                   "Verification and integration")
WORKFLOW_KEYS = ("direction", "experiments", "verification")
WORKFLOW_WEIGHTS = np.array([0.25, 0.5, 0.25])
# RSI: research direction (the stage that chooses what to try next) needs at
# most `full_human`% of baseline human hours, once coding is automated.
# Capability past it is read as ECI.
MILESTONES = ("RSI",)
DEFAULTS = {
    "onset_low": 6, "onset_high": 24, "years": 10, "seed": 42,
    "software_months": 9.0, "compute_growth": 2.0,
    "training_months": 3.0, "validation_months": 1.0,
    "training_share": 40, "experiment_share": 40,
    "taste_slope": 0.8, "coding_slope": 0.8,
    "transfer": 60, "difficulty": 0.5, "parallelization": 0.5,
    "uncertainty": 40,
    "human_direction": 40.0, "human_experiments": 30.0, "human_verification": 50.0,
    "half_direction": 6.0, "half_experiments": 9.0, "half_verification": 12.0,
    "human_floor": 0.0, "full_human": 5.0,
    # Autopilot: from a draw's RSI, no research stage needs people at all (the
    # hours left in other stages are dropped). 0 keeps each stage's own decline.
    "rsi_autopilot": 1.0,
    "fast_share": 25.0, "fast_months": 1.0,
    "coding_today": 70.0,
    # Steady state: the next model ships with one full cycle (training +
    # validation) of discoveries not yet deployed.
    "pipeline_months": 4.0,
    "rsi_trend_weight": 50.0, "rsi_trend_months": 12.0, "rsi_useful_growth": 100.0,
    # Labor and experiment compute combine as a CES mean with this exponent:
    # -1 is the harmonic mean (strong complements), 0 Cobb-Douglas.
    "substitution": -0.5,
    # Speed of AI on work it has taken over, relative to the people it
    # replaced; sets how much a falling human share speeds research (Amdahl).
    "ai_speed": 30.0,
    # More copies: AI's share of research labor scales with the deployed
    # model's algorithmic efficiency to this power, as the same inference
    # compute runs more AI researchers. 0 leaves the number of copies fixed.
    "ai_copies": 1.0,
    # Share of the non-AI input growth that would hold today's pace with AI
    # frozen (researchers, spending); 0 leaves compute the only growing input.
    "baseline_inputs": 100.0,
    # Change in the judgment elasticity per 10x gain in capability (%): above 0
    # judgment gains compound as AI gets more capable (b falls), below 0 they
    # fade (b rises). 0 keeps today's measured rate throughout.
    "judgment_drift": 0.0,
    # Per-draw spread on the four settings that decide whether takeoff runs
    # away (TAKEOFF_MODEL.md, *Parameter uncertainty*): 1-sigma of ln r,
    # 1-sigma of ln(judgment elasticity), 1-sigma of the drift in percentage
    # points, and 1-sigma of ln(elasticity of substitution 1 / (1 - rho)).
    "returns_spread": 65.0, "judgment_spread": 50.0,
    "drift_spread": 15.0, "substitution_spread": 80.0,
    # Compute growth keeps this share of its log rate from `compute_slow_years`
    # after today (the app sets end-2030, where Epoch's feasibility analysis
    # stops); an assumption, not a fitted rate.
    "compute_slowdown": 50.0, "compute_slow_years": 4.0,
    # Multiplier on each draw's log compute growth after its own RSI date: AI
    # speeding chip design, fabs and spending. It waits `rsi_compute_delay`
    # months (a chip generation, a fab's build), then ramps in linearly over
    # `rsi_compute_ramp` months. 1 leaves compute as projected; the size is an
    # assumption, with no published estimate.
    "rsi_compute_boost": 2.0, "rsi_compute_delay": 18.0, "rsi_compute_ramp": 12.0,
    # Software ceiling (Forethought): returns to research fall linearly to 0
    # as algorithmic efficiency nears a limit drawn uniformly this many orders
    # of magnitude above today's. A high of 0 removes it.
    "ceiling_low_oom": 6.0, "ceiling_high_oom": 16.0,
    # Share of algorithmic gains that pay off only at scale (Gundlach et al.
    # 2025): that share counts in full once training compute has grown
    # `scale_lag_oom` orders of magnitude past where it was when the gain was
    # found. Logit-normal share and lognormal lag per draw; spreads are each
    # 1-sigma (log-odds, log) times 100.
    "scale_dependent": 50.0, "scale_dependent_spread": 100.0,
    "scale_lag_oom": 1.0, "scale_lag_spread": 70.0,
}
# Every spread at zero: each draw uses the settings themselves.
POINT_ESTIMATE = {"uncertainty": 0, "returns_spread": 0, "judgment_spread": 0,
                  "drift_spread": 0, "substitution_spread": 0, "scale_dependent_spread": 0,
                  "scale_lag_spread": 0}
PRESETS = {
    "Central": {},
    "Fast": {"software_months": 6.0, "training_months": 1.0,
             "validation_months": 0.5, "taste_slope": 1.2,
             "transfer": 90, "difficulty": 0.2},
    "Bottlenecked": {"software_months": 18.0, "compute_growth": 1.3,
                     "training_months": 6.0, "validation_months": 2.0,
                     "taste_slope": 0.4,
                     "transfer": 20, "difficulty": 1.0},
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
    if not all(0 <= p[k] <= 200 for k in ("returns_spread", "judgment_spread",
                                          "drift_spread", "substitution_spread")):
        raise ValueError("Parameter spreads must be between 0 and 200%.")
    if p["software_months"] <= 0 or p["training_months"] <= 0 \
            or p["validation_months"] < 0 or p["compute_growth"] < 1:
        raise ValueError("Use positive doubling/run times and compute growth of at least 1×.")
    if p["taste_slope"] < 0 or p["coding_slope"] <= 0:
        raise ValueError("Research quality slope must be nonnegative and coding slope positive.")
    if p["difficulty"] < 0 or not 0 < p["parallelization"] <= 1:
        raise ValueError("Use nonnegative difficulty and parallelization in (0, 1].")
    if int(p["seed"]) != p["seed"] or not 0 <= p["seed"] < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2³²−1.")
    if not 0 <= p["human_floor"] <= 100 or not 0 <= p["full_human"] <= 100:
        raise ValueError("Human-work floor and automation threshold must be between 0 and 100%.")
    for stage in WORKFLOW_KEYS:
        if not p["human_floor"] <= p["human_" + stage] <= 100 or p["half_" + stage] <= 0:
            raise ValueError("Each workflow's starting human work must be at least its floor, "
                             "and its halving time must be positive.")
    if not 0 <= p["fast_share"] <= 100 or p["fast_months"] <= 0:
        raise ValueError("Fast feedback needs a share in [0, 100] and a positive validation cycle.")
    if not 0 <= p["coding_today"] < 95:
        raise ValueError("Today's coding automation must be below the 95% coding milestone.")
    if not 0 <= p["pipeline_months"] <= 12:
        raise ValueError("Initial research pipeline must be between 0 and 12 months of progress.")
    if not 0 <= p["rsi_trend_weight"] <= 100 or not 0 <= p["rsi_useful_growth"] <= 100 \
            or p["rsi_trend_months"] <= 0:
        raise ValueError("RSI trend weights must be percentages and their half-life positive.")
    if not -5 <= p["substitution"] <= 0.9 or p["ai_speed"] < 1 \
            or not 0 <= p["baseline_inputs"] <= 200 or not -90 <= p["judgment_drift"] <= 300 \
            or not 0 <= p["compute_slowdown"] <= 100 or p["compute_slow_years"] < 0 \
            or not 0.5 <= p["rsi_compute_boost"] <= 5 \
            or not 0 <= p["rsi_compute_delay"] <= 60 or not 0 <= p["rsi_compute_ramp"] <= 60 \
            or not 0 <= p["ceiling_high_oom"] <= 60 \
            or not (p["ceiling_high_oom"] == 0 or 0 < p["ceiling_low_oom"] <= p["ceiling_high_oom"]) \
            or p["rsi_autopilot"] not in (0, 1) or not 0 <= p["ai_copies"] <= 2 \
            or not 0 <= p["scale_dependent"] <= 100 or not 0 <= p["scale_dependent_spread"] <= 400 \
            or not 0 < p["scale_lag_oom"] <= 10 or not 0 <= p["scale_lag_spread"] <= 400:
        raise ValueError("Use substitution in [-5, 0.9], AI speed of at least 1x, and "
                         "baseline input growth in [0, 200]%.")
    return p


def calibrate_feedback(experiment_growth, compute_growth, capability_growth,
                       yield_change=1.0, validated_growth=None, transfer=.6,
                       coding_slope=.8, parallel=.5, substitution=-0.5,
                       automation_growth=1.0):
    """Effective quality elasticity after the model's own research equation
    has accounted for compute, coding help and `automation_growth` (the
    Amdahl labor gain from people's falling share over the same window).

    Aggregate experiments are a proxy, not causal identification. A supplied
    validated-progress ratio supersedes that proxy. Ratios compare matched
    periods; current assistance is already in today's starting progress rate.
    """
    values = [experiment_growth, compute_growth, capability_growth, yield_change]
    if validated_growth is not None:
        values.append(validated_growth)
    values.append(automation_growth)
    if not all(np.isfinite(v) and v > 0 for v in values) or capability_growth <= 1:
        raise ValueError("Calibration needs positive ratios and capability growth above 1×.")
    if not 0 < transfer <= 1 or not 0 < parallel <= 1 or coding_slope < 0:
        raise ValueError("Calibration requires positive transfer and valid elasticities.")
    useful_growth = experiment_growth * yield_change if validated_growth is None else validated_growth
    residual = useful_growth / compute_growth ** parallel
    coding = ((1 - transfer + transfer * capability_growth ** coding_slope)
              * (1 - transfer + transfer * automation_growth)) ** parallel
    quality = residual / _ces(coding, 1.0, substitution)
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


def judgment_gain_log(effective, slope, drift):
    """Log judgment gain at capability `effective` (natural log of effective
    compute): the integral of an elasticity that starts at `slope` and changes
    by `drift` (a fraction) per 10x of capability."""
    k = np.log1p(drift) / np.log(10)
    flat = np.abs(k) < 1e-12
    safe = np.where(flat, 1.0, k)
    # Capped short of overflow: a draw that far out has already diverged.
    return np.where(flat, slope * effective,
                    slope * np.expm1(np.minimum(safe * effective, 700)) / safe)


def _ces_log(x, y, rho):
    """Log of the CES mean of exp(x) and exp(y), equal to 1 when both are 1."""
    flat = np.abs(rho) < 1e-9
    safe = np.where(flat, 1.0, rho)
    return np.where(flat, (x + y) / 2,
                    (np.logaddexp(safe * x, safe * y) - np.log(2)) / safe)


def _ces(a, b, rho):
    return np.exp(_ces_log(np.log(a), np.log(b), rho))


def normal_scores(uniform):
    """Standard-normal quantiles of draws uniform on [-1, 1]: the same
    rank in every draw, so paired scenarios stay paired."""
    u = np.clip((np.asarray(uniform) + 1) / 2, 1e-12, 1 - 1e-12)
    return np.vectorize(NormalDist().inv_cdf, otypes=[float])(u)


def workflow_state(capability_log, baseline_rate, starts, half_months, floor,
                   curve=None, reference_half=None):
    """Human hours per stage relative to a fixed baseline project.

    Capability progress is measured in years at today's effective-compute
    growth rate. Without `curve`, each stage halves its hours above the floor
    every `half_months`. With `curve` = (years, share), each stage follows that
    shape instead (share relative to its start), at `reference_half` /
    `half_months` its pace: the halving times then only set each stage's
    speed along the curve.
    """
    equivalent_years = np.maximum(capability_log, 0) / baseline_rate
    if curve is None:
        decline = np.exp(-np.log(2) * equivalent_years[:, None] * 12 / half_months)
    else:
        years, share = curve
        along = equivalent_years[:, None] * reference_half / half_months
        decline = np.interp(along, years, share / share[0])
    return floor + (starts - floor) * decline


def progress_multiple(previous_software, new_software, cycle_years, baseline_rate):
    """Delivered fixed-quality software-efficiency progress per calendar year.

    Log efficiency is proportional to training-compute savings at fixed quality.
    Physical compute growth is excluded. Baseline is the already AI-assisted
    validated software progress rate today.
    """
    return np.maximum(new_software - previous_software, 0) / (cycle_years * baseline_rate)


def log_level(t, early, late, at):
    """Log of an input growing at `early` a year until `at`, then `late`."""
    return early * np.minimum(t, at) + late * np.maximum(np.subtract(t, at), 0)


def grown(t, early, late, at):
    """Integral from 0 to t of exp(log_level): the input accumulated."""
    def e(g, x):
        x = np.asarray(x, dtype=float) * np.ones_like(g)
        return np.divide(np.expm1(g * x), g, out=x.copy(), where=g != 0)
    head = np.minimum(t, at)
    return e(early, head) + np.exp(early * head) * e(late, np.maximum(np.subtract(t, at), 0))


def ceiling_step(log_x, log_amount, k, ceiling):
    """Log of x = 1 - S / ceiling after research `amount` (log_amount) under
    dS/dt = A x^k: returns to research falling linearly to 0 at the ceiling
    (k = difficulty x ceiling). Exact for any k; x reaches 0 only for k < 1."""
    k = np.asarray(k, dtype=float) * np.ones_like(np.asarray(log_x, dtype=float))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        above = -np.logaddexp(-(k - 1) * log_x, np.log(np.abs(k - 1))
                              + log_amount - np.log(ceiling)) / (k - 1)
        m = 1 - k
        left = np.exp(m * log_x) - m * np.exp(np.minimum(log_amount, 700)) / ceiling
        below = np.where(left > 0, np.log(np.maximum(left, 1e-300)) / m, -np.inf)
        unit = log_x - np.exp(np.minimum(log_amount, 700)) / ceiling
    return np.where(np.abs(k - 1) < 1e-9, unit, np.where(k > 1, above, below))


def reference_software(years, rate, compute_rate, parallel, initial_throughput, difficulty,
                       late_rate=None, at=np.inf, ceiling=np.inf):
    """Exact software progress with the same resources and frozen AI skills;
    inputs grow at `compute_rate`, then `late_rate` from `at`, under a
    software `ceiling` (natural log; inf for none)."""
    late = compute_rate if late_rate is None else late_rate
    effort = grown(years, parallel * np.asarray(compute_rate, dtype=float),
                   parallel * np.asarray(late, dtype=float), at)
    amount = rate * initial_throughput * effort
    if not np.all(difficulty > 0):
        return amount
    if np.all(np.isinf(ceiling)):
        return np.log1p(difficulty * amount) / difficulty
    with np.errstate(divide="ignore"):
        log_x = ceiling_step(np.zeros_like(np.asarray(amount, dtype=float)), np.log(amount),
                             difficulty * ceiling, ceiling)
    return -ceiling * np.expm1(log_x)


def simulate(params=None, n=1000, step=1 / 52, onset_years=None, experiment_slopes=None,
             compute_rates=None, historical_compute_rate=None, coding_capability=None,
             coding_weights=None, human_curve=None):
    """Weekly research and successor cycles, with censored milestone draws.

    RSI is the first time research direction is at or under `full_human`% of
    baseline human hours with the coding date arrived. Every draw runs to the
    horizon (only a numerically divergent one stops), so its capability path
    (`capability_log`) covers the whole window. `compute_rates`, one log
    growth rate a year per draw, replaces `compute_growth` and its spread;
    `historical_compute_rate` is the compute growth behind today's steady pace
    (default: each draw's own rate). `human_curve` = (years ahead, human
    share), the shape research hours follow (see `workflow_state`); without
    it they decline exponentially. `coding_weights` (w_compute, w_software)
    measures capability as w_compute x its training-compute part plus
    w_software x its algorithmic part (both natural logs vs today's model)
    for that test, so the level can be an ECI gain; default 1 and 1, log
    effective compute. `coding_capability`, a capability level
    (log effective compute), counts coding as automated once a deployed model
    reaches it, ahead of the inherited date; the result's `coding_at`.

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
    # Common quantiles preserve paired scenario comparisons. Lognormal
    # variation: `uncertainty`/100 is the 1-sigma of each factor's log.
    onset = rng.uniform(p["onset_low"], p["onset_high"], n) / 12
    if onset_years is not None:
        onset = np.asarray(onset_years, dtype=float).copy()
        if onset.shape != (n,) or np.isnan(onset).any() or np.isneginf(onset).any():
            raise ValueError("Provide one onset date per draw; dates must be finite or +inf.")
    # Rows fill in order, so appending a factor leaves the earlier draws unchanged.
    # Rows 4, 6, 7 and 11 fed retired milestones; they stay drawn so the others
    # keep their values.
    raw = rng.uniform(-1, 1, (21, n))
    z = normal_scores(raw)
    spread = p["uncertainty"] / 100
    factors = np.exp(spread * z)
    software_rate = np.log(2) * 12 / p["software_months"] * factors[0]
    compute_rate = np.log(p["compute_growth"]) * factors[1]
    if compute_rates is not None:
        compute_rate = np.asarray(compute_rates, dtype=float).copy()
        if compute_rate.shape != (n,) or not np.isfinite(compute_rate).all() \
                or (compute_rate < 0).any():
            raise ValueError("Provide one finite, non-negative compute growth rate per draw.")
    train_time = p["training_months"] / 12 * factors[2]
    eval_time = p["validation_months"] / 12 * factors[3]
    taste_slope = p["taste_slope"] * np.exp(p["judgment_spread"] / 100 * z[5])
    drift = np.maximum(p["judgment_drift"] / 100 + p["drift_spread"] / 100 * z[16], -0.9)
    # rho through the elasticity of substitution 1 / (1 - rho), lognormal
    # around the setting's, held inside the range `validate` accepts.
    elasticity = np.exp(-np.log1p(-p["substitution"]) + p["substitution_spread"] / 100 * z[17])
    rho = np.clip(1 - 1 / elasticity, -5, 0.9)
    # Human-work priors share the realized capability path, while retaining
    # separate bottlenecks for each critical workflow.
    starts = np.minimum(np.array([p["human_" + s] for s in WORKFLOW_KEYS])[None, :]
                        / 100 * factors[12:15].T, 1)
    floor = p["human_floor"] / 100
    starts = np.maximum(starts, floor)
    halves = np.array([p["half_" + s] for s in WORKFLOW_KEYS])[None, :] * factors[8:11].T
    # Each stage's pace along `human_curve` is the settings' weighted halving
    # time over its own (with its draw's spread).
    reference_half = float(np.array([p["half_" + s] for s in WORKFLOW_KEYS]) @ WORKFLOW_WEIGHTS)

    def humans(capability):
        return workflow_state(capability, baseline_capability_rate, starts, halves, floor,
                              human_curve, reference_half)
    baseline_capability_rate = software_rate + compute_rate
    events = {name: np.full(n, np.inf) for name in MILESTONES}
    outcomes = {name: {key: np.full(n, np.nan) for key in
                      ("human_mean", "human_worst", "human_direction",
                       "human_experiments", "human_verification",
                       "progress_multiple", "compute_matched_advantage")}
                for name in ("Today", "Full coding automation") + MILESTONES}

    def record(name, mask, human, progress, advantage):
        for key, values in (("human_mean", human @ WORKFLOW_WEIGHTS),
                            ("human_worst", human.max(axis=1)),
                            ("human_direction", human[:, 0]),
                            ("human_experiments", human[:, 1]),
                            ("human_verification", human[:, 2]),
                            ("progress_multiple", progress),
                            ("compute_matched_advantage", advantage)):
            outcomes[name][key][mask] = values[mask]

    record("Today", np.ones(n, bool), starts, np.ones(n), np.ones(n))
    coding_recorded = onset <= 0
    # Coding counts as automated at the inherited date, or once a deployed
    # model reaches `coding_capability`, whichever comes first.
    coding = onset.copy()
    record("Full coding automation", onset == 0, starts, np.ones(n), np.ones(n))
    rsi0 = (onset <= 0) & (starts[:, 0] <= p["full_human"] / 100)
    events["RSI"][rsi0] = 0
    record("RSI", rsi0, starts, np.ones(n), np.ones(n))
    # Undeployed discoveries already in progress today. The matched reference
    # receives the identical backlog; it is never a second starting multiplier.
    initial_stock = software_rate * p["pipeline_months"] / 12
    software = initial_stock.copy()
    deployed = np.zeros(n)
    # Monthly observations of available models, never unfinished training runs.
    workflow_grid = np.unique(np.r_[np.arange(0, p["years"], 1 / 12), p["years"]])
    capability_history = np.zeros((n, len(workflow_grid)))
    validated_history = np.zeros_like(capability_history)
    # The deployed model's algorithmic efficiency as it pays off at its scale.
    realized_history = np.zeros_like(capability_history)
    # Monthly frontier compute (log, vs today) and the research stock found by
    # then, for the scale-dependent lag; months not yet reached read +inf.
    compute_seen = np.full_like(capability_history, np.inf)
    compute_seen[:, 0] = 0.0
    stock_seen = np.zeros_like(capability_history)
    stock_seen[:, 0] = initial_stock
    row_offset = np.arange(n) * 1e7       # wider than any row's range
    seen_flat = [None]                    # compute_seen flattened, rebuilt on change
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
    finished = np.zeros(n, dtype=bool)
    last_labor = np.ones(n)
    last_experiments = np.ones(n)
    shares = np.array([p["training_share"], p["experiment_share"],
                       100 - p["training_share"] - p["experiment_share"]]) / 100
    # Frontier compute (log, vs today) when the current run, the pending
    # model's run and the deployed model's run started: a model's scale.
    run_start_scale = np.zeros(n)
    pending_scale = np.zeros(n)
    deployed_scale = np.zeros(n)
    transfer = p["transfer"] / 100
    # difficulty = lambda / r, so a lognormal r gives a lognormal difficulty.
    beta = p["difficulty"] * np.exp(p["returns_spread"] / 100 * z[15])
    # Software ceiling in natural-log efficiency above today's stock, uniform
    # between the settings (row 18); inf without one.
    ceiling = (np.log(10) * (p["ceiling_low_oom"] + (p["ceiling_high_oom"] - p["ceiling_low_oom"])
                             * (raw[18] + 1) / 2)
               if p["ceiling_high_oom"] > 0 and p["difficulty"] > 0 else np.full(n, np.inf))
    capped = np.isfinite(ceiling)
    # Scale-dependent share of algorithmic gains, logit-normal per draw (row 19).
    share_pct = p["scale_dependent"]
    with np.errstate(divide="ignore"):
        share_logit = (np.log(share_pct) - np.log(100 - share_pct) if 0 < share_pct < 100
                       else (np.inf if share_pct >= 100 else -np.inf))
    scale_share = 1 / (1 + np.exp(-(share_logit + p["scale_dependent_spread"] / 100 * z[19])))
    # The compute growth (natural log) a scale-dependent gain waits for (row 20).
    scale_lag = np.log(10) * p["scale_lag_oom"] * np.exp(p["scale_lag_spread"] / 100 * z[20])
    parallel = p["parallelization"]
    initial_throughput = np.broadcast_to(_ces((shares[2] / 0.2) ** parallel,
                                              (shares[1] / 0.4) ** parallel, rho), (n,)).copy()
    # Today's pace has held steady while compute grew at its historical rate.
    # The other research inputs grow (or lag) by whatever holds that steady
    # state, d(software)/dt = rate * pace with pace = inputs^parallel /
    # exp(beta * software), so with AI frozen only a change in compute growth
    # from its historical rate moves the pace. Never applied to training.
    hist_rate = (compute_rate if historical_compute_rate is None
                 else np.full(n, float(historical_compute_rate)))
    # Software growth per unit of compute growth in the history behind today's
    # pace: the scale at which scale-dependent gains have paid off so far.
    scale_ratio = np.divide(software_rate, hist_rate, out=np.full(n, np.inf), where=hist_rate > 0)

    def pays_off(rows, scale_now, trained_software, fast_software, reference_trained):
        """For draws `rows`, a model's algorithmic part as it pays off at its
        compute. Fast gains (tooling, kernels: Gundlach et al.'s
        scale-invariant kind) count at once. Of the gains that came through
        training, the scale-dependent share counts in full only for gains
        found while frontier compute was `scale_lag` below the frontier
        compute when this model's run began (before today, history's balanced
        path: today's stock plus `scale_ratio` x that compute), plus what
        history's balanced pace adds over one lag, and never below the
        frozen-AI reference's trained software for the same model, which
        already carries history's own pace. Gains found faster wait until
        models train at that much more compute, then count in full."""
        lag = scale_lag[rows]
        target = scale_now - lag
        # Each row of compute_seen is non-decreasing, so offsetting row i by
        # i x span sorts them all into one array: one search finds every row.
        if seen_flat[0] is None:
            seen_flat[0] = (np.where(np.isfinite(compute_seen), compute_seen, 1e6)
                            + row_offset[:, None]).ravel()
        last = len(workflow_grid) - 1
        idx = np.searchsorted(seen_flat[0], np.minimum(target, 1e6 - 1) + row_offset[rows],
                              side="right") - rows * (last + 1) - 1
        c0 = compute_seen[rows, np.maximum(idx, 0)]
        s0 = stock_seen[rows, np.maximum(idx, 0)]
        c1 = compute_seen[rows, np.minimum(idx + 1, last)]
        s1 = stock_seen[rows, np.minimum(idx + 1, last)]
        # Past the last month recorded, interpolate toward the present.
        ahead = ~np.isfinite(c1) | (idx >= last)
        c1 = np.where(ahead, log_level(t[rows], compute_rate[rows], compute_late[rows], slow_at)
                      + boost_log[rows], c1)
        s1 = np.where(ahead, software[rows], s1)
        ratio = scale_ratio[rows]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.clip(np.divide(target - c0, c1 - c0, out=np.zeros(len(rows)),
                                     where=c1 > c0), 0, 1)
            earlier = np.where(idx >= 0, s0 + frac * (s1 - s0),
                               initial_stock[rows] + ratio * target)
            due = np.where(np.isinf(ratio), np.inf, np.maximum(
                (1 - fast_fraction) * (earlier + ratio * lag), reference_trained))
        return (fast_software + trained_software
                - scale_share[rows] * np.maximum(trained_software - due, 0))

    def realized(rows=None):
        """(training-compute part, algorithmic part as it pays off, capability)
        of the deployed model, refreshed in `now` for draws `rows` (all by
        default)."""
        rows = np.arange(n) if rows is None else rows
        compute_part_now = deployed[rows] - deployed_software[rows]
        paid = pays_off(rows, deployed_scale[rows], deployed_software[rows], fast_deployed[rows],
                        reference_delivered[rows])
        now[0][rows], now[1][rows] = compute_part_now, paid
        now[2][rows] = compute_part_now + paid
        return now

    extra_rate = p["baseline_inputs"] / 100 * (
        beta * software_rate * initial_throughput / parallel - hist_rate)
    input_rate = np.maximum(compute_rate + extra_rate, 0)
    extra_rate = input_rate - compute_rate
    # From `slow_at`, compute keeps `compute_slowdown`% of its log growth; the
    # other inputs keep their own, so research inputs slow by compute's share.
    slow_at = p["compute_slow_years"]
    compute_late = compute_rate * p["compute_slowdown"] / 100
    input_late = np.maximum(compute_late + extra_rate, 0)
    extra_late = input_late - compute_late
    # Amdahl: AI labor on research speeds up as the human share h falls, by
    # (h0 + (1 - h0)/s) / (h + (1 - h)/s), normalized to today's share h0.
    ai_speed = p["ai_speed"]
    human0 = starts @ WORKFLOW_WEIGHTS
    amdahl0 = np.log(human0 + (1 - human0) / ai_speed)
    # Ongoing useful research production is distinct from batched deliveries.
    # Today's production baseline already includes existing AI assistance.
    pace_history = np.repeat(initial_throughput[:, None], len(workflow_grid), axis=1)
    reference_pace_history = pace_history.copy()
    # log pace = resources + judgment - difficulty, recorded with the pace so
    # the chart can show what moves it. Resources carry the trend blend and
    # split exactly as compute + labor + ai: CES is homogeneous, so the input
    # growth labor and experiments share (compute, and the non-AI inputs)
    # comes off it whole; the rest is AI help through the CES.
    factor_history = {k: np.zeros_like(capability_history)
                      for k in ("resources", "judgment", "difficulty",
                                "compute", "labor", "ai")}
    factor_history["resources"][:] = np.log(initial_throughput)[:, None]
    factor_history["ai"][:] = np.log(initial_throughput)[:, None]
    last_factors = {k: v[:, 0].copy() for k, v in factor_history.items()}
    observation_index = np.ones(n, dtype=int)
    last_pace = initial_throughput.copy()
    last_reference_pace = initial_throughput.copy()
    # Today's model's training compute: a three-month run at the reference
    # 40% allocation. This constant must not depend on the chosen run duration.
    reference_train_compute = 0.4 * 3 / 12
    # Each draw advances at most one week, stopping exactly at its own run
    # completion/deployment events. This avoids adding a rounding delay every
    # cycle (which could move an entire milestone to the next successor).
    t = np.zeros(n)
    stopped_at = np.full(n, np.inf)
    # Extra log compute from `rsi_compute_boost` after each draw's RSI; the
    # frozen-AI reference never gets it.
    boost = p["rsi_compute_boost"] - 1
    boost_log = np.zeros(n)
    boost_history = np.zeros_like(capability_history)
    now = (np.zeros(n), np.zeros(n), np.zeros(n))
    realized()                  # the deployed model, refreshed when it changes
    while np.any(t < p["years"] - 1e-12):
        train_end = run_start + train_time
        deploy_end = train_end + eval_time
        next_event = np.minimum(np.where(finished, deploy_end, train_end), fast_next)
        next_event = np.minimum(next_event, np.where(coding > t + 1e-12, coding, np.inf))
        next_observation = workflow_grid[np.minimum(observation_index, len(workflow_grid) - 1)]
        next_event = np.minimum(next_event, next_observation)
        end = np.minimum(np.minimum(t + step, next_event), p["years"])
        active = t < p["years"] - 1e-12
        end = np.where(active, end, t)
        dt = np.maximum(end - t, 0)
        base_compute_log = log_level(t + dt / 2, compute_rate, compute_late, slow_at)
        # Share of the boost in force at mid-step: 0 until the delay after RSI
        # has passed, then rising linearly to 1 over the ramp.
        since = t + dt / 2 - events["RSI"] - p["rsi_compute_delay"] / 12
        ramp = p["rsi_compute_ramp"] / 12
        boosting = np.isfinite(events["RSI"]) & (t >= events["RSI"] - 1e-12)
        in_force = np.where(boosting, np.clip(since / ramp, 0, 1) if ramp > 0
                            else (since >= 0).astype(float), 0.0)
        boost_step = np.where(boosting, boost * in_force * np.where(t + dt / 2 < slow_at, compute_rate,
                                                         compute_late) * dt, 0.0)
        compute_log = base_compute_log + boost_log + boost_step / 2
        extra_log = log_level(t + dt / 2, extra_rate, extra_late, slow_at)
        # The deployed model only changes at the end of a step, where `now`
        # is refreshed, so the last step's reading still holds here.
        effective = now[2].copy()
        human_now = humans(effective)
        if p["rsi_autopilot"]:
            human_now[np.isfinite(events["RSI"]) & (t >= events["RSI"] - 1e-12)] = 0.0
        share_now = human_now @ WORKFLOW_WEIGHTS
        # More copies of the AI researcher on the same compute as its
        # algorithms improve: AI's share of the work speeds up with them.
        # In logs, so a runaway's copies never overflow.
        log_speed = np.log(ai_speed) + p["ai_copies"] * np.maximum(now[1], 0)
        with np.errstate(divide="ignore"):
            amdahl_log = amdahl0 - np.logaddexp(np.log(share_now),
                                                np.log1p(-share_now) - log_speed)
        # People still directing research keep today's judgment; AI's applies
        # to the rest, so research quality is the direction-weighted mix.
        direction = human_now[:, 0]
        # The RSI anchor supplies a smooth reduction in coding human hours.
        # Its productivity gain and endogenous coding gains overlap: use the
        # larger, not their product. Existing assistance is normalized to 1.
        coding_fraction = np.clip(np.divide(t + dt / 2, coding,
            out=np.ones(n), where=coding > 0), 0, 1)
        coding_fraction = np.where(coding <= 0, 0, coding_fraction)
        rsi_coding_log = np.log((1 - p["coding_today"] / 100) / .05) * coding_fraction
        coding_gain = np.maximum(rsi_coding_log, p["coding_slope"] * effective)
        if transfer:
            coding_log = np.logaddexp(np.log1p(-transfer) if transfer < 1 else -np.inf,
                                     np.log(transfer) + coding_gain)
            ai_taste_log = np.logaddexp(np.log1p(-transfer) if transfer < 1 else -np.inf,
                                        np.log(transfer) + judgment_gain_log(
                                            effective, taste_slope, drift))
            with np.errstate(divide="ignore"):   # a share of 0 or 1 drops a term
                quality_log = np.logaddexp(np.log(direction),
                                           np.log1p(-direction) + ai_taste_log)
            automation_log = np.logaddexp(np.log1p(-transfer) if transfer < 1 else -np.inf,
                                          np.log(transfer) + amdahl_log)
        else:
            coding_log = np.zeros(n)
            quality_log = np.zeros(n)
            automation_log = np.zeros(n)
        labor_log = parallel * (compute_log + extra_log + np.log(shares[2] / 0.2)
                                + coding_log + automation_log)
        experiment_log = parallel * (compute_log + extra_log + np.log(shares[1] / 0.4))
        research_log = _ces_log(labor_log, experiment_log, rho) + quality_log
        judgment_log = quality_log.copy()
        compute_part = parallel * compute_log
        labor_part = parallel * extra_log
        if experiment_slopes is not None and transfer:
            mid = t + dt / 2
            resource_log = np.log(initial_throughput) + parallel * input_rate * mid
            residual = (experiment_slopes - parallel * compute_rate) * mid
            proxy_log = resource_log + p["rsi_useful_growth"] / 100 * residual
            weight = p["rsi_trend_weight"] / 100 * np.exp(
                -np.log(2) * mid * 12 / p["rsi_trend_months"])
            # Substitute between two estimates of total research effort; do
            # not multiply the trend into feedback already calibrated from it.
            research_log = (1 - weight) * research_log + weight * proxy_log
            judgment_log = (1 - weight) * quality_log
            compute_part = (1 - weight) * compute_part + weight * parallel * compute_rate * mid
            labor_part = (1 - weight) * labor_part + weight * parallel * extra_rate * mid
        # Difficulty: beta per unit of progress, rising toward the ceiling,
        # where returns to research reach 0 (penalty = -beta C ln(1 - S/C)).
        gained = software - initial_stock

        def difficulty_at(progress):
            """(log x, penalty) at `progress` above today's stock."""
            with np.errstate(divide="ignore", invalid="ignore"):
                lx = np.where(capped, np.log1p(-np.minimum(progress / ceiling, 1)), 0.0)
                return lx, np.where(capped, -beta * np.where(capped, ceiling, 0) * lx,
                                    beta * progress)

        log_x, penalty = difficulty_at(gained)
        reference_progress = reference_software(t, software_rate, input_rate,
                                                parallel, initial_throughput, beta,
                                                input_late, slow_at, ceiling)
        _, ref_penalty = difficulty_at(reference_progress)
        reference_pace = np.exp(np.minimum(np.log(initial_throughput)
            + parallel * (base_compute_log + extra_log) - ref_penalty, 700))
        log_amount = np.log(software_rate) + research_log + np.log(
            dt, out=np.full(n, -np.inf), where=dt > 0)
        # Derivative of the same discovery process being integrated below,
        # in multiples of today's software-progress rate. Snapshot near the
        # end of each monthly integration interval (at most one week long).
        pace = np.exp(np.minimum(research_log - penalty, 700))
        if p["difficulty"] > 0:
            plain = initial_stock + np.logaddexp(
                beta * gained, np.log(beta) + log_amount) / beta
            with np.errstate(divide="ignore", invalid="ignore"):
                near = initial_stock - ceiling * np.expm1(
                    ceiling_step(log_x, log_amount, beta * ceiling, ceiling))
            updated = np.where(capped, near, plain)
        else:
            updated = software + np.exp(np.minimum(log_amount, 100))
        # Stop numerically divergent draws explicitly, rather than silently
        # treating a numerical ceiling as a capability plateau.
        diverged = active & (updated > 1e6)
        # A runaway has automated coding whatever its inherited date says, and
        # its direction share is at the floor: RSI is reached unless the floor
        # sits above the bar, where no draw can ever reach it.
        rsi_possible = floor <= p["full_human"] / 100
        runaway_rsi = diverged & ~np.isfinite(events["RSI"]) & rsi_possible
        events["RSI"][runaway_rsi] = end[runaway_rsi]
        software = np.minimum(updated, 1e6)
        train_budget = (grown(t + dt, compute_rate, compute_late, slow_at)
                        - grown(t, compute_rate, compute_late, slow_at)) \
            * np.exp(boost_log + boost_step / 2)
        boost_log = boost_log + boost_step
        used = np.where(finished, 0, shares[0] * train_budget)
        run_compute += used
        training_used += used
        done = active & (~finished) & (end >= train_end - 1e-12)
        candidate_compute = np.log(np.maximum(run_compute / reference_train_compute, 1e-12))
        candidate = (1 - fast_fraction) * snapshot + candidate_compute
        reference_snapshot = initial_stock + reference_software(run_start, software_rate, input_rate,
                                                 parallel, initial_throughput, beta,
                                                 input_late, slow_at, ceiling)
        # A successor ships only if it beats the deployed model as both pay off
        # at their own scale (a short run can carry more algorithms on less
        # compute and still be worse).
        adopted = np.zeros(n, dtype=bool)
        hit = np.flatnonzero(done)
        if hit.size:
            adopted[hit] = candidate_compute[hit] + pays_off(
                hit, run_start_scale[hit], ((1 - fast_fraction) * snapshot)[hit],
                fast_deployed[hit], ((1 - fast_fraction) * reference_snapshot)[hit]) > now[2][hit]
        pending[done] = np.where(adopted[done], candidate[done], deployed[done])
        pending_scale[done] = np.where(adopted[done], run_start_scale[done], deployed_scale[done])
        pending_software[done] = np.where(adopted[done], (1 - fast_fraction) * snapshot[done], deployed_software[done])
        reference_candidate = (1 - fast_fraction) * reference_snapshot + np.log(
            np.maximum(run_compute / reference_train_compute, 1e-12))
        reference_pending_capability[done] = np.maximum(reference_deployed[done], reference_candidate[done])
        reference_pending_software[done] = np.where(
            reference_candidate[done] > reference_deployed[done],
            (1 - fast_fraction) * reference_snapshot[done], reference_delivered[done])
        finished |= done
        ready = active & finished & (end >= deploy_end - 1e-12)
        # Fast gains keep landing on the deployed model during validation, so
        # check again at deployment: a successor that has fallen behind it as
        # both pay off is dropped.
        pending_compute = pending - pending_software
        behind = np.zeros(n, dtype=bool)
        hit = np.flatnonzero(ready)
        if hit.size:
            behind[hit] = pending_compute[hit] + pays_off(
                hit, pending_scale[hit], pending_software[hit], fast_deployed[hit],
                reference_pending_software[hit]) < now[2][hit] - 1e-12
        pending[behind] = deployed[behind]
        pending_software[behind] = deployed_software[behind]
        pending_scale[behind] = deployed_scale[behind]
        deployed[ready] = pending[ready]
        deployed_scale[ready] = pending_scale[ready]
        fast_ready = active & (end >= fast_next - 1e-12)
        fast_deployed[fast_ready] = fast_fraction * fast_snapshot[fast_ready]
        reference_fast[fast_ready] = fast_fraction * (initial_stock[fast_ready] + reference_software(
            np.maximum(0, fast_next[fast_ready] - fast_time), software_rate[fast_ready],
            input_rate[fast_ready], parallel, initial_throughput[fast_ready], beta[fast_ready],
            input_late[fast_ready], slow_at, ceiling[fast_ready]))
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
        changed = np.flatnonzero(ready | fast_ready)
        if changed.size:
            realized(changed)
        compute_now, software_now, effective = (v.copy() for v in now)
        if coding_capability is not None:
            level = effective
            if coding_weights is not None:
                level = coding_weights[0] * compute_now + coding_weights[1] * software_now
            capable = active & (level >= coding_capability) & (coding > end)
            coding[capable] = end[capable]
        rows = np.flatnonzero(ready | fast_ready)
        future = workflow_grid[None, :] >= end[rows, None] - 1e-12
        for history, values in ((capability_history, effective),
                                (realized_history, software_now),
                                (validated_history, validated),
                                (reference_history, reference_validated)):
            history[rows] = np.where(future, values[rows, None], history[rows])
        # A diverged draw's state is a numerical ceiling: record nothing from it.
        live = active & ~diverged
        human = humans(effective)
        coding_hit = active & (end >= coding) & ~coding_recorded
        record("Full coding automation", coding_hit, human, delivered_rate, advantage)
        coding_recorded |= coding_hit
        hit = (live & (end >= coding) & (human[:, 0] <= p["full_human"] / 100)
               & ~np.isfinite(events["RSI"]))
        events["RSI"][hit] = end[hit]
        record("RSI", hit, human, delivered_rate, advantage)
        tracking = events["RSI"] >= end
        last_labor[tracking] = labor_log[tracking]
        last_experiments[tracking] = experiment_log[tracking]
        snapshot[ready] = software[ready]
        run_start[ready] = end[ready]
        run_start_scale[ready] = (log_level(end, compute_rate, compute_late, slow_at)
                                  + boost_log)[ready]
        run_compute[ready] = 0
        finished[ready] = False
        observed = active & (end >= next_observation - 1e-12)
        rows = np.flatnonzero(observed)
        pace_history[rows, observation_index[rows]] = pace[rows]
        seen_now = (log_level(end, compute_rate, compute_late, slow_at) + boost_log)[rows]
        compute_seen[rows, observation_index[rows]] = seen_now
        if seen_flat[0] is not None:     # fill the same cells of the sorted copy
            seen_flat[0][rows * len(workflow_grid) + observation_index[rows]] = (
                seen_now + row_offset[rows])
        stock_seen[rows, observation_index[rows]] = software[rows]
        boost_history[rows, observation_index[rows]] = boost_log[rows]
        resources_log = research_log - judgment_log
        for key, values in (("judgment", judgment_log),
                            ("resources", resources_log),
                            ("difficulty", penalty),
                            ("compute", compute_part), ("labor", labor_part),
                            ("ai", resources_log - compute_part - labor_part)):
            factor_history[key][rows, observation_index[rows]] = values[rows]
            last_factors[key][active] = values[active]
        reference_pace_history[rows, observation_index[rows]] = reference_pace[rows]
        observation_index[rows] += 1
        last_pace[active] = pace[active]
        last_reference_pace[active] = reference_pace[active]
        stopped_at[active & diverged] = end[active & diverged]
        t = np.where(diverged, p["years"], end)
    total_budget = grown(p["years"], compute_rate, compute_late, slow_at)
    stopped = workflow_grid[None, :] > stopped_at[:, None]
    pace_history = np.where(stopped, last_pace[:, None], pace_history)
    reference_pace_history = np.where(stopped, last_reference_pace[:, None], reference_pace_history)
    for key in factor_history:
        factor_history[key] = np.where(stopped, last_factors[key][:, None], factor_history[key])
    # Boost observed through each draw's last month, held after it stops.
    held = np.where(stopped, boost_log[:, None], boost_history)
    compute_history = log_level(workflow_grid[None, :], compute_rate[:, None],
                                compute_late[:, None], slow_at) + held
    # Retain all draws in each percentile, including slow/non-arriving paths.
    def humans_at(j):
        hours = humans(capability_history[:, j])
        if p["rsi_autopilot"]:
            hours[events["RSI"] <= workflow_grid[j] + 1e-12] = 0.0
        return hours

    human_quantiles = [np.quantile(humans_at(j), [.1, .5, .9], axis=0)
                       for j in range(len(workflow_grid))]
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
        "params": p, "version": MODEL_VERSION, "onset": onset, "coding_at": coding,
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
            "capability_log": capability_history,
            "rate_quantiles": finite_quantiles(rate_history),
            "cumulative_quantiles": np.quantile(
                validated_history / software_rate[:, None] * 12, [.1, .5, .9], axis=0).T,
            "matched_quantiles": finite_quantiles(matched_history),
            "matched_coverage": np.mean(np.isfinite(matched_history), axis=0),
            "validated_log": validated_history,
            # The algorithmic part of capability_log, as it pays off at scale.
            "software_realized_log": realized_history,
            "validated_baseline_months": validated_history / software_rate[:, None] * 12,
            "pace": pace_history,
            # Log compute relative to today, post-RSI boost included.
            "compute_log": compute_history,
            "pace_factors": factor_history,
            "reference_pace": reference_pace_history,
            "reference_log": reference_history,
            "rate": rate_history,
            "matched": matched_history,
        },
        "progress_multiple": delivered_rate,
        "compute_matched_advantage": advantage,
        "draws": {"taste_slope": taste_slope, "difficulty": beta,
                  "judgment_drift": drift, "substitution": rho,
                  "ceiling_oom": ceiling / np.log(10), "scale_dependent": scale_share,
                  "scale_lag_oom": scale_lag / np.log(10)},
        # Each draw's log compute growth a year before and after `slow_at`;
        # its compute is exp(log_level(t, rate, late, slow_at)) x today's.
        "compute": {"rate": compute_rate, "late": compute_late, "slow_at": slow_at},
        "stopped_at": stopped_at,
    }


def progress_years(result, base_rate, window=1.0, level=None):
    """Per draw, the years of progress at `base_rate` (log capability a year,
    or `level`'s units a year) its deployed capability makes in the `window`
    years after RSI: Forethought's "years of AI progress compressed into one
    year". `level` (draws x months) replaces the capability path, as when it
    is read as ECI. NaN where RSI never comes or the window runs past the
    horizon; +inf where the draw diverges inside it.
    """
    wp = result["workflow_progress"]
    grid = wp["years"]
    cap = wp["capability_log"] if level is None else level
    start = result["events"]["RSI"]
    end = start + window
    out = np.full(len(start), np.nan)
    for i in np.flatnonzero(np.isfinite(start) & (end <= grid[-1] + 1e-9)):
        if result["stopped_at"][i] <= end[i]:
            out[i] = np.inf
        else:
            out[i] = (np.interp(end[i], grid, cap[i])
                      - np.interp(start[i], grid, cap[i])) / base_rate
    return out


def cdf(samples, grid):
    """CDF over all draws, including +inf (not reached) in the denominator."""
    return np.searchsorted(np.sort(samples), grid, side="right") / len(samples)


def quantile(samples, q, horizon=np.inf):
    """Empirical inverse CDF, preserving censored probability mass."""
    value = np.sort(samples)[max(0, int(np.ceil(q * len(samples))) - 1)]
    return float(value) if value <= horizon else np.inf
