# Research feedback from today through takeoff

Version: `takeoff-v4-present-feedback`. Work remains on `codex/grounded-takeoff`;
checkpoint `6e22ad2` preserves the earlier explorer. These are scenario forecasts with
explicit proxy calibration, not statistically identified probabilities.

## Today, coding automation, and existing assistance

Every simulation starts today and ends at the selected calendar horizon (default
December 31, 2031). The exact final RSI samples remain the coding-automation dates,
including weights, conditioning, penalty, clock, past dates, and the late tail.
They no longer shift an otherwise identical post-coding simulation forward in time.

Research is partially automated today. Initial human-hour shares (40/30/50% by stage),
useful-project completion (60%), coding automation (70%), and the current software
progress rate (one doubling per nine months) are assumptions. The current progress rate
already includes existing AI assistance. An assumed two months of undeployed discoveries
initializes both the actual and same-compute reference pipelines. Those discoveries must
still pass validation; the model does not invent a pre-today trajectory.

For future RSI dates, remaining coding hours decline smoothly from the assumed current
share to 5% at the coding milestone. This supplies an incremental coding-productivity
channel. Endogenous coding gains and this ramp overlap, so the model uses their maximum,
not their product. Past coding dates are already satisfied, with existing assistance
normalized into today's baseline. Never-arriving coding dates still have research paths.
Full R&D automation requires the coding date to have arrived; this anchoring constraint
is not a joint statistical fit of coding and research capabilities.

## Fast and slow feedback

Research combines coding labor and experiment compute through a harmonic mean, multiplies
by research quality, and discounts increasingly difficult discoveries. Agents, experiments,
and training have disjoint shares of one growing compute budget.

Discoveries are measured in log fixed-quality algorithmic efficiency. By default, 25%
can be delivered through fast improvements (tooling, kernels, and small updates) and 75%
through successor training. The split is an assumption about deployable efficiency gains,
not a claim that all tooling improvements translate into compute savings. It sums to 100%,
and an improvement never appears in both channels.

The fast path validates its start-of-cycle discoveries over an assumed one month. The
slow path freezes discoveries at training start, accumulates physical training compute,
and deploys only after training plus validation (three plus one month by default).
The next cycle picks up later discoveries. Fast improvements enhance the available research
system without requiring a new frontier model. Breadth still depends on successor capability.
Training capacity remains reserved during validation; fast work uses the experiment/agent
budgets rather than creating additional resources.

## Acceleration and autonomy are different outcomes

All four acceleration panels use 2025 units. The first preserves the RSI experiment
activity fan. The second shows ongoing useful research production before validation,
including work still in training/validation queues. The fourth shows the same-compute
reference's own production pace, not the actual/reference ratio. These pace series use
the derivative of the simulated discovery process, with the same increasing-difficulty
penalty, sampled near monthly boundaries within the weekly integration resolution.
They start from today's already-assisted rate and can fall if difficulty outpaces gains.
No minimum of 1× is imposed.

The reporting anchor is each draw's RSI activity multiple today times an assumed useful
yield per experiment relative to 2025 (default 1). This is an explicit proxy conversion,
not measured discovery growth. It rescales each draw before computing percentiles and
does not multiply existing assistance into dynamics a second time. Under default resource
allocation and yield assumptions, both modeled pace series start at the inherited activity
multiple, rather than appearing near zero while waiting for the first new successor.

The third panel counts additional validated deliveries after today, measured in months
of research at the 2025 pace. It starts at zero because it measures an additional amount,
not a pace multiplier. Gains still wait for validation; none are backfilled into the
cumulative series. Table crossings refer to 2×/5× 2025 production pace. The internal
three-month delivered-rate and actual/reference diagnostics remain available for model
analysis and validated milestone definitions; they are not the chart's ongoing pace.

Full R&D automation still requires every research stage to need at most 5% of baseline
human hours and useful-project completion to reach at least 90%, after the coding milestone.
The stages are direction, experiment design/interpretation, and verification/integration.
The 1% floor and stage halving times remain adjustable. Halving times use months of capability
progress at today's initial growth rate, not calendar months; accelerated capability progress
compresses calendar durations. The available research system includes validated fast gains.

Superhuman AI research requires full automation plus 12× today's progress and 2× the
same-compute reference for two consecutive successor cycles. Cycle progress counts changes
in total delivered fast-plus-slow efficiency over the entire cycle. Broad superintelligence
retains the speculative bridge of three plus two capability doublings in successors.

## Evidence calibration

`openai_experiment_velocity.csv` contains 32 source-tooltip observations, already used by
the RSI tab. OpenAI experiments per active researcher rise from 0.72× on January 5, 2026 to
1.60× on August 10: 2.22× growth between these endpoints. Values are four-week averages and
use 2025 as the normalization baseline. The calibration uses that endpoint ratio; it is
not a causal regression, and overlapping observations do not become independent evidence.

The sidebar exposes assumed compute growth over the same period (2×), effective capability
growth (2×), and useful yield per experiment (unchanged). It removes compute's modeled
contribution, then coding's modeled contribution, before solving for the remaining quality
elasticity. Defaults blend that estimate 50/50 with the chosen elasticity prior. If the
residual would require declining quality, the fit stops at zero and reports the boundary.
Zero AI transfer disables this calibration. Historical gains estimate a forward elasticity;
they are not multiplied into today's starting productivity a second time.

A supplied ratio of independently validated fixed-quality efficiency gains per month replaces
the experiment proxy for historical calibration. The separate forward activity channel
reuses the RSI experiment fan's exact deterministic draws, shifted to today. Its ~2× level
is relative to 2025 experiment activity; today's validated-progress baseline remains 1×.
Only future growth affects dynamics, never a second multiplier for existing assistance.
The near-term projection and mechanistic research-effort estimates are blended in log space
(50% initial weight, halving every 12 months by default). Compute growth is subtracted
before applying the adjustable useful-effort conversion and included once in the result.
This is a proxy-to-effort assumption, not measured discovery growth. Zero trend weight
restores the mechanistic model; zero AI transfer also disables this channel. The source
activity fan is displayed unchanged regardless of the chosen conversion or blending weight.

A supplied ratio of independently validated fixed-quality efficiency gains per month replaces
the experiment proxy. Zero means no such evidence was supplied. The baseline software
doubling-time control can separately be set from measured current validated progress.

OpenAI reports interventions in more than half of successful 4–8 hour tasks. That observation
does not identify human hours per complete project. Users can instead enter matched-project
baseline, earlier, and current human hours, including interventions, corrections, and rescue.
Fit buttons set a selected stage's current human share and halving time using the assumed
capability growth over the observation period. Empty inputs default to zero (unavailable),
and inconsistent or unsupported fits leave existing assumptions intact. Project scope,
quality standards, and capability comparison must be matched by the user.

Anthropic's 8× code-per-engineer observation is contextual evidence, not an 8× research
multiplier. Whole-project success and independently validated gain series are not supplied
by these reports. Those parameters remain assumptions until measurements are provided.
RSI and calibration partly share evidence and are not independent confirmations.

Sources: [OpenAI](https://openai.com/index/research-acceleration-view-inside-openai/),
[Anthropic](https://www.anthropic.com/institute/recursive-self-improvement).

## Reporting and limitations

All event times are years from today. Calendar CDFs use them directly; coding-relative
plots subtract each draw's coding date and retain unreached draws in the denominator.
Their follow-up ends at the calendar horizon, so late coding dates have less follow-up.
Workflow plots show monthly medians and middle-80% intervals across scenarios. Marginal
medians meeting every target do not establish joint automation. Completed paths retain
last modeled workflow values and last measured rates. Numerically divergent assumptions
produce an explicit error rather than a fictitious probability of stalling; very strong
feedback with a late imposed coding milestone can reach this limit.

Exports include the version, time origin, effective parameters, calibration assumptions,
evidence dates, inherited RSI samples, and calendar/coding-relative event times. The source
fingerprint reloads changed model code and invalidates Streamlit's result cache.

Human-hour curves are operational criteria, not a staffing or queueing model. Quality and
capability remain aggregate quantities. Project success is not an experiment-level failure
simulation. Hardware innovation, manufacturing, policy, and a measured cross-domain bridge
to superintelligence remain outside scope. Proxy calibration does not establish a causal
feedback coefficient or calibrated confidence intervals.
