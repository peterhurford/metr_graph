# Research feedback from today through takeoff

Version: `takeoff-v7-amdahl-ces`. Work remains on `codex/grounded-takeoff`;
checkpoint `6e22ad2` preserves the earlier explorer. These are scenario forecasts with
explicit proxy calibration, not statistically identified probabilities.

## Today, coding automation, and existing assistance

Every simulation starts today and ends at the selected calendar horizon (default
December 31, 2031). The exact final RSI samples remain the coding-automation dates,
including weights, conditioning, penalty, clock, past dates, and the late tail — with
the AL5 card's weight set to zero, since full autonomy is not a coding milestone.
They no longer shift an otherwise identical post-coding simulation forward in time.

Research is partially automated today. Initial human-hour shares (calibrated below),
coding automation (70%), and the current software
progress rate (one doubling per nine months) are assumptions. The current progress rate
already includes existing AI assistance. One full cycle (training plus validation, four
months) of undeployed discoveries initializes both the actual and same-compute reference
pipelines, the steady state, so the next model ships at today's pace. Those discoveries must
still pass validation; the model does not invent a pre-today trajectory.

For future RSI dates, remaining coding hours decline smoothly from the assumed current
share to 5% at the coding milestone. This supplies an incremental coding-productivity
channel. Endogenous coding gains and this ramp overlap, so the model uses their maximum,
not their product. Past coding dates are already satisfied, with existing assistance
normalized into today's baseline. Never-arriving coding dates still have research paths.
RSI requires the coding date to have arrived; this anchoring constraint is not a joint
statistical fit of coding and research capabilities.

## Fast and slow feedback

Research combines labor and experiment compute through a CES mean (`substitution`, default
-0.5; -1 is the old harmonic mean, under which unlimited labor at most doubles output),
multiplies by research quality, and discounts increasingly difficult discoveries. Agents,
experiments, and training have disjoint shares of one growing compute budget.

Labor speeds up as the human share h of research hours falls, by Amdahl's law:
(h0 + (1 - h0)/s) / (h + (1 - h)/s), with s = `ai_speed` (default 30x) and h0 today's
share, through the same transfer discount as coding and judgment. RSI is where h is
lowest, so it is where this term is largest.

Difficulty is set from the returns to research r (the tab's control, default 1.0, the
revised Forethought central estimate): difficulty = parallelization / r, the
semi-endogenous relation in which each doubling of cumulative effort buys r doublings of
efficiency. Non-AI inputs (researchers, spending) grow at `baseline_inputs`% of the rate
that holds today's pace with AI frozen, as the past trend has; at 100% the same-compute
reference stays at today's pace instead of decaying.

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

The reporting anchor is a set research pace today relative to 2025 (`tk_pace_today`,
default 1.7×, an assumption). Each draw keeps its RSI activity
multiple's spread around that median. It is a stated normalization, not measured discovery
growth. It rescales each draw before computing percentiles and does not multiply existing
assistance into dynamics a second time, so it moves the 2×/5× crossings but no milestone.
Both modeled pace series start at that level, rather than appearing near zero while
waiting for the first new successor.

The third panel counts additional validated deliveries after today, measured in months
of research at the 2025 pace. It starts at zero because it measures an additional amount,
not a pace multiplier. Gains still wait for validation; none are backfilled into the
cumulative series. Table crossings refer to 2×/5× 2025 production pace. The internal
three-month delivered-rate and actual/reference diagnostics remain available for model
analysis and validated milestone definitions; they are not the chart's ongoing pace.

**RSI** is the first time every research stage needs at most 5% of its baseline human
hours, with the coding date arrived. Human hours are a readout, not an input: they never
feed back into research speed. Every draw runs to the horizon (only a numerically
divergent one stops), so its capability path continues past RSI.
The stages are direction, experiment design/interpretation, and verification/integration.
The 1% floor and stage halving times remain adjustable. Halving times use months of capability
progress at today's initial growth rate, not calendar months; accelerated capability progress
compresses calendar durations. The available research system includes validated fast gains.

**Capability as ECI.** Each draw's deployed effective compute (training compute × algorithmic
efficiency, relative to today's model) is read as ECI: today's US frontier score plus a
fixed number of points per unit of log effective compute, set so today's modeled growth
equals the ECI frontier's fitted points per year. The measured exchange rate (about 6.4
points per 10× training compute) would give a third of that slope, so the choice roughly
triples every post-RSI gain read as ECI; both readings belong next to any bar picked off it.
Superintelligence is to be defined as an ECI level on this chart.

## Evidence calibration

Four priors come from the RSI tab. Today's human hours per stage are scaled so their
25/50/25 average matches the R&D automation index's latest month, with each AL level
assumed to take 100/90/70/40/15/0% of unassisted human time (AL0–AL5); stage halving times
are scaled to that share's past-year halving. The research-quality prior is the next-step
study's win-odds growth (a Bradley-Terry strength ratio) and the coding elasticity the staff
survey's output growth, each divided through the same transfer form by assumed
effective-capability growth.

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
last modeled workflow values and last measured rates. A draw whose research runs past
numerical range records nothing further. It counts as unresolved only while a milestone
could still arrive inside the horizon — not when every open milestone waits on a coding
date past it — and the tab reports up to 1% unresolved draws in its caption as not
arriving, refusing to report above that.

Exports include the version, time origin, effective parameters, calibration assumptions,
evidence dates, inherited RSI samples, and calendar/coding-relative event times. The source
fingerprint reloads changed model code and invalidates Streamlit's result cache.

Human-hour curves are operational criteria, not a staffing or queueing model. Quality and
capability remain aggregate quantities. Project success is not an experiment-level failure
simulation. Hardware innovation, manufacturing and policy remain outside scope. Proxy calibration does not establish a causal
feedback coefficient or calibrated confidence intervals.
