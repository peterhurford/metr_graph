# Takeoff parameter guide — September 8, 2026

These are the current Central defaults, not necessarily a browser session's settings.
Most exact values are illustrative assumptions introduced during development. The
rationales below explain their intended role, not a claim that evidence identifies them.
See `checkpoints/takeoff-defaults-2026-09-08.json` for exact machine-readable settings as of
`takeoff-v4`; v5 adds `ladder_weight` = 50 and varies `difficulty`.

**Varied** means the global parameter-uncertainty control samples that quantity.
**Fixed** means it stays at the chosen value throughout a simulation ensemble.

## Starting conditions and delivery

| Parameter | Default | Meaning and rationale | Uncertainty |
|---|---:|---|---|
| Coding human hours automated today | 70% | Substantial assistance with meaningful human work remaining. Placeholder, not inferred from AI-authored code shares. | Fixed |
| Undeployed research today | 2 baseline months | Backlog awaiting validation/incorporation; avoids an empty initial pipeline. Amount not measured. | Months fixed; stock depends on sampled progress rate |
| Fast feedback share | 25% | Minority of efficiency gains usable through tools, kernels, and small updates; 75% waits for successors. Assumed split. | Fixed |
| Fast validation cycle | 1 month | Shorter than successor training; exact delay assumed. | Fixed |
| Baseline software doubling time | 9 months | Current algorithmic efficiency improvement, including existing AI assistance; not fitted to validated discoveries. | Underlying progress rate varied |
| Successor training duration | 3 months | Illustrative months-long training run, not a measured average. | Varied |
| Validation/deployment delay | 1 month | Additional checking and incorporation time; assumed. | Varied |

## Human involvement and project success

Shares refer to human hours on comparable baseline projects, not employees replaced.
Halving times measure months of capability progress at today's initial pace; faster
capability progress compresses calendar durations. Halving applies above the chosen floor.

| Parameter | Default | Meaning and rationale | Uncertainty |
|---|---:|---|---|
| Direction: human work today | 40% | Assumes substantial remaining human direction; no matched-project estimate. | Varied |
| Experiments: human work today | 30% | Assumes more automation of design/interpretation than the other stages; unmeasured. | Varied |
| Verification: human work today | 50% | Makes verification/integration the largest initial human bottleneck; conservative hypothesis. | Varied |
| Direction: human-work halving time | 6 months | Assumed learning speed; not fitted. | Varied |
| Experiments: human-work halving time | 9 months | Assumes slower elimination of residual work. | Varied |
| Verification: human-work halving time | 12 months | Assumes verification remains difficult longer. This materially delays full automation. | Varied |
| Persistent human-work floor | 1% | Leaves residual oversight; not evidence of an irreducible requirement. | Fixed |
| Useful-project completion today | 60% | Placeholder whole-project reliability, not coding-task accuracy. | Varied |
| Failure-rate halving time | 12 months | Assumed pace of declining remaining project failures. | Varied |

The 40/30/50 shares and 6/9/12 ordering are bottleneck hypotheses without strong
empirical justification. Matched-stage measurements can override them.

## Feedback and resources

| Parameter | Default | Meaning and rationale | Uncertainty |
|---|---:|---|---|
| Research-quality elasticity prior | 0.8 | Sublinear capability-to-quality response before the transfer discount; exact value assumed. Blended with calibration. | Resulting blended elasticity varied |
| Coding elasticity | 0.8 | Sublinear capability-to-coding response; not fitted. | Fixed |
| Additional AI research output that transfers | 60% | Discounts gains not applicable to useful research; exact discount arbitrary. | Fixed |
| Increasing research difficulty | 0.5 | Additional discoveries become harder as efficiency accumulates; positive value is a modeling hypothesis, not a calibrated coefficient. This is β in semi-endogenous growth models; feedback compounds only while the blended quality elasticity exceeds it. | Varied |
| Parallelization exponent | 0.5 | Square-root returns to coding labor and experiment compute; a simple coordination/parallelization assumption. | Fixed |
| Compute growth per year | 2× | Scenario of annual doubling; not inherited from the dashboard's compute projections. | Log growth rate varied |
| Compute for training | 40% | Illustrative balanced resource allocation. | Fixed |
| Compute for experiments | 40% | Illustrative allocation. | Fixed |
| Compute for agents | 20% | Remainder after training and experiments, not independent. | Fixed, derived |

Difficulty multiplies research productivity by exp(-d * new log efficiency). At d=0.5,
an additional algorithmic efficiency doubling increases the effort required for comparable
further log progress by about 1.41×, holding inputs constant. This can outweigh resource
growth in the frozen-AI reference. Sublinear elasticity, transfer discounts, square-root
resource returns, and difficulty stack together; their combined conservatism is not fitted.
Difficulty is the most sensitive parameter in the model: holding the rest at Central, broad
superintelligence by end-2031 runs from 96% at 0.35 to 35% at 0.71 and 3% at 1.0. Varying
it puts roughly 9% of scenarios below the compounding threshold.

## Milestone definitions

These mostly define what gets called a milestone rather than estimate its arrival speed.

| Parameter | Default | Meaning and rationale | Uncertainty |
|---|---:|---|---|
| Weight on Anthropic's AL5 date | 50% | Share of scenarios whose full R&D automation is the RSI tab's AL5 date rather than the human-hours gate. The two disagree by over a year at the median; equal credence is a judgement. | Fixed; each scenario draws its definition |
| Maximum human work for full R&D | 5% | Every stage must require very little baseline human labor; chosen to mean near-complete automation. | Fixed |
| Required useful-project completion | 90% | Avoids declaring unreliable low-human-work systems fully automated; exact threshold is definitional. | Fixed |
| Required progress acceleration | 12× today | A year of baseline algorithmic progress per month; legible, demanding definition, not a natural boundary. | Fixed |
| Required same-compute advantage | 2× | Demands substantial improvement beyond resource growth; definitional. | Fixed |
| Consecutive qualifying successor cycles | 2 | Minimal persistence beyond one successful cycle; not statistically calibrated. | Fixed |
| Consecutive cycles for sustained feedback | 2 | Internal repeated-feedback diagnostic; not a gate for full R&D and omitted from graphs. | Fixed |
| AI research to broad expertise gap | 3 capability doublings | Highly speculative cross-domain bridge with no measured conversion. | Varied |
| Additional broad superiority gap | 2 doublings | Further speculative superiority beyond broad expertise. | Varied |

## Evidence and reporting controls

| Parameter | Default | Meaning and rationale | Uncertainty |
|---|---:|---|---|
| Weight on inherited RSI activity trend | 50% | Initial compromise between extrapolated activity and mechanistic effort; not an optimized fit. | Fixed |
| RSI trend weight half-life | 12 months | Gradually hands over to mechanistic dynamics; one year is assumed. | Fixed |
| Non-compute activity growth transferring to useful effort | 100% | All compute-adjusted log activity growth transfers. Optimistic mapping, not measured useful output. | Fixed |
| Weight on empirical elasticity estimate | 50% | Equal blend with quality prior; not Bayesian estimation. | Fixed |
| Compute growth over evidence window | 2× | Assumed change over the historical observation window; not the annual forecast parameter and not measured from the report. | Fixed |
| Capability growth over evidence window | 2× | Assumed effective-capability change needed to infer elasticity; not fitted to a capability series. | Fixed |
| Useful yield per experiment: end/start | 1× | Neutral placeholder of unchanged experiment value. | Fixed |
| Measured validated progress: end/start | 0 = unavailable | A positive matched-period measurement replaces the experiment proxy. Zero is missing data, not no progress. | Fixed |
| Useful yield per experiment today/2025 | 1× | Reporting conversion from RSI activity to a research-pace anchor. Changes chart units, not simulated dynamics. | Fixed |
| Matched-project baseline human hours | 0 = unavailable | Baseline for fitting a selected stage; no invented measurement. | Fixed input |
| Matched-project earlier human hours | 0 = unavailable | Earlier hours including oversight, corrections, and rescue. | Fixed input |
| Matched-project current human hours | 0 = unavailable | Current comparable hours. Fitting updates that stage's starting share and halving time. | Fixed input; resulting stage parameters varied |

Historical calibration uses the observed experiment series' endpoint ratio, not a causal
regression. Its inferred elasticity depends on assumed compute, capability, and useful yield.
The separate forward blend treats the proxy and mechanistic output as alternatives, not
multiplicative gains. RSI and calibration partly share evidence, not independent confirmations.

## Exactly how uncertainty is implemented

Global uncertainty defaults to **40**, an illustrative broad spread, not calibrated coverage.
For each scenario, 16 independent U values are drawn uniformly on [-1, 1]. Multipliers are
F = exp(0.40 * U). Full bounds are about 0.67–1.49×; central-80% bounds are 0.73–1.38×;
median multiplier is 1. Each draw stays fixed through the scenario, not month-to-month noise.

The 16 varied quantities are: software-progress rate; log compute-growth rate; training time;
validation delay; initial useful-project success; calibrated quality elasticity; the two
breadth gaps; three human-work halving times; failure halving time; three initial human
shares; and research difficulty. All other controls are fixed within the ensemble, though users can vary them manually.

The doubling-time input is inverted into a rate before variation: nine months produces
roughly 6.0–13.4-month doubling times. Compute growth is 2^F, about 1.59–2.81× annually,
not 2*F. Human shares are capped at 100% and floored at the persistent floor; success is
capped at 100%. These clips distort the otherwise symmetric log-space distribution.

Inherited uncertainty remains separate:

- Coding dates use exact final RSI draws, including component uncertainty, mixture weights,
  conditioning, subjective penalty, and clock. The Takeoff slider does not remove this.
- The shared RSI experiment fan uses a lognormal doubling time with illustrative central-80%
  bounds at half/twice the fitted value, plus lognormal position uncertainty approximately
  divided/multiplied by 1.3 over the central 80%. Its fixed local seed is 20260907.
- Setting Takeoff uncertainty to zero removes the 16 parameter variations, not RSI uncertainty
  or the per-scenario choice of full-R&D definition.
- Takeoff's seed is 20260906 for reproducibility only. Default production sample count is 5,000.

The prior draws are independent, but outputs become correlated through shared capabilities
and dynamics. There is no full joint posterior, modeled covariance of uncertain assumptions,
or calibrated tail coverage. The displayed bands reflect these scenario assumptions.

## Other defaults and conventions

- End year 2031 was requested by the user, not forecast as a transition date.
- The coding anchor represents 95% of baseline coding human hours automated.
- Human-hour reporting weights are 25/50/25; they do not affect the per-stage automation gate.
- Sustained feedback requires at least a 10% output improvement per qualifying cycle.
- Integration steps are at most weekly, with monthly observations; these are numerical choices.
- Standalone fallback coding dates are uniform over 6–24 months with a 10-year horizon.
  The dashboard overrides those with RSI draws and the selected calendar end year.
- Fast and Bottlenecked presets are alternative assumptions without assigned probabilities;
  their exact overrides are saved in the defaults JSON.
