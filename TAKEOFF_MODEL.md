# Research feedback from today through takeoff

Version: `takeoff-v31-direction-waits-for-coding`. Checkpoint `6e22ad2` preserves the earlier explorer. These are scenario forecasts with
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
RSI requires the coding date to have arrived, or a deployed model at ECI 250: a model at
ultraintelligence has automated coding whatever the inherited date says. This anchoring
constraint is not a joint statistical fit of coding and research capabilities.

The RSI tab's coding dates and this model's research-direction curves read the same
underlying progress, so a late coding date should go with a scenario slow to hand direction
over. The app runs the model once, ranks scenarios by the month direction would first be
under the RSI bar with coding no gate (`direction_ready`; never-ready ones by capability
four years out), reassigns the coding dates along that ranking through a Gaussian copula
(correlation 0.9 by default, a judgment), and runs again. Only the pairing changes; the dates' distribution is the RSI tab's exactly.

## Fast and slow feedback

Research combines labor and experiment compute through a CES mean (`substitution`, default
-0.2, elasticity 0.83 as in Erdil et al.'s chess estimate and Davidson's illustration, under
which unlimited labor at most multiplies output about 30x; -0.5 caps it at 4x, -1 at 2x),
multiplies by research quality, and discounts increasingly difficult discoveries. Agents,
experiments, and training have disjoint shares of one growing compute budget.

Better algorithms also run more AI researchers on the same compute: AI's speed on the
work it has taken over is multiplied by the deployed model's algorithmic efficiency (vs
today's) to the power `ai_copies` (default 1). Since only AI's share speeds up, this adds
about 4% a year to research labor today, while people do about a third of the work, and
counts in full once autopilot takes over at RSI. The staff survey's coding fit is netted of
that 4%, and the experiment calibration credits copies before research judgment. With it,
ultraintelligence comes 2 months sooner (May against Jul 2030) and 24% of scenarios make 3
years of progress in RSI's first year (17% without).

Labor speeds up as the human share h of research hours falls, by Amdahl's law:
(h0 + (1 - h0)/s) / (h + (1 - h)/s), with s = `ai_speed` (default 30x) and h0 today's
share, through the same transfer discount as coding and judgment. RSI is where h is
lowest, so it is where this term is largest.

Compute growth comes from the Data Centers tab by default: each draw takes one sampled US
largest-training-run trajectory (recorded steps, catalogued plans under slip, then the
fitted trend) averaged over a fixed five years, whatever the chart horizon. Off, it is the
flat `compute_growth`. From the end of 2030, where Epoch's analysis finds today's scaling
feasible, compute keeps `compute_slowdown`% (default 50) of its log growth, about 3× a year
becoming 1.7×. That is an assumption: no published estimate gives the post-2030 rate. From each scenario's own RSI
date, `rsi_compute_boost` (default 2) multiplies its log compute growth, standing in for AI
speeding chip design, fabs and spending: Forethought's chip-technology and chip-production
feedback loops. It waits 18 months (a chip generation, a fab's build) and ramps in over a
year; the frozen-AI reference never gets it. The size is an assumption. With the delay the
ultraintelligence median moves a month at most; at once, 2x brings it 6 months sooner,
since scale-dependent gains then wait less. The
other research inputs keep their own growth, so research inputs slow by compute's share.

Research judgment is gated by who directs research: people still in the direction stage
keep today's judgment, and AI judgment (rising with capability, uncapped) applies to the
rest, so quality = h_dir + (1 - h_dir) x AI judgment. Ungated, judgment compounding with
capability drove takeoff whether or not people had left; gated, it takes over as
research direction is handed off, which is RSI.

Two settings tie that handoff to judgment. `judgment_today` (default 1) is AI judgment against
people's today, L, so quality = (h_dir + (1 - h_dir) L J) / (h_dir0 + (1 - h_dir0) L), with J
AI judgment over today's AI's: today's quality stays 1 and the calibrated pace is untouched,
and L only changes what each share handed over buys. The default counts AI and people alike;
benchmark readings that AI already out-judges experts (TasteVal's 2.3x compute multiplier)
are not taken as a level. `direction_adoption` (k, default 0.5) moves research direction
along its human-hours curve J^k times as fast, so labs delegate sooner as AI judgment grows:
at 4x today's judgment the stage halves twice as fast. J is a function of capability, so
RSI is still a capability threshold, only a lower one. Nothing measures k. At 0.5 it brings
RSI about 3 months sooner and ultraintelligence about half a month: handing direction over
buys research only as fast as AI judgment grows, so the RSI date is mostly a marker of the
takeoff, not what paces it. L of 0.75 or 1.5 moves ultraintelligence about a month either
way.

Difficulty is set from the returns to research r (the tab's control, default 1.0, the
revised Forethought central estimate): difficulty = 0.5 / r, the semi-endogenous relation
in which each doubling of cumulative effort buys r doublings of efficiency at the default
input exponent of 0.5. It is held there so the parallelization setting changes only how
much extra input buys, not how fast ideas get harder. Today's pace has held steady while
compute grew at its historical rate (the data-center series' own fit, 3.1x/yr). Two other
inputs are explicit. Human research staff grow 30% a year until a scenario's RSI (and
change -30% a year after); more staff finish the share of work people still do sooner, so
they matter until autopilot and not after. Experiment compute grows at a fixed share of the
frontier's log growth, fitted so labor and experiments together hold today's pace steady
with AI frozen: about 0.4 of it (1.6x a year against 3.1x), slowing in step after 2030.
That share is the one unmeasured number this calibration needs; it reads the gap between
fast frontier growth and a steady trend as experiment compute lagging frontier runs, not
as weaker returns to research, so r stays at Forethought's 1.0.

Discoveries are measured in log fixed-quality algorithmic efficiency. By default, 25%
can be delivered through fast improvements (tooling, kernels, and small updates) and 75%
through successor training. The split is an assumption about deployable efficiency gains,
not a claim that all tooling improvements translate into compute savings. It sums to 100%,
and an improvement never appears in both channels.

Cycles speed up with AI (default on): a model's validation shrinks with the share of
verification AI does (to about a day on autopilot), and each new training run shortens when
capability grows faster than today, by today's growth rate over the current one, since a
run should not outlast the time better methods take to beat it. Shorter runs train on less
compute, which the adoption rule weighs. It moves ultraintelligence 3 months sooner (Nov
against Feb 2030) and doubles the share making 10 years of progress in RSI's first year.
Labs re-split the compute not used for training between running AI researchers and
experiments as AI changes what each is worth, following the labor-compute mean's
first-order condition from today's split (on by default). With AI labor abundant and the
two complements, the agents' share falls from 20% to about 5% by 2030 at the median. It
moves little (the share making 10 years of progress in RSI's first year by about a point),
because experiments can gain at most half again on their 40% share while training keeps its
own. A scenario's software ceiling is also tied to its returns to research (correlation
0.5): a nearer ceiling comes with lower r. That leaves the median alone and makes the
high-r tail more explosive: among r > 2 scenarios where labor substitutes for compute, 81%
make 10+ years of progress in RSI's first year, against 68% unlinked.
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
by default OpenAI's experiments per experimenter, its fitted trend carried from the last
reading to today, about 2×; experiments are a proxy for research). Each draw keeps its RSI activity
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

**RSI** is the first time research direction (the stage that chooses what to try next)
needs at most 5% of its baseline human hours, with the coding date arrived. AI cannot direct
research before it can code, so until a draw's coding date its direction hours hold at that
bar (`test_research_direction_waits_for_coding`). Human hours
feed back into research through the Amdahl labor term and through who directs research.
From RSI on, autopilot (the default) drops every stage's remaining human hours: at RSI
experiment design still needs about 5% of its hours and verification about 11%, and from
then on none do. That lifts research pace about 15% just after RSI (2.35x today's against
2.05x), a gain difficulty erases within a year; ultraintelligence does not move. Every draw
runs to the horizon, so its capability path continues past RSI.
The stages are direction, experiment design/interpretation, and verification/integration.
Human hours follow the R&D automation index's own projection rather than an exponential:
the share of tasks at each level over the coming years (AL4 at the milestone card's median
rate, AL5 as AL4 shifted by a gap), times the human time each level takes in each stage,
gives each stage its own curve. Direction keeps most of its hours at AL4 (AI works from a
person's high-level prompt and the person decides what ships), so its curve holds near
its start until AL5 arrives; experiments, handed over at AL4, fall sooner. AL5 has never
been measured, so each scenario draws its own AL4-to-AL5 gap over the AL5 card's 80%
range. Research direction, the stage RSI is defined on, also reads an independent source,
the next-step study: direction needs people in proportion to the odds a researcher still
picks the better step, above the odds left at the study's own ceiling. Its dates and the
ladder's are averaged, half each. Alone, each puts direction's 5% bar a median ~1.6 years
out (ladder 1.57, next-step 1.71), so the weight moves the tails, not the median: at 0,
0.5 and 1 the median RSI stays May 2028 while its 90th percentile is Oct 2029, Nov 2029
and Feb 2030. Halving or doubling the AL5 gap moves the median RSI
from May 2028 to April or August 2028. Each stage moves along its curve at the index's
past-year pace, in months of capability progress at today's rate, so faster progress
compresses it. A
persistent human-work floor (default 0) and the stage paces remain adjustable. The available research system includes validated fast gains.

**Capability as ECI.** Each draw's deployed model is read as ECI from its two parts, both
relative to today's model: today's US frontier score, plus 6.4 points per 10x of training
compute (the Compute/capabilities tab's frontier-grade exchange rate), plus 31 points per
10x of algorithmic efficiency, the rate that makes today's modeled growth equal the ECI
frontier's fitted 15.7 points a year. Both rates fit that history; algorithms earn more
because compute explains only about 3 of the 15.7 points (the tab's own pooled fit puts
algorithms at 12 points a year, close to the 12.6 left over). One rate for both, 17.6 points
per 10x, also fits the history but credits the mostly algorithmic takeoff less: it put
ultraintelligence 7 months later. Coding counts as automated once a model reaches the ultraintelligence threshold under the same reading.
Ultraintelligence is defined as an ECI level on this chart: 250 by default, adjustable in the sidebar.

## Parameter uncertainty

Every factor is lognormal around its setting, from one fixed set of uniform draws per
scenario mapped to normal scores, so paired comparisons survive any change of setting.
`uncertainty` (default 40%) is the 1σ of each log for rates, cycle times and workflow
hours. Four settings that decide whether takeoff runs away get their own spreads, sized
to the literature:

| Setting | 1σ | Range it gives | Source |
|---|---|---|---|
| Returns to research r | 0.65 in ln r | ~90%: 0.34–2.9× the setting | Forethought 0.33–3 around 1.0; Davidson & Houlden 0.4–3.6 around 1.2 |
| Judgment elasticity | 0.7 in its log | 80%: 0.4–2.5× | Its three readings (next-step ~0.29, TasteVal before and after its break ~0.25 and ~0.76) disagree by ~3×; the band holds all of them |
| Judgment-rate change | 15 points per 10× | 80%: ±19 points | No measurement; sized so it adds to, rather than sets, the runaway share |
| Labor–compute elasticity 1/(1−ρ) | 0.8 in its log | ~90%: 0.22–3.1 around 0.83 | Erdil et al. 0.83 (chess), Davidson's −0.2; Whitfill & Wu ~0 (frontier experiments) to 2.6 (compute) |
| Software ceiling | uniform | 6–16 orders of magnitude above today's efficiency | Forethought: 6–16 above full automation, returns falling linearly to 0 there |

| Scale-dependent share of algorithmic gains | 1.0 in log-odds | 80%: ~22–78% around 50% | Gundlach et al. 2025: most historical gains grew with compute |
| Compute such a gain waits for | 0.7 in its log | 80%: 0.4–2.5 around 10x | No measurement |

A scale-dependent gain is worth something only once models train at larger compute, as
Transformers and Chinchilla scaling were (Gundlach et al. 2025). Of the gains a model gets
through training, that share counts in full once models train at about 10x the compute the
frontier had when the gain was found (about two years at today's growth); fast gains such as
tooling and kernels, the paper's scale-invariant kind, count at once. Nothing AI-frozen
research would have found waits, since history already balanced it with compute; gains AI
finds faster wait, then count in full. This is separate from ρ, which sets how fast gains
are discovered. At the 50% median it moves ultraintelligence from Mar to Aug 2030 and the
share making 3 years of progress in RSI's first year from 27% to 16%; 2x compute growth from
RSI with no delay brings ultraintelligence back to Feb 2030.

The ceiling raises difficulty so returns to research fall linearly to zero as a scenario's
algorithmic efficiency nears it; no scenario then runs away without bound. It moves
ultraintelligence from Jan to Mar 2030 and cuts the share making 10 years of progress in
RSI's first year from 5% to 2%.

The judgment elasticity is still calibrated once at the ρ setting, not per draw.

FOOM here means much progress fast after RSI, measured as the first-year progress card
does (years of 2025-level ECI progress in the year after RSI). Pace doubling after RSI is
the wrong measure: a scenario that runs to its software ceiling within months reads as
slowing a year later. At today's defaults (seed 42) 42% of scenarios make 3+ years and 13%
make 10+. Split by each scenario's returns to research r and labor-compute elasticity
sigma = 1/(1 - rho), the share making 10+ years is:

| | sigma < 0.5 | 0.5-1 | >= 1 | all |
|---|---|---|---|---|
| r < 1 | 0% | 0% | 4% | 2% |
| r 1-2 | 0% | 3% | 44% | 19% |
| r > 2 | 3% | 12% | 81% | 37% |
| all | 1% | 3% | 29% | 13% |

Substitution is the gate: below sigma 1 almost nothing FOOMs whatever r is, because labor
alone can multiply research only so far against fixed experiment compute; at sigma >= 1
(about 40% of scenarios at the default rho of -0.2 and its spread) FOOM is common, and r
scales it. The model's FOOM odds therefore rest mostly on the weight sigma >= 1 gets.

Forethought's own measure, years of progress made in the year after RSI, is counted in
2025-level ECI progress: the 13.5 points the US frontier gained over 2025, about 5.8x a year
of effective compute here. Read through the two ECI rates, under the software ceiling and
the scale-dependent share, its median is 1.8; 16% of scenarios make 3 years or more and
1% make 10, against Forethought's 57% and 18%, whose model has the ceiling but not the scale-dependent share. The yardsticks differ: Forethought counts about 10x a
year of effective compute as a year, holds compute fixed and starts at full automation of AI
R&D. Two features account for most of the gap. At today's defaults (24% and 2%), letting
every gain skip the training-run wait on a one-week cycle gives 56% and 24%; labor and
compute as substitutes (rho 0.5, no spread) gives 53% and 5%; both give 94% and 63%, and
r at Forethought's 2025 central 1.2 on top 96% and 70%. Either alone now about matches
Forethought; both together overshoot it, since the model also has more AI copies and
autopilot, which Forethought's labor already stands for.

## Evidence calibration

Four priors come from the RSI tab. Research work splits into nine activities, each with
the human hours a task at each level still takes, as a share of an unassisted researcher's
total (the app's `_TK_ACTIVITIES`). The unassisted column is how researchers split their
time; no study measures it, so it triangulates Epoch's researcher interviews (2024:
direction takes little dedicated time but is crucial, engineering and debugging the
time-consuming part) and Microsoft's developer time study (about half the day
coordination). The level columns move time rather than only cut it: at AL3 people write no
code and spend more directing and reviewing AI's work (13% of unassisted time, from 0).
The stages sum the activities: direction (high-level direction and coordination) 20%,
experiments (planning, code, debugging, directing AI, runs, analysis) 70%, verification
10%, which are the stage weights. Each stage's prior per level (AL0–AL5) is its activities'
sum over that weight, direction 100/100/100/97.5/85/0%, raised to one common power so the
weighted mix of the index's latest month makes people 1.7x as fast as unassisted; that
gives direction 94%, experiments 46%, verification 80%. The order follows Anthropic's
level definitions and Epoch's and Anthropic's staff studies; no source gives the numbers.
On the index's projection people still do ~35% of unassisted hours in mid-2027, capping
research near 3x however fast AI is, and fall under 5% only as AL5 arrives (Mar 2028). The weights barely move
dates: 25/50/25 gave the same median RSI and ultraintelligence. Stage halving times all
take that share's past-year halving. The research-quality prior is the geometric
mean of three readings: the next-step study's win-odds growth (a Bradley-Terry strength
ratio), and TasteVal's compute multiplier (P-Zero Research, 2026) before and after the break
its Chow scan finds at GPT-5.2: 15.7 and 4.6 ECI points per doubling, read at today's ECI
climb. The multiplier saves serial experiment time, so it counts toward quality at the
experiment input's half share of the research CES, without the parallelization penalty.
Both of its regimes are kept because a steady trend in final scores reproduces the break:
near the expert's level each gain in score is worth more compute, so the post-break rate is
partly a property of the benchmark's own score curves (and it slows at shorter budgets).
The coding elasticity is the staff survey's output growth. Each is divided through the same
transfer form by assumed effective-capability growth.

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
numerical range records nothing further; it has reached RSI, unless a human floor above
the bar rules RSI out for every draw.

Exports include the version, time origin, effective parameters, calibration assumptions,
evidence dates, inherited RSI samples, and calendar/coding-relative event times. The source
fingerprint reloads changed model code and invalidates Streamlit's result cache.

Human-hour curves are operational criteria, not a staffing or queueing model. Quality and
capability remain aggregate quantities. Project success is not an experiment-level failure
simulation. Hardware innovation, manufacturing and policy remain outside scope. Proxy calibration does not establish a causal
feedback coefficient or calibrated confidence intervals.
