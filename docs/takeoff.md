# Takeoff tab: design record

The Takeoff scenario explorer and its evidence half (the former RSI tab). Read before changing `render_takeoff`, `render_rsi`, `takeoff_model.py` wiring, or any `_tk_*` / `_rsi_*` / `_pc_*` milestone helper. Equations and calibration live in `TAKEOFF_MODEL.md`.

## Takeoff scenario explorer

`render_takeoff()` uses `takeoff_model.py`, currently `takeoff-v31-direction-waits-for-coding`.
See `TAKEOFF_MODEL.md` for the current equations, calibration, definitions, and caveats.
Simulation time now begins **today**. `onset` retains exact final RSI coding-date draws;
`events` are already years from today. Never add onset to event times. `after_coding`
subtracts finite coding dates, preserving censored draws without inf-minus-inf NaNs.
The simulation, and every card, interval and share, runs to December 31 of
`_TK_SIM_END_YEAR` (2040); `tk_end_year` (default 2030) only crops the charts, via
`_tk_clip` (`test_horizon_resets_and_rsi_settings_survive_reset` holds the cards fixed as
it moves). Gaps have less follow-up for late starters and say so.

`_pc_rsi_onset` preserves hidden RSI widgets, caches source components and the mixture,
and returns sorted exact final samples, including conditioning, penalty, and clock.
This remains shared with the RSI tab. The display horizon does not clip its tail.

`_pc_rsi_onset` mixes the same components as the RSI projection (`_pc_blend_parts`, AL5
left out). Takeoff simulates every `_TK_DRAW_STRIDE`th (2) of those sorted dates, half
`N_SAMPLES`, so its coding CDF is the projection's as shown to within one full-sample draw
(`test_coding_automation_inherits_the_exact_final_rsi_cdf`). `render_takeoff` then runs
twice: `_tk_pair_onset` reassigns those dates so later ones go with a later first-run
`direction_ready` (the month direction would be under the bar were coding no gate; ties by
slower capability) (`tk_coding_pairing`, copula correlation, default 0.9), which moves the
pairing only, never the CDF
(`test_coding_dates_pair_with_direction_without_changing_their_distribution`). The
first run stops a month past `_TK_PAIRING_YEARS`, all the pairing reads
(`test_the_pairing_run_stops_where_the_pairing_reads`). RSI is the first time research direction is at
or under `full_human`% of baseline human hours with the coding date arrived
(`test_capability_paths_span_the_grid_and_rsi_waits_for_coding`); until that date direction's
hours hold at the bar (`test_research_direction_waits_for_coding`).

Every factor is lognormal: `z` is fixed uniform draws mapped to normal scores. r,
the judgment elasticity, its drift and ρ each carry their own literature-sized spread
(`returns_spread`, `judgment_spread`, `drift_spread`, `substitution_spread`; sources in
`TAKEOFF_MODEL.md` *Parameter uncertainty*), so FOOM is possible but not certain
(`test_the_spreads_leave_a_runaway_possible_not_certain`). `POINT_ESTIMATE` zeroes every
spread; deterministic tests use it. The per-draw values come back as `result["draws"]`.
Append new factor rows to `z`, never insert, so earlier draws stay paired across versions.
Row 18 draws each scenario's software ceiling (`ceiling_low_oom`–`ceiling_high_oom`, default
6–16 orders of magnitude above today's efficiency, Forethought's range): difficulty rises so
returns to research fall linearly to 0 there (penalty −βC·ln(1−S/C); `ceiling_step` is the
exact step, `reference_software` the same closed form), so with it no draw diverges
(`test_the_software_ceiling_holds_every_draw_below_its_limit`); a high end of 0 removes it,
which the runaway tests use. Row 19 draws `scale_dependent` (median 50%, logit spread
`scale_dependent_spread`; drawn with `compute_bound_corr`, default 0.5, against ρ's draw so a substitutable scenario has fewer scale-dependent gains, `test_elasticity_and_scale_dependence_share_one_factor`; it moved the share of scenarios whose pace doubles within a year of RSI from 16% to 17%, a measure that undercounts takeoffs exhausting their ceiling inside the year, so prefer the first-year progress card's shares), row 20 `scale_lag_oom` (median 1, lognormal
`scale_lag_spread`): of the gains a model gets through training, that share pays off only
once models train at that many orders of magnitude more compute than the frontier had when
the gain was found (Gundlach et al. 2025). Fast gains (tooling, kernels) are exempt, as the
paper's scale-invariant kind. `pays_off` reads the research stock found by the month the
frontier (`compute_seen`, `stock_seen`, one sorted search across draws) stood a lag below
the frontier when the model's run began (`run_start_scale`), adds history's balanced pace
over one lag, and never holds a model below the frozen-AI reference's trained software, so
AI-frozen research never waits and AI-driven gains wait, then count in full
(`test_scale_dependent_gains_pay_off_only_as_compute_scales`,
`test_scale_dependent_gains_wait_on_compute_rather_than_vanish`). It is about when a
discovered gain pays off, not how fast gains are found (that is ρ). Successors are adopted,
and re-checked at deployment, by capability as it pays off, so `capability_log` never
falls; `software_realized_log` is its algorithmic part, which the ECI reading uses.

The model starts with partial automation, today's already-assisted software progress,
and an assumed existing undeployed backlog shared with the reference. Coding productivity
ramps to the inherited milestone; overlapping endogenous coding gains use a maximum rather
than multiplying the two. Fast and successor improvements partition discoveries and each
validate a start-of-cycle snapshot. A same-compute reference freezes today's AI skills.
RSI gates on coding plus research direction under the human-hours bar
(`test_rsi_turns_on_research_direction_alone`).
From each draw's RSI, `rsi_autopilot` (default on, *Autopilot from RSI*) zeroes every stage's human
hours in the dynamics and the humans chart; the RSI outcomes keep the pre-autopilot hours
(direction, experiments, verification), which the RSI card's hover quotes
(`test_autopilot_drops_every_human_hour_from_rsi`). `serial_speedup` (default on, *Cycles speed up with
AI*) shrinks each validation by the verification stage's people/AI mix against today's and
each new training run by today's capability growth over the current one (capped at 1), so
cycles shorten once research outpaces today; `reallocate_compute` (default on, *Labs
re-split compute as AI changes*) re-splits the non-training compute between agents and
experiments by the CES first-order condition, anchored at today's 20/40 split and driven
only by AI's change against the frozen-AI reference
(`test_reallocation_follows_what_ai_changes`); section 1's *How the compute is split*
(`_tk_split_fig`, from `agent_share_quantiles`) draws it, and the workforce chart scales its AI copies by each draw's `agent_share` (kept per draw, float32). It moves FOOM about a point, since
only 60% of compute is re-splittable. `ceiling_returns_corr` (default 0.5) ties each draw's
ceiling rank to its r (`test_a_nearer_ceiling_comes_with_lower_returns`). Today's first run is unchanged and fixed-cycle
tests set it to 0 (`test_serial_speedup_shortens_cycles_after_rsi_and_converges`). `ai_copies` (default 1) multiplies AI's speed in that
Amdahl term by the deployed model's paid-off algorithmic efficiency to that power, as more
copies on the same compute (log space, so a runaway cannot overflow;
`test_more_copies_speed_only_the_work_ai_does`). `_tk_copies_yearly` nets its ~4%/yr of
today's labor growth out of the staff-survey coding fit, and `_tk_automation_growth` credits
it over the evidence window before research judgment is fitted. `judgment_cap` (default 0,
none) caps AI research judgment at a multiple of today's
(`test_a_judgment_cap_only_ever_slows_research`); AI Futures caps taste near 4× the best
human, which maps to a larger, unmeasured number here. Coding counts as automated at the
inherited date or once a deployed model reaches the ultraintelligence threshold (`coding_capability`, from
`_tk_eci_scale`), returned as `coding_at` with `onset` left exact
(`test_a_model_at_the_coding_capability_has_automated_coding`). Research acceleration
has its own chart and 2×/5× first-crossing events (internal; no card shows them). `_tk_eci` reads each draw's capability
path as ECI at `result["eci_rates"]` (`_tk_eci_rates`): the training-compute part
(`capability_log − validated_log`) at the CC tab's frontier-grade exchange rate
(`_cc_frontier_grade_algo`, margin 3 then 5 then pooled), the algorithmic part
(`validated_log`) at whatever makes today's modeled growth equal the ECI frontier's fitted
rate, which lands near that tab's own fixed-compute fits (`TestTakeoffEciRates`). One rate
for both also fits the history but credits the algorithm-driven takeoff less: it put
ultraintelligence 7 months later. The coding test uses the same reading
(`simulate(coding_weights=…)`). The path is floored at that trend for `_TK_ECI_FLOOR_YEARS`
(the shipping lag), the lift then fading over `_TK_ECI_FLOOR_FADE_YEARS` so no line steps down; the chart's view stops `_TK_ECI_VIEW_PAST_ULTRA` above the threshold.
v7 adds a CES labor-compute mean (`substitution`), Amdahl labor gains as the human share
falls (`ai_speed`), non-AI input growth holding the frozen-AI pace (`baseline_inputs`), a
steady-state pipeline, and an `r` control (`tk_returns`) from which the app sets
difficulty = 0.5 / r (`_TK_R_REF_PARALLEL`, the default exponent), so the parallelization
setting moves returns to inputs only, not difficulty
(`test_parallelization_does_not_move_difficulty`). It still moves dates through the
experiment calibration: a higher exponent lets compute and coding explain more of the
observed experiment growth, leaving a lower fitted judgment elasticity (at `cal_weight` 0
it moves nothing). The pace-factor chart (`_tk_factors_fig`) is capped at
`_TK_FACTOR_VIEW`. Exact zero-feedback equalities in tests need `baseline_inputs=0`.
Non-AI inputs (all scaled by `baseline_inputs`): staff headcount (`hiring_growth`,
`staff_after_rsi`, on each draw's RSI clock when AI transfer is on) speeds the human share
inside the Amdahl term, and experiment compute grows at `exp_share` of the frontier's log
growth, fitted against `historical_compute_rate` so the frozen-AI pace stays steady
(`test_headcount_and_experiment_compute_hold_history_steady`). The frozen-AI reference is
integrated step by step alongside the actual path (`reference_gained`), not by the
closed-form `reference_software`, since its labor and experiments now grow differently. The
pace chart's factors split exactly: compute with neither AI help nor extra staff, staff's
addition at the share people still do (zero on autopilot), AI the rest.
The sidebar leads with the parameters of a one-at-a-time sweep (seed 42, today's
defaults; months to ultraintelligence from Nov 2029): r (0.5/2: +6/−5), the judgment
elasticity (half/double: +2/−3), parallelization (0.8/0.3: 0/0), ρ (−0.5/0: +2/−1),
`ai_copies` (0/2: +3/−2), the training-run wait `fast_share` (12.5/50%: +1/−3; 100%: −7),
`scale_dependent` (25/75%: −3/+3), `serial_speedup` and `rsi_autopilot` (off: +3, +2) and `rsi_compute_boost` (default 2×;
under *Advanced*, the software ceiling moves it −2 if removed, +2 at 3–8 orders of
magnitude). The boost multiplies each draw's log compute growth after its own
RSI date, waiting `rsi_compute_delay` (18 months) then
ramping in over `rsi_compute_ramp` (12), never in the frozen-AI reference
(`test_compute_boost_starts_at_each_draws_rsi_and_spares_the_reference`,
`test_compute_boost_waits_its_delay_then_ramps_in`): delayed, it moves the
ultraintelligence median a month at most; switched on at once, 2× brings it 6 months
sooner, since scale-dependent gains then wait less. The rest,
data-center compute included (~1 month), sit in one *Advanced* expander. Retarget that
grouping, and the expander's caption, if a rerun of the sweep reorders them.
AI judgment is gated by the direction stage's human share, uncapped
(`test_ai_judgment_only_counts_where_ai_directs_research`), at `judgment_today` (default 1)
times people's, with quality normalized to 1 today (`quality0_log`) so the level moves only
what handing direction over buys (`test_judgment_today_only_changes_what_handing_direction_over_buys`);
the default deliberately ignores benchmark claims that AI already out-judges experts.
`direction_adoption` (k, default 0.5, *What decides RSI*) moves direction along its curve
J^k times as fast, J = `ai_judgment_log` (AI judgment over today's AI's, transfer and cap
included; pace clipped at e^30 for runaways), passed to `workflow_state` as
`direction_pace` (`test_delegation_follows_judgment`). Still a function of capability, so RSI
stays a capability threshold. At 0.5 it brings RSI ~3 months sooner and ultraintelligence
~half a month: the RSI date marks the takeoff more than it paces it. `judgment_drift` (default 0)
changes its elasticity by that fraction per 10× of capability (`judgment_gain_log`), so b
can fall or rise as AI improves; the b card's hover gives b at 100× when it is set. A draw
that runs away numerically counts as reaching RSI then (under a human floor above the bar,
never): its capability has passed any coding milestone
(`test_a_runaway_reaches_rsi_unless_the_floor_blocks_it`). Ultraintelligence is the first
month a draw's ECI reaches `tk_ultra_eci` (*Ultraintelligence threshold*, default
`_TK_ULTRA_ECI` = 250, read by `_tk_ultra_eci`, which falls back to 250 outside
`_TK_ULTRA_RANGE`; `_tk_run` passes it to `_tk_eci` as `ultra_level`, which every card,
chart and caption reads, and Pacing's milestone takes it from there). Raising it moves RSI's
late tail through the coding shortcut, not its median
(`test_ultraintelligence_threshold_moves_the_card_chart_and_explainer`). The explainer under
the cards (`_tk_ultra_explainer`) counts in GPT-4-to-GPT-6-Astra jumps (`_TK_ULTRA_JUMP`); its
METR comparison is a one-off fit of log p80 horizon on ECI over the models in both datasets
(~1,000 work-years at 250 using all 20, ~20 using the 13 at ECI 140+), static text to refit
when either dataset moves. ECI at RSI is fixed by construction:
human hours depend only on capability, so RSI is a capability threshold.
Under the three headline cards, a row gives coding → RSI and RSI → ultraintelligence
(`_tk_gap`, the FOOM readout) with 80%/90% intervals, over draws reaching the first by the
horizon, a second past it counting as longer. Beside them, *Progress in RSI's first year*
(`takeoff.progress_years`) is Forethought's readout: capability gained in the year after RSI
in years of 2025-level ECI progress, the yearly chart's unit (`_tk_year_gain`, ~5.8×/yr
effective compute through the ECI mapping), +inf for a runaway inside the window. Forethought counts ~10×/yr, holds compute fixed and starts
at full automation, so its hover says to compare loosely. The loop, brakes and these six cards lead
the tab, above the evidence: they render into a `st.container()` slot made before
`render_rsi`, since they need its settings first.
One seed, `tk_seed` (default 42, *Random seed* in the sidebar), fixes every draw on the tab:
`_render_takeoff` seeds numpy's global generator for the evidence fans, and the cached
`_pc_milestone_components` and `_pc_rsi_projection_samples` take it as an argument and
draw under `_seeded_rng`, which restores the global state afterward so a cache hit and a
miss leave the same stream. The simulator seeds its own local generator from it. The same
settings give the same cards and charts across processes and reruns.
A progress bar (`_tk_loading`, `_TK_LOADING_STEPS`) sits above that slot and steps through
evidence, simulation, re-pairing and drawing; `render_takeoff` clears it in a `finally`
once the last chart is drawn, so it outlasts every number. The cards' slot is `st.container(key="tk_headline")`; the
global `<style>` block enlarges metrics and captions under `.st-key-tk_headline` only. Each
90% CI line goes in its own `tk_ci90_*` container (`_tk_ci`), which that block dims.
App-wide, `_BOOT_BAR` (right after the style block) shows a loading bar on a session's
first run only (`_vp_booted`), cleared by the dispatch before any tab draws.
`tk_dc_compute` (default on) feeds compute growth from the Data Centers tab:
`_tk_dc_compute` samples the US largest-run projection (`_pc_projection`, as Pacing does)
and averages each path over a fixed `_TK_DC_AVG_YEARS` (not the chart horizon, which
must not move dates) into one rate per draw, passed as
`simulate(compute_rates=…)`. From the end of `_TK_COMPUTE_SLOW_YEAR` (2030) each rate keeps
`compute_slowdown`% of itself (`grown`/`log_level`, exact two-rate integrals;
`test_compute_slows_to_its_late_rate_from_the_slowdown_date`). The ECI scale then uses that series' historical fit as the
compute growth behind today's trend, so only a projected change in compute growth moves ECI
(`test_compute_follows_the_data_center_projection_unless_toggled_off`).

Evidence calibration uses the endpoint growth in `openai_experiment_velocity.csv`, adjusts
for assumed historical compute and coding gains, and fits a quality elasticity blended with
the prior. A supplied validated-progress ratio overrides the experiment proxy. Matched-stage
human hours can set current shares/halving times; missing inputs are zero, not invented data.
The `tk_cal_` and `tk_measure_` inputs round-trip in URLs. *Reset to defaults* resets all
Takeoff inputs and preserves RSI. Workflow fit callbacks validate before changing widgets.

The model's source digest is checked on app reruns, reloads stale imported model code, and
is an explicit argument to `_tk_simulate`'s Streamlit cache. Preserve this dependency when
changing the simulator schema. Local seeded RNG and common draws support paired scenarios.
A diverged draw records nothing further
(`test_a_runaway_reaches_rsi_unless_the_floor_blocks_it`); the pace chart's view stops
at `_TK_PACE_VIEW`, since a runaway's pace reaches ~1e300.
Unit tests cover accounting, snapshots, fast feedback, calibration, ordering, censoring,
convergence, and RNG isolation. AppTests cover the layout, calibration, matched hours,
URL sharing, stale imports, and exact inheritance from RSI.

The tab reads as a story, top to bottom: the warning, *The takeoff model* (the loop
`_TK_LOOP_HTML`, its three brakes, the headline cards for coding automation, RSI and
ultraintelligence with 80%/90% CIs, the two gaps between them and the first-year progress),
*The evidence* (the former RSI tab), then *Inside the takeoff model*: the evidence row, **1. Research speeds
up** (`_tk_pace_fig` twice: *Capability relative to 2025* (`capability=True`,
`capability_2025_quantiles`: the pace with the labor, compute and difficulty factors held
at today's, so only AI and judgment move it, axis to `_TK_CAPABILITY_VIEW`;
`test_capability_pace_holds_headcount_compute_and_difficulty_at_todays`), then the net
pace; then the factor chart, whose labor-and-compute factor splits exactly into
human labor, AI capabilities and compute (`pace_factors` `labor`/`ai`/`compute`, summing to
`resources`; `test_pace_factors_multiply_back_to_the_pace`), then *Compute going in*
(`_tk_compute_fig`, each draw's `workflow_progress["compute_log"]`, post-RSI boost included,
with the milestone medians marked)), **2. People step out of the loop**
(`_tk_humans_fig`), **3. The research workforce** (`_tk_workforce_fig`: heads, humans from
`_TK_HUMAN_RESEARCHERS` growing `_TK_HIRING_GROWTH` a year to each draw's RSI and shrinking
by `_TK_POST_RSI_DECLINE` after, and AI copies from `_TK_AI_COPIES_TODAY` (50k, an
estimate; all five workforce assumptions are *Advanced* inputs, `tk_wf_*`, read by
`_tk_workforce_settings`), growing with agent compute and more copies; and human FTEs, each copy worth today
the automation index's share of today's work spread over those copies (~0.13 people), rising
as (ECI / today's)^`_TK_AI_COMPETENCE_POWER`. Counts
read k/M/B/T (`_tk_count`, axis `exponentformat="B"`), carried in `text` so today-forward
can still relabel the dates. A readout, never an input), **4. What it does to capability (ECI)** (plus *2025-level ECI progress in
each year*, `_tk_yearly_fig`: ECI gained per calendar year over what the frontier gained
in `_TK_YEARLY_BASE_YEAR` (`_tk_year_gain`, 13.5 points, below the 15.7-point fitted trend
the ECI chart uses, so these bars read ~1.16× that trend's), a different 2025 unit from
chart 1's experiments; last year is measured off
the frontier, this year measured to today then projected, each kind its own colour, and
its line labels are placed by hand since a log axis reads an annotation's y as the exponent) and **5. When it happens, and
how fast** (`_tk_dist_fig`: density and cumulative share of the RSI and ultraintelligence
dates and the gap between them) and **6. Against Anthropic Staff** (`_tk_staff_survey_rows`:
the August 2026 staff poll's AI share of the work, median and Figure 8's printed 10th–90th,
beside the run's AI share of research hours from `overall_human_quantiles`, the quantiles of
each draw's stages at `WORKFLOW_WEIGHTS`, never a sum of stage quantiles
(`test_all_stage_quantiles_are_of_the_weighted_draws`); a survey date before today reads the
automation index through the assumed hours per level, an estimate. A table, not a chart overlay, and never an input: a share of work
is not a share of hours). `test_default_render_and_stable_rerun` pins the order.

Today's human share comes from the automation index's latest mix of AL levels times the
human time each level still takes in each stage: `_TK_AL_STAGE_HOURS_PRIOR` (stages x
AL0–AL5; direction keeps 85% at AL4, experiments 24%), each stage's sum of
`_TK_ACTIVITIES` (nine activities, each people's hours per level as a share of an unassisted
researcher's total; AL0 sums are `takeoff.WORKFLOW_WEIGHTS`, 20/70/10,
`test_stages_are_their_activities`; at AL3 writing code goes to 0 and directing and
reviewing AI's work peaks, `test_al3_moves_time_from_writing_code_to_directing_ai`),
raised to one common power by
`_tk_fit_al_hours` so the stage-weighted mix gives `tk_human_accel` (default 1.7×, from
Peter's conversations with researchers; *Human research speed-up today*), ends fixed.
Direction keeps the most hours at every level short of AL5
(`test_people_keep_directing_until_al5`). Changing it resets each stage's starting
share and halving time (`_tk_on_human_accel`, `_tk_human_calibration`), and section 2's
table shows the arithmetic (`TestTakeoffHumanSpeedup`). Under it, *Where people's time goes* (`_tk_activity_path`) is a readout the model
never reads: each stage's fitted hours split among its activities by their priors, on the
index's projection at the median AL5 gap, and the Amdahl
ceiling (one over the human share) in mid-2027, before AL5.
Four defaults are read off RSI data at import (`TestTakeoffRsiCalibration`). Today's human
share per stage is the R&D automation index's latest month read through that stage's row
of `_TK_AL_HUMAN_HOURS` (`_tk_stage_shares`). Research hours then follow the index's own projection, not an
exponential: `_tk_ladder_curve` builds the human share over the coming years from the
ladder (AL4 from the projection's start at the "AL4 reaches 90%" card's median rate, AL5 as
AL4 shifted by an AL4→AL5 gap, AL2/AL3 on their own fits, nested), times the hours per
level, one curve per stage (`test_each_stage_follows_its_own_ladder_curve`). `_tk_ladder_curves` gives each scenario its own gap, lognormal over the AL5 card's
80% range from a fixed seed, so RSI carries the unmeasured AL5 rung's uncertainty; one
curve per draw (rows on an even grid). Research direction, the stage RSI is defined on, also
reads the next-step study: `_tk_nextstep_curves` (the researcher's odds of the better step
in excess of the odds left at the study's ceiling, per-draw rate over the card's range),
averaged with the ladder in *dates*, not values (`_tk_blend_in_time`,
`_TK_NEXTSTEP_WEIGHT` 0.5), so RSI lies between the two sources' dates
(`TestTakeoffDirectionCurve`); `human_curve` is (years, weighted curve, direction,
experiments, verification), elements past the second overriding stages in order. The two agree
independently: a median ~1.6 years each to 5% of direction's start.
`simulate(human_curve=…)` moves each stage along that shape in
capability-equivalent years (`workflow_state`; `test_human_hours_can_follow_a_given_curve`).
The stage halving times only set each stage's pace along its curve, as the weighted
halving over its own; all default to the index's past-year halving, so pace is 1. The
judgment elasticity (`taste_slope`, `_tk_taste_default`) is the geometric mean of three
readings, ~0.38: the next-step study's yearly odds growth raised to `_TK_ODDS_TO_JUDGMENT`
(0.5: picking the better step more often is not proportionally more research; ~0.29), and
TasteVal's compute multiplier before and after its GPT-5.2 break (`_tk_tasteval_taste`,
`_TASTEVAL_ECI_PER_DOUBLING` 15.7 and 4.6 ECI per doubling, read at today's ECI climb and
taken to quality at `_TK_TASTEVAL_TO_QUALITY` 0.5, since it saves serial time on the
experiment half of the research CES; ~0.25 and ~0.76). Both regimes are carried because a
steady final-score trend reproduces the break (the paper's Table 2: the multiplier climbs
faster per unit of score near the expert's flattening curve), so the post-break rate gets a
third of the weight; `judgment_spread` (70) keeps all three inside its 80% band
(`TestTakeoffRsiCalibration`). The coding elasticity is the staff survey's yearly output growth. Both go
through `_tk_elasticity` against capability growth over the experiment window, which
comes from evidence (`_tk_evidence_window_growth`: the data-center compute fit times the
software trend, ~7.8×/yr), never a free assumption: an elasticity measured against one
capability rate and applied at another is inflated by their ratio. The experiment fit
runs the model's own research equation over the window, crediting compute, coding help and
the automation index's falling human share (`_tk_automation_growth`) before judgment, and
lands near the next-step fit (~0.27) and below the three-way default.

`_rsi_experiment_draws` uses a local fixed seed so the RSI fan, threshold card, and Takeoff
reuse the same paths without perturbing global RNG. `_tk_rsi_activity` shifts these to today
and annualizes slopes. `_tk_simulate` caches by these arrays as well as model source and RSI
onsets. Forward activity and mechanistic effort are alternative estimates, blended with a
fading weight, not multiplied. Current activity normalization only affects the displayed
2025-baseline fan, not today's already-assisted productivity. `_tk_simulate` still returns
2025-unit quantiles of pace, the reference's own pace and cumulative deliveries.
Its cached result keeps per draw only `capability_log`, `software_realized_log` and
`compute_log` (the factor chart reads `factor_medians`; `_TK_UNREAD` lists what it drops):
each (draws x months) array is ~3.4 MB held in memory, and Community Cloud's floor is 690 MB.
`tk_pace_today` (default `_tk_pace_today_default()`: OpenAI's experiment trend carried to
today) sets today's
pace directly; each draw keeps its activity spread around it. Rebase each draw before taking quantiles; never multiply quantile bands.
The simulator records ongoing production separately from delivery-window rates, which
remain available for internal diagnostics and the validated milestone definitions.

## RSI tab (now the evidence half of Takeoff)

The RSI tab merged into Takeoff: `render_takeoff()` draws the shared *Project through*
(`tk_end_year`), the header and warning, then `render_rsi(end_year)` under *The evidence*,
then *The takeoff model*. The evidence charts project through that year capped at
`_RSI_MAX_END_YEAR` (2031), on a weekly grid (`_fan_days`; a day is under a pixel). The rest of this
section still says "RSI tab" for that half; `_pc_rsi_onset` only reads its widget keys,
since a drawn widget's key cannot be reassigned in the same run. `test_default_render_and_
stable_rerun` pins the order; tests find Takeoff's own charts after the RSI projection's
CDF (`_tk_charts`).

`render_rsi()` runs CoBench, then the R&D automation index,
then the staff survey, then
merged code per person, OpenAI experiment velocity, research direction, then TasteVal
(`_render_rsi_tasteval()`), then *Capabilities Milestones* + *Automated coder projection* (`_pc_render_milestones()`,
moved here from the Pacing tab). The CoBench section plots that eval — Anthropic's
internal AI R&D benchmark — against release
date, with a fitted trend, a projection fan and `_RSI_SUBSTITUTION_BAR` drawn as
its bar (Anthropic's own stated score for a model that could fully substitute for
its research staff — not a benchmark ceiling). Four things are load-bearing:

1. **The fit is logit-space**, like RLI and UK Cyber: CoBench is a bounded success
   rate and a score-space line runs through 100%.
2. **Only a single OLS basis is offered.** Four frontier points cannot distinguish
   a line from a bend, so there is no piecewise or superexponential option and no
   backtest vantage-point selector. Don't add them by copying another tab.
3. **The default rate CI is widened, not the convention.** The segments disagree,
   so `_rsi_dt_ci()` takes the usual fit/2..fit×2 interval and widens it to
   span every segment rate **and** the slope's 80% t-interval (`_dt_t_interval`,
   `_DT_T80` multipliers by residual dof, decaying to the normal limit as points
   accumulate). A t-interval that cannot exclude a flat slope caps the slow edge at
   `_DT_CAP_DAYS` (flat at every horizon the app offers); the staff survey gets the
   same treatment via `_rsi_survey_dt_ci()`. Neither reaches the cap today (two
   residual dof each). It can only widen — `test_dt_ci_default_spans_both_segment_rates`
   and `test_small_sample_ci_widens_both_rsi_fits` hold that (the latter pins the
   live edges; retarget it when a new round moves the interval).
4. **`date_known` drives the "~" prefix** via `_rsi_date_label()`. Mythos Preview has
   no published release record (its date is carried over from AISI's narrow cyber
   figure, as in `aisi_cyber_tlo.csv`) and Model 2 (internal) is unreleased with its name
   redacted; Mythos 5 ships with Fable 5 on 2026-06-09, which puts it *below* the
   running max and off the frontier. Opus 5 and Mythos 5.1 come from the Fable 5.1
   system card's larger problem set, carried onto the Risk Report's by Mythos 5's
   ratio across the two (`_RSI_SYSCARD_RESCALE`) — an assumption each hover states.

CoBench is filtered for difficulty (mostly problems Mythos Preview failed at least
once in three tries) and run at a 300k-token budget, so scores don't compare to
public AI R&D suites — the fine print has to keep saying so.

**The sections chart; the cards date.** Each of the four draws its bar and stops
there, with no per-section ETA or row of projected values under the fan: that would
restate what the milestone card below already says. Every bar is dated once, on its card
(`test_the_sections_chart_and_the_cards_date`); that each card still reproduces
its own section's fit is pinned unit-side, per section, by
`test_eta_reproduces_the_section_defaults`. Don't re-add an in-section ETA.

The tab's second section (`_render_rsi_automation()`) is the most direct of its
series: not a score or an output count but the share of Anthropic's own model-R&D
tasks a Claude judge rates at each rung of Epoch's automation scale, from
`anthropic_rd_automation.csv` via `load_rsi_automation()`. **One chart carries
all of it** — the three measured rungs (AL2+, AL3+, AL4+), the AL4 projection,
and AL5 projected off the rungs' spacing — with both fans dated against a single
bar, `_RSI_AUTO_TARGET` = `_RSI_AL5_TARGET` = 90%. A standalone AL5 section and a
ladder-collapse diagnostic were folded in here;
`test_the_rsi_tab_has_one_automation_chart` keeps them from drifting back apart.
Nine things are load-bearing.

1. **The columns are cumulative, not per-band.** `al3_plus` is “at AL3 or
   above”, i.e. a boundary of the stacked figure. Cumulative because that is
   the monotone bounded quantity a logit fit wants: a single band's height
   rises and then falls as work moves up past it, so the AL3 *band* is
   shrinking while AL3+ still climbs.
2. **AL4's values are transcribed; everything else is digitized.** The figure
   prints a label only for AL4. `al4_dig` reads that same boundary by pixel
   analysis and must match the printed `al4_pct` within ~0.6pt — that is the
   calibration guard on the method used for the rungs with no printed truth.
3. **Saturated and floor months are dropped before fitting**
   (`_rsi_auto_points`, outside `_RSI_AUTO_FLOOR`..`_RSI_AUTO_SAT_HI`). AL2+
   passes 99% in April 2026 and AL1+ in December 2025; one pixel moves a share
   that close to the ceiling further in log-odds than a month of progress does.
   AL1+ is carried in the CSV for the nesting check only and is deliberately
   not in `_RSI_AUTO_LADDER` — every usable month of it is already above 95%.
4. **AL4's unlabelled months are charted at zero and never fitted.** Aug
   2025–Feb 2026 carry no label and no readable band; a logit fit cannot take
   0 anyway, and the post says only that February was “under 1%”.
5. **The position CI starts from the figure's own published interval**
   (`_rsi_auto_pos_sigma`, quoted at 90%, hence the 1.645), widened for the
   bend. The labelled months since May (`_RSI_AUTO_RECENT_N`) climb at about
   half the full fit's rate and the latest sits below the line, so
   `_rsi_auto_start` centres midway between fit and reading with an 80% band
   reaching both, and `_rsi_auto_dt_ci` stretches the slow edge to the recent
   months' own 80% t-interval. Widened, not refitted: four points can't tell a
   bend from noise.
6. **AL5 has no observations at all.** It is absent in every rated month and the
   post says Claude “is not operating fully autonomously for any measured
   subset”. Its curve is the AL4 rung shifted, so the band carries the AL4
   fit's uncertainty and the lag's — *not* whether a ladder measured inside
   supervised work says anything about the step that removes the supervisor.
   That is the dominant uncertainty, it is structural rather than statistical,
   and it is stated in a visible `_fn_line` rather than a hover.
   `test_al5_has_no_column_at_all` fails the moment a refresh adds any AL5 data,
   at which point the section should be rebuilt on it.
7. **The lag is measured at a reference share, and the reference matters a
   little.** `_rsi_auto_lag` compares two rungs' fitted crossings of
   `_RSI_AUTO_LADDER_REF` (50%). The rungs are near-parallel but the higher ones
   climb slightly faster, so the gap fans with the reference; the cards quote
   the sweep over `_RSI_AUTO_REF_SWEEP` instead of implying one number.
   `test_the_rungs_are_near_parallel_in_log_odds` is what keeps the shift model
   meaningful — if the rungs stop being parallel the lag stops being defined.
8. **The zeros censor the lag from below, and the censoring binds.**
   `_rsi_al5_lag` returns both the ladder extrapolation (continue the measured
   step-to-step increment) and `censor`, the shortest lag under which AL5 would
   still sit below the figure's ~`_RSI_AUTO_FLOOR` floor in the last rated
   month. Live, the extrapolation alone would have put AL5 above that floor
   already — a visible band — so the observed absence wins and the
   projection uses `max(ladder, censor)`.
   `test_the_absent_al5_band_censors_the_lag_from_below` pins that it binds;
   retarget it deliberately rather than loosening it. The lag's own 80% band
   runs from the censored value to `_RSI_AL5_TAU_HI_MULT`x it — asymmetric
   because the fast edge is what the data allows and the slow edge is a stated
   judgement, and a symmetric band would put mass below what is already ruled
   out. The slow edge is `_RSI_AL5_TAU_HI_MULT`x the larger of that lag and
   `slow`, the same censored lag read off the recent AL4 months alone
   (`test_lag_slow_edge_reads_the_recent_al4_months`). `rsi_al5_lag_mo`
   exposes the fast edge, since it is the whole answer.
9. **Both fans come off the same sampled AL4 paths**, AL5's being those paths
   shifted per sample, so the two cannot tell different stories about the rung
   they share; `_pc_rsi_al5_eta` likewise calls `_pc_rsi_auto_eta` and shifts
   it, and a zero lag reproduces the AL4 card exactly
   (`test_al5_is_the_al4_curve_shifted`).

Two chart details are deliberate. The two fans share **one legend entry
each** rather than the app's usual three-per-fan: six keys for two objects
covered the fans they labelled, so the legend moved above the plot too. And
AL5's measured zeros are drawn as a **flat line, not markers** — AL4's own
pre-March zeros occupy the same seven points, and as markers the two were
indistinguishable rings sitting on each other. The y-axis is floored at -3 so
that line is not swallowed by the axis.

The caveats the post states about itself ride the caption's hovers rather than a
paragraph: the basket of tasks is frozen on a July 2026 person-time weighting
(the post found no rise in novel tasks between its January and July baskets), and
the levels come from a Claude judge that matched staff owners exactly 59% of the
time — against 35% between two humans — with the disagreement concentrated at
the AL3/AL4 boundary this series is drawn at. Guarded by `TestRsiAutomation`,
`TestRsiAutomationLadder` and `TestRsiAl5`.

The tab's third section (`_render_rsi_survey()`) charts the report's other
substitution series, the internal staff survey (§3.4.2): self-reported output
multiple against no AI assistance, from `_RSI_SURVEY` via `load_rsi_survey()`,
fitted and projected on **log(multiple)** — a multiple has no ceiling to bound it
against, so it compounds the way METR's horizon does rather than saturating like
a percentage. Three things are load-bearing. Opus 4's round is not carried at
all: it reported no number, only that the result fell under the pre-set 3x median
rule-out threshold, and a bound is not a point on a trend. Model 2 (internal)
has an `estimated` point carried over from Mythos Preview — no round was run for it —
which draws hollow as an assumed value but is **included in the fit and the
anchor**: with only three surveyed rounds, ignoring the one flat reading
available overstates the slope, so the flattening is the point, not a bug (the
flag governs styling only). The rounds do not
report the same statistic on the same sample (medians on superusers, then on a
broader sample, then a geometric mean on an opt-in poll), so each point carries
its `note` on hover and the caption says the rounds differ — and the survey is
not discontinued, the report says only that no new round was run for Mythos 5.
The fan runs to the tab's *Project through* year like every chart, but the log axis
stops at `_RSI_MULTIPLE_VIEW`× the bar (the code chart likewise): at the fitted doubling
time the median runs orders of magnitude past the data by end-decade, which would squash
the measured points into the bottom decile. The chart draws
`_PC_RSI_SURVEY_TARGET_X` as its bar, the same one the milestone card dates.

The tab's fourth section (`_render_rsi_code()`) is the counted counterpart to
that survey: lines merged per active contributor per quarter, as a multiple of
the pre-2025 average, from `_RSI_CODE_RAW` via `load_rsi_code()`, fitted on
log(multiple) like the survey and dated against `_RSI_CODE_TARGET` = 30x. Four
things are load-bearing.

1. **Only the bars from `_RSI_CODE_FIT_FROM` are fitted.** The earlier ones are
   the baseline the multiple is taken against — they average to 1 by
   construction, so there is no trend in them to fit, and pooling them halves
   the slope (`test_fit_runs_on_the_2025_bars_only`). The ones the chart shows
   are drawn hollow; the x-axis opens at `_RSI_CODE_CHART_FROM` rather than at
   the first bar, since flat quarters only stretch the axis. All of them stay
   in the data either way — their mean is the calibration guard on the
   digitization.
2. **The hatched final bar is fitted, at its quarter's midpoint like the rest.**
   It averages only the days observed when the post went up, so a midpoint date
   is later than the days it covers — which reads the ramp slow, not hot.
3. **The position CI is the fit's own residual scatter** (`_RSI_CODE_POS_FACTOR`),
   not the survey's one-significant-figure allowance: this series is counted,
   not self-reported. The rate CI is `_dt_ci_t_widened` like the other two.
4. **The target label is added by hand**, here and on the survey chart's
   `_PC_RSI_SURVEY_TARGET_X` bar. On a log axis plotly reads an annotation's y
   as the exponent, so `add_hline(annotation_text=…)` parks the label at 10^30
   and off the chart. This chart's axis is also `dtick=1`: three decades of
   minor tick labels bury the four that matter.

Lines merged is an output proxy a coding model inflates directly, so the caption
and the milestone card's hover both have to keep saying it measures how much code
ships, not how much research it settles. Guarded by `TestRsiCode`.

The tab's fifth section (`_render_rsi_direction()`) charts Anthropic's Claude
Code detour study from `_RSI_DIR_RAW` via `load_rsi_direction()`: `_RSI_DIR_N`
turns where one of its own researchers went the wrong way, replayed to each
model, with a judge that has seen the finished session picking the better next
step. Three things are load-bearing.

1. **The bar is the study's own annotated practical ceiling**, `_RSI_DIR_TARGET`
   = 90%: what an *oracle* — a model shown the resolved session — scores against
   the same researchers, and the only threshold the source names. The fit is
   plain logit space like CoBench's, so the bar is reachable; scaling the odds
   by the ceiling instead was tried and makes it asymptotic, which is a
   different claim than the one the card dates.
   `test_fit_is_logit_space_against_the_studys_own_ceiling` pins both.
2. **The dates decide the frontier, not the figure's row order.** That figure is
   not chronological (it prints Sonnet 4.6 above Opus 4.6, which shipped twelve
   days earlier), so Opus 4.7 — released after Mythos Preview and scoring under
   it — sits off the running max, as Mythos 5 does on CoBench. Mythos Preview
   carries the same unpublished date the CoBench series does, hence the "~".
3. **The level is not "how often Claude out-researches a human."** The turns
   were *selected* for having room for improvement, so what the series carries
   is the trend in that comparison, not its level; the caption's hovers say so,
   and the milestone card's does too. Only the frontier is labelled in the chart
   (`_rsi_dir_label_positions()`, which also drops a crowded frontier point's
   label below the line) — nine models inside twenty-five months collide at a
   fixed `top center`.

The rate CI is `_rsi_dir_dt_ci()`, and the survey's is `_rsi_survey_dt_ci()`;
both are `_dt_ci_t_widened()`, the shared [DT/2, DT×2]-widened-to-the-t-interval
rule.

The sixth section (`_render_rsi_tasteval()`) is TasteVal, outside the automated-coder
blend but feeding Takeoff's judgment elasticity: Table 2's compute multipliers on a log
axis with the paper's two fits split at GPT-5.2 (`_tasteval_fit`) against a 1× expert
line, then the paper's prediction sigmoid (`_tasteval_predict_mase`) against the card's
90% bar. No fan and no ETA: the expert line is already passed and the card dates the
curve (`test_tasteval_section_charts_without_dating`).

The RSI blend's *Capabilities Milestones* row dates the CoBench and next-step
bars through `_pc_rsi_eta()` and `_pc_nextstep_eta()`, which reuse this tab's own
fits and rate CIs rather than fitting their own. Guarded by `TestRsi`,
`TestRsiDirection` and `TestRsiTab`.
