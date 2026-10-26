# China and Pacing tabs: design record

The lithography ceiling, China's crossing simulation, and the Pacing tab. Read before changing `_duv_*`, `_cn_src_*`, `_cc_cn_*` or `_pc_*` pause/race code. Hand-set judgments are recorded in `CHINA_COMPUTE_JUDGMENTS.md`.

## China tab — lithography ceiling (last section)

`_render_cc_duv_ceiling()` charts `cts_duv_production.csv` as published: US-and-allied
AI chips made per year against China's **lithography-bounded ceiling**, with DUVi imports
continuing or banned from 2027, at the report's AI-push allocation (the other two in the
hover). Under them it draws China's actual 2026 output (IFP) and its own trend. Three
things are load-bearing:

1. **The China levels are ceilings, never China's path.** The report assumes every
   non-lithography bottleneck solved and TSMC-grade yields by 2031, and calls its own
   numbers unrealistic near term. They also count chips made per year, not the largest
   single run every other chart here tracks.
2. **China's path is its own trend, capped by the ceiling.** `_duv_policy_cut()` grows
   IFP's 2026 domestic output (`_DUV_CN_2026_H100E`, ~1M H100e) at
   `_DUV_CN_OUTPUT_GROWTH` over a deterministic quantile grid (`_duv_realized_grid()`),
   takes min(trend, ceiling), and returns 1 − (growth under the policy ÷ growth with
   neither ban) over today → the Pacing horizon. That path starts ~4× under the banned
   ceiling, so an import ban binds only once output grows into it: ~3% to 2031 at the
   default 1.6×/yr. **That number is set mostly by the output growth rate**
   (1% at 1.2×/yr, ~10% at 2.4×, ~17% at 4×), which is chip output, not installed compute, so it is
   its own constant rather than `_WC_GROWTH`, and Pacing exposes it as `pc_duv_growth`
   (`test_faster_output_growth_reaches_the_ceiling_sooner`). Don't revert to the
   ceiling-to-ceiling ratio; it charged the ban ~24% for a limit China hasn't reached.
3. **A servicing ban erodes the fleet: wafer-bound output at once, HBM-bound output once
   the fleet falls below it.** `pc_duv_service` lowers the ceiling by (1 − `pc_duv_decay`)
   per year from `_DUV_POLICY_YEAR`. The wafer-bound share of China's trend (output bound by
   7nm wafer capacity, like the non-HBM Ascend 950PR) shrinks by the same factor at once
   (`test_wafer_bound_output_shrinks_with_the_fleet_at_once`); it runs from
   `_DUV_WAFER_SHARE_NOW` (30%) in mid-2026 to `pc_duv_wafer` (`_DUV_WAFER_SHARE_PCT`,
   default 50%) in mid-2028, held after, as China's own HBM eases that limit
   (`_duv_wafer_share`, `test_wafer_bound_share_rises_to_the_slider_by_its_year`). The
   HBM-limited rest is cut only where the eroded ceiling crosses it
   (`test_servicing_ban_waits_for_the_fleet_to_fall_to_hbm_output`). At the defaults: ~21%
   less chip-output growth in 2028, ~18% to 2031 alone, ~23% with the import ban. The
   rate (`_DUV_SERVICE_DECAY_PCT`, 15%) and both shares are judgments the report does
   not give; they sit under the pause panel's *Advanced* beside `pc_duv_growth`, and
   `CHINA_COMPUTE_JUDGMENTS.md` §5 records them.

On the Pacing tab the chip-supply levers (DUVi, HBM, smuggling, legal sales) act year by
year: `_duv_cut_path` gives each calendar year's DUVi cut, the compute mix
(`_cn_src_with_policy(dom_cut_path=…)`) turns all of them into the share of China's in-China
compute growth removed that year (`_cn_src_supply_cut`), and the sim subtracts it from the
compute term (`comp_cut`; `test_supply_cut_goes_year_by_year`), as does the remote-access
setback's regrowth. Only `pc_dom_slow` stays a constant cut. The "these settings" hover
quotes each lever's average cut to the horizon, naming each active DUVi lever even when
its cut rounds to 0%.

Guarded by `TestDuvProduction` (unit), `TestCcDuvCeilingTab` and
`test_duvi_levers_compound_with_the_domestic_slider` (integration).

## China's crossing simulation (Pacing tab's pause panel)

`_cc_cn_crossing_sim` dates when China's ECI frontier reaches a bar. The China tab's own
"When does China reach ECI 166" section that used it was removed; the Pacing tab's pause panel
is its consumer. Load-bearing:

1. **The pace band reality-checks the bottom-up rate** against China's observed slope
   over `_CC_GAP_WINDOWS` (`_cc_cn_pace_band`).
2. **Release steps are carried by level noise, not a release wait.** `level_sd` adds a
   per-sample offset, `_cc_frontier_level_sd` (sqrt 2 x China's scatter around its
   since-2024 trend), phased in over the median release gap; `release_gap_days` stays a
   sim option, unused. A rolling-origin backtest over China's 2025–26 releases puts ~80%
   of levels and dates inside the 80% band (`TestCcCrossingBacktest`); a rate spread
   alone covered ~16%, and a release wait on top skewed dates late. With `channels` the
   offset is its own `level` entry, left out of Pacing's breakdown rows.
3. **The end-year gap card's hover gives the fit-window range** (`_cc_gap_end_range` over
   `_CC_CALIB_SENS_STARTS`): China's slope swings on DeepSeek-R1's Jan 2025 jump.
4. **One channel split, shared.** `_cc_cn_channels` splits China's observed slope into
   compute (at China's compute growth over the window, `_CC_CN_HIST_COMPUTE`, ×2.12, above
   the forward band's ×1.68 mid, so China decelerates; `compute_fwd` is the forward term),
   innovation (residual), diffusion and distillation; the China tab's breakdown
   chart and scenario chart and the Pacing sim's bands all read it, so the tabs size
   distillation alike (`TestCcCnChannels`).
5. **Three-channel algorithmic engine.** `_cc_cn_crossing_sim` splits the measured
   rate into innovation (`_cc_pure_innovation_band`, never decays) + diffusion
   (nodist − pure, decays only after a pause, `_CC_DIFF_ABSORB_YRS` absorption ramp
   via `t_pause`) + distillation (algo − nodist, decays with gap/gap₀). With
   `pure_lo=None` or `t_pause=None` it reduces to the two-channel law, so the
   moving-US crossing is unchanged; the split bites in the Pacing tab's pause panel
   and the scenario table's four rows.
   `_cc_cn_target_years` stays as the constant-rate comparison quoted in the caption.
   `TestCcCnCrossingSim` pins the limits (saturated = constant-rate; frozen US = slower;
   pause-to-frozen-bar later than the crossing with the US still moving). The pace band is shared via
   `_cc_cn_pace_band` with the Pacing tab's pause panel so the two crossings stay ordered.
   The engines section carries a measured distillation control: `_cc_frontier_grade_algo`
   refits on frontier-grade models — within 5 ECI of the running frontier **and** trained
   within `_CC_FG_FLOP_MARGIN` of the running-max training run, against the
   **full-window** frontier (`load_eci_frontier(full_window=True)`). Both criteria are
   load-bearing: against the ECI tab's Feb-2024 cutoff every earlier model would pass as
   "near-frontier", and without the compute screen the subset admits
   the models nearest the frontier at the least compute — the heaviest distillers
   (DeepSeek/Qwen/Kimi). The surviving fingerprint is one-way: the refit's b_time runs
   below pooled (followers ride a teacher); a_partial does **not** rise —
   reasoning-era models reach the frontier at sub-frontier compute — so there is no
   two-way gradient. The refit's pair still replaces the pooled one for every
   frontier-facing projection (US-vs-China slopes, the pause bar mapping and climb, the
   compute terms), with pooled as fallback; `TestCcFrontierGradeAlgo` pins the b_time
   drop, the screen's bite and the coverage guard. Its n counts models (~12 at margin 5),
   and margin 3 is too thin to fit, so the innovation bands (`_cc_innovation_algo_band`,
   `_cc_pure_innovation_band`) fall back to margin 5 before pooled — pooled there
   collapses the pure-innovation band to a point. Measured by country, distillation is
   a *level*, not a rate: `_cc_cn_level_offset` (the country dummy at matched compute
   and date, quoted live in the control caption and the Pacing distillation checkbox)
   puts Chinese models above their US compute-peers while the two iso-compute rates are
   indistinguishable; `TestCcCnLevelOffset` guards it.

## Pacing tab

`render_pacing()` renders the pause counterfactual **first**
(`_pc_render_us_pause()`) and the *Compute Thresholds* race second — the tab's
question is what a US pause on a chosen date buys, and the race is the per-actor
detail behind it. That race answers "when can each entity first mount one ≥T-op
training job", at a 2mo/6mo run length (`_PC_RUN_OPTIONS`, default 2mo, reusing
the loader's `train_flop*` columns — 30% utilization, Epoch 8-bit OP/s).
Deliberately thin: it reuses the Data Centers tab's machinery rather than growing
its own. **T has no control of its own** — it is the US's own largest training run
at the pause (`caps[_DC_CTY_US]`, the *Largest training run* cell of the
state-of-play table), so the two halves of the tab cannot describe different
scales and the bar moves with the pause date and the run length alike.
`_PC_FALLBACK_THRESHOLD` is only the floor for a catalogue with no US capacity to
read.

*Capabilities Milestones* and the RSI blend live at the **bottom of the RSI
tab** (`_pc_render_milestones()`), not here — they are still named `_pc_*` with
the ETA helpers they call, and the machinery is unchanged. Eleven cards, driven by
the RSI tab's own *Milestone dates point at* selector (`rsi_timing`): `_pc_metr_eta()`
for the METR frontier reaching `_PC_METR_TARGET_HRS` — about one work-month — at each
of `_PC_METR_LEVELS` (at the month-scale bar p50's earlier firing is its own card and
weight rather than the exclusion it got at the old 40h bar), `_pc_rli_eta()` for the RLI frontier reaching `_PC_RLI_TARGET_PCT`
(above that tab's own milestone table), `_pc_rsi_eta()` for the CoBench frontier
reaching `_RSI_SUBSTITUTION_BAR` (Anthropic's own full-substitution bar, which the
RSI tab dates too), `_pc_rsi_survey_eta()` for self-reported staff speedup reaching
`_PC_RSI_SURVEY_TARGET_X` (about a doubling and a half past the most recent round),
`_pc_rsi_code_eta()` for merged code per Anthropic contributor reaching
`_RSI_CODE_TARGET`, `_pc_rsi_auto_eta()` for the AL4 share of Anthropic's model
R&D work reaching `_RSI_AUTO_TARGET` (the app's bar, not the post's — see the
RSI tab section in `docs/takeoff.md`), `_pc_rsi_al5_eta()` for the AL5 share — model R&D with no
human in the loop — reaching `_RSI_AL5_TARGET`, the same 90% bar, which is
**not a fitted series** but the AL4 rung shifted by the ladder's own step (see
the R&D automation section in `docs/takeoff.md`; it is a card only, `_PC_RSI_CARD_ONLY`, outside the projection,
its weights and Takeoff's coding blend, and carries the loudest caveat; its hover adds `_tk_staff_al5_note()`, the AL4-to-AL5 gaps Anthropic's staff survey implies through Takeoff's hours per level, `TestStaffAl5Check`), `_pc_nextstep_eta()` for the detour study's frontier reaching
`_RSI_DIR_TARGET` (that study's own practical ceiling — see the RSI tab section in `docs/takeoff.md`), and `_pc_tasteval_predict_eta()` for TasteVal's experiment-prediction error closing `_TASTEVAL_PREDICT_TARGET`% of its gap to the noise floor (section 3.6). That last is **the paper's own sigmoid and printed dates, not a refit**: its 95% interval is read as an 80% band (the bootstrap holds the four tasks fixed), drawn off a quantile grid and appended last so the other cards keep their seeded draws (`test_tasteval_prediction_card_reproduces_the_papers_dates`). Like AL5 it is card-only (`_PC_RSI_CARD_ONLY`): predicting results is a research skill, not coding; the TasteVal section above it draws its curve undated. They render at most `_PC_CARDS_PER_ROW` (4) to a row — wider and a label truncates mid-word — with the rows **balanced** rather than filled greedily, so eleven cards read 4–4–3 (`TestMilestoneCardLayout`). Each reproduces
its own tab at that tab's defaults — METR: GPT-4o-broken segment, DT over
[DT/2, DT*2], position over the current model's CI, p50 slope fits the trend; RLI: single OLS in
logit space, odds-doubling time over [DT/2, DT*2] floored at 5 days, position
± 1 point; RSI CoBench: single OLS in logit space, odds-doubling time over
`_rsi_dt_ci()`'s widened interval (slow edge `_RSI_SLOW_STEP_MULT`× the slowest frontier
step), position ± `_PC_RSI_POS_CI` around the measured score, not the fitted line; RSI
survey: OLS on log(multiple) over every round the tab fits (the carried-over
`estimated` point included, as on the tab), doubling time over
`_rsi_survey_dt_ci()`'s t-widened interval, position over the
fitted multiple ÷ and × `_RSI_SURVEY_POS_FACTOR`; merged code: the same,
over the quarters from 2025 on, with `_rsi_code_dt_ci()` and
`_RSI_CODE_POS_FACTOR`; R&D automation: single OLS in logit space over the six
labelled months, odds-doubling over `_rsi_auto_dt_ci()`, position over the
figure's own 90% interval on the latest month; AL5: that same draw shifted by a
lag drawn over [censored ladder, `_RSI_AL5_TAU_HI_MULT`x it]; next-step: single OLS in
logit space, odds-doubling over `_rsi_dir_dt_ci()` (widened to span every frontier
step's rate), position around the measured score
± `_RSI_DIR_POS_CI` points (the study's own binomial SE at n=`_RSI_DIR_N`) — rather
than fitting its own, so the tabs can't quote different dates for the same milestone. `test_metr_eta_reproduces_the_metr_tab_defaults`,
`test_rli_eta_reproduces_the_rli_tab_defaults` and
`test_rsi_eta_reproduces_the_rsi_tab_defaults` pin that.

Five of the eleven are dated off *released* models (METR p50 and p80, RLI, the detour
study and TasteVal's prediction fit — publicly benchmarked or run on shipped models); CoBench, the staff survey, merged code, experiment
velocity and both R&D-automation cards (AL4 and AL5) are internal
measurements, driven by work on models the lab has internal access to
before release. So `_pc_report_lag()` pulls the five back
by `_pc_report_lag_days()` (0.7–1.3× the US trained-to-release lag, sampled so the spread lands in the CI) whenever *Milestone dates point at* is not `_PC_TIMING_RELEASE`; the other six
are already on that clock and must not be shifted twice.

A checkbox (`rsi_notyet`, default on, in the blend's *Set your own weights*
expander — below its consumers, so it is read via session state a rerun early,
like the weights themselves) conditions the cards and blend on
"not crossed yet": `_pc_condition_on_today()` rejects samples dating a milestone
at or before today — moving the samplers' own `days_to ≥ 0` truncation from the
(stale) anchor date to the present, by rejection rather than clamping — and the
blend mixes with each weight × survival, which together are exactly the mixture
conditioned on the observation, so a definition that put mass in the ruled-out
region loses credence in proportion. The adjustment is shown, not tabulated as a
probability: the Weight column reads prior → effective share, and the CDF
carries a dotted ghost of the unconditioned blend (`raw_days` through
`_pc_rsi_dist_fig`; its height at today is the mass the update removed — the
grid start widens to the ghost's 0.5th percentile so that stays visible). A
fully-crossed component drops out. A companion number input (`rsi_notyet_ramp`,
default 90d, 0 = off) extends the update into the near future as a soft
likelihood, not a wider hard cut: a sample t days out is kept w.p. t/N inside
the window — the closer a crossing, the more visible its run-up would already
be. On the release clock an active window grows by that range's lower bound
via `_pc_ramp_for()` — a model shipping that soon finished training a
report lag earlier — and the fine print quotes the window actually applied. Survival stays the expected likelihood, so the weight × survival mixing is
unchanged. Assumes a crossing would be known by now — weaker for the internal
evals, and the checkbox help says so.
`test_condition_on_today_truncates_and_reweights`,
`test_condition_ramp_discounts_the_near_future`.

Under the cards sits *Automated coder projection* (`_pc_render_rsi_blend()`):
no single benchmark defines the threshold, so each milestone is treated as one
candidate definition and `_PC_RSI_WEIGHTS` as the credence each gets: 10 for each of the nine
research cards; AL5 is not in the projection at all. The result
is `_pc_rsi_blend()`, the **mixture of their date distributions, not an average
of their medians**, drawn as a CDF (`_pc_rsi_dist_fig()`) between the
cards and the weights table — cumulative rather than a density because a mixture
of ten components is lumpy where they sit and the bin width becomes a
presentation choice, while "X% by date D" is the question the section answers.
Sampled daily with an x-spike so the hover tracks continuously; the axis runs
to the tab's *Project through* year end (`rsi_end_year`, threaded through as
`horizon`; a median past it goes unannotated — the curve ending below 50%
already says so), falling back to the 98th percentile without one.
Each component keeps its own spread, so a late-but-uncertain
milestone widens the blend instead of only shifting it, which an average of the
medians cannot express. Three things are load-bearing. The eta functions take
`samples=True` (via `_pc_eta_out`) so the blend mixes the *same* draw the cards
report, rather than re-rolling and disagreeing with its own table. Weights are
keyed by slug, not by the card label, because the labels are built from the
target constants. And the weights editor's reset **assigns the defaults rather
than popping the keys** — a popped key is re-hydrated straight back out of the
URL on the next run, so on a shared link the reset would never take.

The race half's
headline is the US-vs-China line, so it renders only under the `Country`
attribution; the threshold reaches the display through the chart title.

1. **One roster per attribution, never mixed.** `pc_party` (`_PC_PARTY_OPTIONS` =
   `_DC_PARTY_OPTIONS` + `Country`) picks the entity kind. Tenant/operator race the
   charted companies via `_dc_company_networked_series` at the sidebar's pooling
   level (`pc_pool` reuses `_DC_NETWORK_OPTIONS` verbatim; 'none' → `cluster_of={}`,
   'all' → `None`), with `_dc_hidden_companies` applied and † from
   `_dc_unattributed_companies` — see *Tenant vs operator, and shared tenancy* in `docs/data-centers.md`.
   'Country' races **every** country instead via `_dc_country_steps` (mode `'site'`
   when nothing pools, else `'company'`) on the **unfiltered** site list, China
   listed twice (China-accessible and domestic-only, never plain "China"). Countries
   are never mixed into a company roster, where the US would read as a tenant. The US reference pace for the borrowed trend is passed as
   `ref_steps` (the US country series) since company rosters no longer contain it.
2. **The projection is the by-country model, unchanged.** `_pc_projection()` calls
   `_dc_cty_fit` (since=`_DC_DEFAULTS["dc_cty_since"]`, plan horizon anchored) and
   `_dc_cty_trajectories` (plan slip by `_dc_plan_quality`); paces come from
   `_dc_cty_pace` at its default — China's rows on their evidence band, other
   non-US entities on the US pace — widened by `|g_own − g_ref|/1.282`; short histories re-anchor the US fit
   at their own last step. Don't fork these — drift between the two tabs' cones reads
   as a bug.
3. **Crossing math is per-sample first-hit.** `_pc_crossing_idx` returns the first
   grid index reaching T with `len(grid)` as the never-crossed sentinel (NaN never
   hits); per sample it is non-decreasing in T, so every percentile is too —
   `test_crossing_idx_monotone_in_threshold`. `_pc_idx_date` maps a percentile back
   to a grid date, `None` past the grid (rendered as past the horizon). The grid runs monthly
   from today to `_PC_HORIZON`. An entity whose catalogued steps cross before today
   is "already there" via `_pc_plan_crossing`, not the Monte Carlo. A *Date points
   at* selectbox (`pc_timing`, `_DC_TIMING_OPTIONS` verbatim) shifts the input
   series by `_dc_timing_shift(label, run_days)` before rows are built, exactly as
   the DC tab does — default is run-finished (capacity-online, or +30d for
   release, one click away); the fine print states which milestone the dates mean.

The levers open with *Pace the US* (`pc_pace`, `_PC_PACE_OPTIONS` 100→0, default 100 = no
pacing, 0 = the pause): from the pacing date the US runs its path on `_pc_paced_clock`, `pace`
years a year (plus up to `_PC_WIN_MAX` *Pacing windows* under *Advanced*, `pc_win{i}_pace/start/end`, each its own pace from start to end, overriding the main lever while it runs (so a pause can resume), the slowest winning where windows overlap; they move only the US path, while the pacing date still sets the snapshot, the race bar and the from-pacing-start levers; `test_pacing_windows_slow_the_us_then_return_it_to_normal`, `test_a_pacing_window_slows_the_us_until_it_ends`), each change drawn as a thin purple rule labelled "pause", "N× slower" or "no pacing" (`_pc_pace_changes`, `_pc_pace_rule_label`) beside a dashed unpaced-US line; the pacing date sliders run to `_PC_PAUSE_LAST_YM` (Dec 2035)), so the bar China crosses moves (`_cc_cn_crossing_sim(us_path=…)` returns trained
and released paths; released trails trained by the ship lag, which reproduces the old capped
climb at 0). Crossing dates are percentiles over every sample, a never-crossed one counting as
past the window; under half crossing, the cards read *After <year>* with the share that
does and the breakdown is skipped. Which milestone the dates mean is in the crossing card's
hover (`test_pacing_slows_the_us_and_delays_china`). Pause-scenario tests set
`pc_pace` to 0 in `TestPacingTab._app`. The pacing date and `_pc_render_us_pause()` follow. The US path is the Takeoff tab's: `_pc_takeoff_us` reads `_tk_run`
(seeded, so it neither depends on nor shifts this tab's stream) and each sample follows one
Takeoff scenario's ECI path (`_pc_tk_interp`), the trained bar a ship lag ahead; the chart
marks where the paced US reaches each scenario's ECI at automated coding and at RSI
(`us_ms`). Without a Takeoff run the compute-derived straight climb stands in. China gets automated coding and RSI
at the same ECI: `_pc_speedup_table` reads each scenario's ECI/yr where its path first stands
at an ECI, over today's trend (floored at 1), and `_cc_cn_crossing_sim(speedup=…)` multiplies
China's whole rate by it at China's current ECI, the extra kept as its own `speedup` channel
(a breakdown row) so the channels still sum to the path; `pc_tk_follow` (default 0.8, *Advanced*) sets how much of that extra is the US's own AI-driven ECI/yr at that level rather than a multiple of China's slower rate, `speedup_follow` in the sim (`test_full_tracking_gives_china_the_us_extra_in_eci_a_year`). The extra then scales by China's installed compute under the chip levers over the
no-lever path, to the Takeoff parallelization exponent (`tk_parallelization`, 0.5), since
AI researchers run on compute (`_cn_src_compute_ratio`, `speedup_scale`;
`test_less_compute_shrinks_only_the_speedup_channel`). Under the chart a table dates automated coder, RSI and
ultraintelligence (`_TK_ULTRA_ECI`) for both sides from `ms_years`, the same per-sample first
crossings the chart's marks take medians of (`test_milestone_table_dates_both_sides`)
(`test_speedup_multiplies_chinas_rate_and_is_its_own_channel`). It is capability-indexed and
unpaced: pacing slows the US, not what AI at a given ECI does for research. `pc_pause_mo` and
the `_pc_capacity_at()` reading it implies are rendered by `render_pacing()`, not
by the panel, because the race below needs the same bar; the panel takes
`pause_d` and `caps` as arguments. The US pauses on that date, frozen at whatever
its compute-derived climb reached by then,
per sample (`us_pause_level` and the sim target are arrays — only the climb is
uncertain, the date is chosen). The control is a `select_slider` over
months-from-today with `format_func` labelling each position with the date
(`_pc_add_months`, whole calendar months so no label repeats): it reads as a date
picker while the stored value stays an int, which round-trips through the URL and
can never go stale the way the scenario cut-off *labels* can. With no pacing (pace 100%, no windows) a greyed
stand-in shows today, which the snapshot and race then read, and `pc_withhold` is disabled
and ignored; `pc_pause_mo` stays in state for when pacing returns
(`test_the_pacing_date_and_model_access_grey_out_with_no_pacing`). The default is
`_pc_pause_default_mo()`, the offset to `_PC_PAUSE_DEFAULT_YM` (Jul 2027) rather
than a hardcoded month count, so the reset restores a *date*. The pause is on the
**run-finished** clock; the displayed date carries `timing_label`'s
offset (each side's own release lag on *Model release*). The US climb pace
still comes from the **sidebar-pooled US series**, so the networking selector moves
the bar, scaled by the China tab's US calibration (`_cc_rate_calibration`). Just above *Compute Thresholds* sits the *state of play* table (the panel hands its rows back through `state_out`), off
that same `_pc_capacity_at()` reading (the race's own `_pc_projection` at the pause
date, so the two sections cannot disagree) — three rows (US, China-accessible,
China domestic-only) of each side's
largest training run, frontier ECI (US = the sim's bar, China = its own sampled
paths at that date, the domestic row those paths less the compute the sites
abroad bought — `chan['compute'] - chan['compute_domestic']`, the same shadow term
the breakdown prices, so the two Chinese rows cannot disagree), the ECI→METR
p50/p80 horizons (capped at `_PC_METR_CAP_HRS`,
past which the bridge reads in centuries) and China's lag in months, taken off the
same US line the chart draws. It renders **before** the crossing bail, so a
projection range too narrow for a catch-up still describes the pause. A caption sensitivity reruns China's compute term
at the catalogued China-accessible pace (sites abroad) vs the export-control band.
The bar is the best model the US has *trained* by the pause: the Takeoff path (already
trained), or without one the released-frontier climb (from us_best's release date) plus
the US trained-to-release lag of extra climb. The policy levers below (and their *Advanced* expander) draw in the sidebar, under *Policy levers* above *Compute settings*, through the `levers_box` slot `render_pacing` makes. The US level the distillation gap starts from is the US trend at China's
anchor date (`us_anchor_cn`), not the US's later best. Three scenario checkboxes plus a slowdown slider: `pc_withhold` (default off; label says "from pacing start") — a US
*release freeze from pause-run start*: nothing ships once the final run begins, so
distillation's teacher is the released frontier as of run start (`dist_teacher`, ~one
run's climb + ship lag below the bar); labs do ship during big runs, so this is a
policy assumption, not a pipeline fact — hence the label says freeze;
`pc_stop_dist` and `pc_stop_remote` (default off) — cut the
distillation channel at today (`t_dist_stop` in `_cc_cn_crossing_sim`) to `dist_keep` of
its rate, set by `pc_dist_cut` (share of harvested data stopped, default 95%) through
`_pc_dist_keep`: ~25% less per 10× less data (IFP), so only 100% stops it
(`test_distillation_keeps_a_quarter_less_per_tenfold_data_cut`); and cut remote
compute as a *level setback* (`comp_dead`): China's largest run falls back to its
biggest domestic cluster, the compute term is dead until the domestic buildout regrows
the lost OOMs at its catalogued pace, domestic band thereafter (also suppresses the
China-accessible sensitivity, whose premise it removes). The pace band deliberately
stays on the default compute band — it is shared with the CC crossing.
Beside them, `pc_duv_ban` and `pc_duv_service` (default off) compound the CTS-derived
DUVi import and servicing cuts into it (see *lithography ceiling* above), as do
`pc_cut_smuggling` (0–100%), `pc_stop_legal` and `pc_hbm` (IFP's foreign-HBM controls,
`_PC_HBM_CUT` = 9% of domestic output), from their own *Advanced* dates
(`pc_smug_when` / `pc_legal_when` / `pc_hbm_when`, default *Now*): `_cn_src_channel_cut`
reads the China tab's compute mix (`_cn_src_projection`) for the share of China's in-China
compute growth that channel supplies to the horizon
(`test_smuggling_and_legal_sales_levers_stack_on_the_slowdown`). Under *Advanced* (*Any other slowdown*), `pc_dom_slow` (0–90%, default 0) slows China's *own* buildout further, for whatever the chip levers don't cover —
equipment/fab controls rather than access controls. The cut comes off the **domestic
component** of each band, not the band itself: `g_hi_eff -= g_dom_hi·p`, likewise for
`lo`, so the export-control band keeps whatever access abroad adds on top while a run
already confined to domestic clusters (remote cut on) takes the cut in full, and the
two levers stack instead of one subsuming the other. `g_dom` is slowed too, so the
remote cut's setback takes proportionally longer to regrow, and the breakdown's
`comp_shadow` cap falls with it. Without a catalogued domestic pace to decompose
against, the whole band scales by `1−p`.
An *Advanced* expander dates the two cuts (`pc_dist_when` / `pc_remote_when`,
`_pc_when_options` month labels — strings, so they round-trip through the URL and
reset to a constant `Now` = the checkbox alone; a stale label is dropped like
`pc_pause_mo`). Sliders are `disabled` without their checkbox. It also carries
`pc_cn_run` (2–12 months, min = the bar's own run length, default = it): China need
not match the length the bar was set at. A longer run is a one-off **level** move,
never a faster rate — ×(L/L_us) compute into one model, worth `a_partial` per ×10 —
paid for with L−L_us months of wall clock, so China's whole deliverable path lifts by
`cn_gain` and shifts right by `cn_extra`. The organic sim is untouched (algorithmic
channels don't speed up because a run is longer); the lift is applied by re-reading
the *same* sampled paths against a bar lowered by `cn_gain` via `_pc_cross_years`,
which must reproduce the sim's own interpolated crossings exactly — that pairing is
what makes the stated net free of MC noise. Live the answer is a U-curve with an
interior optimum (a few months beat both matching and a year), which is the point of the control.
A dated remote cut
must not slow the years *before* it: the band stays on the export-control default
until the cut and `comp_slow=(t_cut, a_partial·g_domestic)` takes over after, while
the setback is sized at the cut date (today's OOM gap widened by `(g_mid − g_dom)`
per year of lead), so a later cut costs more to regrow but is never worse for China
overall. Cutting *at* today keeps the old global band cap instead. Above the milestone table under the chart
sits an `st.warning` that it assumes ~SL5 secure weights, shown while `pc_theft_weights` is
off. *US software reaching China* is one channel, `diffusion`, its breakdown row *US software,
leaked or stolen*, cut two ways (both sliders default 0). `pc_theft_share` is the share of US
research stolen from `pc_theft_when`: China jumps by that share of the gap today's ~0.8 OOM run
gap does not explain, plus `a_partial`·log10 of the training compute `pc_theft_realloc` (2.5x,
`_PC_THEFT_REALLOC`) frees, then on the stolen share moves at the faster of its own algorithmic
channels and the US's ECI/yr times its software share (`theft_soft` in the sim). `pc_leak_cut`
stops that share of the open leak (papers, talent; ~31% of US software progress today, stashed
as `_pc_leak_share` for its help) from `pc_leak_when`, fading over `pc_leak_absorb` years
(default `_CC_DIFF_ABSORB_YRS`) to the share left (`test_methods_toggle_keeps_diffusion_running`).
The leak runs on the share not stolen, so it greys out at 100% stolen
(`test_theft_levers_speed_china_and_lift_the_sl5_caveat`, `test_theft_only_ever_helps_china`,
`test_stolen_software_lands_with_the_leak`). `pc_theft_weights` lifts China to the US trained level
at `pc_wt_when`, once or monthly (`pc_wt_mode`), in its own *Theft* row; §10 of
`CHINA_COMPUTE_JUDGMENTS.md`. The leak also follows the US: at each
year it is the paced US's ECI/yr on the Takeoff path (AI speed-up included) over today's
trend, times the share the leak lets through, averaged over that same absorption window
(`_diff_flow`, passed as `diff_flow`; the pace alone without a Takeoff run), so a pause
dries it up with the lever off, the two never stack, and a speeding US leaks faster
(`test_the_leak_follows_the_us_ai_speed_up`)
(`test_a_pause_dries_up_diffusion_without_the_leak_lever`). `pc_sec_sched` (*Defenses tighten over time*, default off) multiplies
distillation, the leak and the share stolen by `_pc_sec_keep`: a six-month sprint removes `pc_sec_sprint`% (50),
then the rest halves every `pc_sec_halve` years (1); `pc_sec_on` limits it to distillation or US software
(`test_defenses_can_cover_software_theft`).
`_diff_flow` integrates it month by month over the absorption window
(`test_defenses_that_tighten_only_ever_slow_china`; judgment in `CHINA_COMPUTE_JUDGMENTS.md` §9). Each lever's state in
words (distillation teacher, cut dates, compute band) sits in the "these settings"
hover under the table. The panel adds **no release-queue
wait** (unlike the CC 161 section — a queue delays both sides alike); dates follow the
sidebar's *Date points at* (`off_us`, `off_us_rel`, `off_cn`; see *Trained-to-release
lags* in `docs/conventions.md`); the
chart (actual points included) rides the same clock, x-axis titled by milestone.
The three crossing cards (date, time to surpass, and the US lead in total installed compute at the crossing or the horizon's end, `_pc_compute_lead`, the ratio table's own installed reading: mainland plus the compute mix's remote stock abroad (`_pc_cn_abroad`), so only the chip and remote-access levers and the crossing date move it, `test_compute_lead_counts_china_abroad_until_remote_access_is_cut`; `test_crossing_cards_sit_above_the_milestone_table`) sit above the SL5 warning and the milestone table, the pacing settings in the first card's hover; the panel ends with `_pc_render_why()`, *Why this US-China gap?*: each source's ECI/yr at the pause,
its average from the pause to the crossing, and at the crossing (`_pc_rate_at`), then a row
turning each total into months to close the gap; the average's months are the actual
catch-up (before the card's one-month release lag). Each source's average is its own median post-pause ECI over
the span; the medians fall short of the median run's gap, and an *Other* row
(`_PC_WHY_OTHER`) carries the difference so rows add to the total. Distillation reads
"Blocked" at the crossing when withholding or the stop lever has dried it up. The compute
split (`comp_shadow`, a domestic-only counterfactual) has no rate of its own, so the
measured compute rate shows once, on the first compute row
(`test_why_states_the_gap_and_the_slowing_rate`, `test_why_breakdown_tracks_the_sliders`).
Under it, drawn with or without a crossing (`test_compute_ratios_show_without_a_crossing`), `_pc_render_compute_ratios()` (*largest run, trained and trainable*: US clusters keep growing at `g_us` while its largest trained run grows at `pace` of that, frozen if paused; the gap is an overhang row) charts and tabulates how far ahead the US is
(US ÷ China, log axis, to *Project through*): largest trainable run (cluster capacity,
`_pc_capacity_at`, read for today and the pause off one projection via `also=`) for China's
usable run and its domestic clusters, against a US run frozen at the pause; and total
installed compute in millions of H100e, the Data Centers tab's world shares of Epoch's chip-sales
total, 31.7M H100e by June 2026 (`_WC_WORLD_H100E`, grown by `_wc_world_h100e`; ratios `_pc_wc_ratio`), the US still
installing through the pause and the DUV/domestic levers slowing China's. Dashed lines and a
last table column give the same date with no compute levers. The table (`Compares` column)
gives one row per comparison, each cell both values then the ratio, columns dated by the
pacing start and the crossing card's date (end of *Project through* when China does not catch up; no pacing, a bare date); no caption sits under it. Capacity is not the China
tab's ~50× *actual-run* gap (`test_compute_ratios_show_what_the_compute_levers_cost_china`). Under it, `_pc_render_compute_mix()` stacks China's installed compute by source
(the China tab's mix) under these levers, `_cn_src_with_policy`, against the no-lever
total (`test_compute_mix_shows_what_the_chip_levers_remove`).
Note for tests: this table renders *after* the race table, so address
the race table by its columns (`TestPacingTab._entities`), never by position.
China
races the paused frontier in an ECI chart (US kink + fan + crossing diamond); the
paused stock stays distillable while a gap remains, then China runs on its
indigenous band plus its export-control-bound compute term. Projections are dashed; the dotted *US, no pacing* and *China if the US kept going*
are the counterfactuals, drawn only where the settings move a country off them, China's
from the same sim and draws with the US unpaced and every lever off
(`test_the_dotted_lines_show_only_what_the_settings_changed`). A *Project through*
dropdown (`pc_end_year`, options 2027–2035, default 2031) sets the crossing-search horizon and — like every
other tab's *Project through* — the x-axis of **both** charts: the timeline runs to the
grid end (+120d pad) instead of auto-fitting to content, and the pause panel takes
`horizon=` (sim `horizon_yrs` from China's anchor, floored at 1yr) so its fan ends on the
year the P(cross by …) column is quoted for.

*What compute buys* (`pc_payoff`, under *Advanced*) sets the catch-up's ECI per 10×
training compute (`_pc_compute_payoff`): our frontier-grade fit (~6.3), IFP's 10.8 months
of US progress (~14), or their mean, the default. Channels are rescaled to China's observed
slope, so it moves what the chip levers buy, not the no-lever crossing
(`test_compute_payoff_default_is_the_mean_of_ours_and_ifps`).

Guarded by `TestPacing` (unit) and `TestPacingTab` (integration).
