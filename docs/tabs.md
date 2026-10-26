# Tabs, and the smaller tabs' design records

The full tab table, plus the UK Cyber, ECI organization-matching and Frontier Thresholds records.

## Tabs and Render Functions

| Tab | Function | Data Source | Metric |
|---|---|---|---|
| METR Horizon | `render_metr()` | `benchmark_results_1_1.yaml` → `load_frontier()` | log₂(minutes) |
| ECI | `render_eci()` | `epoch_capabilities_index.csv` → `load_eci_frontier()` | linear score: the projection, whose data-driven CIs reset when the projected frontier changes (`test_a_new_entity_takes_its_own_rate_ci`) and whose linear rate CI defaults to the frontier's slowest and fastest year on record, each window run to the first release a year on so a wait between releases is not read as a slow year (`_eci_yearly_gain_range`, `test_rate_ci_defaults_to_the_slowest_and_fastest_year_on_record`; half to twice the fit under `_ECI_MIN_GAIN_WINDOWS` (6) year-long windows, where overlapping years pin it too tight, or on a piecewise segment), the fan's median held at the fit's own rate so the projection carries the trend on without a kink (`_split_from_ci`, `test_projection_carries_the_trend_on_without_a_kink`), *months behind* (`_render_eci_gap_history`), then *Where each lab stands* (`_render_eci_leaderboard`), all three driven by *Entity* (the reference) and *Compare to* (the subject). The gap views read each entity's full-window frontier (`_eci_gap_frontier`); *Frontier (any lab)* (`_ECI_FRONTIER_LABEL`) as *Entity* reproduces the old Company Gap tab. From the subject's latest release the months-behind chart carries a band off the projection's own sampled paths (`_render_eci_tab` returns them; `_eci_projected_gap`): the release's gap, plus a month a month, less each path's ECI gain at the reference's own projected median rate (`_eci_ref_rate`, the same centring as the subject's band, so A-vs-B and B-vs-A drift opposite ways: `test_two_labs_cannot_both_fall_behind_each_other`), so the median is a straight line rather than following the reference's frontier steps (`test_projected_gap_runs_from_the_release_gap_at_the_reference_rate`). Each release's gap (`_eci_gap_history`) is the months since the reference first *released* a model at or above its score, never a level interpolated between two sparse releases, so no lab trails one inside the frontier by more than the frontier (`test_a_lab_never_trails_another_by_more_than_the_frontier_holding_it`). The gap now (`_eci_gap_now`, the table's one column) is the ECI gap to the reference's best in months at that rate, since months since a release grow with the model's age (`test_behind_now_is_the_eci_gap_at_the_reference_rate_not_the_models_age`). Both views name the reference in full (`_eci_ref_name`: "US best", never "US", which reads as excluding the US labs in it). On the projection only frontier points are named, alternating above-left and below-right of the rising trend (`_eci_label_positions`), and right-edge labels closer than ~13px stack into one (`_eci_edge_label_groups`; `test_a_model_is_named_once_and_close_edge_labels_stack`) |
| Remote Labor Index | `render_rli()` | `_RLI_RAW` → `load_rli_data()` | logit-transformed score |
| Takeoff, evidence half (was the RSI tab) | `render_rsi(end_year)` | `_RSI_RAW` → `load_rsi_data()`; `anthropic_rd_automation.csv` → `load_rsi_automation()`; `_RSI_SURVEY`; `_RSI_CODE_RAW` → `load_rsi_code()`; `_RSI_DIR_RAW` → `load_rsi_direction()`; `openai_experiment_velocity.csv` → `load_rsi_experiments()`; `_TASTEVAL_RAW` → `load_tasteval()` | CoBench score % (logit-projected), AL2+/AL3+/AL4+ shares of R&D tasks % (logit-projected) plus AL5 shifted off their lag ladder, all on one chart, staff speedup ×, merged code per contributor ×, OpenAI experiments per experimenter ×, next-step win rate %, TasteVal compute multiplier × and prediction error (MASE) |
| UK Cyber | `render_ukcyber()` | `aisi_cyber_narrow.csv` → `load_ukcyber()`; `aisi_cyber_tlo.csv` → `load_ukcyber_tlo()` | success rate % + open-weight lag in months; plus a TLO cyber-range cross-check in steps (`_render_ukcyber_tlo()`) and a callout for models only the range has measured (`_render_ukcyber_newest_open()`) |
| Revenue | `render_revenue()` | `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE`; `_CN_REVENUE` | ARR in billions; optional summed line (`rev_combined`, off by default) via `_rev_combined_series()`; then a data-only Chinese-lab chart (`_render_rev_china()`: MiniMax, Zhipu API, Alibaba MaaS, Moonshot, DeepSeek), run rates only, each point classed company-stated or press-reported (`TestRevenueChina`); closes on US vs China (`_render_rev_us_china()`): each side summed by `_rev_sum_series` (China from the date every lab reports, each lab held flat past its last report), with the ratio and the date the US labs stood at China's total, then the top-N vs top-N ratio over time (`_rev_ratio_path`), held flat and nowcast to today at each lab's last-interval growth (`_rev_value_nowcast`) (`TestRevenueUsChina`) |
| Employment | `render_employment()` | RLI frontier + slider assumptions | unemployment % / jobs lost |
| Data Centers | `render_data_centers()` | `data_centers.csv` + timelines → `load_data_centers()` | H100-equiv / power / cost; carries a US-vs-China by-country projection (`_dc_render_country_panel()`), the world-share charts (`_render_world_shares()`), and ends with *Compute ⟷ ECI* and *ECI Forecasts* (`_render_compute_eci()`, see below) |
| China (slug `china`, was Compute/capabilities/diffusion) | `render_compute_capabilities()` | data centers (`dc_all`) + ECI | US vs China through the compute lens, then China's compute mix by source 2022–2030 (`_render_cc_compute_sources()`: projections faded, our own numbers hatched; `TestChinaComputeSources`), and ends with the CTS lithography ceiling (`_render_cc_duv_ceiling()`) |
| Pacing | `render_pacing()` | data centers (`dc_all`) | China's catch-up to a US pause, then the date each entity first commands a run of the paused US scale |
| Frontier Thresholds | `frontier_thresholds.render(st)` | `frontier_thresholds.json` | dated threshold events per lab and risk category; no projection (see *Frontier Thresholds* below) |

## UK Cyber tab caveats

Five load-bearing constraints; don't "simplify" them away:

1. **The frontier is closed-weight only** — open-weight models are the subject being measured against it, so `load_ukcyber()` excludes them from the running max.
2. **Lag is interpolated between the bracketing frontier models and carries a bracket.** `_ukc_frontier_crossing()` interpolates between the last model below a score and the first above; `ukc_lag_rows()` returns `lag_months` plus `lag_lo`/`lag_hi` from those two. Snapping to the next model up (the earlier implementation) equates scores that can be far apart, understating a lag across a wide frontier gap. Nothing shipped inside the gaps, so the bracket is real uncertainty, not noise. (Rejected: last-model-below, defensible but maximally pessimistic; OLS-curve dating, which imports global fit error into a local question.)
3. **`lag_lo` is the AISI-compatible bound.** AISI's printed annotations use next-model-up, exactly `lag_lo`, so `test_optimistic_bracket_reproduces_aisi_published_lags` doubles as the calibration check on the digitization. Changing the lag method must not break it.
4. **The TLO cross-check compares on `lag_lo`, not the point estimate.** `aisi_cyber_tlo.csv` reuses the same helpers by storing steps in `cyber_score`, but the two datasets' frontiers differ in density, so point estimates aren't comparable — a score inside a wide gap gets inflated, one below every frontier model gets no interpolation at all (`lag_hi` is `None`, a lower bound) — and the ordering inverts for sampling reasons rather than capability. On `lag_lo` the two reproduce AISI's published "4 to 7 months" headline exactly. `test_reproduces_both_figure_titles` guards this.
5. **The suites don't cover the same models, and a callout says so.** Kimi K3 got a selective set (ExploitBench + TLO), not the 70-task narrow suite, so it's missing from the chart, projection and 90% ETA by data availability, not by a filter. `ukc_open_only_on_tlo()` finds the newest open-weight model in TLO but not narrow rows; `_render_ukcyber_newest_open()` shows it as a bordered callout with the `lag_lo`–`lag_hi` bracket (TLO's frontier is too sparse for a point estimate). It returns `None` once narrow catches up, and ignores a TLO-only model *older* than every open-weight point on the chart. Both it and `_render_ukcyber_tlo()` read `ukc_tlo_lag_rows()`, so they can't drift onto different frontiers. Guarded by `TestUkCyberOpenOnlyOnTlo`.

Country and openness are perfectly confounded (no US open-weight, no Chinese closed-weight models). The tab says "China" because the two Chinese models are also the only open-weight ones; `_UKC_CONFOUND_PLAIN` is folded into the fine-print caption so that's stated wherever it does. If it ever needs banner prominence again, promote that same constant rather than writing a second wording to keep in sync.

`ukc_target_eta()` ("when do open-weight models reach `_UKC_TARGET`") = the frontier's interpolated crossing plus the min/max measured lag. `ukc_target_eta_direct()` fits the open-weight points themselves as a cross-check only — they sit close together in time, which makes that slope very sensitive, so it is never the headline.

## ECI tab — organization matching

Every per-company view resolves a company by *substring* match on Epoch's `Organization`
field, through `load_eci_frontier(orgs=…)` / `_ECI_ENTITY_SPECS`. Don't make it an
exact-key lookup: Epoch spells Google several different ways, so a map keyed on one spelling
drops models and draws a different frontier point.

**Adding a company is one row in `_ECI_COMPANIES`.** `_ECG_COLORS`, `_ECG_COUNTRY`,
`_ECI_ENTITY_SPECS` and `_ECI_ENTITY_SLUG` are derived from it at import — don't hand-write
them. `_ECI_COUNTRY_ENTITIES` stays separate (the "best" aggregates and *Frontier (any lab)*
are country filters, not companies) and must come first, since `_ECI_ENTITY_OPTIONS[0]` is
the tab's default.

Keep each company's `orgs` list minimal — substring matching makes longer variants redundant
("Google" already catches "Google DeepMind,Google"). Matching is unambiguous only while no
`Organization` string names two companies; `TestEciOrgMatching` asserts that against the live
CSV, plus the Google spellings and slug round-tripping.

**One frontier point per release date.** `_best_per_date()` reduces a date's models to
its best-scoring one *before* the running max, so a same-day pair puts only the flagship
on the frontier (GPT-5.4 Pro, not GPT-5.4). Otherwise both clear the max, planting a
vertical pair in the OLS whose membership depends on CSV row order. Every ECI frontier goes through it: `load_eci_frontier`,
`load_eci_compute`, `_cc_country_frontier` and `_cc_company_frontier_models`.
`test_one_frontier_model_per_date` and `test_frontier_does_not_depend_on_row_order`
guard it.

## Frontier Thresholds

`?tab=thresholds` dispatches to `frontier_thresholds.render(st)`. The 55 events in
`frontier_thresholds.json` transcribe the user-supplied September 6, 2026 timeline;
they are not independently source-verified. Keep model/configuration, domains,
framework version, assessment type, date qualifications, and safeguards together.
Do not map the lab-specific labels onto a common severity axis. The default view selects one risk category (`ft_category`) and renders separate
bordered lab panels. `CATEGORY_MILESTONES` supplies curated domain-specific labels
and hover text; never reuse a mixed-domain threshold string in these panels.
Charts explicitly disable rangesliders and legends. `CHART_LABEL_IDS` keeps labels
sparse while additional milestones get smaller points. Each panel shows a compact
table and a selectable, visible explanation. The full chronology is collapsed.


The `ft_` defaults participate in URL hydration/sync. The timeline deliberately
bypasses `_tf`: the shared Today forward projection control must not hide historical
events. `test_frontier_thresholds.py` checks semantic distinctions, overlapping
reassessments, CSV exports, URL filters, reset, empty states, and tab switching.
