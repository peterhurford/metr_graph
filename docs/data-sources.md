# Data sources and how to update

Every feed, where it comes from and how to refresh it. The step-by-step refresh is `.claude/commands/update_data.md`.

## Data Sources and How to Update

Six external feeds; the rest are derived. `.claude/commands/update_data.md` has the full
recipe, including the AISI cyber data deliberately *not* ingested.

| File / table | Source | How to refresh |
|---|---|---|
| `benchmark_results_1_1.yaml` | METR | Overwrite from `https://metr.org/assets/benchmark_results_1_1.yaml` |
| `epoch_capabilities_index.csv` | Epoch AI | **Converted, not extracted**: `python3 convert_eci.py eci_scores.csv all_ai_models.csv epoch_capabilities_index.csv` from `https://epoch.ai/data/eci_scores.csv` and `https://epoch.ai/data/all_ai_models.csv`. One row per model; training compute joins by name, with `ALIASES` for renames (`test_convert_eci.py`; the script prints stale ones). Epoch recomputes scores live, so existing rows drift on each pull |
| `data_centers.csv` | Epoch AI | Overwrite from `https://epoch.ai/data/data_centers/data_centers.csv` |
| `data_center_timelines.csv` | Epoch AI | Same, `…/data_center_timelines.csv`. Column order varies between pulls; the loader uses `DictReader`, so that's safe. **One curated deletion — see below** |
| `_RLI_RAW` (hardcoded) | Scale Labs RLI leaderboard (`labs.scale.com/leaderboard/rli`) / `remotelabor.ai` | Hand-edit rows |
| `_RSI_RAW` (hardcoded) | Anthropic, Redacted Risk Report (Aug 2026), §3.4.3 Fig 3.4.3.A (`_RSI_SOURCE_URL`) | **Not downloadable** — scores read off the figure, Anthropic prints no table. Hand-edit rows |
| `_RSI_DIR_RAW` (hardcoded) | Anthropic, [*When AI builds itself*](https://www.anthropic.com/institute/recursive-self-improvement) (`_RSI_DIR_SOURCE_URL`) | **Not downloadable** — the figure prints its own bar values, so the rows are read off the labels, not pixel-digitized. Dates are the models' Epoch-catalogued release dates (`test_dates_are_the_published_release_dates`), not the figure's row order, which is not chronological. Hand-edit rows |
| `_TASTEVAL_*` (hardcoded) | P-Zero Research, [*TasteVal*](https://arxiv.org/abs/2610.06824) (`_TASTEVAL_SOURCE_URL`) | **Not downloadable**, tasks private: the break's ECI-per-doubling rates (Figure 11), section 3.6's 90% date and CI, and Table 2's multipliers (`_TASTEVAL_RAW`, main setting per model, Epoch release dates), transcribed from the text; the prediction sigmoid's midpoint and rate are read off Figure 7's vector path (`mase_release.svg`). `TestTasteVal` checks the fit reproduces the paper's 14.0 and 3.0 months a doubling and the sigmoid its October 2030. Hand-edit when a version reports new models |
| `_RSI_CODE_RAW` (hardcoded) | Same post's *Code contributed per person, by quarter* figure (`_RSI_CODE_SOURCE_URL`) | **Not downloadable** — the figure labels its bars only from 2025Q1 on; the earlier ones are read off the axis. Their mean must come back at ~1 (`test_pre_2025_bars_average_to_the_baseline_they_define`) and the post states the 2026Q2 figure in prose, which pins the other end. Hand-edit rows |
| `anthropic_rd_automation.csv` | Anthropic, [*Measuring the pace of AI development*](https://www.anthropic.com/institute/measuring-pace-of-ai-development) (`_RSI_AUTO_SOURCE_URL`) | **Not downloadable** — the R&D Automation Index, as **cumulative** shares (`al3_plus` = “rated AL3 or above”, the stacked figure's band boundary). The six AL4 months the figure labels are *transcribed*; every other number is pixel-digitized from the chart asset the page serves. Two calibration guards: each 90% interval brackets its own printed label, and `al4_dig` — the AL4 boundary read the same way as the unlabelled rungs — matches the printed `al4_pct` within ~0.6pt (`test_digitised_al4_reproduces_the_printed_labels`), which is what validates the method on the rungs that have no printed truth. **AL5 has no column**: it is absent in every month, and that absence is used as a censoring bound, not as a zero to fit. Refreshing is a re-read of the figure when Anthropic re-versions the index — the version is in `_RSI_AUTO_VERSION` and the file's `#` header. Hand-edit rows |
| `openai_experiment_velocity.csv` | OpenAI research-acceleration [post](https://openai.com/index/research-acceleration-view-inside-openai/) (`_RSI_EXPERIMENT_SOURCE_URL`) | **Not downloadable**: the *Experiment velocity* chart's tooltip values, transcribed by hand (four-week trailing experiments per active experimenter, 2025 = 1x); method and caveats in the file's `#` header. Hand-edit rows |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (hardcoded) | Press reports | Hand-edit `(date, ARR_in_billions)` tuples |
| `_CN_REVENUE` (hardcoded) | Company results calls/filings and press | Hand-edit `(date, $B, basis, note)` points. Run rates only: never annualize audited period revenue into it, and for Alibaba take the MaaS ARR, not "AI-related product revenue" (includes GPU rental) |
| `cts_duv_production.csv` | CTS, [*DUV Immersion Lithography*](https://techstatecraft.org/duv) (`_DUV_SOURCE_URL`) | **Not downloadable** as a file, but exact: the values are transcribed from the base64 chart script the page embeds after Figure 1, not digitized. `test_transcription_matches_the_reports_own_2035_labels` checks them against the figure's aria-label prose. Hand-edit rows |
| `china_compute_sources.csv` | Epoch [chip owners](https://epoch.ai/data/ai-chip-owners), Epoch's [smuggling estimate](https://epoch.ai/publications/chip-smuggling), SemiAnalysis's offshore leased power, [IFP](https://ifp.org/how-to-increase-americas-ai-lead-over-china/) for 2026–27 | Rebuild 2022–25 domestic/legal rows from Epoch's owners zip (`China` rows by year); 2026–27 rows transcribe IFP's channel stocks, flows as stock differences. The `basis` column sets the styling (ours hatched, projections faded). Keep stocks the running sum of flows, except smuggling (`test_stocks_are_the_running_sum_of_flows`) |
| `aisi_cyber_tlo.csv` | UK AISI Figure 2 + [Kimi K3 assessment](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) | **Not downloadable** — 9 rows digitized from `fig2-ranges.png`, one value quoted from prose. Calibration and validation checks are in the file's `#` header, guarded by `TestUkCyberTlo`. Dates are **published release dates**; the figure's x-axis is tokens |
| `aisi_cyber_narrow.csv` | UK AISI [open-weight cyber gap post](https://www.aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber) | **Not downloadable** — AISI publishes no numbers; values digitized from `fig1-narrow.png` by pixel analysis. Refreshing is a *figure-unchanged check*: re-fetch the PNG, confirm gridline rows and marker colours still match, re-digitize only if the figure changed. `test_digitized_dates_match_known_releases` and `test_optimistic_bracket_reproduces_aisi_published_lags` are the calibration guards. Hand-editing a row is fine if AISI states a number in prose |

Before overwriting a CSV wholesale, diff by key column so no curated rows are lost (ECI = `Model version`; DC metadata = `Name`; timelines = `Data center` + `Date`). After any data change, sanity-check the relevant loader via `_VP_TESTING=1 python3 -c "import visualize_projection as v; ..."`, then run the tests.

### Timeline rows with no metadata row

`load_data_centers()` is **timelines-driven** (metadata only supplies the company label), so a
timeline name Epoch never catalogues materializes a site with no owner and no country — and if it
duplicates a catalogued one, it double-counts it. Check
`set(timelines['Data center']) - set(metadata['Name'])` on every refresh.

Empty as of the 2026-09-29 pull. If a re-scoped site leaves stale timeline rows covering capacity
now catalogued under a new name, delete them locally until Epoch fixes the export.
