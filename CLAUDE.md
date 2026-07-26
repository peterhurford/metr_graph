# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run visualize_projection.py
```

Opens at http://localhost:8501.

## Running Tests

```bash
pytest -v                                    # all tests
pytest test_visualize_projection.py -v       # unit tests (fast, uses fake Streamlit)
pytest test_integration.py -v                # integration tests (slower, uses Streamlit AppTest)
pytest test_visualize_projection.py::TestPretty::test_known_name -v  # single test
```

Unit tests use a fake Streamlit module (`_FakeStreamlit` / `_Noop`) so the app can be imported without a running server. The env var `_VP_TESTING=1` skips rendering at module level. Integration tests use Streamlit's `AppTest` headless runtime (30s timeout).

## Project Structure

- **`visualize_projection.py`** (~8400 lines) — Single-file Streamlit app containing all logic
- **`test_visualize_projection.py`** (~2350 lines) — Unit tests with fake Streamlit
- **`test_integration.py`** (~860 lines) — Integration tests with Streamlit AppTest
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 benchmark data (~23 models)
- **`epoch_capabilities_index.csv`** — Epoch ECI data (~714 model-variants)
- **`data_centers.csv`** — Epoch Frontier Data Centers metadata (one row per data center)
- **`data_center_timelines.csv`** — Epoch Frontier Data Centers capacity timelines (multiple dated rows per data center)
- **`aisi_cyber_narrow.csv`** — AISI narrow cyber task success rates (12 models). **Chart-digitized, not a published feed** — see the file's `#` header
- **`aisi_cyber_tlo.csv`** — AISI/CAISI cyber-range "The Last Ones" avg steps of 32 (10 models). Same caveat; 9 rows digitized from Figure 2, Kimi K3 quoted from the CAISI post
- **`requirements.txt`** — `streamlit`, `numpy`, `plotly`, `pyyaml`

No build system, no CI/CD, no package manager beyond requirements.txt.

## Architecture

Nine-tab Streamlit dashboard selected via sidebar radio (`active_tab`, `_TAB_OPTIONS`) with URL deep-linking (`?tab=<slug>`). Each tab has its own render function, sidebar controls, and (where applicable) projection engine. URL slugs (`_SLUG_FOR_TAB`): `metr`, `eci`, `ecigap`, `rli`, `ukcyber`, `employment`, `revenue`, `datacenters`, `computecap`.

### Tabs and Render Functions

Line numbers drift as the file grows — grep `^def render_` to find the current location rather than trusting these.

| Tab | Function (approx line) | Data Source | Metric |
|---|---|---|---|
| METR Horizon | `render_metr()` (~866) | `benchmark_results_1_1.yaml` → `load_frontier()` | log₂(minutes) |
| Epoch ECI | `render_eci()` (~2620) | `epoch_capabilities_index.csv` → `load_eci_frontier()` | linear score |
| Remote Labor Index | `render_rli()` (~2738) | hardcoded `_RLI_RAW` → `load_rli_data()` | logit-transformed score (0-100 bounded) |
| UK Cyber | `render_ukcyber()` | `aisi_cyber_narrow.csv` → `load_ukcyber()`; `aisi_cyber_tlo.csv` → `load_ukcyber_tlo()` | success rate % (0-100 bounded, logit-projected) + open-weight lag in months; plus a TLO cyber-range cross-check in steps (`_render_ukcyber_tlo()`) |
| Revenue | `render_revenue()` (~3897) | hardcoded `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` | ARR in billions |
| Employment | `render_employment()` (~4353) | derived from RLI frontier + slider assumptions (no external feed) | unemployment % / jobs lost |
| ECI Company Gap | `render_eci_gap()` (~5223) | `epoch_capabilities_index.csv` (filtered by org/country) | linear score gap |
| Data Centers | `render_data_centers()` (~5669) | `data_centers.csv` + `data_center_timelines.csv` → `load_data_centers()` | H100-equiv / power / cost |
| Compute vs Capabilities | `render_compute_capabilities()` (~7344) | data centers (`dc_all`) + ECI | train-FLOP frontier vs ECI |

### Data Sources and How to Update

Six external data feeds back the tabs; the rest are derived. Canonical sources and refresh method:

| File / table | Canonical source | How to refresh |
|---|---|---|
| `benchmark_results_1_1.yaml` | METR | Download `https://metr.org/assets/benchmark_results_1_1.yaml` and overwrite |
| `epoch_capabilities_index.csv` | Epoch AI | Extract `epoch_capabilities_index.csv` from `https://epoch.ai/data/benchmark_data.zip` and overwrite. Epoch recomputes scores live, so existing rows drift slightly on each pull |
| `data_centers.csv` | Epoch AI | Download `https://epoch.ai/data/data_centers/data_centers.csv` and overwrite |
| `data_center_timelines.csv` | Epoch AI | Download `https://epoch.ai/data/data_centers/data_center_timelines.csv` and overwrite. Column order differs from older pulls; the loader uses `DictReader` (by header name) so this is safe |
| `_RLI_RAW` (hardcoded) | Scale Labs RLI leaderboard (`labs.scale.com/leaderboard/rli`) / `remotelabor.ai` | Hand-edit new rows |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (hardcoded) | Press reports (The Information, Reuters, etc.) | Hand-edit `(date, ARR_in_billions)` tuples |
| `aisi_cyber_tlo.csv` | UK AISI Figure 2 (same post) + [UK AISI/CAISI Kimi K3 assessment](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) (2026-07-23) | **Not downloadable.** 9 rows digitized from `fig2-ranges.png`; Kimi K3's 17.0 quoted from the CAISI post's prose. Calibration and the four published-number validation checks are in the file's `#` header and guarded by `TestUkCyberTlo`. Dates here are **published release dates**, not chart-derived — the figure's x-axis is tokens |
| `aisi_cyber_narrow.csv` | UK AISI blog, [open-weight cyber gap post](https://www.aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber) (2026-07-17) | **Not downloadable.** AISI publishes no numbers for this chart — values were digitized from `fig1-narrow.png` by pixel analysis. Refreshing is a *figure-unchanged check*, not a download: re-fetch the PNG and confirm gridlines still sit at row 547 (=100%) / row 1760 (=0%) and that each row's (date, score) still lands on its marker colour. Only re-digitize if the figure actually changed, then re-verify against `test_digitized_dates_match_known_releases` and `test_optimistic_bracket_reproduces_aisi_published_lags` — those two are the calibration guards. Hand-editing a row is fine if AISI states a number in prose. See `.claude/commands/update-data.md` for the full recipe and for AISI cyber data that exists but is deliberately not ingested |

Before overwriting a CSV wholesale, diff by key column to confirm no locally-curated rows would be lost (ECI key = `Model version`; DC metadata key = `Name`; timelines key = `Data center` + `Date`). After any data change, sanity-check with `_VP_TESTING=1 python3 -c "import visualize_projection as v; ..."` calling the relevant loader, then run the tests.

### Key Sections of visualize_projection.py

Line numbers below are approximate — grep for the function/table name to locate the current line.

- Shared helpers — `pretty()`, `log2min_to_label()`, `fmt_hrs()`, `fit_line()`, `_fit_slope_p50_intercept_display()`, distribution samplers, `_ss_number_input()`, `superexp_trajectory()`, `_logit()`/`_inv_logit()`
- Backtesting helpers — `_backtest_stats()`, `_bt_color_for()`, `_add_backtest_traces()`, `_backtest_summary()`
- Data loading — `load_frontier()` / `load_metr_all()` (YAML), `load_eci_frontier()` and `load_eci_compute()` (ECI CSV, with dedup + running-max frontier), `load_rli_data()`, `load_data_centers()`, `load_ukcyber()`
- UK Cyber lag helpers — `_ukc_frontier_crossing()` (interpolated crossing + bracketing models), `_ukc_frontier_match_for_score()` / `_ukc_frontier_below_for_score()` (bracket ends), `ukc_lag_rows()`, `ukc_target_eta()` / `ukc_target_eta_direct()`
- Data init + tab selector (`_TAB_OPTIONS`, `_TAB_SLUG`, `_SLUG_FOR_TAB`)
- The nine `render_*()` functions (see table above)
- Dispatch at end of file (skipped when `_VP_TESTING=1`)

### Projection Engine (repeated per tab)

Each tab supports three projection bases: **Linear** (single OLS), **Piecewise linear** (multi-segment OLS, last segment extrapolated), and **Superexponential** (doubling time decays via `superexp_trajectory()` with a floor). All sample 5,000 trajectories and render Plotly fan charts with 50%/80%/90% CI bands.

### Session State and Reset

Widget defaults live in `_RESET_DEFAULTS` dicts per tab. Each tab has `_RESET_KEYS` listing session state keys. The reset button pops all keys and calls `st.rerun()`. Custom number inputs use `_ss_number_input()` to persist values via session state.

### Internal Units

- **METR**: Performance in log₂(minutes), displayed as hours. Work-time: 1d=8h, 1w=40h, 1mo=176h, 1y=2000h
- **ECI**: Linear score (~57-154 range). DPP = days per +1 ECI point
- **RLI**: Score 0-100, projected in logit space to respect bounds
- **UK Cyber**: Success rate 0-100%, projected in logit space (same bounded treatment as RLI)
- **Revenue**: ARR in billions USD

### UK Cyber tab caveats

Three constraints are load-bearing; don't "simplify" them away:

1. **The frontier is closed-weight only.** Open-weight models are the subject being measured against it, so `load_ukcyber()` excludes them from the running max.
2. **Lag is interpolated between the bracketing models, and carries a bracket.** `_ukc_frontier_crossing()` interpolates between the last frontier model below a score and the first above it. Snapping to the next model up (the earlier implementation) equates scores that can be far apart: DeepSeek-V4-Pro's 55.7% falls in a 10-point gap between GPT-5 (52.5%) and Opus 4.5 (62.6%), so snapping credited it with ~7 points it doesn't have and understated its lag by ~2.4 months. The distortion scales with how far the nearest model is — GLM-5.2 sits 2.3 points under Opus 4.6 and barely moved (4.3 → 4.7 mo). `ukc_lag_rows()` returns `lag_months` (interpolated point estimate) plus `lag_lo`/`lag_hi` from the two bracketing models; no model was released in the gap, so that width is real uncertainty, not noise.
3. **`lag_lo` is the AISI-compatible bound.** AISI's printed annotations (4.3mo, 5.0mo) use next-model-up, which is exactly `lag_lo`. `test_optimistic_bracket_reproduces_aisi_published_lags` asserts on it, so it doubles as the calibration check on the digitization. Changing the lag method must not break that guard.

Rejected alternatives, for the record: matching to the *last model below* is defensible but maximally pessimistic (DeepSeek 8.5mo); dating the crossing off the fitted OLS curve imports global fit error into a local question (the curve "reaches" 55.7% in Jun 2025, when the best observed model was GPT-5 at 52.5%, giving 9.9mo).

4. **The TLO cross-check compares on `lag_lo`, not on the point estimate.** `aisi_cyber_tlo.csv` carries AISI's other cyber measure (avg steps of 32 on the "The Last Ones" cyber range) and reuses `_ukc_frontier_crossing()` / `ukc_lag_rows()` unchanged by storing steps in `cyber_score`. But the interpolated point estimates are **not comparable across the two datasets**: DeepSeek-V4-Pro's narrow score falls in a 10-point frontier gap (inflating it to 7.4mo) while its TLO score sits below every frontier model, so no interpolation happens at all (6.8mo, and `lag_hi` is `None` — it is a lower bound only). On point estimates the ordering therefore inverts, for reasons about frontier sampling rather than capability. Compared on `lag_lo` — the next-model-up convention AISI's own chart titles use — the two reproduce AISI's headline exactly: narrow tasks 4.3–5.1mo ("4-5 months prior"), cyber range 6.7–6.8mo ("7 months prior"), together "4 to 7 months". `test_reproduces_both_figure_titles` guards this.

The dataset has no US open-weight and no Chinese closed-weight models, so country and openness are perfectly confounded. The tab headline says "China" because the two Chinese models are also the only open-weight ones — `_UKC_CONFOUND_PLAIN` is folded into the fine-print caption so that's stated wherever the tab says "China". It was previously a standalone `st.warning` banner; if the caveat ever needs that prominence again, promote the same constant rather than adding a second wording to keep in sync.

`ukc_target_eta()` answers "when do open-weight models reach `_UKC_TARGET` (90%)": the frontier's interpolated crossing of the target, plus the min/max measured lag. `ukc_target_eta_direct()` fits the open-weight points themselves as a cross-check only — two models 53 days apart make that slope very sensitive, so it is never the headline.

### Backtesting

"Project as of" model selector lets you project from a historical vantage point. `_backtest_stats()` compares actual future models against projected trajectories, color-coded by which CI band they fall in.
