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

- **`visualize_projection.py`** (~7900 lines) — Single-file Streamlit app containing all logic
- **`test_visualize_projection.py`** (~1730 lines) — Unit tests with fake Streamlit
- **`test_integration.py`** (~744 lines) — Integration tests with Streamlit AppTest
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 benchmark data (~23 models)
- **`epoch_capabilities_index.csv`** — Epoch ECI data (~714 model-variants)
- **`data_centers.csv`** — Epoch Frontier Data Centers metadata (one row per data center)
- **`data_center_timelines.csv`** — Epoch Frontier Data Centers capacity timelines (multiple dated rows per data center)
- **`requirements.txt`** — `streamlit`, `numpy`, `plotly`, `pyyaml`

No build system, no CI/CD, no package manager beyond requirements.txt.

## Architecture

Nine-tab Streamlit dashboard selected via sidebar radio (`active_tab`, `_TAB_OPTIONS`) with URL deep-linking (`?tab=<slug>`). Each tab has its own render function, sidebar controls, and (where applicable) projection engine. URL slugs (`_SLUG_FOR_TAB`): `metr`, `eci`, `ecigap`, `rli`, `employment`, `prinz`, `revenue`, `datacenters`, `computecap`.

### Tabs and Render Functions

Line numbers drift as the file grows — grep `^def render_` to find the current location rather than trusting these.

| Tab | Function (approx line) | Data Source | Metric |
|---|---|---|---|
| METR Horizon | `render_metr()` (~866) | `benchmark_results_1_1.yaml` → `load_frontier()` | log₂(minutes) |
| Epoch ECI | `render_eci()` (~2620) | `epoch_capabilities_index.csv` → `load_eci_frontier()` | linear score |
| Remote Labor Index | `render_rli()` (~2738) | hardcoded `_RLI_RAW` → `load_rli_data()` | logit-transformed score (0-100 bounded) |
| Prinz | `render_prinz()` (~3707) | hardcoded `_PRINZ_RAW` → `load_prinz_data()` | prinzbench score (0-99) |
| Revenue | `render_revenue()` (~3897) | hardcoded `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` | ARR in billions |
| Employment | `render_employment()` (~4353) | derived from RLI frontier + slider assumptions (no external feed) | unemployment % / jobs lost |
| ECI Company Gap | `render_eci_gap()` (~5223) | `epoch_capabilities_index.csv` (filtered by org/country) | linear score gap |
| Data Centers | `render_data_centers()` (~5669) | `data_centers.csv` + `data_center_timelines.csv` → `load_data_centers()` | H100-equiv / power / cost |
| Compute vs Capabilities | `render_compute_capabilities()` (~7344) | data centers (`dc_all`) + ECI | train-FLOP frontier vs ECI |

### Data Sources and How to Update

Five external data feeds back the tabs; the rest are derived. Canonical sources and refresh method:

| File / table | Canonical source | How to refresh |
|---|---|---|
| `benchmark_results_1_1.yaml` | METR | Download `https://metr.org/assets/benchmark_results_1_1.yaml` and overwrite |
| `epoch_capabilities_index.csv` | Epoch AI | Extract `epoch_capabilities_index.csv` from `https://epoch.ai/data/benchmark_data.zip` and overwrite. Epoch recomputes scores live, so existing rows drift slightly on each pull |
| `data_centers.csv` | Epoch AI | Download `https://epoch.ai/data/data_centers/data_centers.csv` and overwrite |
| `data_center_timelines.csv` | Epoch AI | Download `https://epoch.ai/data/data_centers/data_center_timelines.csv` and overwrite. Column order differs from older pulls; the loader uses `DictReader` (by header name) so this is safe |
| `_RLI_RAW` (hardcoded) | Scale Labs RLI leaderboard (`labs.scale.com/leaderboard/rli`) / `remotelabor.ai` | Hand-edit new rows |
| `_PRINZ_RAW` (hardcoded) | prinzbench "full" bar chart | Hand-edit; scores read off the chart are ±1; dates sourced from the ECI CSV |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (hardcoded) | Press reports (The Information, Reuters, etc.) | Hand-edit `(date, ARR_in_billions)` tuples |

Before overwriting a CSV wholesale, diff by key column to confirm no locally-curated rows would be lost (ECI key = `Model version`; DC metadata key = `Name`; timelines key = `Data center` + `Date`). After any data change, sanity-check with `_VP_TESTING=1 python3 -c "import visualize_projection as v; ..."` calling the relevant loader, then run the tests.

### Key Sections of visualize_projection.py

Line numbers below are approximate — grep for the function/table name to locate the current line.

- Shared helpers — `pretty()`, `log2min_to_label()`, `fmt_hrs()`, `fit_line()`, `_fit_slope_p50_intercept_display()`, distribution samplers, `_ss_number_input()`, `superexp_trajectory()`, `_logit()`/`_inv_logit()`
- Backtesting helpers — `_backtest_stats()`, `_bt_color_for()`, `_add_backtest_traces()`, `_backtest_summary()`
- Data loading — `load_frontier()` / `load_metr_all()` (YAML), `load_eci_frontier()` and `load_eci_compute()` (ECI CSV, with dedup + running-max frontier), `load_rli_data()`, `load_data_centers()`, `load_prinz_data()`
- Data init + tab selector (`_TAB_OPTIONS`, `_TAB_SLUG`, `_SLUG_FOR_TAB`)
- The nine `render_*()` functions (see table above)
- Dispatch at end of file (skipped when `_VP_TESTING=1`)

### Projection Engine (repeated per tab)

Each tab supports three projection bases: **Linear** (single OLS), **Piecewise linear** (multi-segment OLS, last segment extrapolated), and **Superexponential** (doubling time decays via `superexp_trajectory()` with a floor). All sample 20,000 trajectories and render Plotly fan charts with 50%/80%/90% CI bands.

### Session State and Reset

Widget defaults live in `_RESET_DEFAULTS` dicts per tab. Each tab has `_RESET_KEYS` listing session state keys. The reset button pops all keys and calls `st.rerun()`. Custom number inputs use `_ss_number_input()` to persist values via session state.

### Internal Units

- **METR**: Performance in log₂(minutes), displayed as hours. Work-time: 1d=8h, 1w=40h, 1mo=176h, 1y=2000h
- **ECI**: Linear score (~57-154 range). DPP = days per +1 ECI point
- **RLI**: Score 0-100, projected in logit space to respect bounds
- **Revenue**: ARR in billions USD

### Backtesting

"Project as of" model selector lets you project from a historical vantage point. `_backtest_stats()` compares actual future models against projected trajectories, color-coded by which CI band they fall in.
