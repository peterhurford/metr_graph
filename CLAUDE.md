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

- **`visualize_projection.py`** (~3185 lines) — Single-file Streamlit app containing all logic
- **`test_visualize_projection.py`** (~1494 lines) — Unit tests with fake Streamlit
- **`test_integration.py`** (~308 lines) — Integration tests with Streamlit AppTest
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 benchmark data (~23 models)
- **`epoch_capabilities_index.csv`** — Epoch ECI data (~265 models)
- **`requirements.txt`** — `streamlit`, `numpy`, `plotly`, `pyyaml`

No build system, no CI/CD, no package manager beyond requirements.txt.

## Architecture

Four-tab Streamlit dashboard selected via sidebar radio (`active_tab`) with URL deep-linking (`?tab=metr|eci|rli|revenue`). Each tab has its own render function, sidebar controls, and projection engine.

### Tabs and Render Functions

| Tab | Function | Data Source | Metric | Trend Unit |
|---|---|---|---|---|
| METR Horizon | `render_metr()` (line 484) | YAML → `load_frontier()` | log₂(minutes) | doubling time (days) |
| Epoch ECI | `render_eci()` (line 1296) | CSV → `load_eci_frontier()` | linear score | days per point |
| Remote Labor Index | `render_rli()` (line 2058) | hardcoded `_RLI_RAW` → `load_rli_data()` | logit-transformed score (0-100 bounded) | doubling time in logit space |
| Revenue | `render_revenue()` (line 2921) | hardcoded in function | ARR in billions | doubling time |

### Key Sections of visualize_projection.py

- **Lines 28-200**: Shared helpers — `pretty()`, `log2min_to_label()`, `fmt_hrs()`, `fit_line()`, `_fit_slope_p50_intercept_display()`, distribution samplers, `_ss_number_input()`, `superexp_trajectory()`, `_logit()`/`_inv_logit()`
- **Lines 204-290**: Backtesting helpers — `_backtest_stats()`, `_bt_color_for()`, `_add_backtest_traces()`, `_backtest_summary()`
- **Lines 293-430**: Data loading — `load_frontier()` (YAML), `load_eci_frontier()` (CSV with dedup + running-max frontier), `load_rli_data()` (hardcoded)
- **Lines 433-480**: Data init + tab selector
- **Lines 484-1293**: `render_metr()`
- **Lines 1296-2055**: `render_eci()`
- **Lines 2058-2920**: `render_rli()`
- **Lines 2921-3173**: `render_revenue()`
- **Lines 3177-3185**: Dispatch (skipped when `_VP_TESTING=1`)

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
