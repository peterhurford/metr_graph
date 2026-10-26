# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run visualize_projection.py    # http://localhost:8501
```

## Random seed

Every tab's rerun draws its Monte Carlo inside `_seeded_rng(_APP_SEED)` (the dispatch), so
the same settings give the same numbers; Takeoff reseeds from its own `tk_seed`. No cached
function may draw unseeded, or a cache hit would shift the stream
(`test_toggling_a_box_back_restores_the_numbers`).

## Running Tests

```bash
pytest -v                                    # all tests
pytest test_visualize_projection.py -v       # unit tests (fake Streamlit)
pytest test_integration.py -v                # integration tests (AppTest)
pytest test_visualize_projection.py::TestPacing::test_defaults_cover_reset_keys_and_url_tracking -v  # single test
pytest -v -n0                                # serial — needed for --pdb / live output
_VP_SAMPLES=5000 pytest -v                   # production sample count (see below)
```

Test deps: `pip install -r requirements-dev.txt` (adds `pytest-xdist`). Unit tests use a fake Streamlit module (`_FakeStreamlit` / `_Noop`) so the app imports without a server; `_VP_TESTING=1` skips rendering at module level. Integration tests use Streamlit's `AppTest` headless runtime (30s timeout).

## Project Structure

- **`visualize_projection.py`** — single-file Streamlit app containing all logic
- **`test_visualize_projection.py`** / **`test_integration.py`** — unit tests (fake Streamlit) and integration tests (AppTest)
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 data
- **`epoch_capabilities_index.csv`** — Epoch ECI data, written by **`convert_eci.py`** from Epoch's `eci_scores.csv` + `all_ai_models.csv`
- **`data_centers.csv`** / **`data_center_timelines.csv`** — Epoch Frontier Data Centers: one metadata row per site, many dated capacity rows per site
- **`CHINA_COMPUTE_JUDGMENTS.md`** — the record behind every hand-set China/SEA compute pace and share: value, evidence, derivation, revisit triggers
- **`cts_duv_production.csv`** — CTS lithography report's Table 45: US AI chip output vs China's lithography-bounded ceiling, DUVi imports continuing or banned (see *China tab — lithography ceiling*)
- **`china_compute_sources.csv`** — China's AI compute by source (domestic, legal sales, smuggled, remote access), flow and stock in H100e: Epoch to 2025, IFP for 2026–27, with `_cn_src_projection()` extending to 2030. Drawn by `_render_cc_compute_sources()` as 100% mixes with totals; derivations in the file's `#` header
- **`.streamlit/config.toml`** — turns on websocket compression (~4× less chart JSON on the wire)
- **`aisi_cyber_narrow.csv`** / **`aisi_cyber_tlo.csv`** — AISI narrow cyber success rates (12 models); AISI/CAISI cyber-range "The Last Ones" avg steps of 32 (10 models). Both **chart-digitized, not published feeds** — see each file's `#` header

No build system, no CI/CD, no package manager beyond requirements.txt (`streamlit`, `numpy`, `plotly`, `pyyaml`).

## Architecture

Eleven-tab Streamlit dashboard selected via sidebar radio (`active_tab`, `_TAB_OPTIONS`, ECI first and the default; tabs from `_TAB_MAIN_N` on have their chips hidden by CSS and are picked by buttons under a *More* expander that set the same radio; `_TAB_LABEL` shows ECI as *Epoch ECI*) with URL deep-linking (`?tab=<slug>`, `?to=<section>`). Each tab has its own render function, sidebar controls, and (where applicable) projection engine. Slugs (`_SLUG_FOR_TAB`): `metr`, `eci` (`ecigap`, the merged ECI Company Gap tab, is an alias for it), `rli`, `takeoff` (`rsi`, the merged RSI tab, is an alias for it),  `thresholds`, `revenue`, `datacenters`, `china` (`computecap`, its old name, is an alias), `pacing` (`pausing`, its old name, is an alias), `ukcyber`, `employment`.

## Design records: open the right one before editing

This file is the index. The design records live in `docs/` and are **not loaded
automatically**: before changing code in an area below, read its file. Each records, for
every non-obvious choice, what it is, the wrong-looking alternative, and the test that holds
it, so most surprising-looking code is deliberate. Update the file in the same commit.

| If you touch… | Read first |
|---|---|
| `render_takeoff`, `takeoff_model.py` wiring, any `_tk_*`; `render_rsi` and every `_rsi_*`, `_render_rsi_*`, `_tasteval_*`, `load_rsi_*`; the *Capabilities Milestones* cards and *Automated coder projection* blend (`_pc_render_milestones`, `_pc_render_rsi_blend`, `_pc_rsi_*`, `_pc_*_eta`, `_pc_condition_on_today`, `_PC_RSI_WEIGHTS`) | `docs/takeoff.md` (equations: `TAKEOFF_MODEL.md`) |
| `load_data_centers`, any `_dc_*` (company aliases, tenant/operator, hidden companies, network clusters, by-country panel), the buildout-vs-release panel and Compute ⟷ ECI (`_cc_company_buildout`, `_cc_forward_match`, `_cc_lab_*`, `_cc_frontier_grade_algo`, `_cc_rate_calibration`, `_render_compute_eci`), world shares (`_wc_*`, `_WC_*`) | `docs/data-centers.md` |
| `render_compute_capabilities`, `_duv_*`, `_cn_src_*`, `_cc_cn_*` (crossing sim, channels, pace band), `render_pacing` and the pause/race/lever code (`_pc_render_us_pause`, `_pc_projection`, `_pc_crossing_idx`, `_pc_capacity_at`, `_pc_render_why`, `_pc_render_compute_*`, every `pc_*` widget) | `docs/china-pacing.md` (judgments: `CHINA_COMPUTE_JUDGMENTS.md`) |
| `_ship_lag*`, `_anchor_slug`, `_render_anchor_links`, `?to=` links, `today_fwd` / `_tf` / `_today_forward`, `_fn_caption` / `_fn_line` / `_fn_span`, any caption or hover | `docs/conventions.md` |
| `render_ukcyber` / `_ukc_*` / `ukc_*`, `render_eci` / `_eci_*` / `_ECI_COMPANIES` / `_best_per_date`, `frontier_thresholds.py`; the full per-tab table | `docs/tabs.md` |
| any data file, loader or hardcoded `_*_RAW` / `_*_REVENUE` table | `docs/data-sources.md` (step by step: `.claude/commands/update_data.md`) |

Keep this file an index: new design detail goes in the matching `docs/` file, not here
(Claude Code caps all instruction files at 150k characters together).

## Rules that apply everywhere

- **Caveats go on the thing they qualify**: `help=` on a metric or widget, else
  `_fn_caption(text, (phrase, note))` anchored on a phrase in the visible line, else
  `_fn_line`. Legends stay visible (`docs/conventions.md`).
- **Today-forward** is a post-process, `_tf()` around `st.plotly_chart`; only projection
  charts opt in (`docs/conventions.md`).
- **Every ECI frontier goes through `_best_per_date()`**, and company matching on Epoch's
  `Organization` is by substring, never an exact key (`docs/tabs.md`).
- **Before overwriting a CSV wholesale, diff by key column** (`docs/data-sources.md`).
- **Editing these files or the app by script:** a naive paren-counter will not do (an
  f-string like `.split(' (')` carries an unbalanced paren). Tokenize, or edit by hand.

## Tabs

| Tab (slug) | Function | Data source |
|---|---|---|
| METR Horizon (`metr`) | `render_metr()` | `benchmark_results_1_1.yaml` → `load_frontier()` |
| ECI (`eci`) | `render_eci()` | `epoch_capabilities_index.csv` → `load_eci_frontier()` |
| Remote Labor Index (`rli`) | `render_rli()` | `_RLI_RAW` → `load_rli_data()` |
| Takeoff (`takeoff`) | `render_takeoff()`; evidence half `render_rsi(end_year)` | `takeoff_model.py`; the RSI tables and CSVs |
| UK Cyber (`ukcyber`) | `render_ukcyber()` | `aisi_cyber_narrow.csv`, `aisi_cyber_tlo.csv` |
| Revenue (`revenue`) | `render_revenue()` | `_OPENAI_REVENUE`, `_ANTHROPIC_REVENUE`, `_CN_REVENUE` |
| Employment (`employment`) | `render_employment()` | RLI frontier + slider assumptions |
| Data Centers (`datacenters`) | `render_data_centers()` | `data_centers.csv` + timelines → `load_data_centers()` |
| China (`china`) | `render_compute_capabilities()` | `dc_all` + ECI, `china_compute_sources.csv`, `cts_duv_production.csv` |
| Pacing (`pacing`) | `render_pacing()` | `dc_all`, Takeoff's `_tk_run` |
| Frontier Thresholds (`thresholds`) | `frontier_thresholds.render(st)` | `frontier_thresholds.json` |

Metrics, sections and guarding tests per tab: `docs/tabs.md`.

## Key Sections of visualize_projection.py

Roughly in file order: shared helpers (`pretty()`, `fit_line()`, distribution samplers,
`_ss_number_input()`, `superexp_trajectory()`, `_logit()`/`_inv_logit()`); backtesting
(`_backtest_*`); loaders (`load_frontier()`, `load_eci_frontier()` / `load_eci_compute()`,
`load_rli_data()`, `load_data_centers()`, `load_ukcyber()`); data-center aggregation
(`_dc_envelope()`, `_dc_company_series()`, `_dc_company_networked_series()`); UK Cyber lag
(`_ukc_*`, `ukc_*`); compute-vs-capabilities (`_cc_*`); data init and tab selector; the
`render_*()` functions (grep `^def render_`; Frontier Thresholds lives in `frontier_thresholds.py`);
dispatch at end of file (skipped when `_VP_TESTING=1`).

## Projection Engine (repeated per tab)

Three bases: **Linear** (single OLS), **Piecewise linear** (multi-segment OLS, last segment extrapolated), **Superexponential** (doubling time decays via `superexp_trajectory()` with a floor). All sample 5,000 trajectories into Plotly fan charts with 50%/80%/90% CI bands.

## Session State and Reset

Widget defaults live in per-tab `_<TAB>_DEFAULTS` (e.g. `_METR_DEFAULTS`); each tab's `_<TAB>_RESET_KEYS` lists its session state keys. The reset button pops all keys and calls `st.rerun()`. Custom number inputs use `_ss_number_input()` to persist via session state.

## Backtesting

"Project as of" projects from a historical vantage point; `_backtest_stats()` compares actual later models against the projected trajectories, colored by which CI band they land in.

## Internal Units

- **METR**: log₂(minutes), displayed as hours. Work-time: 1d=8h, 1w=40h, 1mo=176h, 1y=2000h
- **ECI**: linear score. DPP = days per +1 ECI point
- **RLI** / **UK Cyber**: 0-100, projected in logit space to respect the bounds
- **Revenue**: ARR in billions USD
- **Data Centers, "Capacity (time to GPT-5 / Mythos)"**: stored as *training runs per
  2-month window* (`gpt5s` / `mythos`), displayed as *time to train one model*
  (`kind='traintime'` → `_dc_fmt_value` → `_fmt_duration_days`; days = `_DAYS_2MO` / runs).
  Storing the count keeps every "largest site" aggregation a plain max and the
  networked-sites section a plain **sum** (two equal sites = two runs per window = half the
  time to train one model); inverting it breaks both. Don't invert it to make the axis read
  forward either — the axis is relabelled via `_dc_duration_ticks()` through
  `_dc_layout(kind=…)`, with a caption noting the labelled time shrinks as the line rises
- **Data-center axis labels**: every metric is stored, plotted, hovered and tabulated in
  raw units, and `_dc_axis_ticks(y_range, log_scale, kind)` is the single place a tick gets
  its text — `_dc_tick_label` (k/M suffixes for `h100`), `_dc_logop_ticks` (`flop`) or
  `_dc_duration_ticks` (`traintime`). Don't reintroduce a per-metric axis divisor: ticks
  would then disagree with `_dc_fmt_value`'s raw-unit bar text, hovers and table. And route the snapshot bar chart's
  x-axis through the same dispatcher — it builds its own axis instead of calling
  `_dc_layout()`, and when it hand-rolled the branching it silently lacked the `flop` case.
  `TestDcAxisTicks` guards both halves

## Why the test suite is fast (and how to not un-fast it)

~8s, down from more than three minutes that was essentially all in the `AppTest` cases. Three things buy
that back; the first two live in `conftest.py`.

1. **One shared bytecode cache.** Streamlit builds a fresh `ScriptCache` on every
   `AppTest.run()` — once in `AppTest._run()`, again in `LocalScriptRunner.__init__` — so
   the app was re-read, AST-rewritten by `magic.add_magic` and recompiled twice per case. `conftest.py` patches **both** sites to one process-wide cache; patch only one and
   the other keeps recompiling and the win disappears. Isolation is unaffected — the
   bytecode is still exec'd into a fresh module each run.
2. **A smaller Monte Carlo.** `N_SAMPLES` reads `_VP_SAMPLES`; `conftest.py` sets it to 400.
   Nothing asserted depends on the count — the CI defaults the tests check come from the
   deterministic OLS fits, and only the Takeoff tab seeds its RNG. `_VP_SAMPLES=5000 pytest` runs
   at production fidelity as a check. Don't lower the *default*: the band edges are
   percentiles and go visibly ragged well before 5000.
3. **Parallelism.** `pytest.ini` passes `-n auto`; the cases are independent, so a straight
   multiple. It costs the unit-test-only run a little worker startup, hence `-n0`.

Adding an `AppTest` case is cheap (~0.1s per `.run()`), so prefer one to skipping coverage.
What is *not* cheap is defeating the shared cache — a test that mutates
`visualize_projection.py` on disk, or builds its own runner from `streamlit.testing`
internals.
