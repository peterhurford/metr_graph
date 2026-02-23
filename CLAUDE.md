# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run visualize_projection.py
```

This launches an interactive Streamlit dashboard (typically on localhost:8501) for visualizing AI capability projections.

## Project Structure

This is a minimal project with no build system, no tests, and no package management:

- **`visualize_projection.py`** — Single-file Streamlit app (~1530 lines) containing all logic
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 benchmark data for ~23 AI models
- **`epoch_capabilities_index.csv`** — Epoch ECI data for ~265 scored AI models

Dependencies (assumed pre-installed): `streamlit`, `numpy`, `plotly`, `pyyaml`

## Architecture

The app has two views (METR Horizon and Epoch ECI) selected via a sidebar radio. Each view loads benchmark data, fits projection models to frontier data, and renders an interactive Plotly fan chart with confidence intervals.

### Two-Tab Structure

- **METR Horizon**: Benchmark performance in log₂(minutes), "doubling time" metric, milestones in work-hours
- **Epoch ECI**: Composite capability score (linear), "days per point" metric, milestones at ECI 155/160/165/170

Sidebar radio (`active_tab`) controls which view renders. Each view has its own `render_metr()` / `render_eci()` function containing sidebar controls + chart + metrics.

### Data Flow

**METR**: YAML → `load_frontier()` → sidebar → OLS/superexp fit → 20K trajectories → fan chart + milestones
**ECI**: CSV → `load_eci_frontier()` (dedup by model name, running-max frontier) → sidebar → OLS/superexp fit → 20K trajectories → fan chart + milestones

### Key Sections of visualize_projection.py

- **Lines 24-155**: Shared helpers — `pretty()`, `log2min_to_label()`, `fmt_hrs()`, `fit_line()`, sampling functions (`_lognormal_from_ci`, `_normal_from_ci`, `_log_lognormal_from_ci`)
- **Lines 157-248**: Data loading — `load_frontier()` (YAML), `load_eci_frontier()` (CSV with dedup + frontier detection)
- **Lines 250-268**: Data init + sidebar tab selector
- **Lines 271-928**: `render_metr()` — METR sidebar, projection engine, Plotly chart, metrics/milestones
- **Lines 931-1522**: `render_eci()` — ECI sidebar, projection engine, Plotly chart, metrics/milestones
- **Lines 1525-1528**: Dispatch

### Internal Units

- **METR**: Performance stored as log₂(minutes), displayed as hours or log₂(min). Work-time units: 1d=8h, 1w=40h, 1mo=176h, 1y=2000h
- **ECI**: Linear score (~57-154 range). DPP = days per +1 ECI point

### Sidebar Controls

Each view has its own sidebar section (shown conditionally): projection basis (Linear/Piecewise/Superexponential), advanced CI options, milestone/label toggles, and a "Project as of" model selector for backtesting.
