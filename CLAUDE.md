# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run visualize_projection.py
```

This launches an interactive Streamlit dashboard (typically on localhost:8501) for visualizing METR Frontier model performance projections.

## Project Structure

This is a minimal two-file project with no build system, no tests, and no package management:

- **`visualize_projection.py`** — Single-file Streamlit app (~870 lines) containing all logic
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 benchmark data for ~23 AI models

Dependencies (assumed pre-installed): `streamlit`, `numpy`, `plotly`, `pyyaml`

## Architecture

The app loads benchmark results from YAML, fits projection models to historical SOTA frontier data, and renders an interactive Plotly fan chart with confidence intervals.

### Data Flow

YAML → `load_frontier()` (cached, filters to SOTA models) → sidebar selects projection model & parameters → fit OLS/superexponential to frontier → sample 20K trajectories → compute percentile bands → Plotly fan chart + milestone probability tables

### Key Sections of visualize_projection.py

- **Lines 25-111**: Display helpers — `pretty()` (model name mapping), `log2min_to_label()`, `fmt_hrs()` (work-time units: 1d=8h, 1w=40h, 1mo=176h, 1y=2000h)
- **Lines 114-158**: Data loading — `fit_line()` (OLS via numpy), `load_frontier()` (YAML parse + SOTA filter)
- **Lines 365-482**: Projection models — three approaches:
  - **Linear**: Single OLS fit, samples doubling times from CI distribution
  - **Piecewise linear**: 2-3 segments with user-selected breakpoints
  - **Superexponential**: Accelerating improvement with DT floor (y = A + K × 2^(d/halflife))
- **Lines 537-764**: Plotly chart construction — fan bands (5-95th percentiles), trend lines, data points, milestones, today line
- **Lines 768-868**: Projection display — metrics row with 6 target dates, milestone arrival probabilities

### Internal Units

- Performance is stored/computed as **log₂(minutes)**
- Display converts to hours (linear mode) or keeps log₂(minutes) (log mode)
- Human-readable labels use work-time units (days, weeks, months, years)

### Sidebar Controls

All interactivity is driven by Streamlit sidebar widgets: projection basis selection, p50/p80 metric toggle, log scale toggle, milestone display options, and a "Project as of" model selector for backtesting projections against earlier data.
