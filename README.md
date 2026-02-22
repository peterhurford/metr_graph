# METR Frontier Projection

Interactive dashboard for visualizing AI model performance on METR-Horizon-v1.1 benchmarks and projecting future capability trajectories.

## Quick Start

```bash
pip install streamlit numpy plotly pyyaml
streamlit run visualize_projection.py
```

Opens at http://localhost:8501.

## What It Does

The app plots the **frontier** (state-of-the-art) of AI model performance over time, measured as **horizon length** -- the longest task duration a model can reliably complete. It fits trend lines to historical data and projects forward with uncertainty bands.

### Projection Models

- **Linear**: Single OLS fit extrapolated forward. Performance doubles every ~120-180 days.
- **Piecewise linear**: Multiple OLS segments with user-selected breakpoints (e.g., one trend pre-GPT-4o, another after). Last segment is extrapolated.
- **Superexponential**: Doubling time itself shrinks over time (y = A + K * 2^(d/halflife)), with a configurable floor to prevent runaway projections.

All three sample 20,000 trajectories to produce fan charts with 50%, 80%, and 90% confidence intervals.

### Sidebar Controls

| Control | What it does |
|---|---|
| **Projection basis** | Choose Linear, Piecewise linear, or Superexponential |
| **Advanced options** | Tune doubling-time CI, position CI, distribution shape, segment breakpoints |
| **Milestones** | Toggle horizontal reference lines at 1 work-day, 1 work-week, 1 work-month |
| **Labels** | Show/hide model name annotations on data points |
| **GPT-4o+ only** | Restrict view to post-GPT-4o models |
| **Use p80** | Switch from p50 (median) to p80 (80th percentile) reliability metric |
| **Log scale** | Toggle y-axis between log2 and linear (hours) |
| **Project as of** | Backtest: project from an earlier model's vantage point |

### Output

- Interactive Plotly fan chart with historical data points and projected trend through EOY 2026
- Metrics row showing projected horizon at 6 target dates (today through EOY 2029)
- Milestone probability tables: likelihood of reaching 1 work-week, 1 work-month, 1 work-year
- Estimated arrival dates for each milestone with 80% confidence intervals

## Data

`benchmark_results_1_1.yaml` contains METR-Horizon-v1.1 results for ~23 models (GPT-2 through Claude 4.6 Opus, spanning 2019-2026). Each model entry includes release date, p50/p80 horizon lengths with confidence intervals, and SOTA flags.

## Time Units

All time labels use work-time conventions:

| Unit | Hours |
|---|---|
| 1 day | 8h |
| 1 week | 40h |
| 1 month | 176h |
| 1 year | 2000h |
