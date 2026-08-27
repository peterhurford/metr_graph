# AI Capability Projections

This is an interactive Streamlit dashboard that plots the **frontier** of AI progress across eleven
different measures — benchmark capability, labor automation, cyber offense, revenue,
compute buildout — fits a trend to each, and projects it forward with uncertainty bands.

> Note: In many cases, these are **projections**, not forecasts: they assume the current trend continues.

## Quick start

```bash
pip install -r requirements.txt
streamlit run visualize_projection.py     # or: make app
```

Opens at http://localhost:8501. Python 3.11 (`.python-version`).

Pick a tab from the sidebar radio. Every tab deep-links: `?tab=pacing` opens the Pacing tab,
and the sidebar's *Share view* button copies a URL carrying your current control settings.

## The tabs

| Tab | `?tab=` | Data source | What it measures |
|---|---|---|---|
| METR Horizon | `metr` | `benchmark_results_1_1.yaml` | Task horizon length — the longest task a model completes reliably. log₂(minutes) |
| Epoch ECI | `eci` | `epoch_capabilities_index.csv` | Epoch's Capabilities Index, a pooled benchmark score |
| ECI Company Gap | `ecigap` | Same CSV, split by organization | How far each lab/country trails the frontier |
| Remote Labor Index | `rli` | `_RLI_RAW` (hardcoded) | Share of real remote-work projects completed. Fitted in logit space |
| RSI | `rsi` | `_RSI_RAW` / `_RSI_SURVEY` (hardcoded) | Anthropic's internal AI-R&D benchmark + staff-survey speedup; ends with the *Capabilities Milestones* cards and the blended RSI projection |
| UK Cyber | `ukcyber` | `aisi_cyber_narrow.csv`, `aisi_cyber_tlo.csv` | AISI cyber success rates, and how many months open-weight models trail the closed frontier |
| Employment | `employment` | RLI frontier + assumptions | Unemployment / jobs displaced under slider assumptions |
| Revenue | `revenue` | `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (hardcoded) | OpenAI and Anthropic ARR |
| Data Centers | `datacenters` | `data_centers.csv` + `data_center_timelines.csv` | H100-equivalents, power, cost, time-to-train; ends with a US-vs-China buildout panel |
| Compute/capabilities/diffusion | `computecap` | Data centers + ECI | Training-compute frontier against capability, and China's ETA to today's US frontier |
| Pacing | `pacing` | Data centers | What a US pause buys: China's catch-up, then when each entity can first mount a run of the paused US scale |

Each tab has its own render function (`grep '^def render_'`), its own sidebar controls, and a
reset button backed by a per-tab `_<TAB>_DEFAULTS` / `_<TAB>_RESET_KEYS` pair.

### Projection engine

Three bases, offered per tab where they make sense:

- **Linear** — one OLS fit, extrapolated.
- **Piecewise linear** — multi-segment OLS at user-chosen breakpoints; the last segment is extrapolated.
- **Superexponential** — doubling time itself decays (`superexp_trajectory()`), with a floor.

All of them sample `N_SAMPLES` (5,000) trajectories into a Plotly fan chart with 50/80/90%
bands. Bounded metrics (RLI, UK Cyber, RSI CoBench) are fitted in **logit** space so the
trend can't run through 100%; unbounded ones (METR horizon, revenue, survey speedup) are
fitted in log space.

"Project as of" backtests: it projects from a historical vantage point and colors later
actual models by which CI band they landed in (`_backtest_stats()`).

## Repository layout

```
visualize_projection.py   the whole app — loaders, engine, all eleven render functions (~14k lines)
test_visualize_projection.py   unit tests against a fake Streamlit module
test_integration.py            integration tests through Streamlit's AppTest runtime
conftest.py               test-speed setup (see below)
CLAUDE.md                 the design record — read this before changing anything non-trivial
.claude/commands/update-data.md   the data-refresh recipe
*.csv, *.yaml             the data feeds
```

There is no build system and no CI. Runtime deps are `streamlit`, `numpy`, `plotly`,
`pyyaml`.

The app is deliberately one file. It is large, but every tab shares the helpers at the top
(`pretty()`, `fit_line()`, the samplers, `_ss_number_input()`, `_fn_caption()`), and
splitting it has not been worth the import churn. Navigate by `grep`, not by scrolling.

## Development

```bash
pip install -r requirements-dev.txt
pytest -v                     # or: make tests        (~11s, 578 tests)
pytest -v -n0                 #     make tests-serial (needed for --pdb or live output)
_VP_SAMPLES=5000 pytest -v    #     make tests-full   (production sample count)
pytest test_visualize_projection.py::TestPretty::test_known_name -v
```

Two environment variables shape a test run:

- `_VP_TESTING=1` skips the render dispatch at the bottom of the file, so the module imports
  without a Streamlit server. Handy for poking at a loader directly:
  ```bash
  _VP_TESTING=1 python3 -c "import visualize_projection as v; print(len(v.load_eci_frontier()))"
  ```
- `_VP_SAMPLES` sets the Monte Carlo trajectory count. `conftest.py` turns it down to 400 for
  tests; nothing asserted depends on the count.

### The suite is fast on purpose

~11s, down from over three minutes. `conftest.py` shares one bytecode cache across every
`AppTest.run()` (patching **both** places Streamlit constructs a `ScriptCache` — patch one
and the win vanishes), shrinks the Monte Carlo, and `pytest.ini` runs `-n auto`. An
`AppTest` case costs about 0.1s, so prefer adding one to skipping coverage. What is
expensive is defeating the shared cache: don't write a test that mutates
`visualize_projection.py` on disk or builds its own runner out of `streamlit.testing`
internals.

### A flaky test you may hit

The app never seeds its RNG, so Monte Carlo medians wobble. At the reduced test sample count
`TestPacingTab::test_chinese_run_length_trades_compute_against_wall_clock` fails roughly one
run in six; it passes consistently under `_VP_SAMPLES=5000`. If a red test looks like that
one, re-run it before chasing it.

## Data

Six data files and three hardcoded tables; every other tab derives from them. `.claude/commands/update-data.md` has
the full refresh recipe, including the AISI cyber data that is deliberately *not* ingested.

| File / table | Source | Type |
|---|---|---|
| `benchmark_results_1_1.yaml` | METR (26 models, 2019–2026; 18 on the frontier) | download |
| `epoch_capabilities_index.csv` | Epoch AI benchmark data | download (inside a zip) |
| `data_centers.csv` / `data_center_timelines.csv` | Epoch AI Frontier Data Centers (85 sites) | download |
| `_RLI_RAW` | Scale Labs Remote Labor Index | hand-edited |
| `_RSI_RAW` / `_RSI_SURVEY` | Anthropic, Redacted Risk Report §3.4 | hand-edited, read off a figure |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` | Press reports and company disclosures | hand-edited |
| `aisi_cyber_narrow.csv` / `aisi_cyber_tlo.csv` | UK AISI cyber blog posts | **digitized from published PNGs** |

Three things to know before touching data:

1. **Some of it is digitized, not published.** AISI prints no numbers for its cyber figures;
   both CSVs were recovered from the images by pixel analysis, and each file's `#` header
   records the calibration and the checks that validate it. Refreshing them is a
   *figure-unchanged check*, not a download. `test_optimistic_bracket_reproduces_aisi_published_lags`
   and friends are the calibration guards — if you change the digitization, they must still pass.
2. **Epoch recomputes ECI live**, so existing rows drift on every pull. That's expected. What
   isn't expected is losing rows: diff by key column before overwriting any CSV
   (ECI = `Model version`, DC metadata = `Name`, timelines = `Data center` + `Date`).
3. **One curated deletion.** `data_center_timelines.csv` is not a byte-faithful mirror:
   Epoch's export still emits stale `Fluidstack Lake Mariner` rows that materialize a phantom
   site double-counting Lake Mariner. Re-delete them on every refresh, checking
   `set(timelines['Data center']) - set(metadata['Name'])`. The other two timeline-only names
   (`EdgeCore Mesa PH03`, `DayOne Kempas`) are legitimate.

After any data change, sanity-check the relevant loader with `_VP_TESTING=1 python3 -c ...`,
then run the tests.

## Contributing

**Read `CLAUDE.md` first.** It is the design record: for every non-obvious choice in the app
it says what the choice is, what the wrong-looking alternative was, and which test holds the
line. Most surprising-looking code in this repo is deliberate and documented there.

Conventions that will come up:

- **Every claim gets a guarding test.** The suite is large because the app's charts encode
  judgment calls; a change that alters a number should move a test with it.
- **Caveats hang on the thing they qualify**, not in a paragraph. Use `st.metric(help=…)`
  where a metric exists, `_fn_caption(text, (phrase, note))` to make a phrase in a caption
  hoverable in place, and `_fn_line()` for a line that must not read as fine print.
  `test_footnotes_anchor_to_their_phrase` holds the app at zero unanchored footnotes.
- **Don't fork shared machinery.** The Pacing and Compute/capabilities tabs are built from the
  Data Centers tab's aggregation functions on purpose — two tabs quoting different numbers for
  the same quantity reads as a bug. Several tests exist only to keep the engines honest across
  tabs.
- **Adding a tab** means: a `render_*()` function, an entry in `_TAB_OPTIONS` and
  `_SLUG_FOR_TAB`, a branch in the dispatch block at the bottom of the file, and a
  `_<TAB>_DEFAULTS` / `_<TAB>_RESET_KEYS` pair. Session-state number inputs go through
  `_ss_number_input()` so they survive a rerun and round-trip through the URL.
- **Adding a company to the ECI tabs** is one row in `_ECI_COMPANIES` — the maps both tabs use
  are derived from it at import. Don't hand-write them.
- **Don't seed the RNG** to make a test deterministic; assert on something the sample count
  can't move, or widen the tolerance.

Run `pytest` before opening a PR. Data refreshes and code changes are easier to review apart,
so please keep them in separate commits.

## Units

| Context | Internal unit |
|---|---|
| METR | log₂(minutes), displayed as hours. Work-time: 1d = 8h, 1w = 40h, 1mo = 176h, 1y = 2000h |
| ECI | Linear score. DPP = days per +1 ECI point |
| RLI / UK Cyber / RSI CoBench | 0–100, projected in logit space |
| Revenue | ARR, billions USD |
| Data centers | H100-equivalents, MW, USD, and training runs per 2-month window (displayed as time-to-train-one-model) |

## License

None yet — the repo ships no license file, so default copyright applies. Ask before reusing
or redistributing.
