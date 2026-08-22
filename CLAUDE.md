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
| UK Cyber | `render_ukcyber()` | `aisi_cyber_narrow.csv` → `load_ukcyber()`; `aisi_cyber_tlo.csv` → `load_ukcyber_tlo()` | success rate % (0-100 bounded, logit-projected) + open-weight lag in months; plus a TLO cyber-range cross-check in steps (`_render_ukcyber_tlo()`) and a callout for open-weight models only the range has measured (`_render_ukcyber_newest_open()`) |
| Revenue | `render_revenue()` (~3897) | hardcoded `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` | ARR in billions |
| Employment | `render_employment()` (~4353) | derived from RLI frontier + slider assumptions (no external feed) | unemployment % / jobs lost |
| ECI Company Gap | `render_eci_gap()` (~5223) | `epoch_capabilities_index.csv` (filtered by org/country) | linear score gap |
| Data Centers | `render_data_centers()` (~5669) | `data_centers.csv` + `data_center_timelines.csv` → `load_data_centers()` | H100-equiv / power / cost |
| Compute vs Capabilities | `render_compute_capabilities()` (~7344) | data centers (`dc_all`) + ECI | train-FLOP frontier vs ECI; ends with China's ETA to `_CC_CN_TARGET_ECI` (`_render_cc_china_target()`) |

### Data Sources and How to Update

Six external data feeds back the tabs; the rest are derived. Canonical sources and refresh method:

| File / table | Canonical source | How to refresh |
|---|---|---|
| `benchmark_results_1_1.yaml` | METR | Download `https://metr.org/assets/benchmark_results_1_1.yaml` and overwrite |
| `epoch_capabilities_index.csv` | Epoch AI | Extract `epoch_capabilities_index.csv` from `https://epoch.ai/data/benchmark_data.zip` and overwrite. Epoch recomputes scores live, so existing rows drift slightly on each pull |
| `data_centers.csv` | Epoch AI | Download `https://epoch.ai/data/data_centers/data_centers.csv` and overwrite |
| `data_center_timelines.csv` | Epoch AI | Download `https://epoch.ai/data/data_centers/data_center_timelines.csv` and overwrite. Column order differs from older pulls; the loader uses `DictReader` (by header name) so this is safe. **One curated deletion — see below** |
| `_RLI_RAW` (hardcoded) | Scale Labs RLI leaderboard (`labs.scale.com/leaderboard/rli`) / `remotelabor.ai` | Hand-edit new rows |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (hardcoded) | Press reports (The Information, Reuters, etc.) | Hand-edit `(date, ARR_in_billions)` tuples |
| `aisi_cyber_tlo.csv` | UK AISI Figure 2 (same post) + [UK AISI/CAISI Kimi K3 assessment](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) (2026-07-23) | **Not downloadable.** 9 rows digitized from `fig2-ranges.png`; Kimi K3's 17.0 quoted from the CAISI post's prose. Calibration and the four published-number validation checks are in the file's `#` header and guarded by `TestUkCyberTlo`. Dates here are **published release dates**, not chart-derived — the figure's x-axis is tokens |
| `aisi_cyber_narrow.csv` | UK AISI blog, [open-weight cyber gap post](https://www.aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber) (2026-07-17) | **Not downloadable.** AISI publishes no numbers for this chart — values were digitized from `fig1-narrow.png` by pixel analysis. Refreshing is a *figure-unchanged check*, not a download: re-fetch the PNG and confirm gridlines still sit at row 547 (=100%) / row 1760 (=0%) and that each row's (date, score) still lands on its marker colour. Only re-digitize if the figure actually changed, then re-verify against `test_digitized_dates_match_known_releases` and `test_optimistic_bracket_reproduces_aisi_published_lags` — those two are the calibration guards. Hand-editing a row is fine if AISI states a number in prose. See `.claude/commands/update-data.md` for the full recipe and for AISI cyber data that exists but is deliberately not ingested |

Before overwriting a CSV wholesale, diff by key column to confirm no locally-curated rows would be lost (ECI key = `Model version`; DC metadata key = `Name`; timelines key = `Data center` + `Date`). After any data change, sanity-check with `_VP_TESTING=1 python3 -c "import visualize_projection as v; ..."` calling the relevant loader, then run the tests.

#### The one curated deletion: `Fluidstack Lake Mariner`

`data_center_timelines.csv` is **not** a byte-faithful mirror of Epoch's export. As of the
2026-08-08 pull, Epoch renamed the site to `Anthropic Lake Mariner` in `data_centers.csv`
and re-scoped it to buildings CB3–5, but the timelines export still emits 8 stale rows under
the old name covering the *whole* site (CB1–5, i.e. including the Core42-leased buildings).
Epoch's live page lists 77 sites and no longer includes `Fluidstack Lake Mariner`.

`load_data_centers()` is **timelines-driven** — metadata only supplies the company label, so a
timeline series with no metadata row still loads as its own site (company falls back to the
first name token). Ingesting verbatim therefore materializes a phantom 78th site that
double-counts Lake Mariner: +0.4% of current H100-equiv, +4.7% of the 2027-03 total. The 8
rows are deleted locally. **Re-delete them on every refresh until Epoch fixes the export**;
check with `set(timelines['Data center']) - set(metadata['Name'])`.

Two other timeline-only names are legitimate and must be kept: `EdgeCore Mesa PH03` (a
long-standing orphan) and `DayOne Kempas` (new in the 2026-08-08 pull). Neither duplicates
another site — only Lake Mariner does.

### Key Sections of visualize_projection.py

Line numbers below are approximate — grep for the function/table name to locate the current line.

- Shared helpers — `pretty()`, `log2min_to_label()`, `fmt_hrs()`, `fit_line()`, `_fit_slope_p50_intercept_display()`, distribution samplers, `_ss_number_input()`, `superexp_trajectory()`, `_logit()`/`_inv_logit()`
- Backtesting helpers — `_backtest_stats()`, `_bt_color_for()`, `_add_backtest_traces()`, `_backtest_summary()`
- Data loading — `load_frontier()` / `load_metr_all()` (YAML), `load_eci_frontier()` and `load_eci_compute()` (ECI CSV, with dedup + running-max frontier), `load_rli_data()`, `load_data_centers()`, `load_ukcyber()`
- Data-center aggregation — `_dc_envelope()` (largest single site overall), `_dc_company_series()` (each company's largest single site), `_dc_company_pooled_series()` (the sum of a company's N largest sites, backing the "networking multiple data centers" section; `n_sites=None` pools all). `n_sites=1` must reproduce `_dc_company_series()` exactly — the two charts sit next to each other and would look broken if they drifted; `TestDcCompanyPooledSeries` asserts it
- UK Cyber lag helpers — `_ukc_frontier_crossing()` (interpolated crossing + bracketing models), `_ukc_frontier_match_for_score()` / `_ukc_frontier_below_for_score()` (bracket ends), `ukc_lag_rows()`, `ukc_tlo_lag_rows()` (shared TLO rows), `ukc_open_only_on_tlo()`, `ukc_target_eta()` / `ukc_target_eta_direct()`
- Compute-vs-capabilities helpers — `_cc_pooled_decomp()`, `_cc_iso_compute_rate()`, `_cc_country_frontier()` / `_cc_country_compute_frontier()`, `_cc_frontier_eci_slope()`, and the China-ETA trio `_cc_cn_target_years()` / `_cc_release_gap_days()` / `_cc_first_reached()`
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
- **Data Centers, "Capacity (time to GPT-5 / Mythos)"**: stored as *training runs per
  2-month window* (`gpt5s` / `mythos`), displayed as *time to train one model*
  (`kind='traintime'` → `_dc_fmt_value` → `_fmt_duration_days`, days = `_DAYS_2MO` / runs).
  Storing the count keeps every "largest data center" aggregation in the tab a plain max
  (envelope, per-company max, ranking, bar sort) — bigger count = faster site, so the
  ordering is identical. It also makes the networked-sites section a plain **sum**: two
  equal sites are two runs per window, i.e. half the time to train one model. Inverting
  the stored value would break both. Nor should you invert it to make the axis read
  forward; the axis is relabelled instead, via `_dc_duration_ticks()` (ticks at round durations,
  passed through `_dc_layout(kind=…)`), and a caption under the first chart says the
  labelled time shrinks as the line rises

### UK Cyber tab caveats

Three constraints are load-bearing; don't "simplify" them away:

1. **The frontier is closed-weight only.** Open-weight models are the subject being measured against it, so `load_ukcyber()` excludes them from the running max.
2. **Lag is interpolated between the bracketing models, and carries a bracket.** `_ukc_frontier_crossing()` interpolates between the last frontier model below a score and the first above it. Snapping to the next model up (the earlier implementation) equates scores that can be far apart: DeepSeek-V4-Pro's 55.7% falls in a 10-point gap between GPT-5 (52.5%) and Opus 4.5 (62.6%), so snapping credited it with ~7 points it doesn't have and understated its lag by ~2.4 months. The distortion scales with how far the nearest model is — GLM-5.2 sits 2.3 points under Opus 4.6 and barely moved (4.3 → 4.7 mo). `ukc_lag_rows()` returns `lag_months` (interpolated point estimate) plus `lag_lo`/`lag_hi` from the two bracketing models; no model was released in the gap, so that width is real uncertainty, not noise.
3. **`lag_lo` is the AISI-compatible bound.** AISI's printed annotations (4.3mo, 5.0mo) use next-model-up, which is exactly `lag_lo`. `test_optimistic_bracket_reproduces_aisi_published_lags` asserts on it, so it doubles as the calibration check on the digitization. Changing the lag method must not break that guard.

Rejected alternatives, for the record: matching to the *last model below* is defensible but maximally pessimistic (DeepSeek 8.5mo); dating the crossing off the fitted OLS curve imports global fit error into a local question (the curve "reaches" 55.7% in Jun 2025, when the best observed model was GPT-5 at 52.5%, giving 9.9mo).

4. **The TLO cross-check compares on `lag_lo`, not on the point estimate.** `aisi_cyber_tlo.csv` carries AISI's other cyber measure (avg steps of 32 on the "The Last Ones" cyber range) and reuses `_ukc_frontier_crossing()` / `ukc_lag_rows()` unchanged by storing steps in `cyber_score`. But the interpolated point estimates are **not comparable across the two datasets**: DeepSeek-V4-Pro's narrow score falls in a 10-point frontier gap (inflating it to 7.4mo) while its TLO score sits below every frontier model, so no interpolation happens at all (6.8mo, and `lag_hi` is `None` — it is a lower bound only). On point estimates the ordering therefore inverts, for reasons about frontier sampling rather than capability. Compared on `lag_lo` — the next-model-up convention AISI's own chart titles use — the two reproduce AISI's headline exactly: narrow tasks 4.3–5.1mo ("4-5 months prior"), cyber range 6.7–6.8mo ("7 months prior"), together "4 to 7 months". `test_reproduces_both_figure_titles` guards this.

5. **The two suites don't cover the same models, and the callout above the chart says so.** AISI/CAISI ran only a selective set on Kimi K3 (ExploitBench + TLO), not the 70-task narrow suite, so it has no narrow-task score — it is absent from the main chart, the projection, and the 90% ETA by data availability, not by a filter. `ukc_open_only_on_tlo()` finds the newest open-weight model in the TLO rows but not the narrow ones and `_render_ukcyber_newest_open()` promotes it to a bordered callout directly under the header, showing the `lag_lo`–`lag_hi` bracket (not the point estimate — TLO's frontier is sparse enough that Kimi K3's bracket is 3.3–5.3mo). It returns `None`, and the callout vanishes, once the narrow suite catches up; it also ignores a TLO-only model that is *older* than every open-weight point on the chart, since that isn't the chart being out of date. Both the callout and `_render_ukcyber_tlo()` get their rows from `ukc_tlo_lag_rows()` so they can't drift onto different frontiers. Guarded by `TestUkCyberOpenOnlyOnTlo`.

The dataset has no US open-weight and no Chinese closed-weight models, so country and openness are perfectly confounded. The tab headline says "China" because the two Chinese models are also the only open-weight ones — `_UKC_CONFOUND_PLAIN` is folded into the fine-print caption so that's stated wherever the tab says "China". It was previously a standalone `st.warning` banner; if the caveat ever needs that prominence again, promote the same constant rather than adding a second wording to keep in sync.

`ukc_target_eta()` answers "when do open-weight models reach `_UKC_TARGET` (90%)": the frontier's interpolated crossing of the target, plus the min/max measured lag. `ukc_target_eta_direct()` fits the open-weight points themselves as a cross-check only — two models 53 days apart make that slope very sensitive, so it is never the headline.

### ECI tabs — organization matching

The Epoch ECI tab and the ECI Company Gap tab read the same CSV and **must resolve
organizations identically**. Both match Epoch's `Organization` field by *substring*:
the ECI tab via `load_eci_frontier(orgs=…)` / `_ECI_ENTITY_SPECS`, the gap tab via
`_ecg_org_display()` / `_ECG_ORG_MAP`.

Don't turn either back into an exact-key lookup. The gap tab used to do exactly that,
and the tabs silently disagreed: Epoch spells Google four ways (`Google DeepMind`,
`Google`, `Google DeepMind,Google`, `Google,Google DeepMind`), so a map keyed only on
`Google DeepMind` dropped four Google models and drew a different 2025 frontier point
(Gemini 2.0 Pro Exp 135.43 / 2025-02-05 instead of Gemini 2.0 Flash Thinking Exp
136.00 / 2025-01-21). Google's *current* gap was unaffected, which is why it went
unnoticed — the divergence was only in the historical curve.

**Adding a company is one row in `_ECI_COMPANIES`.** That registry is the single
source of truth for both tabs; `_ECG_ORG_MAP`, `_ECG_COLORS`, `_ECG_COUNTRY`,
`_ECI_ENTITY_SPECS`, and `_ECI_ENTITY_SLUG` are all derived from it at import, so the
two tabs cannot hold different ideas about which models belong to whom. Don't
hand-write those derived tables. `_ECI_COUNTRY_ENTITIES` stays separate because the
"best" aggregates are country filters, not companies — and it must come first, since
`_ECI_ENTITY_OPTIONS[0]` is the ECI tab's default benchmark.

Keep each company's `orgs` list minimal: substring matching makes longer variants
redundant ("Google" already catches "Google DeepMind,Google"). Matching is only
unambiguous while no `Organization` string contains two different companies —
`TestEcgOrgMatching` asserts that against the live CSV, along with tab-to-tab set
equality, the four Google spellings, registry well-formedness, and slug round-tripping.

There was also an `_ECG_DASH` table of per-company line styles. No render path ever
read it — the gap tab styles one highlighted company at a time via `_ECG_COLORS` — so
it was deleted rather than kept as another table to sync.

### Compute vs Capabilities — the buildout-vs-release-timing panel

`_cc_company_buildout()` (bottom of the Data Centers tab) is a **pure timing
test**: each capacity step of a lab's largest single data center, shifted
forward `_CC_RELEASE_LAG_DAYS` (90d = 60d training + 30d release prep), against
when that lab's models actually shipped. Capability is never compared.

Two directions with two different clocks, and they must not contradict each
other — the tables sit one above the other:

- **Backward** (release → cluster), `_responsible_cluster`: the latest step
  online at least one training run (`_CC_TRAIN_FLOOR_DAYS`, 60d) before the
  release. The extra ~1mo release prep is compressible polish, not a gate.
- **Forward** (cluster → release), `_cc_forward_match()`: three tiers, each
  tried only when the one above finds nothing.
  1. first running-max release from `pred − _CC_EARLY_GRACE_DAYS` (7d) onward;
  2. else the earliest running-max release the *backward* match already gave
     this exact step. Required because the two clocks differ: the 60d floor
     admits a release up to 30d early while the forward grace is 7d, so without
     this tier a step whose model shipped 24d early (Claude Opus 4.8 vs New
     Carlisle) would skip to tier 3 and cite a lesser model;
  3. else the lab's next release of any kind, from `_cc_company_all_releases()`,
     flagged `fallback` and marked † wherever it renders.

Three things are load-bearing:

1. **The grace is 7d, not the 30d the 60d floor would allow.** At 30d clusters
   start claiming the *same, earlier* model (Meta's Eagle Mountain, Temple and
   Prometheus steps all collapse onto Muse Spark) and the forward table goes
   degenerate. 7d changes exactly one pre-existing match — Google New Albany →
   Gemini 2.0 Flash, 6d early — and that one moves *into* agreement with the
   backward match.
2. **Tier 3 exists because "frontier release" is a running max and Epoch
   recomputes ECI live.** A real flagship can end up scored under its own
   predecessor and vanish from the series without its release date changing:
   the 2026-08-18 pull put GPT-5.6 Sol at 161.08 against GPT-5.5 Pro's 161.73,
   so Fairwater Wisconsin matched nothing at all. That is a fact about
   rescoring, not about when OpenAI shipped, and this panel compares dates only.
   Tier-3 matches are drawn hollow, marked †, and **excluded from the headline
   median**, which is a claim about record-setting releases.
3. **`_cc_company_all_releases()` keys on `(Model name, Release date)`.** Model
   name alone merges GPT-4o's May and August 2024 releases; Display name alone
   splits the ~10 reasoning-effort rows of one model, and which suffixed variant
   would win a dedup flips between Epoch pulls. Same-day releases sort by
   descending ECI so a step is offered the flagship (2026-07-09 ships Sol 161.08,
   Terra 158.78 and Luna 156.22 together).

`TestCcForwardMatch` and `TestCcCompanyAllReleases` guard all of the above.

### Compute vs Capabilities — China's ETA to a target ECI

The tab's last section (`_render_cc_china_target()`) answers "when does China cross
`_CC_CN_TARGET_ECI`" (161) with a date distribution instead of the gap metrics the
sections above it report. Three things are load-bearing:

1. **The target is meant to track today's US frontier.** 161 is where the US sits
   as of mid-2026, which is what makes the framing —"China
   matching where the US is *now*" — true. `TestCcCnTargetIsTodaysUsFrontier`
   asserts the constant stays at or under the US frontier and within 5 points of
   it; if an ECI refresh breaks that test, retarget the constant rather than
   loosening the test, and re-check the caption wording. Don't hardcode the
   anchor model in prose — Epoch recomputes live and the anchor moves: the
   2026-08-18 pull dropped GPT-5.6 Sol 161.65 → 161.03 (below GPT-5.5 Pro's
   161.60, so Sol left OpenAI's running-max frontier entirely) and lifted Claude
   Fable 5 to 162.30, which is the current US frontier.
2. **The rate is the same two-engine model as Chart B**, deliberately: algorithmic
   term (iso-compute rates, mode = China's own) + `a_partial` × China's compute
   growth. Don't swap in a direct fit of China's frontier — the section would then
   contradict the chart above it. The bottom-up rate currently runs hot (~14
   ECI/yr) against a frontier that has managed 10–13, so `_cc_cn_target_years()`
   takes a `pace_lo`/`pace_hi` band that the caller derives from China's observed
   slope over `_CC_GAP_WINDOWS`. That reality check is the main uncertainty: the
   two iso-compute fits sit within a point of each other and alone would produce a
   spuriously tight band.
3. **The crossing waits for a release.** The frontier is a step function, so
   clearing the bar needs a model to ship. `release_gap_days` (from
   `_cc_release_gap_days()`, the median recent inter-release gap) adds an
   exponential wait on top of the smooth crossing. This is why the median-crossing
   diamond sits *right* of where the fan meets the target line — the fan is the
   smooth capability path, the diamond and the vertical band are release-inclusive.
   Not a plotting bug; don't "fix" it by aligning them.

The fan traces set `mode='lines'` explicitly. The fan spans only ~6 quarters, and
plotly defaults a Scatter under 20 points to `lines+markers`, which studs the band
outline with stray default-blue dots.

### Backtesting

"Project as of" model selector lets you project from a historical vantage point. `_backtest_stats()` compares actual future models against projected trajectories, color-coded by which CI band they fall in.
