# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run visualize_projection.py    # http://localhost:8501
```

## Running Tests

```bash
pytest -v                                    # all tests
pytest test_visualize_projection.py -v       # unit tests (fake Streamlit)
pytest test_integration.py -v                # integration tests (AppTest)
pytest test_visualize_projection.py::TestPretty::test_known_name -v  # single test
pytest -v -n0                                # serial — needed for --pdb / live output
_VP_SAMPLES=5000 pytest -v                   # production sample count (see below)
```

Test deps: `pip install -r requirements-dev.txt` (adds `pytest-xdist`). Unit tests use a fake Streamlit module (`_FakeStreamlit` / `_Noop`) so the app imports without a server; `_VP_TESTING=1` skips rendering at module level. Integration tests use Streamlit's `AppTest` headless runtime (30s timeout).

## Project Structure

- **`visualize_projection.py`** — single-file Streamlit app containing all logic
- **`test_visualize_projection.py`** / **`test_integration.py`** — unit tests (fake Streamlit) and integration tests (AppTest)
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 data (~23 models)
- **`epoch_capabilities_index.csv`** — Epoch ECI data (~714 model-variants)
- **`data_centers.csv`** / **`data_center_timelines.csv`** — Epoch Frontier Data Centers: one metadata row per site, many dated capacity rows per site
- **`aisi_cyber_narrow.csv`** / **`aisi_cyber_tlo.csv`** — AISI narrow cyber success rates (12 models); AISI/CAISI cyber-range "The Last Ones" avg steps of 32 (10 models). Both **chart-digitized, not published feeds** — see each file's `#` header

No build system, no CI/CD, no package manager beyond requirements.txt (`streamlit`, `numpy`, `plotly`, `pyyaml`).

## Architecture

Ten-tab Streamlit dashboard selected via sidebar radio (`active_tab`, `_TAB_OPTIONS`) with URL deep-linking (`?tab=<slug>`). Each tab has its own render function, sidebar controls, and (where applicable) projection engine. Slugs (`_SLUG_FOR_TAB`): `metr`, `eci`, `ecigap`, `rli`, `ukcyber`, `employment`, `revenue`, `datacenters`, `computecap`, `pacing`.

### Tabs and Render Functions

| Tab | Function | Data Source | Metric |
|---|---|---|---|
| METR Horizon | `render_metr()` | `benchmark_results_1_1.yaml` → `load_frontier()` | log₂(minutes) |
| Epoch ECI | `render_eci()` | `epoch_capabilities_index.csv` → `load_eci_frontier()` | linear score |
| Remote Labor Index | `render_rli()` | `_RLI_RAW` → `load_rli_data()` | logit-transformed score |
| UK Cyber | `render_ukcyber()` | `aisi_cyber_narrow.csv` → `load_ukcyber()`; `aisi_cyber_tlo.csv` → `load_ukcyber_tlo()` | success rate % + open-weight lag in months; plus a TLO cyber-range cross-check in steps (`_render_ukcyber_tlo()`) and a callout for models only the range has measured (`_render_ukcyber_newest_open()`) |
| Revenue | `render_revenue()` | `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` | ARR in billions |
| Employment | `render_employment()` | RLI frontier + slider assumptions | unemployment % / jobs lost |
| ECI Company Gap | `render_eci_gap()` | `epoch_capabilities_index.csv` (by org/country) | linear score gap |
| Data Centers | `render_data_centers()` | `data_centers.csv` + timelines → `load_data_centers()` | H100-equiv / power / cost; ends with a US-vs-China by-country projection (`_dc_render_country_panel()`) |
| Compute vs Capabilities | `render_compute_capabilities()` | data centers (`dc_all`) + ECI | train-FLOP frontier vs ECI; ends with China's ETA to `_CC_CN_TARGET_ECI` (`_render_cc_china_target()`) |
| Pacing | `render_pacing()` | data centers (`dc_all`) | date each entity first commands a threshold-scale training run |

### Data Sources and How to Update

Six external feeds; the rest are derived. `.claude/commands/update-data.md` has the full
recipe, including the AISI cyber data deliberately *not* ingested.

| File / table | Source | How to refresh |
|---|---|---|
| `benchmark_results_1_1.yaml` | METR | Overwrite from `https://metr.org/assets/benchmark_results_1_1.yaml` |
| `epoch_capabilities_index.csv` | Epoch AI | Extract from `https://epoch.ai/data/benchmark_data.zip`. Epoch recomputes scores live, so existing rows drift on each pull |
| `data_centers.csv` | Epoch AI | Overwrite from `https://epoch.ai/data/data_centers/data_centers.csv` |
| `data_center_timelines.csv` | Epoch AI | Same, `…/data_center_timelines.csv`. Column order varies between pulls; the loader uses `DictReader`, so that's safe. **One curated deletion — see below** |
| `_RLI_RAW` (hardcoded) | Scale Labs RLI leaderboard (`labs.scale.com/leaderboard/rli`) / `remotelabor.ai` | Hand-edit rows |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (hardcoded) | Press reports | Hand-edit `(date, ARR_in_billions)` tuples |
| `aisi_cyber_tlo.csv` | UK AISI Figure 2 + [Kimi K3 assessment](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) | **Not downloadable** — 9 rows digitized from `fig2-ranges.png`, one value quoted from prose. Calibration and validation checks are in the file's `#` header, guarded by `TestUkCyberTlo`. Dates are **published release dates**; the figure's x-axis is tokens |
| `aisi_cyber_narrow.csv` | UK AISI [open-weight cyber gap post](https://www.aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber) | **Not downloadable** — AISI publishes no numbers; values digitized from `fig1-narrow.png` by pixel analysis. Refreshing is a *figure-unchanged check*: re-fetch the PNG, confirm gridline rows and marker colours still match, re-digitize only if the figure changed. `test_digitized_dates_match_known_releases` and `test_optimistic_bracket_reproduces_aisi_published_lags` are the calibration guards. Hand-editing a row is fine if AISI states a number in prose |

Before overwriting a CSV wholesale, diff by key column so no curated rows are lost (ECI = `Model version`; DC metadata = `Name`; timelines = `Data center` + `Date`). After any data change, sanity-check the relevant loader via `_VP_TESTING=1 python3 -c "import visualize_projection as v; ..."`, then run the tests.

#### The one curated deletion: `Fluidstack Lake Mariner`

`data_center_timelines.csv` is **not** a byte-faithful mirror of Epoch's export: Epoch renamed the site to `Anthropic Lake Mariner` in the metadata and re-scoped it to buildings CB3–5, but the timelines export still emits 8 stale rows under the old name covering the whole site. `load_data_centers()` is **timelines-driven** (metadata only supplies the company label), so those rows materialize a phantom site that double-counts Lake Mariner — +4.7% of the 2027-03 total. They are deleted locally; **re-delete on every refresh until Epoch fixes the export**, checking `set(timelines['Data center']) - set(metadata['Name'])`. Two other timeline-only names, `EdgeCore Mesa PH03` and `DayOne Kempas`, are legitimate and must be kept.

### Key Sections of visualize_projection.py

Roughly in file order: shared helpers (`pretty()`, `fit_line()`, distribution samplers,
`_ss_number_input()`, `superexp_trajectory()`, `_logit()`/`_inv_logit()`); backtesting
(`_backtest_*`); loaders (`load_frontier()`, `load_eci_frontier()` / `load_eci_compute()`,
`load_rli_data()`, `load_data_centers()`, `load_ukcyber()`); data-center aggregation
(`_dc_envelope()`, `_dc_company_series()`, `_dc_company_networked_series()`); UK Cyber lag
(`_ukc_*`, `ukc_*`); compute-vs-capabilities (`_cc_*`); data init and tab selector; the nine
`render_*()` functions (grep `^def render_`); dispatch at end of file (skipped when `_VP_TESTING=1`).

### Projection Engine (repeated per tab)

Three bases: **Linear** (single OLS), **Piecewise linear** (multi-segment OLS, last segment extrapolated), **Superexponential** (doubling time decays via `superexp_trajectory()` with a floor). All sample 5,000 trajectories into Plotly fan charts with 50%/80%/90% CI bands.

### Session State and Reset

Widget defaults live in per-tab `_RESET_DEFAULTS`; each tab's `_RESET_KEYS` lists its session state keys. The reset button pops all keys and calls `st.rerun()`. Custom number inputs use `_ss_number_input()` to persist via session state.

### Backtesting

"Project as of" projects from a historical vantage point; `_backtest_stats()` compares actual later models against the projected trajectories, colored by which CI band they land in.

### Internal Units

- **METR**: log₂(minutes), displayed as hours. Work-time: 1d=8h, 1w=40h, 1mo=176h, 1y=2000h
- **ECI**: linear score (~57-154). DPP = days per +1 ECI point
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
  `_dc_duration_ticks` (`traintime`). Don't reintroduce a per-metric axis divisor: the H100
  metric used to be called "Compute (x1M H100-equiv)" and divide its ticks by a million
  while `_dc_fmt_value` kept the bar text, hovers and the quarterly table in raw counts, so
  a bar reading "1.11M" sat under a tick reading "1". And route the snapshot bar chart's
  x-axis through the same dispatcher — it builds its own axis instead of calling
  `_dc_layout()`, and when it hand-rolled the branching it silently lacked the `flop` case.
  `TestDcAxisTicks` guards both halves


### Data-center company labels

`load_data_centers()` attributes a site to its primary listed `Users` value, falling back to
`Owner`, then to the first token of the site name. `_DC_COMPANY_ALIASES` then folds distinct
Epoch labels that are one company for presentation — currently just `Google DeepMind` →
`Google`. It applies to the derived label only; the CSVs keep Epoch's spellings.

Google is why it exists. Every Google site is `Owner="Google"`, but only some also carry
`Users="Google DeepMind #speculative"`, so a user-first rule split one TPU fleet in two on
nothing but whether Epoch filled an optional cell (Lancaster, `Users` blank, lists the same
TPU v5e/v5p/v6e/v7 as the tagged sites). That split drew two Google lines with Google blue on
the smaller one, left the pooled Columbus cluster dependent on both its Google sites
happening to share a tag, and had the quarterly table reporting the minority series — 2.6x
low by 2027Q4, contradicting the chart directly above it. The rest of the app already merged
them: `_cc_lab_for_site()` maps owner `Google*` to `Google`, and the ECI tabs substring-match
`"Google"` for the same reason.

### Tenant vs operator, and shared tenancy

Both the Data Centers and Pacing tabs carry an *Attribute each site to* control
(`dc_party` / `pc_party`, options `_DC_PARTY_OPTIONS`). `load_data_centers()` records
three attributions per site: `tenant` (= `company`: first-listed user → owner → name
token), `operator` (owner → token), and `users` — **every** listed user, aliased and
deduped. `_dc_with_party(dcs, party)` builds the view: 'operator' re-points `company`
at the owner; 'tenant' keeps `company` but sets a `companies` membership list to the
full users list (falling back to `[tenant]`).

Load-bearing:

1. **Shared sites count under every listed tenant.** Epoch lists Anthropic, Cursor
   *and* SpaceXAI as Colossus 2 users; crediting only the first made SpaceXAI's roster
   exclude its own flagship building. `_dc_series_for_metric` carries `companies`, and
   the two per-company aggregators (`_dc_company_series`,
   `_dc_company_networked_series`) group each site under **each** member. This never
   double-counts within a line (per-company lines are capability maxima), but lines
   are **not additive across companies** — the captions say so. Nothing on these tabs
   sums across companies; don't add such an aggregation without de-duplicating shared
   sites first.
2. **The raw `dc_all` keeps single membership.** `dc.get('companies', [company])`
   fallback means any code path not going through `_dc_with_party` — notably the
   Compute vs Capabilities tab — behaves exactly as before. `test_shared_site_counts_
   under_every_tenant` pins both halves.
3. **Hidden/unattributed sets are recomputed on the view**, so under 'operator' the
   size gate re-admits owners like Oracle (Stargate Abilene) that the tenant view
   never surfaces.

Guarded by `TestPacing` party tests and `TestDataCentersParty`.

`TestDcCompanyAliases` checks against the live CSV that every Google-owned site resolves to
one company, that both spellings are still present upstream (so the map isn't dead weight),
and that no label is a qualified form of another — the guard that would catch the next
`Meta AI`-vs-`Meta` style split.

### Who appears on the Data Centers tab

`_DC_EXCLUDE_COMPANIES` lists companies that aren't AI labs — colocation and neutral-host
operators whose recorded "company" is the landlord, not whoever trains on the hardware. The
list is no longer an unconditional hide. `_dc_hidden_companies()` drops a listed company
only while its largest single site stays under `_DC_EXCLUDE_MIN_H100` (100k H100-equiv)
within `_DC_EXCLUDE_HORIZON_DAYS` (365d). Hiding them all was wrong once one got big: QTS
Cedar Rapids is the largest single site in Epoch's data, so the tab's headline chart was
naming a smaller site as the record holder. Today the rule adds QTS, DayOne and Microsoft
and still hides Oracle, STACK, Stream, Vantage, EdgeCore and CoreWeave.

Four things are load-bearing:

1. **The threshold reads H100-equivalents, always** — never the metric currently selected.
   The same number in megawatts or dollars means something else, and two of the tab's
   metrics are inverted (bigger site = smaller value), so `>=` would read backwards.
2. **A rolling horizon, not the uncapped peak, and not `dc_end_year`.** Uncapped, every
   listed host eventually clears 100k on 2028+ buildout and the list stops meaning anything
   (`test_uncapped_peaks_would_defeat_the_exclusion`). And the roster is a property of the
   company, not of a slider — keying it to the user's projection window would make sites
   blink in and out as the window moves. The roster is stable well either side of a year:
   nothing changes between a 6- and 12-month horizon, and the next company to qualify does
   so only at ~18 months. Oracle is the nearest miss (~84k against the 100k bar); if an
   Epoch refresh crosses it, retarget the constant deliberately rather than loosening
   `test_current_roster_is_what_the_tab_says_it_is`.
3. **`†` marks capacity with no recorded tenant.** `load_data_centers()` sets `attributed`
   False when the company label came from the site-name fallback rather than Epoch's `Users`
   or `Owner`; `_dc_unattributed_companies()` marks a company only when *every* one of its
   sites is such a fallback, so Microsoft (listed as its own sites' user) draws plain while
   QTS and DayOne carry the mark. Solid-vs-dashed is already actual-vs-planned, hence a
   glyph rather than a line style. The scope caption is built from the companies actually
   plotted, so it can't drift from the data.
4. **The Compute vs Capabilities tab keeps the unconditional exclusion.** `_cc_trainflop_
   frontier()` still drops every name in `_DC_EXCLUDE_COMPANIES`: it is the compute half of
   a compute-vs-capability comparison, so each point has to be attributable to a lab that
   ships models, and QTS/DayOne sites have no recorded tenant at all. Crediting their
   capacity to nobody would raise the frontier with no capability to match it, distorting
   the fitted rates and China's ETA downstream. The two tabs' largest-site lines therefore
   differ on purpose and the DC tab's caption says so;
   `test_compute_capabilities_frontier_stays_lab_only` asserts both halves.

Guarded by `TestDcHiddenCompanies` and `TestDcUnattributedCompanies`.

### Networkable data-center clusters

The Data Centers tab's last chart sums the sites a company could plausibly drive as **one**
training job. `_DC_NETWORK_CLUSTERS` is a curated tuple of `(label, basis, site names)`, with
basis one of `_DC_NETWORK_LEVELS` — `'proximity'` (same campus or metro, read off `Address`),
`'fabric'` (far apart but joined by an announced training fabric — currently only the
Fairwater AI WAN pair), or `'plausible'` (a region with no link announced at all). The tab's selector
picks how far down that list to go; `_dc_network_site_clusters(level=…)` admits every basis
up to and including the one named, and the default stays `'fabric'`. Anything absent is its
own cluster and never pools. Six things are load-bearing:

1. **Curated on purpose.** Address parsing was tried and isn't viable — a third of rows
   yield no `(city, state)`, and the two Cedar Rapids sites sit in differently-named
   municipalities. Mis-grouping invents capacity, so `TestDcNetworkClusters` re-checks every
   name against the live CSV.
2. **Clusters are geography; pooling is per company.** A label names the geography or the
   fabric, not the tenant: the Fairwater pair pools under *OpenAI*, Epoch's first-listed user
   for both sites, which is why it isn't called "Microsoft AI WAN". A cluster spanning two
   companies never merges them, and sites hidden by `_dc_hidden_companies()` take their cluster with
   them — only Cedar Rapids is inert today, for the first reason (a Google site and an
   unattributed QTS one). Richmond and San Antonio were inert for the second reason until
   their hosts cleared the size gate. Kept either way: the geography is real even where
   tenant attribution isn't. Note what this means for the `'plausible'` level: it is a
   *fabric* assumption only. It cannot rescue Cedar Rapids, whose two sites are already
   clustered and fail to pool because Epoch records no tenant for the QTS one. Assuming
   tenancy is a much stronger claim than assuming a link, and is deliberately not on this
   axis.
3. **`cluster_of={}` must reproduce `_dc_company_series()` exactly** — that chart sits
   directly above, so drift reads as a bug. `cluster_of=None` pools the whole fleet and is
   offered only as an explicitly-labelled upper bound.
4. **A site appears in at most one cluster _per basis_**, else "largest group" depends on
   dict order.
5. **A wider basis must contain any cluster it touches whole.** Levels nest by overwriting,
   weakest first, so a region that named only *some* of a metro cluster's sites would re-cut
   that cluster rather than widen it, and the wider setting could then show *less* capacity.
   `test_a_wider_basis_subsumes_the_clusters_it_touches` holds the invariant; the
   monotonicity test on live data is what it buys.
6. **The `'plausible'` radius is set by the announced fabric, not below it.** Every pair
   inside a `'plausible'` region sits within roughly the ~1,200 km of the Fairwater AI WAN:
   refusing a shorter unannounced hop than one already being built is the inconsistent
   position, so the tier exists — but it is off by default and its caption says the groups
   are what a company *could* wire together. Regions are listed only where at least one
   company has two sites in them (an inert one pools nothing, unlike the metro clusters,
   which are kept for the geography), a site in range of two regions goes where its own
   company already is, and rule 5 pins anything that sits in a metro cluster.

Don't widen a `'proximity'` or `'fabric'` cluster to "same company, same region": the
cross-site link has to carry the data-parallel gradient all-reduce, so metro fibre or a
purpose-built fabric is the criterion for the levels that claim a link exists.

### UK Cyber tab caveats

Five load-bearing constraints; don't "simplify" them away:

1. **The frontier is closed-weight only** — open-weight models are the subject being measured against it, so `load_ukcyber()` excludes them from the running max.
2. **Lag is interpolated between the bracketing frontier models and carries a bracket.** `_ukc_frontier_crossing()` interpolates between the last model below a score and the first above; `ukc_lag_rows()` returns `lag_months` plus `lag_lo`/`lag_hi` from those two. Snapping to the next model up (the earlier implementation) equates scores that can be far apart — across a 10-point frontier gap it understated one lag by ~2.4 months. Nothing shipped inside the gaps, so the bracket is real uncertainty, not noise. (Rejected: last-model-below, defensible but maximally pessimistic; OLS-curve dating, which imports global fit error into a local question.)
3. **`lag_lo` is the AISI-compatible bound.** AISI's printed annotations use next-model-up, exactly `lag_lo`, so `test_optimistic_bracket_reproduces_aisi_published_lags` doubles as the calibration check on the digitization. Changing the lag method must not break it.
4. **The TLO cross-check compares on `lag_lo`, not the point estimate.** `aisi_cyber_tlo.csv` reuses the same helpers by storing steps in `cyber_score`, but the two datasets' frontiers differ in density, so point estimates aren't comparable — a score inside a wide gap gets inflated, one below every frontier model gets no interpolation at all (`lag_hi` is `None`, a lower bound) — and the ordering inverts for sampling reasons rather than capability. On `lag_lo` the two reproduce AISI's headline exactly: narrow 4.3–5.1mo, cyber range 6.7–6.8mo, "4 to 7 months". `test_reproduces_both_figure_titles` guards this.
5. **The suites don't cover the same models, and a callout says so.** Kimi K3 got a selective set (ExploitBench + TLO), not the 70-task narrow suite, so it's missing from the chart, projection and 90% ETA by data availability, not by a filter. `ukc_open_only_on_tlo()` finds the newest open-weight model in TLO but not narrow rows; `_render_ukcyber_newest_open()` shows it as a bordered callout with the `lag_lo`–`lag_hi` bracket (TLO's frontier is too sparse for a point estimate). It returns `None` once narrow catches up, and ignores a TLO-only model *older* than every open-weight point on the chart. Both it and `_render_ukcyber_tlo()` read `ukc_tlo_lag_rows()`, so they can't drift onto different frontiers. Guarded by `TestUkCyberOpenOnlyOnTlo`.

Country and openness are perfectly confounded (no US open-weight, no Chinese closed-weight models). The tab says "China" because the two Chinese models are also the only open-weight ones; `_UKC_CONFOUND_PLAIN` is folded into the fine-print caption so that's stated wherever it does. If it ever needs banner prominence again, promote that same constant rather than writing a second wording to keep in sync.

`ukc_target_eta()` ("when do open-weight models reach `_UKC_TARGET`, 90%") = the frontier's interpolated crossing plus the min/max measured lag. `ukc_target_eta_direct()` fits the open-weight points themselves as a cross-check only — two models 53 days apart make that slope very sensitive, so it is never the headline.

### ECI tabs — organization matching

The Epoch ECI tab and the ECI Company Gap tab read the same CSV and **must resolve
organizations identically**, both by *substring* match on Epoch's `Organization` field: the
ECI tab via `load_eci_frontier(orgs=…)` / `_ECI_ENTITY_SPECS`, the gap tab via
`_ecg_org_display()` / `_ECG_ORG_MAP`. Don't turn either back into an exact-key lookup — the
gap tab used to, and the tabs silently disagreed: Epoch spells Google four different ways,
so a map keyed only on `Google DeepMind` dropped four models and drew a different 2025
frontier point. Google's *current* gap was unaffected, which is why it went unnoticed.

**Adding a company is one row in `_ECI_COMPANIES`**, the single source of truth for both
tabs. `_ECG_ORG_MAP`, `_ECG_COLORS`, `_ECG_COUNTRY`, `_ECI_ENTITY_SPECS` and
`_ECI_ENTITY_SLUG` are derived from it at import — don't hand-write them.
`_ECI_COUNTRY_ENTITIES` stays separate (the "best" aggregates are country filters, not
companies) and must come first, since `_ECI_ENTITY_OPTIONS[0]` is the ECI tab's default.

Keep each company's `orgs` list minimal — substring matching makes longer variants redundant
("Google" already catches "Google DeepMind,Google"). Matching is unambiguous only while no
`Organization` string names two companies; `TestEcgOrgMatching` asserts that against the live
CSV, plus tab-to-tab set equality, the Google spellings, and slug round-tripping.

### Buildout by country (bottom of the Data Centers tab)

`_dc_render_country_panel()` is the US-vs-China view: each country's largest training
run over time, extrapolated past the end of its recorded data under a 50%/80% cone, plus a
year-end table of US, China, their ratio and China's lag in months. Load-bearing:

1. **Country is a property of the building.** `load_data_centers()` carries Epoch's
   `Country`; `_dc_site_country()` adds `_DC_COUNTRY_FALLBACK` for the two timeline-only
   names with no metadata row (Kempas, Mesa). The panel is built from the **unfiltered**
   site list — `_dc_hidden_companies()` is not applied, because a landlord's hall in a
   country is capacity in that country whoever trains in it. `test_country_fallback_only_
   names_sites_epoch_left_blank` retires a fallback the moment Epoch fills the cell.
2. **"China-accessible" = China + `_DC_CN_ACCESS_ABROAD`** (DayOne Johor). Epoch's own
   source notes on Nusajaya cite the FT on Alibaba and ByteDance training in Southeast Asia.
   Both lines are always drawn and projected — mainland alone and accessible — with no
   selector (one was tried and removed as redundant); the sites are *moved* out of Malaysia
   rather than copied, and the caption says the campus has other tenants. Assuming tenancy is the
   strong claim here, same as for the networked-clusters `'plausible'` level; don't add
   sites to the tuple without a citation of the same standard.
3. **Pooling follows the sidebar's one networking selector** (`dc_pool_n`), shared with
   the networked-sites chart so the China line reads in that chart's units: `'site'` mode
   when nothing pools, else the largest networked group per company. `_dc_country_steps()`
   also has a `'country'` mode (every site summed — the Pacing-tab state-direction claim)
   that no control exposes yet; `test_pooling_modes_nest` holds site ≤ company ≤ country
   and that `cluster_of={}` reproduces the envelope. The pace and fit-since controls live
   in the sidebar's *Country projection* expander; the horizon is the tab's *Project
   through* year (now offered to 2031), so the panel follows the sidebar's projection
   range like every other chart on the tab.
4. **The extrapolation is a log-linear OLS on monthly samples, anchored at the last
   recorded step.** `_dc_cty_fit()` samples the forward-filled step series monthly (not at
   event dates, which would weight dense periods), fits from the *Fit trend since* year,
   and reports the pace over every `_DC_CTY_FIT_WINDOWS` lookback. Pace uncertainty is
   max(OLS se, window spread / 2.56, `_DC_CTY_SIGMA_G_FLOOR` = 0.10 OOM/yr) — the floor is
   what carries the cone for China, whose windows all coincide. `_dc_cty_trajectories()`
   adds the series' own residual scatter, ramped in over a year. Two biases to keep in the
   caption: a fit from a country's first go-live runs hot (China-accessible's own trend is
   ×5/yr, a from-zero ramp), and one to the edge of the planned data runs cool (the far
   future is under-catalogued — the US since-2026 window is ×2.3/yr vs ×3.2 since 2024).
5. **Cones open at today and centre on the plan.** Sample *i* reads the step plan at
   `d − (d − today)·f_i` (clamped so a shifted plan never falls below what is built
   today), `f_i` ~ Normal(0, `_dc_cty_slip_sigma(quality)`) — 15% of lead for fully
   sourced plans, 35% for pure estimates — and scales it by a symmetric level draw
   (`_DC_CTY_PLAN_LEVEL_SIGMA` = 0.06 OOM per year of lead) so a flat plan still carries
   a small band. The timing noise is symmetric **on purpose**: Epoch dates conservatively
   (it pushes doubtful completions out itself), and the one-sided lognormal-lateness model
   tried first put every planned step at the top edge of its own interval — VNET Ulanqab
   most visibly. Every sampled path is made non-decreasing (built capacity persists) —
   that, not a plot fix, is what keeps the median from dipping at the plan-to-trend
   junction. Slipping a *step* plan is deliberate: log-interpolating between steps was
   tried and anticipated capacity ahead of its date. The dashed plan line is drawn to its
   own last catalogued step, never truncated at the trend anchor — `_dc_split_at` appends
   its end-x after the future steps, so truncating there made the polyline double back on
   itself. A *Show projection cones* checkbox (`dc_cty_cones`) hides the bands. Quality is `_dc_plan_quality()`: the share of a
   country's future rows whose `Construction status` cites a document
   (`_DC_PLAN_SOURCED_RE` — a markdown link, "schedule", "filing", "stated", "permit"…).
   Epoch publishes no confidence column, so this is a prose heuristic; live it splits the
   future rows ~45/55 and `test_live_catalogue_has_both_kinds_of_plan` fails if a refresh
   makes it dead weight. The trend takes over at `_DC_CTY_PLAN_HORIZON_DAYS` (18 months)
   rather than at the last catalogued step, **with the slipped plan as a floor carried past
   its last entry** — anchoring on the last entry held the US line flat through 2029 off one
   site dated 2030, and the fit window is clipped there too so the under-catalogued tail
   doesn't drag the pace down (×2.4/yr vs ×3.2).
6. **The default pace is the US trend borrowed** (`_DC_CTY_PACE_OPTIONS`, 'us'), with the
   cone widened to `|g_own − g_us| / 1.28` so a country's own fit sits at the 80% edge
   rather than vanishing. Own-trend is one click away and has China overtaking the US by
   2029 — that is the ramp bias above, not a finding. The US always uses its own fit.
7. **Lag never drops the samples where China leads.** `_dc_cty_lag_months()` floors an
   unresolved sample (the US running max never reaches China's value inside the grid) at
   one month past the grid end and returns the mask; the table prints "ahead in N% of
   samples" once most are floored. Dropping them as NaN biased the 2030 median to "5 months
   behind" in a row whose ratio read 0.5×.

Past the plan horizon the cone is a trend through the site list (for China the catalogue
ends in 2027), not a forecast of export-control policy.
The tab's *Project through* year caps the recorded data for every country alike; turning
planned buildout off gives a trend-only projection from today. `TestDcByCountry` (unit)
and `TestDataCentersByCountry` (integration).

### Compute vs Capabilities — the buildout-vs-release-timing panel

`_cc_company_buildout()` (bottom of the Data Centers tab) is a **pure timing test**: each
capacity step of a lab's largest single data center, shifted forward `_CC_RELEASE_LAG_DAYS`
(90d = 60d training + 30d release prep), against when that lab's models actually shipped.
Capability is never compared. Two directions with two different clocks — both now read out
through the one timeline chart (forward in the predicted-row hovers, backward in the
connectors and the headline median; the two date tables that used to sit under it were
removed as redundant) — so they must not contradict each other:

- **Backward** (release → cluster), `_responsible_cluster`: the latest step online at least
  one training run (`_CC_TRAIN_FLOOR_DAYS`, 60d) before the release. The extra ~1mo release
  prep is compressible polish, not a gate.
- **Forward** (cluster → release), `_cc_forward_match()`: three tiers, each tried only when
  the one above finds nothing. (1) First running-max release from
  `pred − _CC_EARLY_GRACE_DAYS` (7d) onward. (2) Else the earliest running-max release the
  *backward* match already gave this exact step — needed because the clocks differ: the 60d
  floor admits a release up to 30d early while the forward grace is 7d, so without this tier
  a step whose model shipped ~24d early falls to tier 3 and cites a lesser model. (3) Else
  the lab's next release of any kind from `_cc_company_all_releases()`, flagged `fallback`
  and marked † wherever it renders.

Load-bearing:

1. **The grace is 7d, not the 30d the 60d floor would allow.** At 30d, clusters start
   claiming the *same, earlier* model and the forward match goes degenerate. 7d changes
   exactly one pre-existing match, moving it *into* agreement with the backward match.
2. **Tier 3 exists because "frontier release" is a running max while Epoch recomputes ECI
   live.** A real flagship can be rescored under its own predecessor and vanish from the
   series without its release date changing — it has happened, leaving a cluster matching
   nothing. That's a fact about rescoring, not about when the lab shipped, and this panel
   compares dates only. Tier-3 matches draw hollow, are marked †, and are **excluded from
   the headline median**, which is a claim about record-setting releases.
3. **`_cc_company_all_releases()` keys on `(Model name, Release date)`.** Model name alone
   merges GPT-4o's May and August 2024 releases; Display name alone splits one model's ~10
   reasoning-effort rows, and which suffixed variant wins a dedup flips between Epoch pulls.
   Same-day releases sort by descending ECI so a step is offered the flagship.

`TestCcForwardMatch` and `TestCcCompanyAllReleases` guard all of the above.

### Compute vs Capabilities — China's ETA to a target ECI

`_render_cc_china_target()` answers "when does China cross `_CC_CN_TARGET_ECI`" (161) with a
date distribution rather than the gap metrics above it. Three things are load-bearing:

1. **The target tracks today's US frontier**, which is what makes the framing — "China
   matching where the US is *now*" — true. `TestCcCnTargetIsTodaysUsFrontier` asserts it
   stays at or under the US frontier and within 5 points of it; if an ECI refresh breaks
   that test, retarget the constant rather than loosening the test, and re-check the
   caption. Don't hardcode the anchor model in prose — Epoch recomputes live and the anchor
   moves between pulls, sometimes dropping a model off its own lab's running max entirely.
2. **The rate is the same two-engine model as Chart B**, deliberately: algorithmic term
   (iso-compute rates, mode = China's own) + `a_partial` × China's compute growth. A direct
   fit of China's frontier would contradict the chart above. The bottom-up rate runs hot
   (~14 ECI/yr) against a frontier that has managed 10–13, so `_cc_cn_target_years()` takes
   a `pace_lo`/`pace_hi` band derived from China's observed slope over `_CC_GAP_WINDOWS`.
   That reality check is the main uncertainty — the two iso-compute fits sit within a point
   of each other and alone would give a spuriously tight band.
3. **The crossing waits for a release.** The frontier is a step function, so clearing the
   bar needs a model to ship: `release_gap_days` (`_cc_release_gap_days()`, median recent
   inter-release gap) adds an exponential wait on top of the smooth crossing. That's why the
   median-crossing diamond sits *right* of where the fan meets the target line — the fan is
   the smooth capability path, the diamond and vertical band are release-inclusive. Not a
   plotting bug; don't "fix" it by aligning them.

Fan traces set `mode='lines'` explicitly: the fan spans ~6 quarters, and plotly defaults a
Scatter under 20 points to `lines+markers`, studding the band outline with stray dots.

### Pacing tab

`render_pacing()` answers "when can each entity first mount one ≥T-op training job",
for T from `_PC_THRESHOLDS` (a user-specified menu: 1e27…1e29) and a 2mo/6mo run
length (`_PC_RUN_OPTIONS`, reusing the loader's `train_flop*` columns — 30%
utilization, Epoch 8-bit OP/s). Deliberately thin: it reuses the Data Centers tab's
machinery rather than growing its own.

1. **Entities are the DC tab's, under two extra axes.** Companies via
   `_dc_company_networked_series` at the sidebar's pooling level (`pc_pool` reuses
   `_DC_NETWORK_OPTIONS` verbatim; 'none' → `cluster_of={}`, 'all' → `None`) with
   `_dc_hidden_companies` applied and † from `_dc_unattributed_companies`; then three
   country aggregates (US, China-accessible, China domestic) via `_dc_country_steps`
   (mode `'site'` when nothing pools, else `'company'`) on the **unfiltered** site
   list. `pc_party` picks the attribution via `_dc_with_party()` — see *Tenant vs
   operator, and shared tenancy* above: tenant view credits shared sites to every
   listed user, operator view to the owner alone.
2. **The projection is the by-country model, unchanged.** `_pc_projection()` calls
   `_dc_cty_fit` (since=`_DC_DEFAULTS["dc_cty_since"]`, plan horizon anchored) and
   `_dc_cty_trajectories` (plan slip by `_dc_plan_quality`); non-US entities borrow
   the US pace widened by `|g_own − g_us|/1.282`, short histories re-anchor the US fit
   at their own last step. Don't fork these — drift between the two tabs' cones reads
   as a bug.
3. **Crossing math is per-sample first-hit.** `_pc_crossing_idx` returns the first
   grid index reaching T with `len(grid)` as the never-crossed sentinel (NaN never
   hits); per sample it is non-decreasing in T, so every percentile is too —
   `test_crossing_idx_monotone_in_threshold`. `_pc_idx_date` maps a percentile back
   to a grid date, `None` past the grid (rendered ">2033"). The grid runs monthly
   from today to `_PC_HORIZON`. An entity whose catalogued steps cross before today
   is "already there" via `_pc_plan_crossing`, not the Monte Carlo.

Guarded by `TestPacing` (unit) and `TestPacingTab` (integration).

### Why the test suite is fast (and how to not un-fast it)

~8s, down from 3m20s that was essentially all in the 75 `AppTest` cases. Three things buy
that back; the first two live in `conftest.py`.

1. **One shared bytecode cache.** Streamlit builds a fresh `ScriptCache` on every
   `AppTest.run()` — once in `AppTest._run()`, again in `LocalScriptRunner.__init__` — so
   the app was re-read, AST-rewritten by `magic.add_magic` and recompiled ~300 times per
   suite. `conftest.py` patches **both** sites to one process-wide cache; patch only one and
   the other keeps recompiling and the win disappears. Isolation is unaffected — the
   bytecode is still exec'd into a fresh module each run.
2. **A smaller Monte Carlo.** `N_SAMPLES` reads `_VP_SAMPLES`; `conftest.py` sets it to 400.
   Nothing asserted depends on the count — the CI defaults the tests check come from the
   deterministic OLS fits, and the app never seeds its RNG. `_VP_SAMPLES=5000 pytest` runs
   at production fidelity (~16s) as a check. Don't lower the *default*: the band edges are
   percentiles and go visibly ragged well before 5000.
3. **Parallelism.** `pytest.ini` passes `-n auto`; the cases are independent, so a straight
   ~3.5x. It costs the unit-test-only run ~1.5s of worker startup, hence `-n0`.

Adding an `AppTest` case is cheap (~0.1s per `.run()`), so prefer one to skipping coverage.
What is *not* cheap is defeating the shared cache — a test that mutates
`visualize_projection.py` on disk, or builds its own runner from `streamlit.testing`
internals.
