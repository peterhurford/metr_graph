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
- **`benchmark_results_1_1.yaml`** — METR-Horizon-v1.1 data
- **`epoch_capabilities_index.csv`** — Epoch ECI data
- **`data_centers.csv`** / **`data_center_timelines.csv`** — Epoch Frontier Data Centers: one metadata row per site, many dated capacity rows per site
- **`aisi_cyber_narrow.csv`** / **`aisi_cyber_tlo.csv`** — AISI narrow cyber success rates (12 models); AISI/CAISI cyber-range "The Last Ones" avg steps of 32 (10 models). Both **chart-digitized, not published feeds** — see each file's `#` header

No build system, no CI/CD, no package manager beyond requirements.txt (`streamlit`, `numpy`, `plotly`, `pyyaml`).

## Architecture

Eleven-tab Streamlit dashboard selected via sidebar radio (`active_tab`, `_TAB_OPTIONS`) with URL deep-linking (`?tab=<slug>`, `?to=<section>`). Each tab has its own render function, sidebar controls, and (where applicable) projection engine. Slugs (`_SLUG_FOR_TAB`): `metr`, `eci`, `ecigap`, `rli`, `rsi`, `ukcyber`, `employment`, `revenue`, `datacenters`, `computecap`, `pacing`.

### Section deep links

`?to=<heading anchor>` scrolls to a section; a bare `#fragment` cannot. Community Cloud
serves the app in an iframe whose src copies the query string and drops the fragment, and
every `st.query_params` write rebuilds the URL from path + search alone, dropping it again.
Three things are load-bearing:

1. **The slug is whitelisted, not escaped** (`_anchor_slug()`): it is inlined into the
   injected script as a bare literal.
2. **`to` is consumed on arrival.** Left in the URL, every later rerun would jump the page
   back, and *Share view* would hand out the old section.
3. **`_render_anchor_links()` rewrites each main-column heading's link icon** to an absolute
   `?tab=…&to=…` URL of the outer page, and intercepts the click as an in-page scroll —
   Streamlit's own icon offers the `#fragment` form. Sidebar headings are left alone. The
   arrival scroll polls for the heading (the page streams in) and re-settles briefly (Plotly
   resizes after insert), stopping on the reader's first scroll.

`TestAnchorSlug` and `TestSectionDeepLinks` guard it; the latter also holds every heading
inside the whitelist, since the icon hands out a link the app must then accept. The METR tab
has no main-column headings, so it has no section to link to.

### Tabs and Render Functions

| Tab | Function | Data Source | Metric |
|---|---|---|---|
| METR Horizon | `render_metr()` | `benchmark_results_1_1.yaml` → `load_frontier()` | log₂(minutes) |
| Epoch ECI | `render_eci()` | `epoch_capabilities_index.csv` → `load_eci_frontier()` | linear score |
| Remote Labor Index | `render_rli()` | `_RLI_RAW` → `load_rli_data()` | logit-transformed score |
| RSI | `render_rsi()` | `_RSI_RAW` → `load_rsi_data()`; `_RSI_SURVEY`; `_RSI_CODE_RAW` → `load_rsi_code()`; `_RSI_DIR_RAW` → `load_rsi_direction()` | CoBench score % (logit-projected), staff speedup ×, merged code per contributor ×, next-step win rate % |
| UK Cyber | `render_ukcyber()` | `aisi_cyber_narrow.csv` → `load_ukcyber()`; `aisi_cyber_tlo.csv` → `load_ukcyber_tlo()` | success rate % + open-weight lag in months; plus a TLO cyber-range cross-check in steps (`_render_ukcyber_tlo()`) and a callout for models only the range has measured (`_render_ukcyber_newest_open()`) |
| Revenue | `render_revenue()` | `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` | ARR in billions; optional summed line (`rev_combined`, off by default) via `_rev_combined_series()` |
| Employment | `render_employment()` | RLI frontier + slider assumptions | unemployment % / jobs lost |
| ECI Company Gap | `render_eci_gap()` | `epoch_capabilities_index.csv` (by org/country) | linear score gap |
| Data Centers | `render_data_centers()` | `data_centers.csv` + timelines → `load_data_centers()` | H100-equiv / power / cost; carries a US-vs-China by-country projection (`_dc_render_country_panel()`) and ends with the region-share stack (`_dc_render_region_share()`) |
| Compute/capabilities/diffusion (slug `computecap`) | `render_compute_capabilities()` | data centers (`dc_all`) + ECI | train-FLOP frontier vs ECI; carries China's ETA to `_CC_CN_TARGET_ECI` (`_render_cc_china_target()`) and ends with the global compute distribution, today and projected (`_render_cc_world_shares()`) |
| Pacing | `render_pacing()` | data centers (`dc_all`) | China's catch-up to a US pause, then the date each entity first commands a run of the paused US scale |

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
| `_RSI_RAW` (hardcoded) | Anthropic, Redacted Risk Report (Aug 2026), §3.4.3 Fig 3.4.3.A (`_RSI_SOURCE_URL`) | **Not downloadable** — scores read off the figure, Anthropic prints no table. Hand-edit rows |
| `_RSI_DIR_RAW` (hardcoded) | Anthropic, [*When AI builds itself*](https://www.anthropic.com/institute/recursive-self-improvement) (`_RSI_DIR_SOURCE_URL`) | **Not downloadable** — the figure prints its own bar values, so the rows are read off the labels, not pixel-digitized. Dates are the models' Epoch-catalogued release dates (`test_dates_are_the_published_release_dates`), not the figure's row order, which is not chronological. Hand-edit rows |
| `_RSI_CODE_RAW` (hardcoded) | Same post's *Code contributed per person, by quarter* figure (`_RSI_CODE_SOURCE_URL`) | **Not downloadable** — the figure labels its bars only from 2025Q1 on; the earlier ones are read off the axis. Their mean must come back at ~1 (`test_pre_2025_bars_average_to_the_baseline_they_define`) and the post states the 2026Q2 figure in prose, which pins the other end. Hand-edit rows |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (hardcoded) | Press reports | Hand-edit `(date, ARR_in_billions)` tuples |
| `aisi_cyber_tlo.csv` | UK AISI Figure 2 + [Kimi K3 assessment](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) | **Not downloadable** — 9 rows digitized from `fig2-ranges.png`, one value quoted from prose. Calibration and validation checks are in the file's `#` header, guarded by `TestUkCyberTlo`. Dates are **published release dates**; the figure's x-axis is tokens |
| `aisi_cyber_narrow.csv` | UK AISI [open-weight cyber gap post](https://www.aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber) | **Not downloadable** — AISI publishes no numbers; values digitized from `fig1-narrow.png` by pixel analysis. Refreshing is a *figure-unchanged check*: re-fetch the PNG, confirm gridline rows and marker colours still match, re-digitize only if the figure changed. `test_digitized_dates_match_known_releases` and `test_optimistic_bracket_reproduces_aisi_published_lags` are the calibration guards. Hand-editing a row is fine if AISI states a number in prose |

Before overwriting a CSV wholesale, diff by key column so no curated rows are lost (ECI = `Model version`; DC metadata = `Name`; timelines = `Data center` + `Date`). After any data change, sanity-check the relevant loader via `_VP_TESTING=1 python3 -c "import visualize_projection as v; ..."`, then run the tests.

#### Timeline rows with no metadata row

`load_data_centers()` is **timelines-driven** (metadata only supplies the company label), so a
timeline name Epoch never catalogues materializes a site with no owner and no country — and if it
duplicates a catalogued one, it double-counts it. Check
`set(timelines['Data center']) - set(metadata['Name'])` on every refresh.

Empty as of the 2026-08-29 pull. It previously held `Fluidstack Lake Mariner`, stale rows covering
the whole site after Epoch re-scoped it to `Anthropic Lake Mariner` (CB3–5); those were deleted
locally until Epoch fixed the export by splitting out `Core42 Lake Mariner` (CB1–2). The same pull
dropped `EdgeCore Mesa PH03` and `DayOne Kempas`, which had been legitimate timeline-only names.

### Key Sections of visualize_projection.py

Roughly in file order: shared helpers (`pretty()`, `fit_line()`, distribution samplers,
`_ss_number_input()`, `superexp_trajectory()`, `_logit()`/`_inv_logit()`); backtesting
(`_backtest_*`); loaders (`load_frontier()`, `load_eci_frontier()` / `load_eci_compute()`,
`load_rli_data()`, `load_data_centers()`, `load_ukcyber()`); data-center aggregation
(`_dc_envelope()`, `_dc_company_series()`, `_dc_company_networked_series()`); UK Cyber lag
(`_ukc_*`, `ukc_*`); compute-vs-capabilities (`_cc_*`); data init and tab selector; the ten
`render_*()` functions (grep `^def render_`); dispatch at end of file (skipped when `_VP_TESTING=1`).

### Projection Engine (repeated per tab)

Three bases: **Linear** (single OLS), **Piecewise linear** (multi-segment OLS, last segment extrapolated), **Superexponential** (doubling time decays via `superexp_trajectory()` with a floor). All sample 5,000 trajectories into Plotly fan charts with 50%/80%/90% CI bands.

### Session State and Reset

Widget defaults live in per-tab `_RESET_DEFAULTS`; each tab's `_RESET_KEYS` lists its session state keys. The reset button pops all keys and calls `st.rerun()`. Custom number inputs use `_ss_number_input()` to persist via session state.

### Backtesting

"Project as of" projects from a historical vantage point; `_backtest_stats()` compares actual later models against the projected trajectories, colored by which CI band they land in.

### Internal Units

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
happening to share a tag, and had the quarterly table reporting the minority series, contradicting the chart directly above it. The rest of the app already merged
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
   Compute/capabilities/diffusion tab — behaves exactly as before. `test_shared_site_counts_
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
only while its largest single site stays under `_DC_EXCLUDE_MIN_H100` within
`_DC_EXCLUDE_HORIZON_DAYS`. Hiding them all was wrong once one got big: QTS
Cedar Rapids is the largest single site in Epoch's data, so the tab's headline chart was
naming a smaller site as the record holder. The roster the rule currently produces is
pinned by `test_current_roster_is_what_the_tab_says_it_is`.

Four things are load-bearing:

1. **The threshold reads H100-equivalents, always** — never the metric currently selected.
   The same number in megawatts or dollars means something else, and two of the tab's
   metrics are inverted (bigger site = smaller value), so `>=` would read backwards.
2. **A rolling horizon, not the uncapped peak, and not `dc_end_year`.** Uncapped, every
   listed host eventually clears the bar on 2028+ buildout and the list stops meaning anything
   (`test_uncapped_peaks_would_defeat_the_exclusion`). And the roster is a property of the
   company, not of a slider — keying it to the user's projection window would make sites
   blink in and out as the window moves. The roster is stable well either side of a year.
   If an Epoch refresh moves a company across the bar, retarget the constant deliberately
   rather than loosening `test_current_roster_is_what_the_tab_says_it_is`.
3. **`†` marks capacity with no recorded tenant.** `load_data_centers()` sets `attributed`
   False when the company label came from the site-name fallback rather than Epoch's `Users`
   or `Owner`; `_dc_unattributed_companies()` marks a company only when *every* one of its
   sites is such a fallback, so Microsoft (listed as its own sites' user) draws plain while
   QTS and DayOne carry the mark. Solid-vs-dashed is already actual-vs-planned, hence a
   glyph rather than a line style. The scope caption is built from the companies actually
   plotted, so it can't drift from the data.
4. **The Compute/capabilities/diffusion tab keeps the unconditional exclusion.** `_cc_trainflop_
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

### Caveats belong on the thing they qualify

Prefer a hover to a paragraph — applied across every tab. Keep in the visible
line only what a reader needs *before* deciding to look closer: the claim the
section makes, and anything the page would be misleading without. The method, the
provenance and the caveats go in hovers. Chart **legends stay visible** — they are
keys, not caveats.

Three mechanisms, in order of preference:

1. **`st.metric(..., help=…)`** where a metric exists. Native tooltip, no raw
   HTML. The *Capabilities Milestones* row is the worked example: ten cards,
   each with its own note (`_notes` by slug, plus `_pc_clock_note()` for the
   release-vs-internal split that used to be a sentence naming every milestone on
   both sides), and a two-line caption under them. Widget `help=` does the same
   for controls, which is why every sidebar control has one.
2. **`_fn_caption(text, *notes)`** for prose qualifying a chart, where there is no
   widget to hang `help=` on. Each note is a `(phrase, note)` pair, and **the
   phrase is looked up in the visible line and made hoverable in place** — a
   footnote hangs off the words it qualifies, not off a marker parked at the end
   of the sentence. So write the line to contain the phrase; a phrase that isn't
   there falls back to a trailing `?`, which is the exception, not the design
   (`test_footnotes_anchor_to_their_phrase` holds the live app at zero
   fallbacks). Matching is whole-word: a bare `find` once wrapped "check" inside
   "Cross-checked", splitting the word across a tag. It stays an `st.caption`, not
   an `st.markdown` — the element type is what tells fine print from body text, to
   the reader and to the tests that address one and not the other.
3. **`_fn_line(...)`** for a body-prominent line that must not read as fine print
   (the pause panel's *Assumes*, the buildout headline). `st.warning` takes no
   HTML, so the efficiency caveats use `_fn_line` with a warning glyph instead of
   a yellow box.

`_fn_span()` wraps the anchored phrase (dotted underline, `.vp-fn-a`); `_fn()` is
the standalone fallback marker (`.vp-fn`). Their CSS lives in the global `<style>`
block at the top of the file. The bubble is dark in both themes on purpose — it
floats above the page, and no Streamlit variable reliably tracks the user's
theme.

`test_every_milestone_card_carries_its_caveats_on_hover` pins the split for the
milestone row: every card has a hover naming its fit and its clock, and the
caption stays short.

**Editing these blocks by script:** a naive paren-counter will not do — an
f-string like `.split(' (')` puts an unbalanced paren inside a literal, and a
scan that counts it swallowed 5,500 lines once. Tokenize, or edit by hand.

### RSI tab

`render_rsi()` is titled *RSI* and runs CoBench, then the staff survey, then
merged code per person, then research direction, then
*Capabilities Milestones* + *RSI projection (tentative)* (`_pc_render_milestones()`,
moved here from the Pacing tab). The CoBench section plots that eval — Anthropic's
internal AI R&D benchmark — against release
date, with a fitted trend, a projection fan and `_RSI_SUBSTITUTION_BAR` drawn as
its bar (Anthropic's own stated score for a model that could fully substitute for
its research staff — not a benchmark ceiling). Four things are load-bearing:

1. **The fit is logit-space**, like RLI and UK Cyber: CoBench is a bounded success
   rate and a score-space line runs through 100%.
2. **Only a single OLS basis is offered.** Three frontier points cannot distinguish
   a line from a bend, so there is no piecewise or superexponential option and no
   backtest vantage-point selector. Don't add them by copying another tab.
3. **The default rate CI is widened, not the convention.** The two segments disagree
   by nearly an order of magnitude, so `_rsi_dt_ci()` takes the usual fit/2..fit×2 interval and widens it to
   span both segment rates **and** the slope's 80% t-interval (`_dt_t_interval`,
   `_DT_T80` multipliers by residual dof, decaying to the normal limit as points
   accumulate). A t-interval that cannot exclude a
   flat slope — true today for both this fit and the staff survey's, which gets the
   same treatment via `_rsi_survey_dt_ci()` — caps the slow edge at `_DT_CAP_DAYS`
   (flat at every horizon the app offers), which is what puts
   "no crossing before 2028" inside the default fan rather than outside it. It can
   only widen — `test_dt_ci_default_spans_both_segment_rates` and
   `test_small_sample_ci_widens_both_rsi_fits` hold that (the latter pins the
   live caps; retarget it when a new round tightens the interval).
4. **`date_known` drives the "~" prefix** via `_rsi_date_label()`. Mythos Preview has
   no published release record (its date is carried over from AISI's narrow cyber
   figure, as in `aisi_cyber_tlo.csv`) and Model 2 (internal) is unreleased with its name
   redacted; Mythos 5 ships with Fable 5 on 2026-06-09, which puts it *below* the
   running max and off the frontier.

CoBench is filtered for difficulty (mostly problems Mythos Preview failed at least
once in three tries) and run at a 300k-token budget, so scores don't compare to
public AI R&D suites — the fine print has to keep saying so.

**The sections chart; the cards date.** Each of the four draws its bar and stops
there — the per-section "when does it reach X" ETA pair and the ECI-style row of
projected values both used to sit under every fan, restating what the milestone
card below already says. Every bar is now dated once, on its card
(`test_the_sections_chart_and_the_cards_date`); that each card still reproduces
its own section's fit is pinned unit-side, per section, by
`test_eta_reproduces_the_section_defaults`. Don't re-add an in-section ETA.

The tab's second half (`_render_rsi_survey()`) charts the report's other
substitution series, the internal staff survey (§3.4.2): self-reported output
multiple against no AI assistance, from `_RSI_SURVEY` via `load_rsi_survey()`,
fitted and projected on **log(multiple)** — a multiple has no ceiling to bound it
against, so it compounds the way METR's horizon does rather than saturating like
a percentage. Three things are load-bearing. Opus 4's round is not carried at
all: it reported no number, only that the result fell under the pre-set 3x median
rule-out threshold, and a bound is not a point on a trend. Model 2 (internal)
has an `estimated` point carried over from Mythos Preview — no round was run for it —
which draws hollow as an assumed value but is **included in the fit and the
anchor**: with only three surveyed rounds, ignoring the one flat reading
available overstates the slope, so the flattening is the point, not a bug (it
was excluded once, on the opposite reasoning; the flag now governs styling
only). The rounds do not
report the same statistic on the same sample (medians on superusers, then on a
broader sample, then a geometric mean on an opt-in poll), so each point carries
its `note` on hover and the caption says the rounds differ — and the survey is
not discontinued, the report says only that no new round was run for Mythos 5.
And the fan runs `_RSI_SURVEY_HORIZON_DAYS` past the last round rather than to the
tab's *Project through* year: at the fitted doubling time the median runs several
orders of magnitude past the data by end-decade, which on a log axis squashes the
three actual points into the bottom decile. The chart draws
`_PC_RSI_SURVEY_TARGET_X` as its bar, the same one the milestone card dates.

The tab's third section (`_render_rsi_code()`) is the counted counterpart to
that survey: lines merged per active contributor per quarter, as a multiple of
the pre-2025 average, from `_RSI_CODE_RAW` via `load_rsi_code()`, fitted on
log(multiple) like the survey and dated against `_RSI_CODE_TARGET` = 30x. Four
things are load-bearing.

1. **Only the bars from `_RSI_CODE_FIT_FROM` are fitted.** The earlier ones are
   the baseline the multiple is taken against — they average to 1 by
   construction, so there is no trend in them to fit, and pooling them halves
   the slope (`test_fit_runs_on_the_2025_bars_only`). The ones the chart shows
   are drawn hollow; the x-axis opens at `_RSI_CODE_CHART_FROM` rather than at
   the first bar, since flat quarters only stretch the axis. All of them stay
   in the data either way — their mean is the calibration guard on the
   digitization.
2. **The hatched final bar is fitted, at its quarter's midpoint like the rest.**
   It averages only the days observed when the post went up, so a midpoint date
   is later than the days it covers — which reads the ramp slow, not hot.
3. **The position CI is the fit's own residual scatter** (`_RSI_CODE_POS_FACTOR`),
   not the survey's one-significant-figure allowance: this series is counted,
   not self-reported. The rate CI is `_dt_ci_t_widened` like the other two.
4. **The target label is added by hand**, here and on the survey chart's
   `_PC_RSI_SURVEY_TARGET_X` bar. On a log axis plotly reads an annotation's y
   as the exponent, so `add_hline(annotation_text=…)` parks the label at 10^30
   and off the chart. This chart's axis is also `dtick=1`: three decades of
   minor tick labels bury the four that matter.

Lines merged is an output proxy a coding model inflates directly, so the caption
and the milestone card's hover both have to keep saying it measures how much code
ships, not how much research it settles. Guarded by `TestRsiCode`.

The tab's fourth section (`_render_rsi_direction()`) charts Anthropic's Claude
Code detour study from `_RSI_DIR_RAW` via `load_rsi_direction()`: `_RSI_DIR_N`
turns where one of its own researchers went the wrong way, replayed to each
model, with a judge that has seen the finished session picking the better next
step. Three things are load-bearing.

1. **The bar is the study's own annotated practical ceiling**, `_RSI_DIR_TARGET`
   = 90%: what an *oracle* — a model shown the resolved session — scores against
   the same researchers, and the only threshold the source names. The fit is
   plain logit space like CoBench's, so the bar is reachable; scaling the odds
   by the ceiling instead was tried and makes it asymptotic, which is a
   different claim than the one the card dates.
   `test_fit_is_logit_space_against_the_studys_own_ceiling` pins both.
2. **The dates decide the frontier, not the figure's row order.** That figure is
   not chronological (it prints Sonnet 4.6 above Opus 4.6, which shipped twelve
   days earlier), so Opus 4.7 — released after Mythos Preview and scoring under
   it — sits off the running max, as Mythos 5 does on CoBench. Mythos Preview
   carries the same unpublished date the CoBench series does, hence the "~".
3. **The level is not "how often Claude out-researches a human."** The turns
   were *selected* for having room for improvement, so what the series carries
   is the trend in that comparison, not its level; the caption's hovers say so,
   and the milestone card's does too. Only the frontier is labelled in the chart
   (`_rsi_dir_label_positions()`, which also drops a crowded frontier point's
   label below the line) — nine models inside twenty-five months collide at a
   fixed `top center`.

The rate CI is `_rsi_dir_dt_ci()`, and the survey's is `_rsi_survey_dt_ci()`;
both are `_dt_ci_t_widened()`, the shared [DT/2, DT×2]-widened-to-the-t-interval
rule.

The RSI blend's *Capabilities Milestones* row dates the CoBench and next-step
bars through `_pc_rsi_eta()` and `_pc_nextstep_eta()`, which reuse this tab's own
fits and rate CIs rather than fitting their own. Guarded by `TestRsi`,
`TestRsiDirection` and `TestRsiTab`.

### UK Cyber tab caveats

Five load-bearing constraints; don't "simplify" them away:

1. **The frontier is closed-weight only** — open-weight models are the subject being measured against it, so `load_ukcyber()` excludes them from the running max.
2. **Lag is interpolated between the bracketing frontier models and carries a bracket.** `_ukc_frontier_crossing()` interpolates between the last model below a score and the first above; `ukc_lag_rows()` returns `lag_months` plus `lag_lo`/`lag_hi` from those two. Snapping to the next model up (the earlier implementation) equates scores that can be far apart, understating a lag across a wide frontier gap. Nothing shipped inside the gaps, so the bracket is real uncertainty, not noise. (Rejected: last-model-below, defensible but maximally pessimistic; OLS-curve dating, which imports global fit error into a local question.)
3. **`lag_lo` is the AISI-compatible bound.** AISI's printed annotations use next-model-up, exactly `lag_lo`, so `test_optimistic_bracket_reproduces_aisi_published_lags` doubles as the calibration check on the digitization. Changing the lag method must not break it.
4. **The TLO cross-check compares on `lag_lo`, not the point estimate.** `aisi_cyber_tlo.csv` reuses the same helpers by storing steps in `cyber_score`, but the two datasets' frontiers differ in density, so point estimates aren't comparable — a score inside a wide gap gets inflated, one below every frontier model gets no interpolation at all (`lag_hi` is `None`, a lower bound) — and the ordering inverts for sampling reasons rather than capability. On `lag_lo` the two reproduce AISI's published "4 to 7 months" headline exactly. `test_reproduces_both_figure_titles` guards this.
5. **The suites don't cover the same models, and a callout says so.** Kimi K3 got a selective set (ExploitBench + TLO), not the 70-task narrow suite, so it's missing from the chart, projection and 90% ETA by data availability, not by a filter. `ukc_open_only_on_tlo()` finds the newest open-weight model in TLO but not narrow rows; `_render_ukcyber_newest_open()` shows it as a bordered callout with the `lag_lo`–`lag_hi` bracket (TLO's frontier is too sparse for a point estimate). It returns `None` once narrow catches up, and ignores a TLO-only model *older* than every open-weight point on the chart. Both it and `_render_ukcyber_tlo()` read `ukc_tlo_lag_rows()`, so they can't drift onto different frontiers. Guarded by `TestUkCyberOpenOnlyOnTlo`.

Country and openness are perfectly confounded (no US open-weight, no Chinese closed-weight models). The tab says "China" because the two Chinese models are also the only open-weight ones; `_UKC_CONFOUND_PLAIN` is folded into the fine-print caption so that's stated wherever it does. If it ever needs banner prominence again, promote that same constant rather than writing a second wording to keep in sync.

`ukc_target_eta()` ("when do open-weight models reach `_UKC_TARGET`") = the frontier's interpolated crossing plus the min/max measured lag. `ukc_target_eta_direct()` fits the open-weight points themselves as a cross-check only — they sit close together in time, which makes that slope very sensitive, so it is never the headline.

### ECI tabs — organization matching

The Epoch ECI tab and the ECI Company Gap tab read the same CSV and **must resolve
organizations identically**, both by *substring* match on Epoch's `Organization` field: the
ECI tab via `load_eci_frontier(orgs=…)` / `_ECI_ENTITY_SPECS`, the gap tab via
`_ecg_org_display()` / `_ECG_ORG_MAP`. Don't turn either back into an exact-key lookup — the
gap tab used to, and the tabs silently disagreed: Epoch spells Google several different
ways, so a map keyed only on `Google DeepMind` dropped models and drew a different
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

### Share of catalogued capacity by region (last chart on the tab)

`_dc_render_region_share()` stacks each region's share of the selected metric,
summed over every site, on a monthly grid. It reads the **unfiltered,
geography-only** series the country panel does — capacity in a country is
capacity whoever trains in it — so DayOne Johor sits in *SEA* here while the
panel above also reads it as China-accessible. Four things are load-bearing:

1. **The buckets partition the catalogue.** `_DC_REGIONS` names US, China, SEA,
   Europe/UK and UAE; everything else — including a site Epoch leaves without a
   country — falls to *Other*, so the region totals add to the catalogue's own
   total at every date (`test_regions_partition_the_catalogue`). Europe/UK is
   the EU, the UK and the rest of Europe (Norway today), which is why it isn't
   called EU+UK; `test_every_catalogued_country_is_placed_deliberately` names
   what the residual bucket currently holds, so a refresh that adds a country
   worth naming surfaces there.
2. **The coverage caveat is body text above the chart, not caption fine
   print** — an `_fn_line` with a warning glyph, since `st.warning` takes no
   HTML and the hovers are worth more than the yellow box. Nearly every
   catalogued site is American, so the line says non-US shares are floors and
   puts the why in hovers. Its count comes from the plotted sites, not written
   down.
3. **Shares are computed, not `groupnorm`.** The trace carries the percentage
   and the raw level as `customdata`, so the hover can't disagree with the
   axis. Months where nothing is built are dropped — a share of nothing is
   undefined, not 0%.
4. **Only additive metrics mean anything.** Every `_DC_METRICS` key is one; the
   two "time to train" metrics because they store runs per window, which
   `_dc_share_label()` names in place of their inverted label.

Guarded by `TestDcRegionShare` (unit) and `TestDataCentersRegionShare`
(integration).

### Buildout by country (bottom of the Data Centers tab)

`_dc_render_country_panel()` is the US-vs-China view: each country's largest training
run over time, extrapolated past the end of its recorded data under a 50%/80% cone, plus a
year-end table of US, China, their ratio and China's lag in months. Load-bearing:

1. **Country is a property of the building.** `load_data_centers()` carries Epoch's
   `Country`; `_dc_site_country()` adds `_DC_COUNTRY_FALLBACK` for any timeline-only
   name with no metadata row (empty today). The panel is built from the **unfiltered**
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
   max(OLS se, window spread / 2.56, `_DC_CTY_SIGMA_G_FLOOR`) — the floor is
   what carries the cone for China, whose windows all coincide. `_dc_cty_trajectories()`
   adds the series' own residual scatter, ramped in over a year. Two biases to keep in the
   caption: a fit from a country's first go-live runs hot (China-accessible's own trend is
   a from-zero ramp), and one to the edge of the planned data runs cool (the far future is
   under-catalogued, so a recent-window fit reads slower than a longer one).
5. **Cones open at today and centre on the plan.** Sample *i* reads the step plan at
   `d − (d − today)·f_i` (clamped so a shifted plan never falls below what is built
   today), `f_i` ~ Normal(0, `_dc_cty_slip_sigma(quality)`) — a larger fraction of lead
   for pure estimates than for fully sourced plans — and scales it by a symmetric level
   draw (`_DC_CTY_PLAN_LEVEL_SIGMA`, per year of lead) so a flat plan still carries
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
   Epoch publishes no confidence column, so this is a prose heuristic;
   `test_live_catalogue_has_both_kinds_of_plan` fails if a refresh makes it dead weight.
   The trend takes over at `_DC_CTY_PLAN_HORIZON_DAYS`
   rather than at the last catalogued step, **with the slipped plan as a floor carried past
   its last entry** — anchoring on the last entry held the US line flat through 2029 off one
   site dated 2030, and the fit window is clipped there too so the under-catalogued tail
   doesn't drag the pace down.
6. **The default pace is the US trend borrowed** (`_DC_CTY_PACE_OPTIONS`, 'us'), with the
   cone widened to `|g_own − g_us| / 1.28` so a country's own fit sits at the 80% edge
   rather than vanishing. Own-trend is one click away and has China overtaking the US
   inside the window — that is the ramp bias above, not a finding. The US always uses its own fit.
7. **Lag never drops the samples where China leads.** `_dc_cty_lag_months()` floors an
   unresolved sample (the US running max never reaches China's value inside the grid) at
   one month past the grid end and returns the mask; the table prints "ahead in N% of
   samples" once most are floored. Dropping them as NaN biased the median toward "behind"
   in rows whose ratio said China was already ahead.

Past the plan horizon the cone is a trend through the site list (China's catalogue ends
earlier than the US's), not a forecast of export-control policy.
The tab's *Project through* year caps the recorded data for every country alike; turning
planned buildout off gives a trend-only projection from today. `TestDcByCountry` (unit)
and `TestDataCentersByCountry` (integration).

### Compute/capabilities/diffusion — the buildout-vs-release-timing panel

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
  a step whose model shipped a few weeks early falls to tier 3 and cites a lesser model. (3) Else
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
   merges GPT-4o's May and August 2024 releases; Display name alone splits one model's
   reasoning-effort rows, and which suffixed variant wins a dedup flips between Epoch pulls.
   Same-day releases sort by descending ECI so a step is offered the flagship.

`TestCcForwardMatch` and `TestCcCompanyAllReleases` guard all of the above.

### Compute/capabilities/diffusion — how it derives from the Data Centers tab

The CC tab's compute inputs are the DC tab's machinery, not a parallel implementation.
Four contracts, each guarded:

1. **Site→lab attribution reads the loader's fields.** `_cc_lab_attribution()` derives
   from `dc_all`'s `operator`/`users` (already folded by `_DC_COMPANY_ALIASES`) plus the
   one CC-only identity `_CC_LAB_ALIASES` (`SpaceXAI` → `xAI`); it never re-reads the CSV.
   Attribution is **operator-first, then primary tenant** — Colossus 2 is *xAI's* buildout
   though Anthropic is its first-listed user — deliberately unlike the DC/Pacing
   shared-tenancy rule, because the timing panel asks whose *buildout* predicts whose
   releases. Fallback-labelled sites (`attributed` False) never map to a lab.
   `TestCcDcDerivation` pins both the field-derivation and the Colossus precedence.
2. **Milestones are the shared envelope.** `_cc_lab_dc_milestones()` =
   `_dc_series_for_metric` + `_dc_envelope` over the lab's sites, filtered to record
   steps — one milestone per date (the old hand-rolled scan could emit two same-date
   records in dict order).
3. **Dates mean "Training run finished".** The +2mo shifts in `_cc_trainflop_frontier()`
   and `render_compute_capabilities()` go through `_dc_timing_shift("Training run
   finished")`, the DC tab's own milestone vocabulary.
4. **Country paces are cross-checked, not forked.** `_cc_country_pace_check()` runs the
   by-country engine (`_dc_country_steps` + `_dc_cty_fit`, no pooling, plan-horizon
   clipped, unfiltered sites) for US / China-accessible / China-domestic. Captions quote
   it next to the segment fits and the hand-set `_CC_CN_COMPUTE_LO/HI` band; tests keep
   the two engines honest — US segment fit within 0.15 OOM/yr of the country fit
   (`test_us_pace_cross_check_agrees_across_tabs`), and the China band at or below the
   catalogued paces (`TestCcCnComputeBand`: HI ≤ China-accessible ramp, LO ≤ domestic
   pace). The band stays hand-set on purpose: it claims coherent-*single-run* growth
   (networking is the export-controlled step), while the catalogue counts buildings —
   and the China-accessible fit is a from-zero ramp the DC tab itself flags as hot. If a
   refresh breaks either test, retarget the constants deliberately.

Provenance is stated in captions deep-linking `?tab=datacenters`; the frontier
itself is deliberately not charted (`test_dc_derivation_is_stated`). Lab-only
exclusion unchanged (`test_compute_capabilities_frontier_stays_lab_only`).
Sidebar: *Project through* (`cc_end_year`, default 2029) moves the projection
horizon only — the catalogue cap never drops below end-2028 and every fit era ends
by Jan 2029, so segment fits never change (`test_project_through_horizon`); the
China-ETA panel keeps its own adaptive horizon. *Run length* (`cc_run`,
`_PC_RUN_OPTIONS`, default 2-month) switches to `train_flop_6mo` + one-run shift —
a constant scale-and-slide, same leaders (`test_run_window_scales_levels_only`);
China's capacity band scales by the window ratio; `_cc_country_pace_check` stays
on 2-month (paces identical).

### Compute/capabilities/diffusion — global compute distribution (last section)

`_render_cc_world_shares()` answers what every chart above it cannot: how the
world's *installed* AI compute divides between the US, mainland China, SEA, the
UAE and everywhere else — a bar chart for today, then a stacked area carrying
it forward to the tab's horizon. The catalogue those other charts read covers a
minority of the global stock and is far denser for the US (Epoch says so
itself), so this section is a judgment layer over published country estimates,
not a reading of the CSVs. Six things are load-bearing:

1. **The centrals come from cited public estimates, and the band is their
   disagreement.** `_WC_REGIONS` carries `(label, central, p10, p90)`; the
   sources are in the block comment above it — Epoch's leading-cluster share by
   country (US ~75% / China ~15%), Epoch's chip-smuggling report (660k H100e
   ≈ ⅓ of China's compute and ≈3% of the global stock, i.e. China ≈9%), the
   AI-2027 tracker (~12%) and RAND (~15%). China's 9–18% spans them; no single
   source is that wide. Retarget the constants when the estimates move rather
   than nudging them to taste. Every region's provenance is one line in
   `_WC_NOTES`, shown on its bar's hover rather than written out as prose.
2. **The regions' growth rates are correlated, and unequally.**
   `_WC_COMMON_LOAD` is each region's loading on one global shock (chip
   supply, a capex cycle), so two regions' log-rates correlate at the product
   of their loadings; marginal rate spreads are untouched, only the joint
   moves. Independent rates said the plausible low-US worlds were ones where
   somebody else ran away with the buildout — backwards, since most non-US
   capacity is US firms building abroad on US-supplied chips. China's loading
   is the low one (domestic silicon, controls already binding), so it is the
   region whose share rises when the US stalls: conditional on the US in its
   bottom decile at the horizon, China absorbs 35% of the shortfall against
   14% under independence. Guarded by
   `test_a_low_us_world_is_a_slow_world_not_a_handover`. Note what the
   published-estimate spread does *not* get: a common factor on the levels
   would cancel exactly under renormalization, and the disagreement there is
   genuinely per-region (the sources argue about China specifically).
   **The per-region CIs are marginals of a composition** — every sample sums
   to 100, but the p90s do not (150% at the horizon), so both captions say the
   range is one region at a time.

3. **The projection draws the central-rate scenario, not a percentile.**
   `_wc_central_shares()` grows each region at its `_WC_GROWTH` central rate
   and normalizes — so it sums to 100 with no rescaling and is literally the
   line the caption describes. `_wc_share_paths()` samples the rates for the
   hover's 80% CI only. Medians were tried and don't sum to 100 (needing a
   rescale that then disagrees with the hover); means sum correctly but a
   lognormal rate's tail doubles the UAE. Growth rates are cross-checked
   against the catalogue in the comment: the US aggregates to ~1.9×/yr and
   China's domestic largest-site fit is ~1.8×/yr, against this tab's own
   `_CC_CN_COMPUTE_LO/HI` cluster band.
4. **It is by location, not owner.** Same convention as the Data Centers tab's
   region chart, so Chinese labs' capacity abroad counts under SEA, and most of
   Europe/UK is US firms building there. The area chart opens at `_CC_X_START`
   (Jan 2025) like the rest of the tab, so the stretch left of the "Today"
   divider is the same rates run *backwards* — there is no measured history of
   these shares. It doubles as a check the caption quotes: back-cast to
   mid-2025 China lands near Epoch's ~15% reading for May 2025. That half is
   washed out by a white `vrect` at `layer='above'` (a shaded band would sit
   *under* the filled areas and never show), and the divider is heavier than
   the other tabs' dotted one on purpose: it is the only thing separating two
   halves that otherwise look alike.
5. **The buckets are the Data Centers tab's** (`_wc_region_of`), so the two
   charts are comparable region by region. Europe/UK is carved out of the
   residual rather than left in *Other*: it is the largest part of what the
   anchors leave after the US and China, and most of it is US firms building
   abroad.
6. **The tracked-site comparison is in H100e, not the tab's train FLOP**
   (`_wc_catalogued_shares`), because the published anchors are stated in
   H100e. It reads the catalogue *as built today*, so a region whose first
   campus is still future-dated (the UAE) shows near zero.

No controls: an editable-centrals expander was tried and removed. Guarded by
`TestCcWorldShares` (unit) and `TestCcWorldSharesTab` (integration).

### Compute/capabilities/diffusion — China's ETA to a target ECI

`_render_cc_china_target()` answers "when does China cross `_CC_CN_TARGET_ECI`" with a
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
   against what the frontier has actually managed, so `_cc_cn_target_years()` takes
   a `pace_lo`/`pace_hi` band derived from China's observed slope over `_CC_GAP_WINDOWS`.
   That reality check is the main uncertainty — the two iso-compute fits sit within a point
   of each other and alone would give a spuriously tight band.
3. **The crossing waits for a release.** The frontier is a step function, so clearing the
   bar needs a model to ship: `release_gap_days` (`_cc_release_gap_days()`, median recent
   inter-release gap) adds an exponential wait on top of the smooth crossing. That's why the
   median-crossing diamond sits *right* of where the fan meets the target line — the fan is
   the smooth capability path, the diamond and vertical band are release-inclusive. Not a
   plotting bug; don't "fix" it by aligning them. The *months-behind* card is ship-to-ship
   on both sides and must stay so: a "smooth" variant that interpolated the US released
   steps was tried and read low — across a same-day release pair the interpolation
   collapses to the ship date, silently carrying the US's real wait (GPT-5.5 Pro's run
   finished ~2 mo before it shipped, per the Mythos model card; the wait is paid in prep
   and overshoot, invisible in released steps) while charging China none.
4. **Three-channel algorithmic engine.** `_cc_cn_crossing_sim` splits the measured
   rate into innovation (`_cc_pure_innovation_band`, never decays) + diffusion
   (nodist − pure, decays only after a pause, `_CC_DIFF_ABSORB_YRS` absorption ramp
   via `t_pause`) + distillation (algo − nodist, decays with gap/gap₀). With
   `pure_lo=None` or `t_pause=None` it reduces to the two-channel law, so the
   moving-US crossing is unchanged; the split bites in the Pacing pause panel
   and the scenario table's four rows.
   `_cc_cn_target_years` stays as the constant-rate comparison quoted in the caption.
   `TestCcCnCrossingSim` pins the limits (saturated = constant-rate; frozen US = slower;
   pause-to-frozen-bar later than the crossing with the US still moving). The pace band is shared via
   `_cc_cn_pace_band` with the Pacing pause panel so the two crossings stay ordered.
   The engines section carries a measured distillation control: `_cc_frontier_grade_algo`
   refits on frontier-grade models — within 5 ECI of the running frontier **and** trained
   within `_CC_FG_FLOP_MARGIN` of the running-max training run, against the
   **full-window** frontier (`load_eci_frontier(full_window=True)`). Both criteria are
   load-bearing: the ECI tab's Feb-2024 cutoff used to auto-admit every earlier model as
   "near-frontier", and without the compute screen the subset admits
   the models nearest the frontier at the least compute — the heaviest distillers
   (DeepSeek/Qwen/Kimi). The surviving fingerprint is one-way: the refit's b_time runs
   below pooled (followers ride a teacher); a_partial does **not** rise —
   reasoning-era models reach the frontier at sub-frontier compute — so the old two-way
   gradient claim is dead. The refit's pair still replaces the pooled one for every
   frontier-facing projection (US-vs-China slopes, the pause bar mapping and climb, the
   compute terms), with pooled as fallback; `TestCcFrontierGradeAlgo` pins the b_time
   drop, the screen's bite and the coverage guard. Measured by country, distillation is
   a *level*, not a rate: `_cc_cn_level_offset` (the country dummy at matched compute
   and date, quoted live in the control caption and the Pacing distillation checkbox)
   puts Chinese models above their US compute-peers while the two iso-compute rates are
   indistinguishable; `TestCcCnLevelOffset` guards it.

Fan traces set `mode='lines'` explicitly: the fan spans ~6 quarters, and plotly defaults a
Scatter under 20 points to `lines+markers`, studding the band outline with stray dots.

### Pacing tab

`render_pacing()` renders the pause counterfactual **first**
(`_pc_render_us_pause()`) and the *Compute Thresholds* race second — the tab's
question is what a US pause on a chosen date buys, and the race is the per-actor
detail behind it. That race answers "when can each entity first mount one ≥T-op
training job", at a 2mo/6mo run length (`_PC_RUN_OPTIONS`, default 2mo, reusing
the loader's `train_flop*` columns — 30% utilization, Epoch 8-bit OP/s).
Deliberately thin: it reuses the Data Centers tab's machinery rather than growing
its own. **T has no control of its own** — it is the US's own largest training run
at the pause (`caps[_DC_CTY_US]`, the *Largest training run* cell of the
state-of-play table), so the two halves of the tab cannot describe different
scales and the bar moves with the pause date and the run length alike.
`_PC_FALLBACK_THRESHOLD` is only the floor for a catalogue with no US capacity to
read.

*Capabilities Milestones* and the RSI blend live at the **bottom of the RSI
tab** (`_pc_render_milestones()`), not here — they are still named `_pc_*` with
the ETA helpers they call, and the machinery is unchanged. Ten cards, driven by
the RSI tab's own *Milestone dates point at* selector (`rsi_timing`): `_pc_metr_eta()`
for the METR frontier reaching `_PC_METR_TARGET_HRS` — about one work-month — at each
of `_PC_METR_LEVELS` (at the month-scale bar p50's earlier firing is its own card and
weight rather than the exclusion it got at the old 40h bar), `_pc_eci_eta()` for the
US-best ECI frontier reaching each of `_PC_ECI_TARGETS` (today's frontier plus **two
and three more jumps the size of GPT-5 → the current frontier**, `_PC_ECI_JUMP_FROM`
being the near end; each card's footnote counts its own jumps off the live scores, and
`test_eci_target_is_two_more_frontier_jumps` pins both against the live CSV since
Epoch rescores live; well above anything that tab draws, and a lower card sat close
enough to today's frontier that it dated model releases, not RSI, so its weight
moved here). The cards are labelled *ECI reaches …*, not *US ECI* — the US-best
frontier is stated in the hover. The 187.5 slug keeps the half-point
(`eci_187_5`) — it keys the blend weight, and `{t:.0f}` would round it to a slug
no weight matches, `_pc_rli_eta()` for the RLI frontier reaching `_PC_RLI_TARGET_PCT`
(above that tab's own milestone table), `_pc_rsi_eta()` for the CoBench frontier
reaching `_RSI_SUBSTITUTION_BAR` (Anthropic's own full-substitution bar, which the
RSI tab dates too), `_pc_rsi_survey_eta()` for self-reported staff speedup reaching
`_PC_RSI_SURVEY_TARGET_X` (about a doubling and a half past the most recent round),
`_pc_rsi_code_eta()` for merged code per Anthropic contributor reaching
`_RSI_CODE_TARGET`, `_pc_nextstep_eta()` for the detour study's frontier reaching
`_RSI_DIR_TARGET` (that study's own practical ceiling — see the RSI tab section),
and `_pc_revenue_eta()` for the **leading** company's ARR reaching `_PC_REV_TARGET_B`
(the Revenue tab's own top milestone) — the one bar here that isn't a benchmark, but still dated off
released models, since ARR is what shipped models earn. They render in **two rows** — on one line every label squeezes to two words. Each reproduces
its own tab at that tab's defaults — METR: GPT-4o-broken segment, DT over
[DT/2, DT*2], position over the current model's CI, p50 slope fits the trend;
ECI: single OLS, +Pts/Yr over [PPY/2, PPY*2], position ± 2; RLI: single OLS in
logit space, odds-doubling time over [DT/2, DT*2] floored at 5 days, position
± 1 point; RSI CoBench: single OLS in logit space, odds-doubling time over
`_rsi_dt_ci()`'s widened interval, position ± `_PC_RSI_POS_CI`; RSI
survey: OLS on log(multiple) over every round the tab fits (the carried-over
`estimated` point included, as on the tab), doubling time over
`_rsi_survey_dt_ci()`'s t-widened interval, position over the
fitted multiple ÷ and × `_RSI_SURVEY_POS_FACTOR`; merged code: the same,
over the quarters from 2025 on, with `_rsi_code_dt_ci()` and
`_RSI_CODE_POS_FACTOR`; next-step: single OLS in
logit space, odds-doubling over `_rsi_dir_dt_ci()`, position
± `_RSI_DIR_POS_CI` points (the study's own binomial SE at n=`_RSI_DIR_N`);
revenue: OLS on
log2(ARR) over every point, DT lognormal over [max(10, DT×0.65), DT×1.5],
position normal at the tab's 0.3 log2 σ — rather than
fitting its own, so the tabs can't quote different dates for the same
milestone. The revenue milestone takes whichever company crosses first
**per sample** (independent draws — a common shock would narrow the answer
on nothing measured) and re-anchors both to the later of the two series'
last dates, since they end on different days. `test_metr_eta_reproduces_the_metr_tab_defaults`,
`test_eci_eta_reproduces_the_eci_tab_defaults`,
`test_rli_eta_reproduces_the_rli_tab_defaults` and
`test_rsi_eta_reproduces_the_rsi_tab_defaults` pin that; the cross-tab
`test_pacing_quotes_the_same_milestone` compares the two CoBench dates with a
tolerance, since both are Monte Carlo medians off an unseeded RNG.

Seven of the ten are dated off *released* models (METR, ECI, RLI and the detour
study — publicly benchmarked or run on shipped models — and revenue, since ARR is
earned by shipped models); CoBench, the staff survey and merged code are internal
measurements, the last two driven by models Anthropic has internal access to
before release. So `_pc_report_lag()` pulls the seven back
by `_PC_REPORT_LAG_DAYS` (sampled over the range so the spread lands in the CI) whenever *Milestone dates point at* is not `_PC_TIMING_RELEASE`; the other three
are already on that clock and must not be shifted twice.

A checkbox (`rsi_notyet`, default on, in the blend's *Set your own weights*
expander — below its consumers, so it is read via session state a rerun early,
like the weights themselves) conditions the cards and blend on
"not crossed yet": `_pc_condition_on_today()` rejects samples dating a milestone
at or before today — moving the samplers' own `days_to ≥ 0` truncation from the
(stale) anchor date to the present, by rejection rather than clamping — and the
blend mixes with each weight × survival, which together are exactly the mixture
conditioned on the observation, so a definition that put mass in the ruled-out
region loses credence in proportion. The adjustment is shown, not tabulated as a
probability: the Weight column reads prior → effective share, and the CDF
carries a dotted ghost of the unconditioned blend (`raw_days` through
`_pc_rsi_dist_fig`; its height at today is the mass the update removed — the
grid start widens to the ghost's 0.5th percentile so that stays visible). A
fully-crossed component drops out, generalizing the by-hand removal METR
p50 once got at the old, lower bar. A companion number input (`rsi_notyet_ramp`,
default 90d, 0 = off) extends the update into the near future as a soft
likelihood, not a wider hard cut: a sample t days out is kept w.p. t/N inside
the window — the closer a crossing, the more visible its run-up would already
be. On the release clock an active window grows by `_PC_REPORT_LAG_DAYS[0]`
via `_pc_ramp_for()` — a model shipping that soon finished training a
report lag earlier — and the fine print quotes the window actually applied. Survival stays the expected likelihood, so the weight × survival mixing is
unchanged. Assumes a crossing would be known by now — weaker for the internal
evals, and the checkbox help says so.
`test_condition_on_today_truncates_and_reweights`,
`test_condition_ramp_discounts_the_near_future`.

Under the cards sits *RSI projection (tentative)* (`_pc_render_rsi_blend()`):
no single benchmark defines the threshold, so each milestone is treated as one
candidate definition and `_PC_RSI_WEIGHTS` as the credence each gets. The result
is `_pc_rsi_blend()`, the **mixture of their date distributions, not an average
of their medians**, drawn as a CDF (`_pc_rsi_dist_fig()`) between the
cards and the weights table — cumulative rather than a density because a mixture
of ten components is lumpy where they sit and the bin width becomes a
presentation choice, while "X% by date D" is the question the section answers.
Sampled daily with an x-spike so the hover tracks continuously; the axis runs
to the tab's *Project through* year end (`rsi_end_year`, threaded through as
`horizon`; a median past it goes unannotated — the curve ending below 50%
already says so), falling back to the 98th percentile without one.
Each component keeps its own spread, so a late-but-uncertain
milestone widens the blend instead of only shifting it, which an average of the
medians cannot express. Three things are load-bearing. The eta functions take
`samples=True` (via `_pc_eta_out`) so the blend mixes the *same* draw the cards
report, rather than re-rolling and disagreeing with its own table. Weights are
keyed by slug, not by the card label, because the labels are built from the
target constants. And the weights editor's reset **assigns the defaults rather
than popping the keys** — a popped key is re-hydrated straight back out of the
URL on the next run, so on a shared link the reset would never take.

The race half's
headline is the US-vs-China line, so it renders only under the `Country`
attribution; the threshold reaches the display through the chart title.

1. **One roster per attribution, never mixed.** `pc_party` (`_PC_PARTY_OPTIONS` =
   `_DC_PARTY_OPTIONS` + `Country`) picks the entity kind. Tenant/operator race the
   charted companies via `_dc_company_networked_series` at the sidebar's pooling
   level (`pc_pool` reuses `_DC_NETWORK_OPTIONS` verbatim; 'none' → `cluster_of={}`,
   'all' → `None`), with `_dc_hidden_companies` applied and † from
   `_dc_unattributed_companies` — see *Tenant vs operator, and shared tenancy* above.
   'Country' races **every** country instead via `_dc_country_steps` (mode `'site'`
   when nothing pools, else `'company'`) on the **unfiltered** site list, China
   listed twice (China-accessible and domestic-only, never plain "China"). Countries
   used to be appended after the companies; that read as the US being a tenant, so
   they were split out. The US reference pace for the borrowed trend is passed as
   `ref_steps` (the US country series) since company rosters no longer contain it.
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
   to a grid date, `None` past the grid (rendered as past the horizon). The grid runs monthly
   from today to `_PC_HORIZON`. An entity whose catalogued steps cross before today
   is "already there" via `_pc_plan_crossing`, not the Monte Carlo. A *Date points
   at* selectbox (`pc_timing`, `_DC_TIMING_OPTIONS` verbatim) shifts the input
   series by `_dc_timing_shift(label, run_days)` before rows are built, exactly as
   the DC tab does — default is run-finished (capacity-online, or +30d for
   release, one click away); the fine print states which milestone the dates mean.

The tab opens with the pause date and `_pc_render_us_pause()`. `pc_pause_mo` and
the `_pc_capacity_at()` reading it implies are rendered by `render_pacing()`, not
by the panel, because the race below needs the same bar; the panel takes
`pause_d` and `caps` as arguments. The US pauses on that date, frozen at whatever
its compute-derived climb reached by then,
per sample (`us_pause_level` and the sim target are arrays — only the climb is
uncertain, the date is chosen). The control is a `select_slider` over
months-from-today with `format_func` labelling each position with the date
(`_pc_add_months`, whole calendar months so no label repeats): it reads as a date
picker while the stored value stays an int, which round-trips through the URL and
can never go stale the way the scenario cut-off *labels* can. The default is
`_pc_pause_default_mo()`, the offset to `_PC_PAUSE_DEFAULT_YM` (Jul 2027) rather
than a hardcoded month count, so the reset restores a *date*. The pause is on the
**run-finished** clock; the displayed date carries `timing_label`'s
offset, which is what keeps the US–China gap milestone-invariant. The US climb pace
still comes from the **sidebar-pooled US series**, so the networking selector moves
the bar. Between the slider and the subheader sits the *state of play* table, off
that same `_pc_capacity_at()` reading (the race's own `_pc_projection` at the pause
date, so the two sections cannot disagree) — three rows (US, China-accessible,
China domestic-only) of each side's
largest training run, frontier ECI (US = the sim's bar, China = its own sampled
paths at that date, the domestic row those paths less the compute the sites
abroad bought — `chan['compute'] - chan['compute_domestic']`, the same shadow term
the breakdown prices, so the two Chinese rows cannot disagree), the ECI→METR
p50/p80 horizons (capped at `_PC_METR_CAP_HRS`,
past which the bridge reads in centuries) and China's lag in months, taken off the
same US line the chart draws. It renders **before** the crossing bail, so a
projection range too narrow for a catch-up still describes the pause. A caption sensitivity reruns China's compute term
at the catalogued China-accessible pace (sites abroad) vs the export-control band.
The bar is the best model the US has *trained* by the pause: the released-frontier
climb (measured from us_best's release date, not the China-anchored grid) plus
`_PC_SHIP_LAG_DAYS` (the realized trained→shipped lag, GPT-5.5 Pro / Mythos card)
of extra climb. Three scenario checkboxes plus a slowdown slider: `pc_withhold` (default **on**) — a US
*release freeze from pause-run start*: nothing ships once the final run begins, so
distillation's teacher is the released frontier as of run start (`dist_teacher`, ~one
run's climb + ship lag below the bar); labs do ship during big runs, so this is a
policy assumption, not a pipeline fact — hence the label says freeze;
`pc_stop_dist` and `pc_stop_remote` (default off) — cut the
distillation channel at today (`t_dist_stop` in `_cc_cn_crossing_sim`), and cut remote
compute as a *level setback* (`comp_dead`): China's largest run falls back to its
biggest domestic cluster, the compute term is dead until the domestic buildout regrows
the lost OOMs at its catalogued pace, domestic band thereafter (also suppresses the
China-accessible sensitivity, whose premise it removes). The pace band deliberately
stays on the default compute band — it is shared with the CC crossing.
Alongside them, `pc_dom_slow` (0–90%, default 0) slows China's *own* buildout —
equipment/fab controls rather than access controls. The cut comes off the **domestic
component** of each band, not the band itself: `g_hi_eff -= g_dom_hi·p`, likewise for
`lo`, so the export-control band keeps whatever access abroad adds on top while a run
already confined to domestic clusters (remote cut on) takes the cut in full, and the
two levers stack instead of one subsuming the other. `g_dom` is slowed too, so the
remote cut's setback takes proportionally longer to regrow, and the breakdown's
`comp_shadow` cap falls with it. Without a catalogued domestic pace to decompose
against, the whole band scales by `1−p`. Guarded by
`test_domestic_slowdown_lever_delays_the_crossing`.
An *Advanced* expander dates the two cuts (`pc_dist_when` / `pc_remote_when`,
`_pc_when_options` month labels — strings, so they round-trip through the URL and
reset to a constant `Now` = the checkbox alone; a stale label is dropped like
`pc_pause_mo`). Sliders are `disabled` without their checkbox. It also carries
`pc_cn_run` (2–12 months, min = the bar's own run length, default = it): China need
not match the length the bar was set at. A longer run is a one-off **level** move,
never a faster rate — ×(L/L_us) compute into one model, worth `a_partial` per ×10 —
paid for with L−L_us months of wall clock, so China's whole deliverable path lifts by
`cn_gain` and shifts right by `cn_extra`. The organic sim is untouched (algorithmic
channels don't speed up because a run is longer); the lift is applied by re-reading
the *same* sampled paths against a bar lowered by `cn_gain` via `_pc_cross_years`,
which must reproduce the sim's own interpolated crossings exactly — that pairing is
what makes the stated net free of MC noise. Live the answer is a U-curve with an
interior optimum — a year is worse than matching — which is the point of the control.
A dated remote cut
must not slow the years *before* it: the band stays on the export-control default
until the cut and `comp_slow=(t_cut, a_partial·g_domestic)` takes over after, while
the setback is sized at the cut date (today's OOM gap widened by `(g_mid − g_dom)`
per year of lead), so a later cut costs more to regrow but is never worse for China
overall. Cutting *at* today keeps the old global band cap instead. An **Assumes**
line (not buried in the caption) states weight security (theft would put China at the
bar), the drying channels, and the compute band. The panel adds **no release-queue
wait** (unlike the CC 161 section — a queue delays both sides alike); dates follow the
sidebar's *Date points at* in lockstep (US events run-finished + `shift − run`, ECI
release dates + `shift − run − 30d`), so the US–China gap is milestone-invariant; the
chart (actual points included) rides the same clock, x-axis titled by milestone.
The panel ends with `_pc_render_why()`, a **decomposition of the same samples**, not a
second model: `_cc_cn_crossing_sim` fills an optional `channels` dict with the four rate
terms' cumulative ECI (`_CC_CHANNELS`, summing to `traj − anchor` exactly —
`test_channels_account_for_the_whole_climb`), `_pc_at_years` reads each at its own
sample's crossing for the *ECI closed* / *Share* columns, and *Without it* re-crosses
`traj − channel` via `_pc_cross_years`. **Compute splits in two**, because remote access
is a lever the panel offers and folding it into one compute row hid it: `comp_shadow`
(comp_cap, dead) makes the sim accumulate a fifth column, the compute term recomputed
with each sample's *own* pace capped at the domestic ceiling and its own setback window.
Capping the same sample rather than redrawing is load-bearing — when the run already has
no access abroad the cap never binds, so the two coincide exactly and the abroad row
reads a clean 0.0, and it's held ≤ the actual term per step so the remainder can't go
negative while a setback regrows. The shadow is built whether or not the checkbox is on.
The two columns answer different questions and
**Without it is deliberately non-additive** — kill compute and the gap stays wider, so
distillation runs at full strength longer; the caption says so. The total's months run
from China's last model, not from the pause the cards count from — also captioned. A
longer Chinese run appears as its own row (`years_base` is its counterfactual). Live at
defaults innovation dominates, then diffusion, then distillation, then the two compute
rows; the frontier-grade fix shrank distillation and grew diffusion, since the
no-teacher band's floor rose while the pure-innovation floor fell. Note for tests: this table renders *after* the race table, so address
the race table by its columns (`TestPacingTab._entities`), never by position.
China
races the paused frontier in an ECI chart (US kink + fan + crossing diamond); the
paused stock stays distillable while a gap remains, then China runs on its
indigenous band plus its export-control-bound compute term. A *Projection range*
expander (`pc_end_year`, options 2027–2031, default 2031) sets the crossing-search horizon and — like every
other tab's *Project through* — the x-axis of **both** charts: the timeline runs to the
grid end (+120d pad) instead of auto-fitting to content, and the pause panel takes
`horizon=` (sim `horizon_yrs` from China's anchor, floored at 1yr) so its fan ends on the
year the P(cross by …) column is quoted for.

Guarded by `TestPacing` (unit) and `TestPacingTab` (integration).

### Why the test suite is fast (and how to not un-fast it)

~8s, down from more than three minutes that was essentially all in the `AppTest` cases. Three things buy
that back; the first two live in `conftest.py`.

1. **One shared bytecode cache.** Streamlit builds a fresh `ScriptCache` on every
   `AppTest.run()` — once in `AppTest._run()`, again in `LocalScriptRunner.__init__` — so
   the app was re-read, AST-rewritten by `magic.add_magic` and recompiled twice per case. `conftest.py` patches **both** sites to one process-wide cache; patch only one and
   the other keeps recompiling and the win disappears. Isolation is unaffected — the
   bytecode is still exec'd into a fresh module each run.
2. **A smaller Monte Carlo.** `N_SAMPLES` reads `_VP_SAMPLES`; `conftest.py` sets it to 400.
   Nothing asserted depends on the count — the CI defaults the tests check come from the
   deterministic OLS fits, and the app never seeds its RNG. `_VP_SAMPLES=5000 pytest` runs
   at production fidelity as a check. Don't lower the *default*: the band edges are
   percentiles and go visibly ragged well before 5000.
3. **Parallelism.** `pytest.ini` passes `-n auto`; the cases are independent, so a straight
   multiple. It costs the unit-test-only run a little worker startup, hence `-n0`.

Adding an `AppTest` case is cheap (~0.1s per `.run()`), so prefer one to skipping coverage.
What is *not* cheap is defeating the shared cache — a test that mutates
`visualize_projection.py` on disk, or builds its own runner from `streamlit.testing`
internals.
