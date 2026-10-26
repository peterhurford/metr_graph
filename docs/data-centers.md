# Data Centers tab: design record

Company attribution, the roster, networked clusters, the by-country projection, the buildout-vs-release panel, Compute ⟷ ECI and world shares. Read before changing any `_dc_*`, `_cc_*` or `_wc_*` helper.

## Data-center company labels

`load_data_centers()` attributes a site to its primary listed `Users` value, falling back to
`Owner`, then to the first token of the site name. `_DC_COMPANY_ALIASES` then folds distinct
Epoch labels that are one company for presentation — currently just `Google DeepMind` →
`Google`. It applies to the derived label only; the CSVs keep Epoch's spellings.

Every Google site is `Owner="Google"`, but only some also carry
`Users="Google DeepMind #speculative"`, so without the alias a user-first rule splits one TPU
fleet in two on whether Epoch filled an optional cell: two Google lines, a pooled cluster that
depends on its sites sharing a tag, and a quarterly table that disagrees with the chart above
it. The rest of the app merges them the same way: `_cc_lab_for_site()` maps owner `Google*` to
`Google`, and the ECI tabs substring-match `"Google"`.

## Tenant vs operator, and shared tenancy

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
   China tab and the Compute ⟷ ECI section — behaves exactly as before. `test_shared_site_counts_
   under_every_tenant` pins both halves.
3. **Hidden/unattributed sets are recomputed on the view**, so under 'operator' the
   size gate re-admits owners like Oracle (Stargate Abilene) that the tenant view
   never surfaces.

Guarded by `TestPacing` party tests and `TestDataCentersParty`.

`TestDcCompanyAliases` checks against the live CSV that every Google-owned site resolves to
one company, that both spellings are still present upstream (so the map isn't dead weight),
and that no label is a qualified form of another — the guard that would catch the next
`Meta AI`-vs-`Meta` style split.

## Who appears on the Data Centers tab

`_DC_EXCLUDE_COMPANIES` lists companies that aren't AI labs — colocation and neutral-host
operators whose recorded "company" is the landlord, not whoever trains on the hardware. `_dc_hidden_companies()` drops a listed company
only while its largest single site stays under `_DC_EXCLUDE_MIN_H100` within
`_DC_EXCLUDE_HORIZON_DAYS`, because one can grow into the record holder: QTS
Cedar Rapids is the largest single site in Epoch's data, and an unconditional hide would
have the headline chart name a smaller site as the record. The roster the rule currently produces is
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
4. **Compute ⟷ ECI and the China tab keep the unconditional exclusion.** `_cc_trainflop_
   frontier()` still drops every name in `_DC_EXCLUDE_COMPANIES`: it is the compute half of
   a compute-vs-capability comparison, so each point has to be attributable to a lab that
   ships models, and QTS/DayOne sites have no recorded tenant at all. Crediting their
   capacity to nobody would raise the frontier with no capability to match it, distorting
   the fitted rates and China's ETA downstream. The two tabs' largest-site lines therefore
   differ on purpose and the DC tab's caption says so;
   `test_compute_capabilities_frontier_stays_lab_only` asserts both halves.

Guarded by `TestDcHiddenCompanies` and `TestDcUnattributedCompanies`.

## Networkable data-center clusters

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

## Buildout by country (bottom of the Data Centers tab)

`_dc_render_country_panel()` is the US-vs-China view: each country's largest training
run over time, extrapolated past the end of its recorded data under a 50%/80% cone, plus a
year-end table of US, China, their ratio and China's lag in months. Load-bearing:

1. **Country is a property of the building.** `load_data_centers()` carries Epoch's
   `Country`; `_dc_site_country()` adds `_DC_COUNTRY_FALLBACK` for any timeline-only
   name with no metadata row (empty today). The panel is built from the **unfiltered**
   site list — `_dc_hidden_companies()` is not applied, because a landlord's hall in a
   country is capacity in that country whoever trains in it. `test_country_fallback_only_
   names_sites_epoch_left_blank` retires a fallback the moment Epoch fills the cell.
2. **"China-accessible" = China + `_DC_CN_ACCESS_ABROAD`** (DayOne Nusajaya and Bridge MY06,
   both in Johor), each at the share Chinese labs could use as one run (`_DC_CN_USABLE`:
   Nusajaya at a median 30%, drawn per run on the Pacing tab; `_dc_country_steps` applies it,
   `TestDcCnUsable`; `CHINA_COMPUTE_JUDGMENTS.md` §7). Epoch's own source notes on Nusajaya cite the FT on Alibaba and ByteDance
   training in Southeast Asia; Epoch lists ByteDance (#likely) as MY06's owner and user.
   Both lines are always drawn and projected — mainland alone and accessible — with no
   selector (one was tried and removed as redundant); the sites are *moved* out of Malaysia
   rather than copied, and the caption says the campus has other tenants. Assuming tenancy is the
   strong claim here, same as for the networked-clusters `'plausible'` level; don't add
   sites to the tuple without a citation of the same standard. DayOne Kempas, the other
   Johor campus, stays in Malaysia for want of one. In `'company'` mode the abroad sites pool
   as one tenant, `_DC_CN_ABROAD_PARTY`, so only the `'plausible'` Johor cluster joins them,
   and any `'plausible'` group in a country-step detail is tagged "(hypothetical link)"
   (`test_sites_abroad_pool_as_one_party_only_on_a_hypothetical_link`).
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
6. **Past the plans, China runs on an evidence band; everyone else borrows the US trend**
   (`_dc_cty_pace`, default option 'band'). Mainland and domestic-only use
   `_CC_CN_COMPUTE_LO/HI` (shared with Pacing and the China tab); China-accessible,
   whose largest site is offshore Nvidia capacity, uses `_DC_CTY_CN_ACCESS_PACE`, derived from
   SemiAnalysis's offshore leasing path. Both are read as 80% intervals, widened by
   `|g_own − g_ref| / 1.28` so a country's own fit sits at the 80% edge. The derivations,
   sources and revisit triggers live in `CHINA_COMPUTE_JUDGMENTS.md`; update it with any change.
   Guards: `test_china_extrapolates_on_the_export_control_band`,
   `test_china_largest_site_stays_inside_the_named_plans` (1–3 GW at end-2029) and
   `test_us_pace_is_above_the_offshore_band_on_live_data`. The US-for-everyone and own-trend
   options are one click away; own-trend has China overtaking the US inside the window — that
   is the ramp bias above, not a finding. The US always uses its own fit.
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

## Data Centers — the buildout-vs-release-timing panel

`_cc_company_buildout()` (bottom of the Data Centers tab) is a **pure timing test**: each
capacity step of a lab's largest single data center, shifted forward `_CC_RELEASE_LAG_DAYS`
(90d = 60d training + 30d release prep), against when that lab's models actually shipped.
Capability is never compared. Two directions with two different clocks — both read out
through the one timeline chart (forward in the predicted-row hovers, backward in the
connectors and the headline median) — so they must not contradict each other:

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
   series without its release date changing, leaving a cluster matching
   nothing. That's a fact about rescoring, not about when the lab shipped, and this panel
   compares dates only. Tier-3 matches draw hollow, are marked †, and are **excluded from
   the headline median**, which is a claim about record-setting releases.
3. **`_CC_TRAINED_ON` pins a release to its site.** The backward rule reads the
   lab's *largest* record step, so a model trained on a smaller site lands on
   whichever record happened last (the one pin today sends GPT-6 Astra to Stargate
   Abilene rather than Fairwater Atlanta). A pin, keyed `(lab, Model name)`,
   takes the named site's latest record step online a training run before the
   release and falls back to the rule when it has none; the release hover says
   the site was pinned. `TestCcResponsibleCluster` checks each pin binds against
   the live CSVs.
4. **`_cc_company_all_releases()` keys on `(Model name, Release date)`.** Model name alone
   merges GPT-4o's May and August 2024 releases; Display name alone splits one model's
   reasoning-effort rows, and which suffixed variant wins a dedup flips between Epoch pulls.
   Same-day releases sort by descending ECI so a step is offered the flagship.

`TestCcForwardMatch`, `TestCcResponsibleCluster` and `TestCcCompanyAllReleases` guard all of the above.

## Compute ⟷ ECI and the China tab — how they derive from the Data Centers catalogue

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
   and `_render_compute_eci()` go through `_dc_timing_shift("Training run
   finished")`, the DC tab's own milestone vocabulary.
4. **Country paces are cross-checked, not forked.** `_cc_country_pace_check()` runs the
   by-country engine (`_dc_country_steps` + `_dc_cty_fit`, no pooling, plan-horizon
   clipped, unfiltered sites) for US / China-accessible / China-domestic. Captions quote
   it next to the segment fits and the hand-set `_CC_CN_COMPUTE_LO/HI` band; tests keep
   the two engines honest — US segment fit within twice the country fit's `sigma_g`
   (`test_us_pace_cross_check_agrees_across_tabs`), and the China band at or below the
   catalogued paces (`TestCcCnComputeBand`: HI ≤ China-accessible ramp, LO ≤ domestic
   pace). The band stays hand-set on purpose: it claims coherent-*single-run* growth
   (networking is the export-controlled step), while the catalogue counts buildings —
   and the China-accessible fit is a from-zero ramp the DC tab itself flags as hot. If a
   refresh breaks either test, retarget the constants deliberately.

**Build-up rates are calibrated.** Each country's ECI rate (algorithmic term +
exchange rate × compute growth) is multiplied by `_cc_rate_calibration`: its
observed frontier slope since `_CC_CALIB_SINCE` over that build-up rate (US ×1.31,
China ×1.37 live), so projected lines fit their history. The distillation scenario's
China rows are the calibrated rate less the channels the decomposition measures
(distillation, then diffusion). The Pacing tab's pause panel scales its US climb by the same
US factor. `TestCcRateCalibration`.

*Compute ⟷ ECI* (exchange rate, iso-ECI/iso-compute charts, the two engines) and
*ECI Forecasts* (`_cc_eci_forecast`) close the Data Centers tab via
`_render_compute_eci()`, on a fixed 2-month run, `dc_future`, and a horizon of
end-`max(2029, dc_end_year)`; the catalogue cap never drops below end-2028 and every
fit era ends by Jan 2029, so the segment fits don't move with it
(`TestDataCentersComputeEci`). The frontier itself is deliberately not charted.
The China tab's provenance captions deep-link `?tab=datacenters`
(`test_dc_derivation_is_stated`). Lab-only exclusion unchanged
(`test_compute_capabilities_frontier_stays_lab_only`). China sidebar: *Project
through* (`cc_end_year`, default 2029) moves the US-vs-China horizon
(`test_project_through_horizon`); the China-ETA panel keeps its own adaptive horizon. *Run length* (`cc_run`,
`_PC_RUN_OPTIONS`, default 2-month) switches to `train_flop_6mo` + one-run shift —
a constant scale-and-slide, same leaders (`test_run_window_scales_levels_only`);
China's capacity band scales by the window ratio; `_cc_country_pace_check` stays
on 2-month (paces identical).

## Data Centers — share of world AI compute by region

`_render_world_shares()` (above *Compute ⟷ ECI*, moved from the China tab; it
replaced a catalogue-only region stack, whose split survives as the diamonds) shows how the
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
   AI-2027 tracker (~12%), RAND (~15%) and IFP's chip count (~7% inside China).
   China's 6–15% spans them, centred at 9% on the chip counts; the US central is
   76% (IFP ~77%). Retarget the constants when the estimates move rather
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
   bottom decile at the horizon, China absorbs 26% of the shortfall against
   11% under independence. Guarded by
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
   May 2025 China lands at ~12%, under Epoch's ~15% cluster-share reading. That half is
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
`TestCcWorldShares` (unit) and `TestDataCentersRegionShare` (integration).
