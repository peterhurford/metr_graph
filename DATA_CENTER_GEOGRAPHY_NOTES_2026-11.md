# Data-center geography and the RSI date: notes, Nov 2026

Analysis only, parked for later; no app code changed. It comes from two sessions run Nov 4–6, 2026 against master `392663b`: @e8 #ce6d (where the labs' data centers are) and @d8 #b752 (whether Texas matters for the RSI date). Both read Epoch's Frontier Data Centers data (`data_centers.csv`, `data_center_timelines.csv`) as of the Nov 2, 2026 refresh. The scratch scripts named below live in session scratchpads that get cleaned up, so these numbers are the record.

## Summary

- **Where the compute is:** the US has 86% of the H100e (H100-equivalents) in Epoch's catalogue today (13.5M of 15.8M) and 90% of the planned peak (59.9M of 66.3M). The big clusters are West Texas/New Mexico, the Midwest belt, and the Mid-South (Memphis and north Louisiana).
- **Where the largest run is:** always in the US. On the app's default networking, Memphis (TN) leads to Oct 2027. Then Goodnight (TX), Hyperion (LA) and the Fairwater WAN (WI+GA) each lead in turn, and OpenAI's Abilene cluster (TX) leads from Nov 2028 on.
- **The Texas freeze:** stopping all Texas data-center growth moves the median RSI date about +1 day (Mar 21 → Mar 22, 2028), under every networking mode. Even halving all US compute growth delays RSI only about 6 weeks in the Takeoff model.
- **After 2028:** Epoch has almost nothing dated after 2028, so every "2029–2030" figure below is the last announced plan held flat, not a forecast.

## Part 1: Where the frontier-lab data centers are (@e8 #ce6d)

### Questions

- **Q1.** "Where are the biggest most RSI-relevant data centers geographically located?"
- **Q2.** "Over Jan 2026 – 2030 eoy, for each of Google, Anthropic, OpenAI, xAI and Meta, where is their largest single training run geographically (including via networking)?"

All figures are H100e.

### Q1: by country

| Country | Today | Planned peak (each site's largest timeline point) |
|---|---|---|
| US | 13.5M (86%) | 59.9M (90%) |
| Malaysia | 1.4M (9%) | 1.9M |
| China | 0.4M (2%) | 1.0M |
| UAE | — | 2.0M (Stargate UAE) |
| UK | 0.1M | — |
| Portugal | — | 0.5M |
| **Total** | **15.8M** | **66.3M** |

### Q1: largest frontier-lab sites today

| Site | Location | H100e |
|---|---|---|
| Colossus 2 | Memphis, TN | 1.11M |
| Stargate Abilene (OpenAI) | Abilene, TX | 1.02M |
| New Carlisle (Anthropic on Amazon) | New Carlisle, IN | 0.82M |
| Fairwater Atlanta (OpenAI/Microsoft) | Atlanta, GA | 0.77M |
| Google Pryor | Pryor, OK | 0.64M |
| Meta Prometheus | New Albany, OH | 0.54M |
| Fairwater Wisconsin | Mount Pleasant, WI | 0.45M |

### Q1: largest planned sites (peak)

| Site | Location | H100e |
|---|---|---|
| Stargate New Mexico | NM | 5.1M |
| Stargate Shackelford | TX | 4.6M |
| Fairwater Wisconsin | WI | 4.5M |
| Meta Hyperion | Holly Ridge, LA | 4.1M |
| Google Fort Wayne | IN | 2.9M |
| Stargate Michigan | MI | 2.9M |
| Stargate Wisconsin | WI | 2.6M |
| Stargate Milam | TX | 2.5M |
| Goodnight (Google) | TX | 2.3M |
| Meta Cedar Rapids | IA | 2.3M |
| Stargate UAE | Abu Dhabi | 2.0M |

**Clusters:** West Texas and New Mexico; the Midwest belt (central Ohio, Indiana, southeast Wisconsin, Iowa, Michigan); the Mid-South (Memphis, north Louisiana, Mississippi). Abroad: Abu Dhabi, and Johor in Malaysia (no tenant recorded for the Johor DayOne sites).

### Q2: each lab's largest networked run

**Method:** the app's own functions: `_dc_with_party(load_data_centers(), 'tenant')` → `_dc_series_for_metric(..., 'h100')` → `_dc_company_networked_series(series, _dc_network_site_clusters(level))`, read at each year-end. The default level is `'fabric'`: same-metro sites plus the one announced long-haul link, Microsoft's AI WAN between Atlanta and Wisconsin.

`'fabric'` level (app default):

| Lab | Jan 2026 | End 2026 | End 2027 | End 2028–2030 |
|---|---|---|---|---|
| Google | Columbus OH metro, 0.54M | Columbus, 0.66M | Goodnight TX, 2.28M | Fort Wayne IN, 2.94M |
| Anthropic | Memphis (Colossus 1+2), 0.55M | Memphis, 1.39M | Memphis, 2.10M | Memphis, 2.10M |
| OpenAI | Fairwater WAN (GA+WI), 0.39M | Fairwater, 1.21M | Abilene TX (Abilene + Shackelford), 2.08M | Abilene, 5.58M (Fairwater leads briefly May–Nov 2028 at 5.30M) |
| xAI | Memphis (Colossus 2 only), 0.28M | 1.11M | 1.82M | 1.82M |
| Meta | Temple TX, 0.17M | New Albany OH (Prometheus), 0.77M | Cedar Rapids IA, 2.25M | Holly Ridge LA (Hyperion), 4.14M |

`'plausible'` level, end 2028 (also pools regional groups with no announced link, so read it as an upper bound):

| Lab | Group | H100e |
|---|---|---|
| OpenAI | Texas & Oklahoma | 9.03M (9.19M in 2029) |
| Meta | Mid-South around Hyperion | 5.17M |
| Anthropic | Mid-South (Memphis + New Carlisle + Amazon Madison/Ridgeland, MS) | 5.15M |
| Google | Great Lakes | 3.97M (Texas & Oklahoma leads in 2027 at 3.34M) |
| xAI | unchanged | 1.82M |

### Caveats

- **Colossus 2 counts in full for both Anthropic and xAI.** Epoch lists both (plus Cursor) as users, and the app credits a shared site to every listed tenant. Colossus 1 lists only Anthropic, so xAI can't pool it.
  - Anthropic without Colossus: New Carlisle IN, 0.47M / 0.82M / 0.82M / 1.75M.
  - xAI with Colossus 1 included: 2.10M from 2027.
- **Almost nothing is dated after 2028.** The only later points are CoreWeave Helios (Jan 2029) and Stream Phoenix (Jan 2030), so the 2029–2030 columns repeat 2028.
- **China is undercounted** (2%), because Epoch covers few Chinese sites. See `china_compute_sources.csv`.
- **Planned peaks are announcements:** about 50M of the 66M H100e is unbuilt.

**Scripts** (scratchpad of #ce6d): `dc.py` (Q1 rankings) and `net4.py` (the final Q2 run, with tenant attribution). `net.py`, `net2.py` and `net3.py` are earlier passes. `net2.py` skipped `_dc_with_party` and gave Colossus 2 to Anthropic alone, so ignore it.

## Part 2: Texas freeze and the RSI date (@d8 #b752)


### Q3: How much would a total Texas data-center freeze move the RSI date?

**Answer: about +1 day on the median (Mar 21 → Mar 22, 2028), under every networking mode.**

"Freeze" means all 12 Texas sites in `data_centers.csv` stop growing after today (their future rows in `data_center_timelines.csv` are dropped). The 12 sites: CoreWeave Denton TX, Coreweave Helios, Crusoe Abilene Expansion, Goodnight, Google Midlothian, Google Red Oak, Meta Temple, Microsoft SAT14, Microsoft SAT40, OpenAI Stargate Abilene, OpenAI Stargate Shackelford, Vantage TX1.

Single site (what Takeoff reads through `_tk_dc_compute`):

| Scenario | US compute growth | RSI p10 | RSI p50 | RSI p90 |
|---|---|---|---|---|
| Baseline | 0.446 OOM/yr | Sep 4, 2027 | Mar 21, 2028 | Aug 20, 2029 |
| Texas frozen at today | 0.442 | Sep 4, 2027 | Mar 22, 2028 | Aug 20, 2029 |
| Texas removed entirely | 0.441 | Sep 4, 2027 | Mar 22, 2028 | Aug 20, 2029 |

Networked sites fed into Takeoff instead (scratch patch only, no app change):

| Networking | Growth, base → frozen | Largest run Nov 2031, base → frozen | RSI p50, base → frozen | Shift |
|---|---|---|---|---|
| Single site | 0.444 → 0.436 OOM/yr | 5.56e29 → 5.51e29 (−1%) | Mar 21 → Mar 22, 2028 | +1 day |
| Nearby + announced fabric (app default) | 0.430 → 0.430 | 6.21e29 → 6.17e29 (−1%) | Mar 25 → Mar 26, 2028 | +1 day |
| Nearby + plausible fabric | 0.395 → 0.390 | 7.77e29 → 7.61e29 (−2%) | Mar 31 → Apr 1, 2028 | +1 day |

Holding the historical compute fit at its pre-freeze value, so recalibration can't absorb the freeze, gives a shift of 0 to 2 days.

**Why the shift is so small:**
- **Other sites nearly match Texas.** Without Texas, the networked record falls to Fairwater (WI+GA), 1.6e28 against Abilene's 1.7e28, about 5% lower.
- **Plans only cover the next 18 months.** After `_DC_CTY_PLAN_HORIZON_DAYS`, the projection follows the US historical trend, which barely depends on Texas. The freeze removes some planned Texas steps but doesn't bend the long-run curve.
- **RSI barely responds to compute growth in this model.** Scaling all US compute growth by ×0.75 moves the median RSI date to Apr 6, 2028 (+16 days). Scaling it by ×0.5 moves it to May 4, 2028 (+44 days). That is the upper bound for any compute shock this model can represent.

**What the model can't say:** if Texas carries much of the real buildout from 2028 on, a freeze could bend the national trend itself. The model assumes the trend continues. Representing that would need a "growth falls X%" input, and the answer would follow from whatever X is chosen.

### Q4: From Jan 2026 to the end of 2030, where is the largest single training run?

Always in the US. These are Epoch's dated rows only (built plus announced), with no trend extrapolation. Metric: the largest 2-month training run. The latest row is dated Jan 1, 2030, and the leader doesn't change after Dec 2028, so the "–2030" rows are the last plan held flat.

Single site:

| Period | Site | State |
|---|---|---|
| Jan–Jun 2026 | Anthropic-Amazon New Carlisle | IN |
| Jun 2026 (brief) | Microsoft Fairwater Atlanta | GA |
| Jun 2026–Oct 2027 | Colossus 2 | TN |
| Oct 2027–Jan 2028 | Goodnight | TX |
| Jan–May 2028 | Meta Hyperion | LA |
| May–Nov 2028 | Microsoft Fairwater Wisconsin | WI |
| Nov–Dec 2028 | OpenAI Stargate Shackelford | TX |
| Dec 2028–2030 | OpenAI Stargate New Mexico (1.6e28 FLOP) | NM |

Nearby + announced fabric (app default):

| Period | Who and where | State |
|---|---|---|
| Jan 2026–Oct 2027 | Anthropic, Memphis cluster (Colossus 1+2) | TN |
| Mar–Apr 2026 (brief) | Anthropic, New Carlisle | IN |
| Apr–Jun 2026, Feb 2027 (brief) | OpenAI, Fairwater AI WAN | WI+GA |
| Oct 2027–Jan 2028 | Google, Goodnight | TX |
| Jan–May 2028 | Meta, Hyperion | LA |
| May–Nov 2028 | OpenAI, Fairwater AI WAN (1.6e28) | WI+GA |
| Nov 2028–2030 | OpenAI, Abilene cluster (Stargate Abilene + Crusoe expansion + Shackelford; 1.7e28) | TX |

- **Nearby only:** the same picture without the Fairwater WAN spells. Abilene takes the lead in Oct 2028.
- **Nearby + plausible fabric:** the hypothetical Mid-South group leads to May 2028 (Anthropic, then Meta), Fairwater leads May–Oct 2028, then the hypothetical "Texas & Oklahoma" group leads at 2.8e28, about 1.6× anything else.

On the single-site basis Takeoff uses, Texas leads for only about 4 of the 60 months. With networking counted, Texas leads from late 2028 on. Even so, the freeze moves RSI by only about a day (see Q3).

### Scripts

Scratch only, nothing committed; run from the repo root with `python <path>`:
- `/private/tmp/claude-501/-Users-peterwildeford-dev-metr-graph/b752ea2c-aa2c-49e4-a7d7-4cc151bc4289/scratchpad/tx_freeze.py`: single-site freeze and removal
- `/private/tmp/claude-501/-Users-peterwildeford-dev-metr-graph/b752ea2c-aa2c-49e4-a7d7-4cc151bc4289/scratchpad/tx_why.py`: the leader's step series and the compute-growth ×0.75 / ×0.5 sensitivity
- `/private/tmp/claude-501/-Users-peterwildeford-dev-metr-graph/b752ea2c-aa2c-49e4-a7d7-4cc151bc4289/scratchpad/largest.py`: largest-run leader by networking mode
- `/private/tmp/claude-501/-Users-peterwildeford-dev-metr-graph/b752ea2c-aa2c-49e4-a7d7-4cc151bc4289/scratchpad/tx_freeze_net.py`: networked freeze, with the historical fit held fixed as well

The scratchpad gets cleaned up, so the numbers above are the record.

### Open

- Takeoff reads compute from the largest single site, while the Data Centers and Pacing tabs default to networked sites. On these results the choice moves the RSI median by 4 to 10 days (Mar 21, 25 or 31). It's a possible consistency fix, not urgent.
