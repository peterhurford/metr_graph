# China and SEA compute: judgment record

This file records the hand-set China and Southeast Asia compute assumptions: each value, the evidence behind it, how it was derived, and what should make us revisit it.
The code comments at each constant carry the short version; this is the long one. When a value changes, update its entry here with the new evidence rather than appending a history.

## Sources

- **SemiAnalysis, *The Chinese AI Infrastructure Boom: Introducing the SemiAnalysis China Datacenter Model*** (Everlyn, Dylan Patel, Patrick Schaabi; Sep 2026; paywalled). It is a bottom-up model of 1,000+ Chinese facilities.
  - Figures quoted in prose are exact. Figures marked *(chart)* were read off bar charts and are good to roughly ±0.1 GW.
  - It cannot be ingested as data, only cited.
- **Epoch Frontier Data Centers catalogue** (`data_centers.csv`, `data_center_timelines.csv`), pull of 2026-09-22.

## Principle

Coverage of China does not matter much for these tabs. What they plot and race is China's **largest single site** (or largest networkable group). Within the plan horizon (`_DC_CTY_PLAN_HORIZON_DAYS`, ~18 months) that comes from Epoch's catalogued plans. Past it, it comes from a pace. So the judgments that matter are:

- whether the catalogue holds China's largest sites;
- what pace to extrapolate on once the plans run out.

## Judgments

### 1. Mainland and domestic-only China pace past the plans: `_CC_CN_COMPUTE_LO/HI` = 0.15–0.30 OOM/yr (×1.4–2.0)

- **Where it's used:** the Compute/capabilities tab's China band. Via `_dc_cty_pace` it is also the default extrapolation for `China` and `China (domestic only)` on the Data Centers by-country panel and in the Pacing race.
- **Why a band and not the borrowed US pace:** the US pace runs about ×3/yr in H100e. It carried China's largest site to about 3 GW by end-2029 and 7 GW by 2030, above every campus SemiAnalysis names:
  - ByteDance's Ulanqab and Horinger, about 1 GW planned each, approved to build over 2025–29, with 2–3 GW upside;
  - Shanxi and Wuhu, about 500 MW each;
  - Huawei Wuhu, a ~3 GW design with about a sixth of it built.
- **Why this band:** SemiAnalysis calls China "chip-gated" rather than power-gated: power, permitting and labor are abundant, and export controls set the limit. That is the band's premise: controls on leading-edge chips and on the networking needed to fuse them into one run.
- **Cross-check:**
  - The five hyperscalers' power goes from ~6 GW (end-2023) to ~13.5 GW (2026, chart) to ~21 GW (~2028), about ×1.3/yr.
  - Growth over the capex cycle (BATB, meaning ByteDance, Alibaba, Tencent and Baidu, went from $35B in 2024 to $50B in 2025, heading for ~$100B in 2026, about half on chips) sits inside the band.
- **Result, largest domestic site in facility power:**
  - end-2029: median ~1.9 GW (1.0–3.2);
  - end-2030: ~3.2 GW.
- **Guards:**
  - `test_china_extrapolates_on_the_export_control_band`;
  - `test_china_largest_site_stays_inside_the_named_plans` (1–3 GW at end-2029);
  - `TestCcCnComputeBand` (the band stays at or below the catalogue's paces).
- **Revisit when:** controls on chips or networking loosen or tighten materially, or domestic silicon supply steps up (Ascend output, HBM). Also revisit if newer reporting names a Chinese single campus running well past 3 GW by 2029.

### 2. China-accessible pace past the plans: `_DC_CTY_CN_ACCESS_PACE` = 0.20–0.35 OOM/yr (×1.6–2.2)

- **Where it's used:** the default extrapolation for `China-accessible` (China plus `_DC_CN_ACCESS_ABROAD`).
- **Why it differs from the domestic band:** in compute terms the accessible line's largest site is offshore. DayOne Nusajaya has ~584k H100e from Oct 2026, against ~388k for VNET Bayin Ulanqab in 2027. It runs Nvidia chips outside the controls, so the export-control band would understate it. SemiAnalysis: "Export controls cap what silicon can run onshore, so ByteDance's Nvidia fleet lives offshore". Renting GPUs to Chinese hyperscalers is "at present a legal and sizable business".
- **Derivation:**

  **Offshore power.** Chinese hyperscalers' leased power in APAC ex-China *(chart)* is ~0.85 GW (2025), ~1.45 (2026), ~2.45 (2027), ~2.8 (2028), ~2.9 (2029). That is ×1.39/yr over 2026–28 and ×1.26/yr over 2026–29. Most of it is in the Singapore–Johor–Batam hub.

  **Compute per watt, same chips as the US fleet.**
  - Low end: the catalogue's US sites go from 14.2 to 20.5 GW of IT power (×1.44) and from 16.7M to 30.2M H100e (×1.81) over end-2026→2027. That is ×1.26/W.
  - High end: the world-share US compute rate (2.3×) over that same ×1.44 power growth gives ×1.6/W.

  **Product:** ×1.26 × ×1.26 = ×1.6 and ×1.39 × ×1.6 = ×2.2, which gives the band.
- **Sanity check:**
  - Median largest accessible site at end-2029 is ~1.8M H100e. Total offshore leasing then (2.9 GW) at the densities above is roughly 3.5–6M H100e, so the largest site is 30–50% of the total. DayOne is ~45% of today's offshore leasing.
  - The band is capped below the US's own largest-site pace (`test_us_pace_is_above_the_offshore_band_on_live_data`). Chinese tenants abroad build on US-supplied chips in someone else's halls.
- **Result:**
  - China-accessible reaches the US's Jul-2027 largest run in Jan 2030 (median; 10th percentile Feb 2029, 90th past 2031);
  - the US lead at end-2029 is ~14×.
  - For comparison: the domestic band gives Jun 2030 (~17×), and the borrowed US pace gave Mar 2029 (5.8×).
- **What it omits:**
  - GPUs rented from Western clouds ("hundreds of thousands"). They are dispersed, so they don't form one site.
  - Any change to the legality of remote access. Pacing's `pc_stop_remote` lever is that downside case.
- **Revisit when:** remote-access rules change, a Chinese tenant is documented on a second large offshore campus (see 4), or SemiAnalysis revises the offshore leasing path.

### 3. World-share growth rates (`_WC_GROWTH`)

Rates are power growth times compute per watt (×1.3–1.6/yr, as in 2).

- **SEA: 2.0× (1.5–3.2), lowered from 2.6× (1.8–4.0).** The offshore leasing path in 2 gives ×1.6–2.0 for the Chinese-tenant part. The top end stays wide because that series omits US hyperscalers in Johor.
  - Effect: SEA's share now drifts from ~2.9% to ~1.4% by 2031 instead of rising to ~5%. The difference goes mostly to the US.
  - Revisit with evidence on US-hyperscaler buildout in Johor and Batam.
- **China domestic: 1.9× (1.4–2.6), kept.** The hyperscaler power path in 1 (×1.3/yr) times compute per watt gives ×1.6–2.1, which contains it.

### 4. Offshore sites counted as China-accessible (`_DC_CN_ACCESS_ABROAD` = DayOne Nusajaya only)

- **Supported:** SemiAnalysis names DayOne among ByteDance's core offshore landlords.
- **Not added:** Oracle Batam, though it sits in the hub SemiAnalysis names. No source ties a Chinese tenant to that specific site, and the tuple requires a site-level citation.

### 5. DUVi import and servicing levers (`pc_duv_ban`, `pc_duv_service`, `_duv_policy_cut`)

- **Source:** Brown & Khan, CTS, *DUV Immersion Lithography* (Sep 2026): Table 45 (in `cts_duv_production.csv`) and Figure 9's IFP estimate of China's 2026 output.
- **The ceiling:** China's AI chips made per year if lithography were the only limit, imports continuing (the report's "Robust" path) or banned from 2027. Its levels assume every other bottleneck solved, so they are never China's path.
- **China's path:** IFP's 62k–160k B300e for 2026 (×2.52 H100e per B300e, the report's TrendForce conversion, so ~0.16–0.40M H100e), growing at `_DUV_CN_OUTPUT_GROWTH` (median 2.4×/yr, 10th–90th 1.5–3.8×), capped at the ceiling. The cut is 1 − (growth under the policy ÷ growth with neither ban) over today → the Pacing horizon, averaged over a quantile grid of start level and rate, at the AI-push allocation.
- **Why its own growth rate:** this is chip *output*, not installed compute, so it does not borrow `_WC_GROWTH` (1.9×). Ascend units go ~0.8M (2025) → ~1.5M (2026, Epoch, ×1.9); SemiAnalysis's 2027 median is 2M Ascend 950s (×1.3), its high case 4M (×2.7); per-chip performance rises on top, by an amount not pinned down here. Epoch's slow case keeps output near 1% of Nvidia's through 2028 on domestic HBM alone. The 2.4× median is a judgment across those.
- **Import ban result, and what it hinges on:** China's path starts ~16× under even the banned ceiling, so the ban binds only once output grows into it. At the default: ~4% less growth to 2031, ~12% to 2035, a crossing moved by about a day. The cut is driven almost entirely by the growth rate: 0.5% to 2031 at 1.5×/yr, ~11% at 4×. The Pacing `pc_duv_growth` slider exposes it; don't quote "the ban barely matters near term" without the rate.
- **Servicing ban:** advanced-fab DUVi capacity falls `pc_duv_decay` per year from 2027 (default 15%, 0–50%), multiplying China's path whether or not the ceiling binds. ~27% cut to 2031 at the defaults; about +1 month on the crossing.
- **The decay rate is a judgment with no source.** The report says servicing controls would degrade the fleet "though there is some uncertainty as to the magnitude". Low values assume China self-services and cannibalizes its ~190 legacy-fab scanners for parts; high ones that lasers, stages and optics fail without foreign consumables. Not checked against any measured attrition rate.
- **Why AI push:** the Pacing panel's China is a state-directed catch-up. With the ceiling rarely binding before 2031, the allocation matters little to the import-ban cut.
- **Revisit when:** CTS re-versions the table; better estimates of China's realized output or its growth (IFP, Epoch, SemiAnalysis 2027 numbers) arrive; any evidence on scanner attrition without servicing; or a ban or servicing control is enacted.

## Checked and deliberately left alone

- **Catalogue coverage.** It holds ~0.94 GW of IT power in China at end-2026, against SemiAnalysis's 24 GW national fleet (~4%; the US is ~25%). That is irrelevant to largest-site views as long as the largest sites are present.
  - Uncatalogued sites that could matter: ByteDance's own campuses (whose built sizes are unpublished), four of Alibaba Zhangbei's five campuses, Huawei Gui'an and Ulanqab, Tencent Shaoguan and Qingyuan.
  - Near term, the catalogue's largest Chinese site (~250 MW now, ~760 MW by end-2027) is the right order of magnitude. It could be low by up to ~2×; that is a guess, not verified.
- **Nearby clusters.** ByteDance's new campus shares the Bayin cluster with VNET Bayin Ulanqab. There is no point adding an Ulanqab cluster to `_DC_NETWORK_CLUSTERS` until Epoch catalogues a second site there.
- **Huawei Wuhu timing.** Epoch steps it to 419 MW (facility) on 2026-12-31. SemiAnalysis's undated chart shows ~0.2 GW IT built of a ~3 GW design. Left to the next Epoch refresh.
- **DayOne Nusajaya's owner.** Epoch's `Owner` is blank again as of the 2026-09-29 pull (it had read "ByteDance #likely, Oracle #likely"), so the site is back under the DayOne fallback label under both attributions. This is an upstream field; it is not overridden here.
- **DayOne Kempas (Johor), added upstream 2026-09-29.** ~480k H100e by Oct 2026, B300-only, no `Owner`/`Users`. Not added to `_DC_CN_ACCESS_ABROAD`: SemiAnalysis names DayOne as a ByteDance landlord at company level, but no source ties a Chinese tenant to this site. Revisit with a site-level citation.

## Known limitation

The bands are growth rates of *compute*. On the Data Centers tab's power and cost metrics the same pace overstates growth by roughly the compute-per-watt factor. For example, China-accessible's largest site reads ~2.4 GW of facility power at end-2029. The train-FLOP and H100e metrics, which Pacing uses, read it as intended.
