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

- **Where it's used:** the China tab's China band. Via `_dc_cty_pace` it is also the default extrapolation for `China` and `China (domestic only)` on the Data Centers by-country panel and in the Pausing race.
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
  - Any change to the legality of remote access. Pausing's `pc_stop_remote` lever is that downside case.
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

- **Source:** Brown & Khan, CTS, *DUV Immersion Lithography* (Sep 2026): Table 45 (in `cts_duv_production.csv`). China's 2026 output: IFP, *How to Increase America's AI Lead Over China* (Oct 2026).
- **The ceiling:** China's AI chips made per year if lithography were the only limit, imports continuing (the report's "Robust" path) or banned from 2027. Its levels assume every other bottleneck solved, so they are never China's path.
- **China's path:** IFP's 2026 domestic output, 0.80–1.26M H100e (`_DUV_CN_2026_H100E`; Epoch measured 0.43M in H1 alone, and CTS's own current-trend figure is ~1M), growing at `_DUV_CN_OUTPUT_GROWTH` (median 1.6×/yr, 10th–90th 1.0–2.6×), capped at the ceiling. The cut is 1 − (growth under the policy ÷ growth with neither ban) over today → the Pausing horizon, averaged over a quantile grid of start level and rate, at the AI-push allocation.
- **Basis check:** IFP counts H100e in dense 8-bit and the 910C at 0.76; Epoch's specs give it 0.81 in 16-bit, so ~1M H100e reads the same on CTS's ceiling. Open question: an IFP footnote values the 910C at 0.39 in 16-bit, unsourced. At half the level the import-ban cut to 2031 is 1.6% (not 3.3%) and import plus servicing 40% (not 42%).
- **Why its own growth rate:** this is chip *output*, not installed compute, so it does not borrow `_WC_GROWTH` (1.9×). IFP has ×1.3 for 2026→27, held down by HBM; Epoch's domestic H100e went ×1.8 over 2025→26; SemiAnalysis's 2027 Ascend units run ×1.3 (median) to ×2.7 (high). The 1.6× median sits across those. It was 2.4× from a 0.16–0.40M start, both too low a level and too fast a rate against IFP and Epoch.
- **Import ban result, and what it hinges on:** China's path starts ~4× under the banned ceiling, so the ban binds only once output grows into it. At the defaults: ~3% less growth to 2031, ~10% to 2035. The cut is driven mostly by the growth rate, which the Pausing `pc_duv_growth` slider exposes; don't quote "the ban barely matters near term" without the rate.
- **Servicing ban:** advanced-fab DUVi capacity falls `pc_duv_decay` per year from 2027 (default 15%, 0–50%), lowering the ceiling. China's output is HBM-limited well below that ceiling (IFP: 4–8× under it), so the erosion costs output only once the shrunken fleet falls below the HBM-limited path. To 2031: ~3% alone, ~9% with the import ban (it was 42% when the erosion multiplied output directly); to 2035, ~13% and ~28%.
- **The decay rate is a judgment with no source.** The report says servicing controls would degrade the fleet "though there is some uncertainty as to the magnitude". Low values assume China self-services and cannibalizes its ~190 legacy-fab scanners for parts; high ones that lasers, stages and optics fail without foreign consumables. Not checked against any measured attrition rate.
- **Why AI push:** the Pausing panel's China is a state-directed catch-up. With the ceiling rarely binding before 2031, the allocation matters little to the import-ban cut.
- **Revisit when:** CTS re-versions the table; better estimates of China's realized output or its growth (IFP, Epoch, SemiAnalysis 2027 numbers) arrive; any evidence on scanner attrition without servicing; or a ban or servicing control is enacted.

### 6. China's compute by source (`china_compute_sources.csv`, `_cn_src_projection`)

- **Sourced:** to 2025, Epoch's chip data (domestic, legal) and smuggling model (159k and 480k H100e in 2024 and 2025). For 2026–27, IFP's end-of-year stocks by channel (Fist, Adamson & Pilz, Oct 2026). Display only: nothing downstream reads these rows.
- **Remote access to 2025, a judgment:** through 2024, the two Johor sites in `_DC_CN_ACCESS_ABROAD`. In 2025, SemiAnalysis's leased offshore power (0.85 GW) at IFP's own end-2026 ratio, 727 H100e per leased MW (1.054M over 1.45 GW). That ratio carries IFP's AI share of leased power (~43–50%) and the clusters outside SemiAnalysis's scope; counting all leased power as AI at the Johor sites' density (790/MW) overstated it. The 2025 jump partly reflects the change of method from the catalogue.
- **2028–2030, a judgment:** domestic output grows at `_DUV_CN_OUTPUT_GROWTH`, capped at CTS's ceiling on today's allocation. Legal sales hold at 2027's level (a policy variable). Smuggling scales with CTS's US chip output, so its share of US output holds at IFP's 2027 level (~0.8%). Remote access adds SemiAnalysis's new offshore leasing (2.45 GW in 2027, 2.8, 2.9, then ~3.0 in 2030) at `_CN_SRC_DENSITY_GROWTH` (1.4×/yr) more H100e per MW.
- **The projection hinges on the domestic rate.** At 2.4×/yr, domestic chips are 70% of the 2030 stock (~45M H100e). IFP's own 2026→27 domestic growth is ~1.3×, because HBM binds.
- **Revisit when:** Epoch publishes 2026 smuggling, IFP or SemiAnalysis revise their channel estimates, or US policy changes on H200 or RTX sales, remote access or HBM.

### 7. Share of the Johor campus usable for one Chinese run (`_DC_CN_USABLE`)

- **Value:** DayOne Nusajaya counts at a median 30% of its capacity in every China-accessible series; the Pacing tab draws each run's share logit-normally (`_DC_CN_USABLE_LOGIT_SD` 1.2, 80% range ~8–67%). Bridge MY06 counts in full.
- **Why:** the campus is multi-tenant, and the evidence is company-level (SemiAnalysis names DayOne among ByteDance's offshore landlords; the FT reports Alibaba and ByteDance training in Southeast Asia), not a lease of the whole site. One coherent run also needs one tenant's contiguous halls and networking. At the median, MY06 (~221k H100e) is China's largest offshore cluster.
- **Effect:** the remote-access setback falls from 0.66 OOM at the full campus to ~0.24 at the median.
- **Revisit when:** a source documents a Chinese tenant's share of Nusajaya, or a cluster built for one Chinese run.

### 8. China's compute growth over the calibration window (`_CC_CN_HIST_COMPUTE`)

- **Value:** ×2.12/yr since 2024, the compute share of China's observed ECI slope in the channel split (`_cc_cn_channels`). The forward band (`_CC_CN_COMPUTE_LO/HI`, ×1.4–2.0, mid ×1.68) is unchanged, so China's forward pace sits below its observed one by the slowdown.
- **Evidence (since 2024):** catalogue China-accessible largest site at the Johor usable share ×2.12; domestic-only ×2.08; full Johor campus ×2.64; largest actual runs ×3.05 (6 records), ×1.41 since 2025 (2 records). The two catalogue fits agree and do not depend on the Johor share. These fits run to the plan horizon, so they include planned sites; fitted to today on built sites only they are faster (×2.30 domestic, ×3.12 accessible at the Johor share), because the planned stretch is flat until a 2027 domestic step. ×2.12 is therefore the low end of the evidence, and the deceleration it implies a lower bound.
- **Effect (Pacing, no pacing, default payoff):** China's lag rises from 5.0 / 4.9 / 7.5 months (coder / RSI / ultraintelligence) at ×1.68 to ~6.5 / 6.8 / 10.2. At ×2.64: 8.1 / 9.0 / 13.4.
- **Revisit when:** new Chinese frontier runs are disclosed, a catalogue refresh moves the China fits, or late disclosure of 2025–26 runs changes the record.

## Checked and deliberately left alone

- **Catalogue coverage.** It holds ~0.94 GW of IT power in China at end-2026, against SemiAnalysis's 24 GW national fleet (~4%; the US is ~25%). That is irrelevant to largest-site views as long as the largest sites are present.
  - Uncatalogued sites that could matter: ByteDance's own campuses (whose built sizes are unpublished), four of Alibaba Zhangbei's five campuses, Huawei Gui'an and Ulanqab, Tencent Shaoguan and Qingyuan.
  - Near term, the catalogue's largest Chinese site (~250 MW now, ~760 MW by end-2027) is the right order of magnitude. It could be low by up to ~2×; that is a guess, not verified.
- **Nearby clusters.** ByteDance's new campus shares the Bayin cluster with VNET Bayin Ulanqab. There is no point adding an Ulanqab cluster to `_DC_NETWORK_CLUSTERS` until Epoch catalogues a second site there.
- **Huawei Wuhu timing.** Epoch steps it to 419 MW (facility) on 2026-12-31. SemiAnalysis's undated chart shows ~0.2 GW IT built of a ~3 GW design. Left to the next Epoch refresh.
- **DayOne Nusajaya's owner.** Epoch's `Owner` is blank again as of the 2026-09-29 pull (it had read "ByteDance #likely, Oracle #likely"), so the site is back under the DayOne fallback label under both attributions. This is an upstream field; it is not overridden here.
- **DayOne Kempas (Johor), added upstream 2026-09-29.** ~480k H100e by Oct 2026, B300-only, no `Owner`/`Users`. Not added to `_DC_CN_ACCESS_ABROAD`: SemiAnalysis names DayOne as a ByteDance landlord at company level, but no source ties a Chinese tenant to this site. Revisit with a site-level citation.

## Known limitation

The bands are growth rates of *compute*. On the Data Centers tab's power and cost metrics the same pace overstates growth by roughly the compute-per-watt factor. For example, China-accessible's largest site reads ~2.4 GW of facility power at end-2029. The train-FLOP and H100e metrics, which Pausing uses, read it as intended.
