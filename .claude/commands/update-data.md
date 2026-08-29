---
description: Systematically refresh all data sources (METR, ECI, data centers, RLI, revenue, UK Cyber) and report what changed
argument-hint: "[optional: a single source name to update just that one]"
---

Update the dashboard's data sources to the latest available values, then report exactly what you found, changed, and left alone.

If `$ARGUMENTS` names a specific source (e.g. `eci`, `metr`, `datacenters`, `rli`, `revenue`, `ukcyber`), update only that one. Otherwise update all of them.

Today's date is authoritative for judging staleness — check the current date, then find each source's newest existing entry so you know what "new" means before researching.

## The seven sources

Four downloadable feeds, two hardcoded tables, one digitized figure. For downloadable feeds, prefer fetching the canonical file directly over hand-transcribing.

| Source | Type | Canonical location |
|---|---|---|
| `benchmark_results_1_1.yaml` (METR) | download | `https://metr.org/assets/benchmark_results_1_1.yaml` |
| `epoch_capabilities_index.csv` (ECI) | download (inside zip) | `epoch_capabilities_index.csv` inside `https://epoch.ai/data/benchmark_data.zip` |
| `data_centers.csv` | download | `https://epoch.ai/data/data_centers/data_centers.csv` |
| `data_center_timelines.csv` | download | `https://epoch.ai/data/data_centers/data_center_timelines.csv` |
| `_RLI_RAW` (in `visualize_projection.py`) | hardcoded | Scale Labs RLI leaderboard `labs.scale.com/leaderboard/rli` / `remotelabor.ai` |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (in `visualize_projection.py`) | hardcoded | Press reports (The Information, Reuters, Bloomberg, CNBC, etc.) |
| `aisi_cyber_narrow.csv` (UK Cyber) | **digitized figure — verify, don't download** | AISI blog post, [open-weight cyber gap](https://www.aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber) |
| `aisi_cyber_tlo.csv` (UK Cyber cross-check) | **digitized figure + quoted prose** | Same post's Figure 2, plus the [UK AISI/CAISI Kimi K3 assessment](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) |

Note: Employment, ECI Company Gap, and Compute vs Capabilities tabs have NO feed of their own — they derive from RLI / ECI / data centers, so updating those feeds updates them automatically. Don't hunt for separate data for them.

### Revenue: source-type discipline

The two revenue series follow different conventions on purpose, and mixing them creates a fake growth spike:

- **Anthropic is company-disclosed only.** Third-party trackers (TickerTrends, YipitData) run well above disclosures — TickerTrends put April 2026 at $35.6B where Anthropic itself disclosed $30B. Do not append a tracker estimate to this series without asking.
- **OpenAI is TickerTrends-derived** from `2025-12-31` (21.4) onward, and runs far above the press line (~$25B as of Feb 2026 per The Information / Sacra / Epoch). Keep continuing that one series rather than splicing in a press figure.
- Always classify a candidate figure before using it: company disclosure / press report / third-party alt-data estimate / **forecast**. Forecasts and guidance ("expected to exceed $50B by end of July") are never observations. Quarterly or annual revenue is the wrong unit — these tables are annualized run-rate only.

### UK Cyber (AISI): what "update" means here

AISI publishes **no numbers** for this figure — `aisi_cyber_narrow.csv` was digitized from `fig1-narrow.png` by pixel analysis. So the check is *did the figure change*, not *is there a new file*:

1. Scrape the figure `<img src>` values from the post (CDN filenames are content-hashed, so don't hardcode them; look for `fig1-narrow.png` and `fig2-ranges.png`).
2. Download `fig1-narrow.png` and confirm the recorded calibration still holds: horizontal gridlines at **row 547 (=100%)** and **row 1760 (=0%)** on the 2840×2160 image, with x-ticks Aug2024=601.5px / Feb2025=1010.0px / Aug2025=1411.0px / Feb2026=1819.5px (0.45080 days/px).
3. Confirm each CSV row's `(date, success_rate)` still lands on its marker: convert to pixel coords and check the pixel matches that model's colour. If all rows hit and the model count is unchanged, **the figure is unchanged and the CSV is current** — say so and stop. Do not re-digitize.
4. Only if the figure changed or gained markers: re-digitize, then re-run the two calibration guards (`test_digitized_dates_match_known_releases`, `test_optimistic_bracket_reproduces_aisi_published_lags`). Those must still pass — see CLAUDE.md for why `lag_lo` is the AISI-compatible bound.

Hand-editing a row is fine **only** if AISI states a number in prose. Never transcribe a value from a press summary of the post.

**Check both AISI posts.** AISI's cyber output is spread across more than the one blog post, and later posts do *not* necessarily extend the narrow-task suite:

| Post | Date | Narrow-task rows? |
|---|---|---|
| [How far behind the frontier are leading open weight models on cyber?](https://www.aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber) | 2026-07-17 | Yes — Figure 1, the source of the CSV |
| [Preliminary assessment of Kimi K3's cyber capabilities](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) (UK AISI / CAISI joint; [NIST mirror](https://www.nist.gov/news-events/news/2026/07/uk-aisi-caisi-preliminary-assessment-kimi-k3s-cyber-capabilities)) | 2026-07-23 | **No** — ran only a selective set (ExploitBench + TLO range), not the 70-task narrow suite |

The NIST mirror serves figures at full resolution if you strip the `styles/<preset>/` path segment from the `<img src>` — and unlike the AISI post, **its figures have printed numbers**, so they need no digitization.

**Cyber ranges (TLO) is now ingested** as `aisi_cyber_tlo.csv` — 9 rows digitized from `fig2-ranges.png` plus Kimi K3 quoted from the CAISI post. Refresh it the same way as the narrow file: verify the figure is unchanged (y-axis calibration: **row 1715 = 0 steps, row 498 = 32 steps** on the 3500×2160 image; endpoint = the *solid* trace at the right edge, since the two dotted traces reaching 32 are "best attempt" runs). New models are more likely to arrive as printed prose in a follow-up post than as a redrawn figure — check the prose first, it needs no digitization. `TestUkCyberTlo` holds four published-number guards; they must keep passing.

**Known-available AISI/CAISI cyber data that is NOT ingested** (current as of the 2026-07-23 post — don't re-discover it each run and don't mistake it for staleness):

- **ExploitBench (CMU, 41 post-2023 V8 CVEs).** Ladder score, printed with CIs: Top U.S. models **76.2% ±7.6**, Kimi K3 **32.2% ±4.2**, GLM-5.2 **24.4% ±4.0**. Milestone counts out of 41 (full exploit / general primitives / V8 primitives / bug reproduction / coverage): U.S. 20/30/38/39/41, Kimi K3 0/0/17/34/41, GLM-5.2 0/0/6/24/41.
- **CAISI "Overall Cyber Capability" Elo series** (`Kimi K3 Over Time.png`), a US-vs-PRC trend with 9 labelled PRC points from DeepSeek R1 through Kimi K3 (~2020 Elo). Different metric and different institution from the narrow tasks — don't conflate the two scales.
- **Figure 1 error bars.** Every marker carries a CI whisker; the CSV stores point estimates only. Digitizable from the same image.
- **Cost figures (prose only).** Per-task at 100% reliability: Opus 4.6 $15.17 vs GLM-5.2 $6.12; Opus 4.5 $12.50 vs DeepSeek-V4-Pro $0.28. Per 100M-token range run: ~$85 Opus 4.5/4.6, ~$46 GLM-5.2, $1.19 DeepSeek-V4-Pro.
- **GPT-5.3-Codex is absent by AISI's choice**, not by oversight — the Figure 1 footnote says it was "omitted for legibility; released same day as Opus 4.6 with similar performance." It has no plottable value. Don't hunt for it.

**Kimi K3 status:** released 2026-07-16, open weights slated for 2026-07-27. Evaluated on ExploitBench and TLO only, so it yields **no narrow-task row**. If AISI later runs the 70-task suite on it, that *would* be a genuine new CSV row — check for that specifically. Note it will be the first open-weight model to enter the CSV from a post other than the original.

## Workflow

1. **Scope + baseline.** For each source in scope, find its newest existing entry (tail the file / grep the table) so you can tell what's genuinely new.

2. **Research in parallel.** For the hardcoded tables (RLI, revenue) and to sanity-check the feeds, launch parallel research agents — one per source — that find the canonical source and report only NEW data points beyond the current newest, with exact values, dates, and citations. Instruct them to be factual and to say "no new data" rather than invent.

3. **Apply downloadable feeds (METR, ECI, data centers).** Download the canonical file to a temp file *in the project directory* (never `/tmp` — the safety classifier flags writes outside the project). Then:
   - Diff by key column against the current file to confirm **no locally-curated rows would be lost** (ECI key = `Model version`; DC metadata key = `Name`; timelines key = `Data center`+`Date`; METR = model keys). If any local-only rows exist, STOP and ask before overwriting.
   - Check drift magnitude on existing rows (Epoch recomputes ECI scores live — small drift is expected and fine).
   - Overwrite the file. Clean up the temp file.

4. **Apply hardcoded tables (RLI, revenue).** Hand-edit new rows in `visualize_projection.py`. Keep each table's existing sort order and formatting/alignment. For revenue and any low-confidence / third-party-estimate figure that materially bends a projection, ASK the user before adding it rather than deciding unilaterally. When you exclude a figure for a reason (forecast, wrong unit, wrong source class), leave a short comment in the table saying so — otherwise the next run re-litigates it.

5. **Check UK Cyber (AISI).** Follow the verification recipe above. This is a *figure-unchanged* check, not a download — expect "already current" to be the normal outcome, and expect the newest AISI post to often carry no narrow-task data at all. Cheap to run; do it every time.

6. **Verify.** After each change, sanity-check with the relevant loader:
   ```bash
   _VP_TESTING=1 python3 -c "import visualize_projection as v; print(len(v.load_eci_frontier()))"
   ```
   (use `load_frontier`, `load_data_centers`, `load_rli_data`, `load_ukcyber` as appropriate). Then run `pytest -q`.

7. **Report.** Give a per-source table: Updated / Already current / Skipped (with reason). For updates, list the specific new entries (name, value, date). For "already current," state you verified against the canonical source. Surface any judgment calls and anything you deliberately left out.

8. **Analyze implications.** Based on the report, see if any key assumptions in e.g. RSI timelines or the pacing plan change.

## Guardrails

- Never invent data values. If research can't confirm a newer figure, report the source as current and move on.
- Diff before overwriting; never silently drop curated rows.
- **New data on a new metric is not a refresh.** If a source starts publishing a measure the app doesn't model (a second benchmark, a different scale, an Elo series), report it and offer — do not invent a schema and start filling it in mid-refresh.
- Don't commit or push unless the user asks.
- If the tab/data-source structure has changed since this command was written, trust the code (`grep '^def render_'`, the loaders, and the hardcoded `_*_RAW` tables) over this list, and flag the drift so CLAUDE.md and this command can be updated.
