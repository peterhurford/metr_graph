---
description: Systematically refresh all data sources (METR, ECI, data centers, RLI, Prinz, revenue) and report what changed
argument-hint: "[optional: a single source name to update just that one]"
---

Update the dashboard's data sources to the latest available values, then report exactly what you found, changed, and left alone.

If `$ARGUMENTS` names a specific source (e.g. `eci`, `metr`, `datacenters`, `rli`, `prinz`, `revenue`), update only that one. Otherwise update all of them.

Today's date is authoritative for judging staleness — check the current date, then find each source's newest existing entry so you know what "new" means before researching.

## The five external feeds + two hardcoded tables

There are five downloadable feeds and three hardcoded tables. For downloadable feeds, prefer fetching the canonical file directly over hand-transcribing.

| Source | Type | Canonical location |
|---|---|---|
| `benchmark_results_1_1.yaml` (METR) | download | `https://metr.org/assets/benchmark_results_1_1.yaml` |
| `epoch_capabilities_index.csv` (ECI) | download (inside zip) | `epoch_capabilities_index.csv` inside `https://epoch.ai/data/benchmark_data.zip` |
| `data_centers.csv` | download | `https://epoch.ai/data/data_centers/data_centers.csv` |
| `data_center_timelines.csv` | download | `https://epoch.ai/data/data_centers/data_center_timelines.csv` |
| `_RLI_RAW` (in `visualize_projection.py`) | hardcoded | Scale Labs RLI leaderboard `labs.scale.com/leaderboard/rli` / `remotelabor.ai` |
| `_PRINZ_RAW` (in `visualize_projection.py`) | hardcoded | prinzbench "full" bar chart (scores ±1; dates from the ECI CSV) |
| `_OPENAI_REVENUE` / `_ANTHROPIC_REVENUE` (in `visualize_projection.py`) | hardcoded | Press reports (The Information, Reuters, Bloomberg, CNBC, etc.) |

Note: Employment, ECI Company Gap, and Compute vs Capabilities tabs have NO feed of their own — they derive from RLI / ECI / data centers, so updating those feeds updates them automatically. Don't hunt for separate data for them.

## Workflow

1. **Scope + baseline.** For each source in scope, find its newest existing entry (tail the file / grep the table) so you can tell what's genuinely new.

2. **Research in parallel.** For the hardcoded tables (RLI, Prinz, revenue) and to sanity-check the feeds, launch parallel research agents — one per source — that find the canonical source and report only NEW data points beyond the current newest, with exact values, dates, and citations. Instruct them to be factual and to say "no new data" rather than invent.

3. **Apply downloadable feeds (METR, ECI, data centers).** Download the canonical file to a temp file *in the project directory* (never `/tmp` — the safety classifier flags writes outside the project). Then:
   - Diff by key column against the current file to confirm **no locally-curated rows would be lost** (ECI key = `Model version`; DC metadata key = `Name`; timelines key = `Data center`+`Date`; METR = model keys). If any local-only rows exist, STOP and ask before overwriting.
   - Check drift magnitude on existing rows (Epoch recomputes ECI scores live — small drift is expected and fine).
   - Overwrite the file. Clean up the temp file.

4. **Apply hardcoded tables (RLI, Prinz, revenue).** Hand-edit new rows in `visualize_projection.py`. Keep each table's existing sort order and formatting/alignment. For revenue and any low-confidence / third-party-estimate figure that materially bends a projection, ASK the user before adding it rather than deciding unilaterally.

5. **Verify.** After each change, sanity-check with the relevant loader:
   ```bash
   _VP_TESTING=1 python3 -c "import visualize_projection as v; print(len(v.load_eci_frontier()))"
   ```
   (use `load_frontier`, `load_data_centers`, `load_rli_data`, `load_prinz_data` as appropriate). Then run `pytest -q`.

6. **Report.** Give a per-source table: Updated / Already current / Skipped (with reason). For updates, list the specific new entries (name, value, date). For "already current," state you verified against the canonical source. Surface any judgment calls and anything you deliberately left out.

## Guardrails

- Never invent data values. If research can't confirm a newer figure, report the source as current and move on.
- Diff before overwriting; never silently drop curated rows.
- Don't commit or push unless the user asks.
- If the tab/data-source structure has changed since this command was written, trust the code (`grep '^def render_'`, the loaders, and the hardcoded `_*_RAW` tables) over this list, and flag the drift so CLAUDE.md and this command can be updated.
