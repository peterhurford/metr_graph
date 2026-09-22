# Takeoff model handoff — September 8, 2026

Resume on **`codex/grounded-takeoff`**, available as
`origin/codex/grounded-takeoff`. Do not merge, commit, or push the model to master
without new authorization. The user authorized switching to master, pulling it,
and adding a local TODO reminder after saving this checkpoint; that is not
authorization to publish the Takeoff model on master.

## Resume safely

1. Check `git status` and preserve any unrelated work before switching branches.
2. `git fetch origin`
3. If master has a local TODO reminder, preserve it before switching: the Takeoff
   branch tracks its own TODO file. For example,
   `git stash push --include-untracked -m "Master TODO resume note" -- TODO.md`.
   If TODO is ignored, use `--all` instead of `--include-untracked` for that path.
   Leave this master-specific stash unapplied on Takeoff.
4. `git switch codex/grounded-takeoff` (or, in a new clone,
   `git switch --track origin/codex/grounded-takeoff`).
5. `git pull --ff-only origin codex/grounded-takeoff`
6. Read this file, `TAKEOFF_MODEL.md`, `TAKEOFF_PARAMETERS.md`, and `CLAUDE.md`.

If the local branch has additional work or has diverged, inspect it instead of
resetting it. Do not overwrite a dirty working tree.

## What is saved

- `takeoff_model.py`: pure NumPy simulator; time begins today, with partial automation.
- `visualize_projection.py`: Streamlit Takeoff tab and exact shared RSI projections.
- `TAKEOFF_MODEL.md`: mechanics, evidence conversion, chart meanings, and limitations.
- `TAKEOFF_PARAMETERS.md`: every current default, its intended rationale, and uncertainty.
- `checkpoints/takeoff-defaults-2026-09-08.json`: machine-readable code defaults,
  presets, seeds, and relevant source hashes. **Not a live browser-session export.**
- `openai_experiment_velocity.csv`: 32 source-tooltip observations, already used by RSI.
- Tests and the existing TODO links, including conversation/research references.

No live browser settings were accessible when this handoff was saved. Custom RSI
weights, conditioning, penalty, and custom Takeoff slider settings are therefore
not claimed to be captured by the defaults JSON. For an exact session, save both
the app's **Download assumptions** and **Download simulation draws** exports.
There is currently no assumptions-JSON upload control; restore values from the
export manually or implement an importer. A shared URL preserves many controls
but not a frozen as-of date. Forecasts rerun weeks later use the new current date.

## Decisions already made

- Keep all model work on this branch. Master was backmerged at `597e1a5`;
  both Takeoff and Frontier Thresholds are preserved.
- Coding automation inherits the exact final RSI blend less its AL5 card, including
  conditioning, weights, subjective penalty, clock, and late/past dates. It is a
  milestone, not the point where research feedback suddenly begins.
- Research starts today with existing assistance and an assumed undeployed backlog.
  Fast validation and slower successor training receive disjoint shares of discoveries.
- Research acceleration and research autonomy are separate. Full R&D requires the
  coding date plus, per scenario, either low human hours in every stage with high
  project completion or the RSI tab's AL5 date (`ladder_weight`, default 50%). Human
  hours never feed back into research speed, so that weight moves full R&D automation
  and nothing after it.
- Research difficulty is varied with the other rates, so some scenarios never compound.
- Sustained research feedback remains an internal diagnostic/table row, not a graph.
- Default projection horizon is December 31, 2031.
- All four research-acceleration panels use **2025 units**: experiment activity,
  ongoing useful research pace before validation, additional validated gains in
  months of 2025 progress, and the reference's own research pace.
- The reference has the same resources and pipeline but frozen AI skills. Its
  slowdown reflects the assumed increasing difficulty of further discoveries.
- Rate curves start from existing assisted research, rather than a cold-start
  delivery window. Cumulative validated progress starts at zero because it counts
  additional deliveries after today. Do not floor all curves at 1 or invent deliveries.
- The 2025 research anchor uses RSI activity and an explicit useful-yield assumption.
  It is not measured discovery growth and must not multiply model dynamics twice.
- Most numerical defaults remain illustrative. Be candid about this; do not
  retroactively claim that the defaults or probability bands were empirically fitted.

## Important implementation details

`events` are already years after today: never add coding onset to them again.
`after_coding` computes finite duration differences and retains censored draws.
Late coding dates have less follow-up before the calendar horizon.

`_rsi_experiment_draws` uses a local fixed seed. The RSI fan, threshold card, and
Takeoff use the same sampled trajectories. `_tk_simulate` caches on inherited
arrays, parameters, reporting conversion, and model source digest. The app reloads
stale imported model code when the digest changes. Preserve this behavior: a live
Streamlit session previously raised `KeyError: workflow_progress` after a schema change.

Rebase raw draws before computing quantiles. Keep the production/effort curves
distinct from validated deliveries and the same-compute advantage used in milestone gates.
The global uncertainty control varies only 16 selected parameters; RSI uncertainty
persists even when this control is zero. Details are in the parameter guide.

## Validation and running

The files added for this handoff are documentation and a defaults snapshot; the
snapshot predates `takeoff-v5-al-ladder`.

Known working interpreter on this machine:
`/Users/peterwildeford/.pyenv/versions/dev/bin/python`.

```sh
/Users/peterwildeford/.pyenv/versions/dev/bin/python -m pytest -n 4 -q
/Users/peterwildeford/.pyenv/versions/dev/bin/python -m streamlit run visualize_projection.py
```

Open `http://localhost:8501/?tab=takeoff`. Elsewhere, install `requirements.txt`
and `requirements-dev.txt` in an appropriate environment. Tests use 400 samples;
the app defaults to 5,000. `_VP_SAMPLES=5000` runs tests at production sample count.

## Suggested next steps — not yet implemented

1. Review weakly grounded defaults with the user, especially difficulty (the dominant
   parameter, and the β of the semi-endogenous growth literature, whose central estimates
   put it near 1 with a quality elasticity near 1.4, not 0.5 and 0.8), the
   workflow starting shares/halving times, transfer, parallelization, and breadth gaps.
2. Add sensitivity analysis before interpreting the Central scenario as a forecast.
   Several discounts currently compound, and important parameters have fixed values.
3. Improve calibration with matched research outputs, human intervention/rescue hours,
   compute use, and validated efficiency gains. Activity and code volume are proxies.
4. Consider correlated uncertainty and uncertain structural assumptions. The largest
   open one: human hours are a readout, so an earlier full R&D automation date (the AL5
   definition) changes no later milestone. Letting remaining human hours gate research
   speed would connect them.
5. Add validated assumptions-JSON import and a frozen as-of-date mode if exact replay
   of historical slider configurations becomes a priority.

There is no currently authorized unfinished model feature beyond this checkpoint
workflow. These are proposals for resuming, not promises that they were completed.

## Earlier checkpoints

- `6e22ad2`: original RSI-linked Takeoff explorer checkpoint.
- `00a4050`: feedback from today, grounded outcomes, calibration, shared RSI activity.
- `597e1a5`: backmerge of master through `d01a970`, including Frontier Thresholds.
- The commit containing this handoff preserves the subsequent 2025 reporting changes.
