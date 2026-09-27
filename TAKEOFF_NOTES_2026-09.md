# Takeoff v5 notes — September 2026

Parked write-up of the `takeoff-v5-al-ladder` revision: what changed, the numbers
before and after, what was deliberately left alone, and an assessment of the
intelligence-explosion literature the model was checked against. Resume from
`TAKEOFF_HANDOFF.md`; this file is the reasoning behind that checkpoint.

## What changed in v5

1. **AL5 is a definition of full R&D automation, not a coding-onset input.** The
   RSI blend's AL5 card dates full R&D automation — the model's own output
   milestone — yet it sat in the inherited coding-onset blend with the largest
   weight (15). `_TK_LADDER_SLUGS` now zeroes it in the coding blend and returns it
   separately as `ladder_years`. Removing it barely moves the onset (median Nov
   2027 either way).
2. **`ladder_weight` (default 50%)** picks, per scenario, whether full R&D
   automation is the human-hours gate (≤5% human hours in every stage, ≥90%
   project completion) or the AL5 date. Either waits for the coding milestone.
   The two are rank-paired and the AL5 draws carry the RSI subjective penalty.
   The tab tabulates the two subsets side by side.
3. **Research difficulty is sampled with the other rates** (16th factor row).
   Fixed and uncalibrated, it left the model with no scenario in which feedback
   fails to compound. At the same ±40% log spread, ~9% of scenarios now sit below
   the compounding threshold (quality elasticity < difficulty).
4. **Runaway draws.** A draw whose research exceeds numerical range records no
   further milestones and counts as unresolved only while a milestone could still
   arrive inside the horizon. The tab captions ≤1% such draws as not arriving and
   errors above `_TK_LIMITED_MAX`. Without this, horizons from 2033 on errored.

## Numbers (5,000 samples, "Training run finished" clock, conditioned)

10% / median / 90%, with the share arriving by end-2031.

| Milestone | v4 | v5 |
|---|---|---|
| Coding automation (inherited) | Feb 2027 / Nov 2027 / Jul 2029 | Jan 2027 / Nov 2027 / Aug 2029 |
| Full R&D automation | Oct 2028 / Mar 2029 / Nov 2029 (99%) | Aug 2027 / Nov 2028 / Dec 2029 (99%) |
| — human-hours definition | as above | Sep 2028 / Mar 2029 / Jan 2030 |
| — AL5 definition | — | Jul 2027 / Nov 2027 / Aug 2029 |
| Superhuman AI research | Sep 2029 / Aug 2030 / beyond (83%) | Aug 2029 / Sep 2030 / beyond (77%) |
| Broad superintelligence | Jan 2030 / Jan 2031 / beyond (76%) | Dec 2029 / Jan 2031 / beyond (71%) |

- The two full-R&D definitions disagree by ~16 months at the median.
- **The ladder weight moves nothing after full R&D automation**: superhuman and
  superintelligence read 77% / 71% at every weight from 0 to 100. Human hours are a
  readout in this model and never feed back into research speed. The drop from
  83% / 76% is entirely from sampling difficulty.
- The fizzle tail shows at later horizons: superintelligence by 2035 is 92% in v5
  against 100% in v4; by 2040, 95%.
- Research pace 2× today: Sep 2027 / Jan 2028 / Jun 2028. 5×: Aug 2028 / Feb 2029 /
  Nov 2029.
- Full R&D → superhuman gap: median 15 months in human-hours-defined draws, 27 in
  AL5-defined ones.

### Sensitivities

- **Difficulty is the dominant parameter.** Fixed values, everything else Central:
  superintelligence by 2031 is 96% at 0.35, 76% at 0.50, 35% at 0.71, 3% at 1.0.
- **Semi-endogenous-growth parameters.** The model's `difficulty` is β and its
  asymptotic quality elasticity is ≈ `taste_slope` (the quantity channel saturates
  on experiment compute), so r = λ/β. Ho & Whitfill's medians (λ ≈ 1.40, β ≈ 1.01)
  give superhuman 70% / superintelligence 64% by 2031, medians Nov 2030 / Apr 2031;
  r = 1.2 (1.2 / 1.0) gives 53% / 47%; r = 1.3 gives 63% / 56%. The Central margin
  (0.8 − 0.5 = 0.3) is close to their λ − β ≈ 0.39, but the model's 2031 shares run
  somewhat above theirs.
- **`software_months`** 9 → 6 gives 78% / 73%; 4.5 gives 76% / 70% (on the
  fixed-difficulty model). A faster baseline lowers the shares slightly; I'd guess
  because the 12× target and the difficulty discount both scale with it, but this
  is unverified. Left at 9.

Harnesses for these runs are in the session scratchpad, not the repo; each is
`blend_and_ladder()` + `tk.simulate()` with overrides, easy to rebuild from
`test_takeoff_model.py`.

## Left alone, on purpose

- Central difficulty 0.5 and quality-elasticity prior 0.8 — the margin matches the
  literature; the lower tail was the missing piece, not the centre.
- `coding_today` 70%: code authorship share is not an hours share.
- Workflow starts 40/30/50: an AL-composition-implied ~22% human share is
  assumption-dependent.
- `cobench_85` and the productivity-multiple cards stay in the onset blend — a
  judgement call worth revisiting.
- Human hours do not gate research speed. Connecting them is the structural change
  that would let the AL5 definition move later milestones (handoff, next step 4).
- `_pc_rsi_components`'s cache `data_key` omits the R&D-automation CSV rows, so a
  refresh of that file may serve stale components until the process restarts.
- `checkpoints/takeoff-defaults-2026-09-08.json` is still the v4 snapshot.

## Assessment against the intelligence-explosion literature

Checked against a September 2026 draft arguing that automating AI R&D could trigger
an intelligence explosion (Chan, Pachocki, Clark, Hinton, Bengio, Davidson et al.;
draft, not for distribution — nothing from it is reproduced here) and the public
work it builds on: Ho & Whitfill's semi-endogenous estimates, Whitfill & Wu on
compute bottlenecks, and Cunningham et al. (Jul 2026) on measured productivity gains.
The short version: the policy asks are right and the speed is overstated.

Where it holds up:

- The semi-endogenous framing (`dA/dt = A^(1−β)·E^λ`, explosion iff r = λ/β > 1) is
  the model already: `difficulty` is β exactly.
- The Central margin over difficulty is close to the literature's λ − β.
- A ~17-month path from full automation to a 10× research pace lines up with the
  model's 15-month median from full R&D automation to superhuman research.
- The visibility asks amount to reporting the indicators the RSI tab is built from.

Where to push back:

1. **The headline rests on a poorly measured number.** Ho & Whitfill's central r
   sits near 1.2–1.4 with sub-field intervals from ~0.4 to ~2.7, and they concede an
   upward bias. In the model, difficulty 0.35 → 1.0 takes superintelligence by 2031
   from 96% to 3%.
2. **Compute is more than a friction.** In the model the *quantity* of research is
   bottlenecked by experiment compute (the Whitfill & Wu point, structurally). Only
   research *quality* escapes it, so the explosion claim is really that research
   taste scales with capability outside the compute constraint. That is the crux,
   and the literature treats it as a caveat.
3. **"Automating AI R&D" is a ramp, not a trigger.** AL4 share is climbing, merged
   code per person is ~8×, experiments per researcher ~1.6×; the model reaches 2×
   research pace around Jan 2028; Cunningham et al. find the gains have not passed
   the explosion threshold. Clocks that start at "full automation" inherit the
   16-month disagreement between the two best datings of it.
4. **AL5 is weaker than its blend weight suggests.** No direct observations (the
   AL4 curve shifted along the ladder); a Claude judge agreeing with task owners
   59% of the time, worst at the AL3/AL4 boundary; and the wrong unit — task shares
   inside supervised work, not hours. 90% of tasks at AL5 can coexist with the
   remaining 10% (direction, verification) being the binding constraint. Reporting
   asks should specify hours and rescue effort, not task shares.

Overall: the ramp is happening now through 2028; whether it then compounds depends
on a parameter nobody has measured well. Treat the model's ~70% by 2031 as a 50–70%
range that turns on difficulty, not as a forecast.

## Resume

- Branch `codex/grounded-takeoff`, backmerged with master. Suite: 792 passed.
- Next: decide whether human hours should gate research speed; revisit the onset
  blend's membership; refresh the defaults checkpoint for v5.
