"""Model invariants and numerical checks for conditional takeoff scenarios."""

import numpy as np
import pytest

import takeoff_model as tk


def test_workflow_trajectory_observes_deployments_and_keeps_all_draws():
    """Human hours fall only after deployments, and every draw stays in the bands."""
    result = tk.simulate(dict(years=2, **tk.POINT_ESTIMATE), n=3,
                         onset_years=np.array([0, 20, np.inf]))
    progress = result["workflow_progress"]
    grid = progress["years"]
    human = progress["human_quantiles"]
    assert grid[0] == 0 and grid[-1] == 2
    assert human.shape == (len(grid), 3, 3)
    assert progress["capability_log"].shape == (3, len(grid))
    starts = np.array([tk.DEFAULTS["human_" + s] for s in tk.WORKFLOW_KEYS]) / 100
    np.testing.assert_allclose(human[0], np.tile(starts, (3, 1)))
    # No improvement before the first fast validation cycle finishes.
    before_deployment = grid < 1 / 12 - 1e-12
    np.testing.assert_allclose(human[before_deployment],
                              np.broadcast_to(human[0], human[before_deployment].shape))
    assert np.all(np.diff(human, axis=0) <= 1e-12)
    assert np.all(human >= tk.DEFAULTS["human_floor"] / 100)
    assert np.all(human[-1] < human[0])
    # Even a never-arriving coding milestone retains a research trajectory.
    never = tk.simulate(dict(years=2, **tk.POINT_ESTIMATE), n=1,
                        onset_years=np.array([np.inf]))["workflow_progress"]
    assert np.all(never["human_quantiles"][-1] < never["human_quantiles"][0])


def test_seeded_draws_and_rng_isolation():
    before = np.random.get_state()
    a = tk.simulate(n=100)
    b = tk.simulate(n=100)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for name in tk.MILESTONES:
        np.testing.assert_array_equal(a["events"][name], b["events"][name])
    np.testing.assert_array_equal(a["onset"], b["onset"])


def test_baseline_growth_and_zero_transfer_remove_feedback():
    """With zero transfer, AI skill slopes cannot change the software path."""
    p = dict(transfer=0, compute_growth=1.0, **tk.POINT_ESTIMATE, years=2, baseline_inputs=0)
    a = tk.simulate(p, n=1)
    rate = np.log(2) * 12 / tk.DEFAULTS["software_months"]
    beta = tk.DEFAULTS["difficulty"]
    expected = rate * tk.DEFAULTS["pipeline_months"] / 12 + np.log1p(beta * rate * p["years"]) / beta
    assert a["software_log"][0] == pytest.approx(expected)
    b = tk.simulate(dict(p, coding_slope=2.0, taste_slope=2.0), n=1)
    np.testing.assert_allclose(a["software_log"], b["software_log"])


def test_compute_budget_conservation():
    for growth in [1.0, 2.0]:
        p = dict(compute_growth=growth, **tk.POINT_ESTIMATE, years=2,
                 training_share=30, experiment_share=50)
        r = tk.simulate(p, n=5)
        np.testing.assert_allclose(r["allocated"].sum(axis=0), r["budget"])
        np.testing.assert_allclose(r["allocated"] / r["budget"],
                                   np.tile([[0.3], [0.5], [0.2]], (1, 5)))
        expected = 2 if growth == 1 else np.expm1(np.log(growth) * 2) / np.log(growth)
        np.testing.assert_allclose(r["budget"], expected)
        assert np.all(r["training_used"] <= r["allocated"][0] + 1e-10)


def test_validation_reserves_but_does_not_spend_training_compute():
    r = tk.simulate(dict(compute_growth=1.0, **tk.POINT_ESTIMATE, transfer=0,
                         years=2, training_months=3.0, validation_months=3.0),
                    n=1, step=1/120)
    # Four 3-month runs in two years; half the reserved training allocation
    # sits idle during validation rather than being spent twice.
    assert r["training_used"][0] == pytest.approx(0.4)
    assert r["allocated"][0, 0] == pytest.approx(0.8)


def test_algorithms_wait_for_the_next_run_and_validation():
    p = dict(compute_growth=1.0, **tk.POINT_ESTIMATE, training_months=3.0,
             validation_months=0.0, transfer=0, fast_share=0, pipeline_months=0)
    first = tk.simulate(dict(p, years=0.25), n=1, step=1/120)
    assert first["software_log"][0] > 0
    assert first["deployed_log"][0] == pytest.approx(0, abs=1e-10)
    second = tk.simulate(dict(p, years=0.5), n=1, step=1/120)
    assert second["deployed_log"][0] == pytest.approx(first["software_log"][0])
    delayed = tk.simulate(dict(p, years=0.5, validation_months=3.0), n=1, step=1/120)
    assert delayed["deployed_log"][0] == pytest.approx(0, abs=1e-10)


def test_later_coding_does_not_postpone_the_start_of_research():
    """Research improves before the coding date; only RSI waits for it."""
    p = dict(years=.5, **tk.POINT_ESTIMATE)
    a = tk.simulate(p, n=1, onset_years=np.array([1.]))
    b = tk.simulate(p, n=1, onset_years=np.array([10.]))
    assert a["software_log"][0] >= b["software_log"][0] > 0
    assert b["fast_software_log"][0] > 0
    assert np.isinf(b["events"]["RSI"][0])


def test_external_onset_preserves_past_dates_and_censored_tail():
    onset = np.array([-1.0, .25, 2.0, 20.0, np.inf])
    b = tk.simulate(n=5, onset_years=onset)
    np.testing.assert_array_equal(b["onset"], onset)
    # A runaway reaches RSI whatever its coding date says
    # (test_a_runaway_reaches_rsi_unless_the_floor_blocks_it).
    held = b["software_log"] < 1e6
    for name in tk.MILESTONES:
        assert np.all(b["events"][name][held] >= np.maximum(onset, 0)[held])
    assert np.isnan(b["outcomes"]["Full coding automation"]["human_mean"][-1])
    assert np.all(np.isfinite(b["outcomes"]["Today"]["human_mean"]))
    durations = tk.after_coding(b["events"][tk.MILESTONES[-1]], onset)
    assert not np.isnan(durations).any()
    assert np.isinf(durations[-1])


def test_rsi_needs_low_human_work_and_the_coding_date():
    """With every stage already under the bar, RSI arrives with coding automation."""
    p = dict(**tk.POINT_ESTIMATE, onset_low=0, onset_high=0, human_floor=0.0, human_direction=0.0,
             human_experiments=0.0, human_verification=0.0)
    assert tk.simulate(p, n=1)["events"]["RSI"][0] == 0
    later = tk.simulate(p, n=1, onset_years=np.array([.5]))["events"]["RSI"][0]
    assert .5 <= later < .5 + 1 / 12


def test_rsi_turns_on_research_direction_alone():
    """RSI is direction handed off: other stages still needing people do not
    block it, and a low average does not count while direction is above it."""
    base = dict(years=0.1, **tk.POINT_ESTIMATE, human_floor=0.0)
    onset = np.array([0.0])
    handed_off = tk.simulate(dict(base, human_direction=0.0, human_experiments=60.0,
                                  human_verification=60.0), n=1, onset_years=onset)
    assert handed_off["events"]["RSI"][0] == 0
    still_human = tk.simulate(dict(base, human_direction=9.0, human_experiments=0.0,
                                   human_verification=0.0), n=1, onset_years=onset)
    assert still_human["outcomes"]["Today"]["human_mean"][0] < tk.DEFAULTS["full_human"] / 100
    assert np.isinf(still_human["events"]["RSI"][0])


def test_persistent_human_floor_above_the_bar_prevents_rsi():
    """A human-work floor above the RSI bar rules RSI out in every draw."""
    r = tk.simulate(dict(human_floor=6.0), n=30)
    assert np.isinf(r["events"]["RSI"]).all()


def test_research_difficulty_is_sampled_with_the_other_rates(monkeypatch):
    seen = []
    original = tk.reference_software

    def capture(*args):
        seen.append(np.asarray(args[5]))       # difficulty
        return original(*args)

    monkeypatch.setattr(tk, "reference_software", capture)
    tk.simulate(dict(years=0.5), n=500)
    drawn = np.log(next(d for d in seen if d.shape == (500,)))
    # Lognormal around the setting, at the returns-to-research spread.
    assert np.median(drawn) == pytest.approx(np.log(0.5), abs=0.1)
    assert drawn.std() == pytest.approx(tk.DEFAULTS["returns_spread"] / 100, abs=0.1)
    tk.simulate(dict(years=0.5, difficulty=0), n=5)  # the undiscounted branch still runs


def test_a_runaway_reaches_rsi_unless_the_floor_blocks_it():
    """A diverged draw has automated coding whatever its inherited date says, so
    it reaches RSI, even before that date. Under a floor above the bar no draw
    can reach RSI, so a runaway never does."""
    explosive = dict(difficulty=0.0, taste_slope=2.0, transfer=100, **tk.POINT_ESTIMATE, years=12)
    early = tk.simulate(explosive, n=1, onset_years=np.array([11.0]))
    assert early["software_log"][0] == 1e6
    assert early["events"]["RSI"][0] < 11.0
    reached = tk.simulate(explosive, n=1, onset_years=np.array([0.5]))
    assert reached["software_log"][0] == 1e6          # it did diverge
    assert np.isfinite(reached["events"]["RSI"][0])
    blocked = dict(explosive, human_floor=6.0)
    waiting = tk.simulate(blocked, n=1, onset_years=np.array([20.0]))
    assert np.isinf(waiting["events"]["RSI"][0])
    inside = tk.simulate(blocked, n=1, onset_years=np.array([0.5]))
    assert np.isinf(inside["events"]["RSI"][0])


def test_progress_years_reads_the_capability_gain_after_rsi():
    """Forethought's readout: capability gained in the year after RSI over the
    base rate. A runaway inside the window counts as more than any number; a
    window running past the horizon, or no RSI, is not measured."""
    r = tk.simulate(dict(tk.POINT_ESTIMATE, years=6), n=3,
                    onset_years=np.array([0.5, 1.0, np.inf]))
    wp, rsi = r["workflow_progress"], r["events"]["RSI"]
    years = tk.progress_years(r, 2.0)
    i = int(np.flatnonzero(np.isfinite(rsi))[0])
    gain = (np.interp(rsi[i] + 1, wp["years"], wp["capability_log"][i])
            - np.interp(rsi[i], wp["years"], wp["capability_log"][i]))
    assert years[i] == pytest.approx(gain / 2.0)
    assert np.isnan(years[2])
    assert np.isnan(tk.progress_years(r, 2.0, window=10)[i])
    explosive = dict(difficulty=0.0, taste_slope=2.0, transfer=100, **tk.POINT_ESTIMATE, years=12)
    runaway = tk.simulate(explosive, n=1, onset_years=np.array([0.5]))
    assert np.isinf(tk.progress_years(runaway, 2.0)[0])


def test_compute_boost_starts_at_each_draws_rsi_and_spares_the_reference():
    """`rsi_compute_boost` multiplies log compute growth from each draw's own
    RSI date, slowdown included; before RSI, and in the frozen-AI reference,
    nothing changes."""
    onset = np.array([0.5, 1.0, 2.0])
    base = dict(tk.POINT_ESTIMATE, years=6, rsi_compute_boost=1.0,
                rsi_compute_delay=0.0, rsi_compute_ramp=0.0)
    a = tk.simulate(base, n=3, onset_years=onset)
    b = tk.simulate(dict(base, rsi_compute_boost=2.0), n=3, onset_years=onset)
    rsi, c = a["events"]["RSI"], a["compute"]
    np.testing.assert_array_equal(rsi, b["events"]["RSI"])
    extra = (tk.log_level(6.0, c["rate"], c["late"], c["slow_at"])
             - tk.log_level(rsi, c["rate"], c["late"], c["slow_at"]))
    gap = b["workflow_progress"]["compute_log"][:, -1] - a["workflow_progress"]["compute_log"][:, -1]
    np.testing.assert_allclose(gap, extra, rtol=1e-6)
    years = a["workflow_progress"]["years"]
    before = years[None, :] < rsi[:, None]
    np.testing.assert_allclose(b["workflow_progress"]["compute_log"][before],
                               a["workflow_progress"]["compute_log"][before])
    np.testing.assert_allclose(b["workflow_progress"]["reference_pace"],
                               a["workflow_progress"]["reference_pace"])
    assert (b["workflow_progress"]["capability_log"][:, -1]
            > a["workflow_progress"]["capability_log"][:, -1]).all()


def test_compute_boost_waits_its_delay_then_ramps_in():
    """After RSI the boost is off for `rsi_compute_delay` months, then ramps
    linearly to full over `rsi_compute_ramp`: the extra log compute is the
    rate times (ramp / 2 + the time at full strength)."""
    onset = np.array([0.5])
    base = dict(tk.POINT_ESTIMATE, years=3.5, rsi_compute_boost=1.0)
    a = tk.simulate(base, n=1, onset_years=onset)
    b = tk.simulate(dict(base, rsi_compute_boost=2.0, rsi_compute_delay=12.0,
                         rsi_compute_ramp=6.0), n=1, onset_years=onset)
    rsi, rate = a["events"]["RSI"][0], a["compute"]["rate"][0]
    gap = b["workflow_progress"]["compute_log"] - a["workflow_progress"]["compute_log"]
    years = a["workflow_progress"]["years"]
    assert np.all(gap[0, years <= rsi + 1.0] == 0)
    assert gap[0, -1] == pytest.approx(rate * (0.25 + (3.5 - rsi - 1.5)), rel=1e-3)


def test_coding_weights_measure_capability_by_its_parts():
    """Unit weights reproduce the plain log-effective-compute test; weighting
    the algorithmic part up reaches the same level sooner."""
    onset = np.full(20, np.inf)
    base = dict(years=6)
    plain = tk.simulate(base, n=20, onset_years=onset, coding_capability=3.0)
    unit = tk.simulate(base, n=20, onset_years=onset, coding_capability=3.0,
                       coding_weights=(1.0, 1.0))
    np.testing.assert_allclose(plain["coding_at"], unit["coding_at"])
    heavy = tk.simulate(base, n=20, onset_years=onset, coding_capability=3.0,
                        coding_weights=(0.5, 2.0))
    assert (heavy["coding_at"] <= plain["coding_at"]).all()
    assert (heavy["coding_at"] < plain["coding_at"]).any()


def test_workflow_halving_time_has_a_measurable_interpretation():
    """One halving time of capability progress halves human work above the floor."""
    starts = np.array([[.4, .3, .5]])
    human = tk.workflow_state(np.array([1.0]), np.array([1.0]), starts,
                              np.full((1, 3), 12.0), .02)
    np.testing.assert_allclose(human, .02 + (starts - .02) / 2)


def test_fast_feedback_precedes_successors_and_is_not_double_counted():
    p = dict(**tk.POINT_ESTIMATE, transfer=0, compute_growth=1, baseline_inputs=0, fast_share=25,
             fast_months=1, training_months=3, validation_months=1, years=.25)
    r = tk.simulate(p, n=1, onset_years=np.array([10.]))
    earlier = tk.simulate(dict(p, years=2/12), n=1, onset_years=np.array([10.]))
    assert r["deployed_software_log"][0] == 0
    assert r["fast_software_log"][0] == pytest.approx(.25 * earlier["software_log"][0])
    for share in [0, 25, 100]:
        r = tk.simulate(dict(p, fast_share=share, years=2), n=5)
        assert np.all(r["fast_software_log"] + r["deployed_software_log"] <= r["software_log"] + 1e-10)
        # At identical resources, splitting delivery paths cannot create an
        # apparent AI advantage when feedback is off.
        h = r["workflow_progress"]
        np.testing.assert_allclose(h["validated_log"], h["reference_log"], atol=1e-10)


def test_acceleration_can_arrive_without_autonomous_research():
    """Research can reach 2x its pace in draws that never reach RSI."""
    r = tk.simulate(dict(years=5, **tk.POINT_ESTIMATE, human_floor=6), n=1,
                    onset_years=np.array([20.]))
    assert np.isfinite(r["acceleration_events"]["2× research progress"][0])
    assert np.isinf(r["events"]["RSI"][0])


def test_rates_need_a_full_window_and_cumulative_progress_never_falls():
    r = tk.simulate(dict(years=2, **tk.POINT_ESTIMATE), n=3)
    h = r["workflow_progress"]
    early = h["years"] < .25 - 1e-12
    assert np.isnan(h["rate"][:, early]).all()
    assert np.isnan(h["rate_quantiles"][early]).all()
    assert np.isfinite(h["rate"][:, ~early]).all()
    assert np.all(np.diff(h["cumulative_quantiles"], axis=0) >= -1e-12)
    np.testing.assert_array_equal(h["cumulative_quantiles"][0], 0)


def test_ongoing_research_pace_does_not_wait_for_successor_deliveries():
    r = tk.simulate(dict(years=.5, **tk.POINT_ESTIMATE, baseline_inputs=0, fast_share=0,
                         training_months=12, validation_months=3, transfer=0,
                         compute_growth=1), n=2)
    h = r["workflow_progress"]
    np.testing.assert_array_equal(h["pace"][:, 0], 1)
    np.testing.assert_array_equal(h["reference_pace"][:, 0], 1)
    assert np.all(h["pace"] > 0)
    assert np.all(h["validated_baseline_months"] == 0)
    np.testing.assert_allclose(h["pace"], h["reference_pace"], rtol=1e-10)


def test_inherited_activity_informs_effort_without_a_second_compute_multiplier():
    p = dict(years=1, **tk.POINT_ESTIMATE, rsi_trend_weight=100)
    slow = tk.simulate(p, n=1, experiment_slopes=np.array([0.0]))
    fast = tk.simulate(p, n=1, experiment_slopes=np.array([np.log(16)]))
    assert fast["software_log"][0] > slow["software_log"][0]
    disabled = dict(p, rsi_trend_weight=0)
    a = tk.simulate(disabled, n=1)
    b = tk.simulate(disabled, n=1, experiment_slopes=np.array([np.log(16)]))
    np.testing.assert_array_equal(a["software_log"], b["software_log"])
    # Disabling additional AI feedback also disables proxy-driven feedback.
    a = tk.simulate(dict(p, transfer=0), n=1)
    b = tk.simulate(dict(p, transfer=0), n=1, experiment_slopes=np.array([np.log(16)]))
    np.testing.assert_array_equal(a["software_log"], b["software_log"])
    with pytest.raises(ValueError):
        tk.simulate(n=2, experiment_slopes=np.array([np.nan, 1.0]))


def test_calibration_removes_compute_and_uses_validated_evidence_when_supplied():
    pure_compute = tk.calibrate_feedback(2, 4, 2)
    assert pure_compute["resource_adjusted_growth"] == pytest.approx(1)
    assert pure_compute["quality_slope"] == 0
    a = tk.calibrate_feedback(2.2, 2, 2)
    worse_yield = tk.calibrate_feedback(2.2, 2, 2, yield_change=.5)
    assert worse_yield["quality_slope"] < a["quality_slope"]
    measured = tk.calibrate_feedback(100, 2, 2, validated_growth=2.2)
    assert measured["quality_slope"] == pytest.approx(a["quality_slope"])
    assert measured["basis"] == "validated progress"


def test_matched_stage_fit_uses_hours_not_intervention_incidence():
    fit = tk.calibrate_workflow(10, 5, 2.55, 2, np.log(2), 1)
    assert fit["human_now"] == pytest.approx(25.5)
    assert fit["half_months"] == pytest.approx(12)
    with pytest.raises(ValueError):
        tk.calibrate_workflow(0, 0, 0, 2, 1, 1)
    with pytest.raises(ValueError):
        tk.calibrate_workflow(10, 2, 5, 2, 1, 1)


def test_resource_growth_alone_creates_no_compute_matched_advantage():
    """Faster progress from compute alone is matched by the same-compute
    reference. The post-RSI boost is AI's doing, which the reference never
    gets, so it is off here."""
    r = tk.simulate(dict(transfer=0, difficulty=0.0, compute_growth=5.0,
                         **tk.POINT_ESTIMATE, years=5, baseline_inputs=0,
                         rsi_compute_boost=1.0), n=1)
    assert r["progress_multiple"][0] > 1
    assert r["compute_matched_advantage"][0] == pytest.approx(1, rel=1e-4)


def test_rsi_outcomes_meet_the_human_hours_bar():
    """Every draw reaching RSI has research direction at or under the bar there."""
    r = tk.simulate(n=100)
    observed = r["outcomes"]["RSI"]
    reached = np.isfinite(r["events"]["RSI"])
    assert reached.any()
    assert np.all(observed["human_direction"][reached] <= tk.DEFAULTS["full_human"] / 100)
    assert np.isnan(observed["human_direction"][~reached]).all()


def test_capability_paths_span_the_grid_and_rsi_waits_for_coding():
    """Each draw's capability path starts at zero, stays finite and never falls
    across the full grid, and RSI never precedes its coding date."""
    r = tk.simulate(dict(years=4), n=200)
    progress = r["workflow_progress"]
    capability = progress["capability_log"]
    assert capability.shape == (200, len(progress["years"]))
    assert progress["years"][-1] == 4
    np.testing.assert_array_equal(capability[:, 0], 0)
    assert np.isfinite(capability).all()
    assert np.all(np.diff(capability, axis=1) >= -1e-12)
    assert np.all(capability[:, -1] > 0)
    held = r["software_log"] < 1e6       # a runaway reaches RSI regardless
    assert held.mean() > 0.5
    assert np.all(r["events"]["RSI"][held] >= np.maximum(r["onset"], 0)[held])


def test_cdf_and_quantile_keep_censored_draws():
    """Non-arrivals (+inf) stay in the denominator and can be the quantile."""
    assert tk.cdf([1, 2, np.inf, np.inf], [0, 1, 2, 20]).tolist() == [0, .25, .5, .5]
    assert tk.quantile([1, 2, np.inf, np.inf], .9) == np.inf
    assert tk.quantile([1, 2, 3], .5, horizon=1) == np.inf


def test_weekly_steps_converge_to_half_week_steps():
    """Milestones land on run, deployment and observation events, so the step
    cannot move them; the integrated software stock converges well inside the
    error a monthly step leaves (~2e-4)."""
    p = dict(years=8)
    weekly = tk.simulate(p, n=200)
    half_week = tk.simulate(p, n=200, step=1/104)
    # A runaway's date is the step its stock crosses the numerical ceiling,
    # so only draws that stay in range must agree exactly.
    held = (weekly["software_log"] < 1e6) & (half_week["software_log"] < 1e6)
    assert held.mean() > 0.5
    for name in tk.MILESTONES:
        np.testing.assert_array_equal(weekly["events"][name][held],
                                      half_week["events"][name][held])
    # Near a blow-up the stock is stiff; compare the ones well short of it.
    calm = held & (weekly["software_log"] < 1e3)
    np.testing.assert_allclose(weekly["software_log"][calm], half_week["software_log"][calm],
                               rtol=3e-5)


def test_longer_validation_delays_milestones():
    a = tk.simulate(dict(**tk.POINT_ESTIMATE, validation_months=0.5), n=1)
    b = tk.simulate(dict(**tk.POINT_ESTIMATE, validation_months=3.0), n=1)
    for name in tk.MILESTONES:
        assert b["events"][name][0] > a["events"][name][0]


@pytest.mark.parametrize("params", [
    {"training_share": 60, "experiment_share": 40},
    {"onset_low": 50, "onset_high": 10}, {"transfer": 101},
    {"software_months": 0}, {"compute_growth": float("nan")},
    {"seed": -1}, {"difficulty": -1}, {"parallelization": 0}, {"taste_slope": -1},
    {"human_direction": -1}, {"human_floor": 50}, {"half_direction": 0},
    {"full_human": 101}, {"fast_share": 101}, {"fast_months": 0},
    {"coding_today": 95}, {"rsi_trend_weight": 101},
])
def test_invalid_assumptions_rejected(params):
    with pytest.raises(ValueError):
        tk.simulate(params, n=1)


def test_pace_factors_multiply_back_to_the_pace():
    """Resources x judgment / difficulty is the recorded pace, month by month."""
    r = tk.simulate(n=200, onset_years=np.full(200, 1.0))
    p = r["workflow_progress"]
    f = p["pace_factors"]
    log_rebuilt = f["resources"] + f["judgment"] - f["difficulty"]
    # The pace itself is capped at exp(700); a runaway's factors overflow.
    ok = np.isfinite(log_rebuilt) & (log_rebuilt < 699)
    assert ok.mean() > 0.6
    np.testing.assert_allclose(np.exp(log_rebuilt[ok]), p["pace"][ok], rtol=1e-6)
    # Labor and compute split exactly into compute, human labor and AI help.
    np.testing.assert_allclose(f["compute"] + f["labor"] + f["ai"], f["resources"], atol=1e-9)
    # With no AI transfer, AI help never moves.
    r0 = tk.simulate(dict(transfer=0, years=4), n=50, onset_years=np.full(50, 1.0))
    ai = r0["workflow_progress"]["pace_factors"]["ai"]
    np.testing.assert_allclose(ai, ai[:, :1].repeat(ai.shape[1], axis=1), atol=1e-9)


def test_ai_judgment_only_counts_where_ai_directs_research():
    """With people directing all research, AI judgment changes nothing; with a
    zero transfer, neither does any judgment elasticity."""
    base = dict(**tk.POINT_ESTIMATE, years=3, transfer=100)
    human_led = dict(base, human_direction=100.0, human_floor=100.0,
                     human_experiments=100.0, human_verification=100.0)
    a = tk.simulate(dict(human_led, taste_slope=0.2), n=1)
    b = tk.simulate(dict(human_led, taste_slope=2.0), n=1)
    np.testing.assert_allclose(a["software_log"], b["software_log"])
    e = tk.simulate(dict(base, taste_slope=2.0), n=1)
    f = tk.simulate(dict(base, taste_slope=0.2), n=1)
    assert e["software_log"][0] > f["software_log"][0]


def test_judgment_drift_bends_the_judgment_rate_per_10x():
    """Zero drift is a constant elasticity; drift d multiplies the elasticity by
    (1 + d) per 10x of capability, so the gain's slope at 10x is slope * (1 + d)."""
    e = np.linspace(0, 6, 7)
    np.testing.assert_allclose(tk.judgment_gain_log(e, 0.3, 0.0), 0.3 * e)
    up, down = tk.judgment_gain_log(e, 0.3, 0.5), tk.judgment_gain_log(e, 0.3, -0.5)
    assert np.all(up[1:] > 0.3 * e[1:]) and np.all(down[1:] < 0.3 * e[1:])
    h = 1e-6
    slope_at_10x = (tk.judgment_gain_log(np.log(10) + h, 0.3, 0.5)
                    - tk.judgment_gain_log(np.log(10), 0.3, 0.5)) / h
    assert slope_at_10x == pytest.approx(0.45, rel=1e-4)



def test_point_estimate_draws_the_settings_themselves():
    """With every spread at zero each draw uses the settings; by default r,
    judgment, its drift and rho each vary per draw."""
    p = dict(years=0.1, taste_slope=0.3, judgment_drift=10.0, substitution=-0.5)
    point = tk.simulate(dict(p, **tk.POINT_ESTIMATE), n=50)["draws"]
    np.testing.assert_allclose(point["taste_slope"], 0.3)
    np.testing.assert_allclose(point["difficulty"], tk.DEFAULTS["difficulty"])
    np.testing.assert_allclose(point["judgment_drift"], 0.1)
    np.testing.assert_allclose(point["substitution"], -0.5)
    spread = tk.simulate(p, n=50)["draws"]
    assert all(np.ptp(v) > 0 for v in spread.values())
    # rho stays inside the range `validate` accepts.
    assert (spread["substitution"] >= -5).all() and (spread["substitution"] <= 0.9).all()


def test_the_spreads_leave_a_runaway_possible_not_certain():
    """At the app's measured judgment elasticity (~0.28), some draws run away
    and most do not; at point estimates none does."""
    p = dict(taste_slope=0.28, years=5)
    runaway = lambda r: np.mean(r["software_log"] >= 1e6 - 1)
    assert 0.02 < runaway(tk.simulate(p, n=400)) < 0.6
    assert runaway(tk.simulate(dict(p, **tk.POINT_ESTIMATE), n=400)) == 0


def test_compute_slows_to_its_late_rate_from_the_slowdown_date():
    """`grown` integrates the two-rate path exactly, and keeping 100% of the
    growth reproduces the single rate."""
    r, late, at = np.array([1.1]), np.array([0.55]), 2.0
    s = np.linspace(0, 5, 200001)
    level = np.exp(tk.log_level(s, r[0], late[0], at))
    assert tk.grown(5.0, r, late, at)[0] == pytest.approx(np.trapezoid(level, s), rel=1e-6)
    assert tk.grown(5.0, r, r, at)[0] == pytest.approx(np.expm1(5.5) / 1.1)
    base = dict(**tk.POINT_ESTIMATE, years=6, compute_slow_years=2.0)
    kept = tk.simulate(dict(base, compute_slowdown=100), n=1)
    never = tk.simulate(dict(base, compute_slow_years=50.0), n=1)
    slowed = tk.simulate(dict(base, compute_slowdown=50), n=1)
    np.testing.assert_allclose(kept["budget"], never["budget"])
    np.testing.assert_allclose(kept["software_log"], never["software_log"])
    assert slowed["budget"][0] < kept["budget"][0]
    assert slowed["software_log"][0] < kept["software_log"][0]


def test_a_model_at_the_coding_capability_has_automated_coding():
    """RSI waits for coding, but coding counts as automated once a deployed
    model reaches `coding_capability`, even with no inherited date."""
    p = dict(**tk.POINT_ESTIMATE, years=8)
    never = tk.simulate(p, n=1, onset_years=np.array([np.inf]))
    assert np.isinf(never["events"]["RSI"][0]) and np.isinf(never["coding_at"][0])
    early = tk.simulate(p, n=1, onset_years=np.array([np.inf]), coding_capability=1.0)
    assert np.isfinite(early["coding_at"][0])
    assert early["events"]["RSI"][0] >= early["coding_at"][0]
    assert np.isinf(early["onset"][0])           # the inherited date is kept as given
