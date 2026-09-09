"""Model invariants and numerical checks for conditional takeoff scenarios."""

import numpy as np
import pytest

import takeoff_model as tk


def test_workflow_trajectory_observes_deployments_and_keeps_all_draws():
    result = tk.simulate(dict(years=2, uncertainty=0), n=3,
                         onset_years=np.array([0, 20, np.inf]))
    progress = result["workflow_progress"]
    grid = progress["years"]
    human = progress["human_quantiles"]
    success = progress["success_quantiles"]
    assert grid[0] == 0 and grid[-1] == 2
    assert human.shape == (len(grid), 3, 3)
    assert success.shape == (len(grid), 3)
    starts = np.array([tk.DEFAULTS["human_" + s] for s in tk.WORKFLOW_KEYS]) / 100
    np.testing.assert_allclose(human[0], np.tile(starts, (3, 1)))
    np.testing.assert_allclose(success[0], tk.DEFAULTS["project_success"] / 100)
    # No improvement before the first fast validation cycle finishes.
    before_deployment = grid < 1 / 12 - 1e-12
    np.testing.assert_allclose(human[before_deployment],
                              np.broadcast_to(human[0], human[before_deployment].shape))
    assert np.all(np.diff(human, axis=0) <= 1e-12)
    assert np.all(np.diff(success, axis=0) >= -1e-12)
    assert np.all(human >= tk.DEFAULTS["human_floor"] / 100)
    assert np.all(success <= 1)
    assert np.all(human[-1] < human[0])
    # Even a never-arriving coding milestone retains a research trajectory.
    never = tk.simulate(dict(years=2, uncertainty=0), n=1,
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
    p = dict(transfer=0, compute_growth=1.0, uncertainty=0, years=2)
    a = tk.simulate(p, n=1)
    rate = np.log(2) * 12 / tk.DEFAULTS["software_months"]
    beta = tk.DEFAULTS["difficulty"]
    expected = rate * tk.DEFAULTS["pipeline_months"] / 12 + np.log1p(beta * rate * p["years"]) / beta
    assert a["software_log"][0] == pytest.approx(expected)
    assert np.isinf(a["events"]["Sustained research feedback"]).all()
    b = tk.simulate(dict(p, coding_slope=2.0, taste_slope=2.0), n=1)
    np.testing.assert_allclose(a["software_log"], b["software_log"])


def test_compute_budget_conservation():
    for growth in [1.0, 2.0]:
        p = dict(compute_growth=growth, uncertainty=0, years=2,
                 training_share=30, experiment_share=50)
        r = tk.simulate(p, n=5)
        np.testing.assert_allclose(r["allocated"].sum(axis=0), r["budget"])
        np.testing.assert_allclose(r["allocated"] / r["budget"],
                                   np.tile([[0.3], [0.5], [0.2]], (1, 5)))
        expected = 2 if growth == 1 else np.expm1(np.log(growth) * 2) / np.log(growth)
        np.testing.assert_allclose(r["budget"], expected)
        assert np.all(r["training_used"] <= r["allocated"][0] + 1e-10)


def test_validation_reserves_but_does_not_spend_training_compute():
    r = tk.simulate(dict(compute_growth=1.0, uncertainty=0, transfer=0,
                         years=2, training_months=3.0, validation_months=3.0),
                    n=1, step=1/120)
    # Four 3-month runs in two years; half the reserved training allocation
    # sits idle during validation rather than being spent twice.
    assert r["training_used"][0] == pytest.approx(0.4)
    assert r["allocated"][0, 0] == pytest.approx(0.8)


def test_algorithms_wait_for_the_next_run_and_validation():
    p = dict(compute_growth=1.0, uncertainty=0, training_months=3.0,
             validation_months=0.0, transfer=0, fast_share=0, pipeline_months=0)
    first = tk.simulate(dict(p, years=0.25), n=1, step=1/120)
    assert first["software_log"][0] > 0
    assert first["deployed_log"][0] == pytest.approx(0, abs=1e-10)
    second = tk.simulate(dict(p, years=0.5), n=1, step=1/120)
    assert second["deployed_log"][0] == pytest.approx(first["software_log"][0])
    delayed = tk.simulate(dict(p, years=0.5, validation_months=3.0), n=1, step=1/120)
    assert delayed["deployed_log"][0] == pytest.approx(0, abs=1e-10)


def test_larger_general_gap_delays_asi_and_does_not_change_rd():
    a = tk.simulate(n=100)
    b = tk.simulate(dict(general_gap=8.0), n=100)
    for name in tk.MILESTONES[:-1]:
        np.testing.assert_array_equal(a["events"][name], b["events"][name])
    assert np.all(b["events"][tk.MILESTONES[-1]] >= a["events"][tk.MILESTONES[-1]])


def test_later_coding_does_not_postpone_the_start_of_research():
    p = dict(years=.5, uncertainty=0)
    a = tk.simulate(p, n=1, onset_years=np.array([1.]))
    b = tk.simulate(p, n=1, onset_years=np.array([10.]))
    assert a["software_log"][0] >= b["software_log"][0] > 0
    assert b["fast_software_log"][0] > 0
    assert np.isinf(b["events"]["Full R&D automation"][0])


def test_external_onset_preserves_past_dates_and_censored_tail():
    onset = np.array([-1.0, .25, 2.0, 20.0, np.inf])
    b = tk.simulate(n=5, onset_years=onset)
    np.testing.assert_array_equal(b["onset"], onset)
    for name in tk.MILESTONES[1:]:
        assert np.all(b["events"][name] >= np.maximum(onset, 0))
    assert np.isnan(b["outcomes"]["Full coding automation"]["human_mean"][-1])
    assert np.all(np.isfinite(b["outcomes"]["Today"]["human_mean"]))
    durations = tk.after_coding(b["events"][tk.MILESTONES[-1]], onset)
    assert not np.isnan(durations).any()
    assert np.isinf(durations[-1])


def test_full_rd_requires_low_human_work_and_successful_projects():
    p = dict(uncertainty=0, onset_low=0, onset_high=0, human_floor=0.0, human_direction=0.0,
             human_experiments=0.0, human_verification=0.0, project_success=95.0)
    r = tk.simulate(p, n=1)
    assert r["events"]["Full R&D automation"][0] == 0
    assert r["events"]["Sustained research feedback"][0] > 0
    # The hidden feedback milestone does not gate operational automation.
    b = tk.simulate(dict(p, feedback_cycles=5), n=1)
    assert b["events"]["Full R&D automation"][0] == 0


def test_one_human_bottleneck_cannot_be_averaged_away():
    p = dict(years=0.1, uncertainty=0, human_floor=0.0, human_direction=0.0,
             human_experiments=0.0, human_verification=9.0, project_success=100.0)
    r = tk.simulate(p, n=1)
    initial = r["outcomes"]["Today"]
    assert initial["human_mean"][0] < tk.DEFAULTS["full_human"] / 100
    assert initial["human_worst"][0] > tk.DEFAULTS["full_human"] / 100
    assert np.isinf(r["events"]["Full R&D automation"][0])


def test_project_completion_is_an_independent_gate():
    p = dict(years=0.1, uncertainty=0, human_floor=0.0, human_direction=0.0,
             human_experiments=0.0, human_verification=0.0, project_success=50.0)
    r = tk.simulate(p, n=1)
    assert np.isinf(r["events"]["Full R&D automation"][0])


def test_persistent_human_floor_prevents_all_later_milestones():
    r = tk.simulate(dict(human_floor=6.0), n=30)
    for name in tk.MILESTONES[1:]:
        assert np.isinf(r["events"][name]).all()


def test_workflow_halving_time_has_a_measurable_interpretation():
    starts = np.array([[.4, .3, .5]])
    human, success = tk.workflow_state(np.array([1.0]), np.array([1.0]), starts,
                                      np.full((1, 3), 12.0), .02,
                                      np.array([.6]), np.array([12.0]))
    np.testing.assert_allclose(human, .02 + (starts - .02) / 2)
    np.testing.assert_allclose(success, .8)


def test_validated_progress_uses_efficiency_and_full_cycle_duration():
    baseline = np.log(2)  # one software doubling per year
    assert tk.progress_multiple(0, np.log(2), 1, baseline) == pytest.approx(1)
    assert tk.progress_multiple(0, np.log(2), 1/12, baseline) == pytest.approx(12)
    assert tk.progress_multiple(0, 0, 1/12, baseline) == 0


def test_fast_feedback_precedes_successors_and_is_not_double_counted():
    p = dict(uncertainty=0, transfer=0, compute_growth=1, fast_share=25,
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
    r = tk.simulate(dict(years=5, uncertainty=0, human_floor=6), n=1,
                    onset_years=np.array([20.]))
    assert np.isfinite(r["acceleration_events"]["2× research progress"][0])
    assert np.isinf(r["events"]["Full R&D automation"][0])


def test_rates_need_a_full_window_and_cumulative_progress_never_falls():
    r = tk.simulate(dict(years=2, uncertainty=0), n=3)
    h = r["workflow_progress"]
    early = h["years"] < .25 - 1e-12
    assert np.isnan(h["rate"][:, early]).all()
    assert np.isnan(h["rate_quantiles"][early]).all()
    assert np.isfinite(h["rate"][:, ~early]).all()
    assert np.all(np.diff(h["cumulative_quantiles"], axis=0) >= -1e-12)
    np.testing.assert_array_equal(h["cumulative_quantiles"][0], 0)


def test_ongoing_research_pace_does_not_wait_for_successor_deliveries():
    r = tk.simulate(dict(years=.5, uncertainty=0, fast_share=0,
                         training_months=12, validation_months=3, transfer=0,
                         compute_growth=1), n=2)
    h = r["workflow_progress"]
    np.testing.assert_array_equal(h["pace"][:, 0], 1)
    np.testing.assert_array_equal(h["reference_pace"][:, 0], 1)
    assert np.all(h["pace"] > 0)
    assert np.all(h["validated_baseline_months"] == 0)
    np.testing.assert_allclose(h["pace"], h["reference_pace"], rtol=1e-10)


def test_inherited_activity_informs_effort_without_a_second_compute_multiplier():
    p = dict(years=1, uncertainty=0, rsi_trend_weight=100)
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


def test_resource_growth_alone_cannot_trigger_superhuman_research():
    r = tk.simulate(dict(transfer=0, difficulty=0.0, compute_growth=5.0,
                         uncertainty=0, years=5), n=1)
    assert r["progress_multiple"][0] > tk.DEFAULTS["progress_target"]
    assert r["compute_matched_advantage"][0] == pytest.approx(1, rel=1e-4)
    assert np.isinf(r["events"]["Superhuman AI research"][0])


def test_superhuman_outcomes_meet_all_operational_thresholds():
    r = tk.simulate(n=100)
    observed = r["outcomes"]["Superhuman AI research"]
    reached = np.isfinite(r["events"]["Superhuman AI research"])
    assert reached.any()
    assert np.all(observed["human_worst"][reached] <= tk.DEFAULTS["full_human"] / 100)
    assert np.all(observed["project_success"][reached] >= tk.DEFAULTS["full_success"] / 100)
    assert np.all(observed["progress_multiple"][reached] >= tk.DEFAULTS["progress_target"])
    assert np.all(observed["compute_matched_advantage"][reached] >= tk.DEFAULTS["advantage_target"])


def test_acceleration_must_persist_across_successor_cycles():
    p = dict(uncertainty=0, progress_cycles=1)
    once = tk.simulate(p, n=1)
    three = tk.simulate(dict(p, progress_cycles=3), n=1)
    difference = (three["events"]["Superhuman AI research"][0]
                  - once["events"]["Superhuman AI research"][0])
    cycle = (tk.DEFAULTS["training_months"] + tk.DEFAULTS["validation_months"]) / 12
    assert difference >= 2 * cycle - 1e-10


def test_milestone_order_and_retained_non_arrivals():
    r = tk.simulate(tk.PRESETS["Bottlenecked"], n=200)
    full, research, asi = (r["events"][name] for name in tk.MILESTONES[1:])
    assert np.all(full <= research)
    assert np.all(research <= asi)
    assert np.isinf(asi).any()
    assert tk.quantile(asi, 0.5) == np.inf
    assert tk.cdf(asi, [10])[0] == np.mean(asi <= 10)
    assert tk.cdf([1, 2, np.inf, np.inf], [0, 1, 2, 20]).tolist() == [0, .25, .5, .5]
    assert tk.quantile([1, 2, np.inf, np.inf], .9) == np.inf
    assert tk.quantile([1, 2, 3], .5, horizon=1) == np.inf


def test_weekly_steps_converge_to_half_week_steps():
    p = dict(uncertainty=0, years=8)
    weekly = tk.simulate(p, n=1)
    half_week = tk.simulate(p, n=1, step=1/104)
    for name in tk.MILESTONES:
        assert abs(weekly["events"][name][0] - half_week["events"][name][0]) < .12


def test_longer_validation_delays_milestones():
    a = tk.simulate(dict(uncertainty=0, validation_months=0.5), n=1)
    b = tk.simulate(dict(uncertainty=0, validation_months=3.0), n=1)
    for name in tk.MILESTONES[1:]:
        assert b["events"][name][0] >= a["events"][name][0]


@pytest.mark.parametrize("params", [
    {"training_share": 60, "experiment_share": 40},
    {"onset_low": 50, "onset_high": 10}, {"transfer": 101},
    {"software_months": 0}, {"compute_growth": float("nan")},
    {"seed": -1}, {"general_gap": -1}, {"taste_slope": -1},
    {"human_direction": -1}, {"human_floor": 50}, {"half_direction": 0},
    {"full_success": 110}, {"project_success": 0}, {"advantage_target": 1},
    {"progress_cycles": 1.5}, {"fast_share": 101}, {"fast_months": 0},
    {"coding_today": 95},
])
def test_invalid_assumptions_rejected(params):
    with pytest.raises(ValueError):
        tk.simulate(params, n=1)
