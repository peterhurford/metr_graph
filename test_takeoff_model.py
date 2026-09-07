"""Model invariants and numerical checks for conditional takeoff scenarios."""

import numpy as np
import pytest

import takeoff_model as tk


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
    expected = np.log1p(beta * rate * p["years"]) / beta
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
             validation_months=0.0, transfer=0)
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


def test_later_onset_shifts_dates_without_changing_takeoff_duration():
    a = tk.simulate(n=20)
    b = tk.simulate(dict(onset_low=18, onset_high=36), n=20)
    np.testing.assert_allclose(b["onset"] - a["onset"], 1)
    for name in tk.MILESTONES:
        np.testing.assert_array_equal(a["events"][name], b["events"][name])


def test_external_onset_preserves_tail_and_past_dates_without_redrawing_dynamics():
    onset = np.array([-1.0, .25, 2.0, 20.0, np.inf])
    a = tk.simulate(n=5)
    b = tk.simulate(n=5, onset_years=onset)
    np.testing.assert_array_equal(b["onset"], onset)
    for name in tk.MILESTONES:
        np.testing.assert_array_equal(a["events"][name], b["events"][name])
    assert np.isinf((b["onset"] + b["events"][tk.MILESTONES[-1]])[-1])


def test_full_rd_triggers_at_human_judgment_parity():
    p = dict(uncertainty=0, taste_at_onset=1.0)
    r = tk.simulate(p, n=1)
    assert r["events"]["Full R&D automation"][0] == 0
    assert r["events"]["Sustained research feedback"][0] > 0
    # A different feedback persistence requirement does not gate R&D parity.
    b = tk.simulate(dict(p, feedback_cycles=5), n=1)
    assert b["events"]["Full R&D automation"][0] == 0


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
    {"seed": -1}, {"general_gap": -1}, {"taste_at_onset": 0},
])
def test_invalid_assumptions_rejected(params):
    with pytest.raises(ValueError):
        tk.simulate(params, n=1)
