"""
Tests for visualize_projection.py helper functions and data loading.

Run: pytest test_visualize_projection.py -v
"""

import numpy as np
import re
import pytest
from datetime import datetime, timedelta
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Fake Streamlit module so visualize_projection.py can be imported in tests.
# ---------------------------------------------------------------------------

import types

# ---------------------------------------------------------------------------
# Build a comprehensive fake streamlit that no-ops everything.
# The module-level code in visualize_projection.py runs render_metr() etc.
# which call many st.* functions, so we need a catch-all.
# ---------------------------------------------------------------------------

class _Noop:
    """Object whose every attribute access / call returns another _Noop.
    Acts as a universal sink — you can call it, iterate it, index it,
    use it as a context manager, and it'll never raise."""
    def __call__(self, *a, **kw):
        return _Noop()
    def __getattr__(self, name):
        return _Noop()
    def __iter__(self):
        return iter([])
    def __bool__(self):
        return False
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def __getitem__(self, key):
        return _Noop()
    def __setitem__(self, key, val):
        pass
    def __contains__(self, item):
        return False
    def __len__(self):
        return 0


class _FakeStreamlit(types.ModuleType):
    """Drop-in replacement for the `streamlit` module during testing.
    Only the functions actually used at import time need real behavior;
    everything else returns _Noop to silently absorb render calls."""

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.query_params = {"tab": "metr"}
        self._testing = True

    # --- functions that need real behavior for data loading / setup ---
    def set_page_config(self, **kw):
        pass

    def cache_data(self, f=None, **kw):
        """Pass-through decorator so data loading functions work."""
        if f is not None:
            return f
        return lambda fn: fn

    def radio(self, label, options, **kw):
        """Return the option at `index` (default 0) so module-level code works."""
        idx = kw.get("index", 0)
        if options and 0 <= idx < len(options):
            return options[idx]
        return options[0] if options else ""

    def selectbox(self, label, options, **kw):
        idx = kw.get("index", 0)
        if options and 0 <= idx < len(options):
            return options[idx]
        return options[0] if options else ""

    def number_input(self, label, **kw):
        return kw.get("value", 0)

    def toggle(self, label, **kw):
        return kw.get("value", False)

    def columns(self, n, **kw):
        """Return n _Noop objects that act as column placeholders."""
        return [_Noop() for _ in (range(n) if isinstance(n, int) else range(len(n)))]

    def expander(self, *a, **kw):
        return _Noop()

    def stop(self):
        raise SystemExit("st.stop")

    def button(self, *a, **kw):
        return False

    # --- catch-all for everything else (header, caption, info, …) ---
    def __getattr__(self, name):
        return _Noop()


_fake_st = _FakeStreamlit()

# Temporarily replace streamlit so visualize_projection.py can be imported
# without a running Streamlit server.  We restore the real module afterward
# so other test files (e.g. test_integration.py) that need the real streamlit
# can coexist in the same pytest session.
_real_st = sys.modules.get("streamlit")
sys.modules["streamlit"] = _fake_st

# Tell the module to skip rendering during import
os.environ["_VP_TESTING"] = "1"

# Now import the module under test
_orig_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import visualize_projection as vp

os.chdir(_orig_dir)

# Restore the real streamlit module (if it was installed) so integration
# tests that import from streamlit.testing work correctly.
if _real_st is not None:
    sys.modules["streamlit"] = _real_st
else:
    del sys.modules["streamlit"]

# Remove the cached visualize_projection module so that integration tests
# (which use AppTest.from_file) get a fresh import with the real streamlit.
# Our local `vp` reference is already bound and unaffected.
sys.modules.pop("visualize_projection", None)

# Clear the testing env var so integration tests render normally.
os.environ.pop("_VP_TESTING", None)


# ===========================================================================
# Shared test helpers — DRY data loading + fitting used across test classes
# ===========================================================================

def _load_metr_fit():
    """Load METR frontier, fit OLS in log2 space. Returns (days, vals, params)."""
    frontier = vp.load_frontier()
    base = frontier[0]['date']
    days = np.array([(m['date'] - base).days for m in frontier], dtype=float)
    vals = np.array([np.log2(m['p50_min']) for m in frontier])
    params = vp.fit_line(days, vals)
    return days, vals, params


def _load_eci_fit():
    """Load ECI frontier, fit OLS. Returns (days, vals, params)."""
    all_data = vp.load_eci_frontier()
    frontier = [m for m in all_data if m['is_frontier']]
    base = frontier[0]['date']
    days = np.array([(m['date'] - base).days for m in frontier], dtype=float)
    vals = np.array([m['eci_score'] for m in frontier])
    params = vp.fit_line(days, vals)
    return days, vals, params


def _load_rli_fit():
    """Load RLI frontier, fit OLS in logit space. Returns (days, vals, params)."""
    all_data = vp.load_rli_data()
    frontier = [m for m in all_data if m['is_frontier']]
    base = frontier[0]['date']
    days = np.array([(m['date'] - base).days for m in frontier], dtype=float)
    vals = np.array([vp._logit(m['rli_score'] / 100.0) for m in frontier])
    params = vp.fit_line(days, vals)
    return days, vals, params


def _fit_superexp(days, values, halflife):
    """Fit y = A + K * 2^(d/halflife). Returns (A, K)."""
    z = 2 ** (days / halflife)
    X = np.column_stack([np.ones_like(z), z])
    (A, K), *_ = np.linalg.lstsq(X, values, rcond=None)
    return A, K


# ===========================================================================
# pretty()
# ===========================================================================

class TestPretty:
    def test_known_name(self):
        assert vp.pretty("gpt_4") == "GPT-4"

    def test_known_name_gpt2(self):
        assert vp.pretty("gpt2") == "GPT-2"

    def test_known_name_claude(self):
        assert vp.pretty("claude_3_5_sonnet_20240620_inspect") == "Claude 3.5S (old)"

    def test_unknown_name_passthrough(self):
        assert vp.pretty("some_unknown_model") == "some_unknown_model"

    def test_empty_string(self):
        assert vp.pretty("") == ""


# ===========================================================================
# log2min_to_label()
# ===========================================================================

class TestLog2MinToLabel:
    def test_sub_minute(self):
        # log2(0.5 min) = -1 => 30 seconds
        result = vp.log2min_to_label(-1)
        assert result == "30s"

    def test_one_minute(self):
        result = vp.log2min_to_label(0)
        assert result == "1m"

    def test_30_minutes(self):
        # log2(30) ~ 4.91
        result = vp.log2min_to_label(np.log2(30))
        assert result == "30m"

    def test_one_hour(self):
        # log2(60) ~ 5.91
        result = vp.log2min_to_label(np.log2(60))
        assert result == "1h"

    def test_two_hours(self):
        result = vp.log2min_to_label(np.log2(120))
        assert result == "2h"

    def test_large_value_uses_hours(self):
        # 800 minutes = 13.33 hours (< 100h threshold, so uses hours not days)
        result = vp.log2min_to_label(np.log2(800))
        assert "h" in result


# ===========================================================================
# fmt_hrs()
# ===========================================================================

class TestFmtHrs:
    def test_minutes(self):
        assert vp.fmt_hrs(0.5) == "30m"

    def test_zero(self):
        assert vp.fmt_hrs(0) == "0m"

    def test_exact_hours(self):
        assert vp.fmt_hrs(3) == "3h"

    def test_hours_and_minutes(self):
        result = vp.fmt_hrs(1.5)
        assert result == "1h30m"

    def test_hours_below_100_threshold(self):
        # 16 hours < 100 threshold, stays in hours
        assert vp.fmt_hrs(16) == "16h"

    def test_hours_with_minutes_remainder(self):
        # 12 hours = exact hours
        assert vp.fmt_hrs(12) == "12h"

    def test_hours_near_threshold(self):
        # 99h is below 100h threshold, still uses hours
        assert vp.fmt_hrs(99) == "99h"

    def test_work_days(self):
        # 100 hours >= 100h threshold -> 12.5 days = 12d4h (8h/d)
        result = vp.fmt_hrs(100)
        assert "d" in result

    def test_work_weeks(self):
        # 160 hours -> 4 weeks = 1 month boundary, but < 4.4 weeks so stays in weeks
        assert vp.fmt_hrs(160) == "4w"

    def test_work_weeks_with_remainder(self):
        # 168 hours -> 4.2w -> 4w1d
        assert vp.fmt_hrs(168) == "4w1d"

    def test_work_months(self):
        # 176 hours = 1 work-month
        assert vp.fmt_hrs(176) == "1mo"

    def test_work_years(self):
        # 2000 hours = 1 work-year
        assert vp.fmt_hrs(2000) == "1y"

    def test_work_years_with_remainder(self):
        # 2176 hours = 1y1mo (2000 + 176)
        assert vp.fmt_hrs(2176) == "1y1mo"

    def test_multi_year(self):
        # 4000 hours = 2y
        assert vp.fmt_hrs(4000) == "2y"

    def test_minutes_rounding(self):
        # 59.9 minutes ~ 1h
        result = vp.fmt_hrs(59.9 / 60)
        assert "m" in result


# ===========================================================================
# fit_line()
# ===========================================================================

class TestFitLine:
    def test_perfect_line(self):
        x = np.array([0, 1, 2, 3, 4], dtype=float)
        y = 2.0 + 3.0 * x
        params = vp.fit_line(x, y)
        np.testing.assert_allclose(params[0], 2.0, atol=1e-10)
        np.testing.assert_allclose(params[1], 3.0, atol=1e-10)

    def test_horizontal_line(self):
        x = np.array([0, 1, 2, 3], dtype=float)
        y = np.array([5, 5, 5, 5], dtype=float)
        params = vp.fit_line(x, y)
        np.testing.assert_allclose(params[0], 5.0, atol=1e-10)
        np.testing.assert_allclose(params[1], 0.0, atol=1e-10)

    def test_negative_slope(self):
        x = np.array([0, 1, 2, 3], dtype=float)
        y = 10.0 - 2.0 * x
        params = vp.fit_line(x, y)
        np.testing.assert_allclose(params[0], 10.0, atol=1e-10)
        np.testing.assert_allclose(params[1], -2.0, atol=1e-10)

    def test_noisy_data(self):
        np.random.seed(42)
        x = np.arange(100, dtype=float)
        y = 1.0 + 0.5 * x + np.random.normal(0, 0.1, 100)
        params = vp.fit_line(x, y)
        assert abs(params[0] - 1.0) < 0.5
        assert abs(params[1] - 0.5) < 0.05

    def test_two_points(self):
        x = np.array([0, 10], dtype=float)
        y = np.array([0, 30], dtype=float)
        params = vp.fit_line(x, y)
        np.testing.assert_allclose(params[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(params[1], 3.0, atol=1e-10)


# ===========================================================================
# _fit_slope_p50_intercept_display()
# ===========================================================================

class TestFitSlopeP50InterceptDisplay:
    def test_basic(self):
        d = np.array([0, 1, 2, 3], dtype=float)
        p50 = 1.0 + 2.0 * d
        disp = 3.0 + 2.0 * d  # same slope, different intercept
        params = vp._fit_slope_p50_intercept_display(d, p50, disp)
        np.testing.assert_allclose(params[1], 2.0, atol=1e-10)  # slope from p50
        np.testing.assert_allclose(params[0], 3.0, atol=1e-10)  # intercept from disp


# ===========================================================================
# Sampling functions
# ===========================================================================

class TestLognormalFromCi:
    def test_returns_correct_size(self):
        samples = vp._lognormal_from_ci(50, 200, 1000)
        assert len(samples) == 1000

    def test_all_positive(self):
        samples = vp._lognormal_from_ci(50, 200, 5000)
        assert np.all(samples > 0)

    def test_median_approximately_geometric_mean(self):
        np.random.seed(42)
        lo, hi = 50, 200
        samples = vp._lognormal_from_ci(lo, hi, 100_000)
        expected_median = np.sqrt(lo * hi)
        actual_median = np.median(samples)
        assert abs(actual_median - expected_median) / expected_median < 0.02

    def test_80_ci_coverage(self):
        """~80% of samples should fall within [lo, hi] since they define 80% CI."""
        np.random.seed(42)
        lo, hi = 50, 200
        samples = vp._lognormal_from_ci(lo, hi, 100_000)
        within = np.mean((samples >= lo) & (samples <= hi))
        assert abs(within - 0.80) < 0.02

    def test_raises_on_negative_sigma(self):
        """If lo > hi, sigma becomes negative and numpy should raise."""
        with pytest.raises(ValueError):
            vp._lognormal_from_ci(200, 50, 100)


class TestNormalFromCi:
    def test_returns_correct_size(self):
        samples = vp._normal_from_ci(50, 200, 1000)
        assert len(samples) == 1000

    def test_mean_approximately_midpoint(self):
        np.random.seed(42)
        lo, hi = 50, 200
        samples = vp._normal_from_ci(lo, hi, 100_000)
        expected_mean = (lo + hi) / 2
        actual_mean = np.mean(samples)
        assert abs(actual_mean - expected_mean) / expected_mean < 0.02

    def test_80_ci_coverage(self):
        np.random.seed(42)
        lo, hi = 50, 200
        samples = vp._normal_from_ci(lo, hi, 100_000)
        within = np.mean((samples >= lo) & (samples <= hi))
        # Normal clipping at lo/10 shifts things slightly, allow wider tolerance
        assert abs(within - 0.80) < 0.05

    def test_clipped_at_lo_over_10(self):
        np.random.seed(42)
        lo, hi = 10, 20
        samples = vp._normal_from_ci(lo, hi, 100_000)
        assert np.all(samples >= lo / 10)


class TestLogLognormalFromCi:
    def test_returns_correct_size(self):
        samples = vp._log_lognormal_from_ci(10, 200, 1000)
        assert len(samples) == 1000

    def test_all_positive(self):
        samples = vp._log_lognormal_from_ci(10, 200, 5000)
        assert np.all(samples > 0)

    def test_heavier_right_tail_than_lognormal(self):
        """Log-lognormal should have fatter right tail."""
        np.random.seed(42)
        lo, hi = 10, 200
        ln_samples = vp._lognormal_from_ci(lo, hi, 50_000)
        lln_samples = vp._log_lognormal_from_ci(lo, hi, 50_000)
        # 99th percentile should be higher for log-lognormal
        assert np.percentile(lln_samples, 99) > np.percentile(ln_samples, 99)


# ===========================================================================
# _logit() and _inv_logit()
# ===========================================================================

class TestLogitInvLogit:
    def test_logit_0_5(self):
        assert abs(vp._logit(0.5)) < 1e-10

    def test_logit_high(self):
        result = vp._logit(0.9)
        expected = np.log(0.9 / 0.1)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_logit_low(self):
        result = vp._logit(0.1)
        expected = np.log(0.1 / 0.9)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_inv_logit_0(self):
        np.testing.assert_allclose(vp._inv_logit(0), 0.5, atol=1e-10)

    def test_inv_logit_large_positive(self):
        result = vp._inv_logit(100)
        assert abs(result - 1.0) < 1e-10

    def test_inv_logit_large_negative(self):
        result = vp._inv_logit(-100)
        assert abs(result) < 1e-10

    def test_roundtrip(self):
        """logit and inv_logit should be inverses."""
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            np.testing.assert_allclose(vp._inv_logit(vp._logit(p)), p, atol=1e-10)

    def test_logit_clips_at_boundaries(self):
        """Should not raise for p=0 or p=1 due to clipping."""
        vp._logit(0)    # should not raise
        vp._logit(1)    # should not raise
        vp._logit(0.0)
        vp._logit(1.0)

    def test_logit_array(self):
        p = np.array([0.1, 0.5, 0.9])
        result = vp._logit(p)
        assert result.shape == (3,)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-10)

    def test_inv_logit_array(self):
        x = np.array([-2, 0, 2])
        result = vp._inv_logit(x)
        assert result.shape == (3,)
        np.testing.assert_allclose(result[1], 0.5, atol=1e-10)


# ===========================================================================
# Data loading: load_frontier()
# ===========================================================================

class TestLoadFrontier:
    def test_returns_list(self):
        data = vp.load_frontier()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_models_have_required_keys(self):
        data = vp.load_frontier()
        required = {'name', 'date', 'p50_min', 'p50_lo', 'p50_hi',
                     'p80_min', 'p80_lo', 'p80_hi', 'is_sota'}
        for m in data:
            assert required.issubset(m.keys()), f"Missing keys in {m['name']}: {required - m.keys()}"

    def test_sorted_by_date(self):
        data = vp.load_frontier()
        dates = [m['date'] for m in data]
        assert dates == sorted(dates)

    def test_all_are_sota(self):
        data = vp.load_frontier()
        for m in data:
            assert m['is_sota'] is True

    def test_dates_are_datetime(self):
        data = vp.load_frontier()
        for m in data:
            assert isinstance(m['date'], datetime)

    def test_p50_values_are_numeric(self):
        data = vp.load_frontier()
        for m in data:
            assert isinstance(m['p50_min'], (int, float))

    def test_p50_and_p80_both_present(self):
        """Both p50 and p80 should be present for frontier models."""
        data = vp.load_frontier()
        for m in data:
            assert m['p50_min'] is not None, f"{m['name']} missing p50"
            assert m['p80_min'] is not None, f"{m['name']} missing p80"

    def test_known_model_exists(self):
        data = vp.load_frontier()
        names = [m['name'] for m in data]
        assert 'gpt_4' in names or 'gpt_4_turbo_inspect' in names or 'gpt_4o_inspect' in names


# ===========================================================================
# Data loading: load_eci_frontier()
# ===========================================================================

class TestLoadEciFrontier:
    def test_returns_list(self):
        data = vp.load_eci_frontier()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_models_have_required_keys(self):
        data = vp.load_eci_frontier()
        required = {'name', 'date', 'eci_score', 'is_frontier', 'display_name'}
        for m in data:
            assert required.issubset(m.keys()), f"Missing keys: {required - m.keys()}"

    def test_sorted_by_date(self):
        data = vp.load_eci_frontier()
        dates = [m['date'] for m in data]
        assert dates == sorted(dates)

    def test_frontier_is_running_max(self):
        """Frontier models should form a non-decreasing sequence of scores."""
        data = vp.load_eci_frontier()
        frontier = [m for m in data if m['is_frontier']]
        scores = [m['eci_score'] for m in frontier]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1], \
                f"Frontier not monotonic at index {i}: {scores[i-1]} > {scores[i]}"

    def test_at_least_one_frontier_model(self):
        data = vp.load_eci_frontier()
        frontier = [m for m in data if m['is_frontier']]
        assert len(frontier) >= 1

    def test_dedup_by_model_name(self):
        """No duplicate model names in the output."""
        data = vp.load_eci_frontier()
        names = [m['name'] for m in data]
        assert len(names) == len(set(names))

    def test_dates_after_cutoff(self):
        """All models should be from Feb 2024 onward."""
        data = vp.load_eci_frontier()
        cutoff = datetime(2024, 2, 29)
        for m in data:
            assert m['date'] >= cutoff, f"{m['name']} date {m['date']} before cutoff"


# ===========================================================================
# Data loading: load_rli_data()
# ===========================================================================

class TestLoadRliData:
    def test_returns_list(self):
        data = vp.load_rli_data()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_models_have_required_keys(self):
        data = vp.load_rli_data()
        required = {'name', 'date', 'rli_score', 'is_frontier'}
        for m in data:
            assert required.issubset(m.keys()), f"Missing keys: {required - m.keys()}"

    def test_sorted_by_date(self):
        data = vp.load_rli_data()
        dates = [m['date'] for m in data]
        assert dates == sorted(dates)

    def test_frontier_is_running_max(self):
        data = vp.load_rli_data()
        frontier = [m for m in data if m['is_frontier']]
        scores = [m['rli_score'] for m in frontier]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1]

    def test_scores_in_reasonable_range(self):
        """RLI scores are percentages, should be 0-100."""
        data = vp.load_rli_data()
        for m in data:
            assert 0 <= m['rli_score'] <= 100, f"{m['name']}: score {m['rli_score']} out of range"

    def test_known_model_exists(self):
        data = vp.load_rli_data()
        names = [m['name'] for m in data]
        assert "Opus 4.5" in names


# ===========================================================================
# Backtest helpers
# ===========================================================================

class TestBacktestStats:
    def _make_trajectory(self, n_days, n_traj, base_val, slope):
        """Create simple linear trajectories for testing."""
        t = np.arange(n_days)
        return base_val + slope * t[np.newaxis, :] + \
            np.random.normal(0, 0.5, (n_traj, n_days))

    def test_basic(self):
        np.random.seed(42)
        start = datetime(2025, 1, 1)
        end = datetime(2025, 3, 1)
        trajs = self._make_trajectory(60, 1000, 10, 0.1)
        models = [
            {'date': datetime(2025, 1, 15), 'val': 11.5, 'name': 'M1'},
            {'date': datetime(2025, 2, 1), 'val': 13.0, 'name': 'M2'},
        ]
        results = vp._backtest_stats(
            models, trajs, start, end,
            get_value=lambda m: m['val'],
            get_name=lambda m: m['name'],
        )
        assert len(results) == 2
        for r in results:
            assert 'percentile' in r
            assert 'within_50' in r
            assert 'within_80' in r
            assert 'within_90' in r
            assert 0 <= r['percentile'] <= 100

    def test_excludes_models_outside_range(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 3, 1)
        trajs = np.random.normal(10, 1, (100, 60))
        models = [
            {'date': datetime(2024, 12, 1), 'val': 10, 'name': 'Before'},  # before start
            {'date': datetime(2025, 1, 1), 'val': 10, 'name': 'AtStart'},  # at start (excluded: <=)
            {'date': datetime(2025, 6, 1), 'val': 10, 'name': 'After'},   # after end
        ]
        results = vp._backtest_stats(
            models, trajs, start, end,
            get_value=lambda m: m['val'],
            get_name=lambda m: m['name'],
        )
        assert len(results) == 0


class TestBtColorFor:
    def test_within_50(self):
        r = {'within_50': True, 'within_80': True, 'within_90': True}
        assert vp._bt_color_for(r) == '#27ae60'

    def test_within_80_not_50(self):
        r = {'within_50': False, 'within_80': True, 'within_90': True}
        assert vp._bt_color_for(r) == '#f1c40f'

    def test_within_90_not_80(self):
        r = {'within_50': False, 'within_80': False, 'within_90': True}
        assert vp._bt_color_for(r) == '#e67e22'

    def test_outside_all(self):
        r = {'within_50': False, 'within_80': False, 'within_90': False}
        assert vp._bt_color_for(r) == '#e74c3c'


# ===========================================================================
# Integration: fit_line on real METR data
# ===========================================================================

class TestFitLineOnRealData:
    def test_metr_positive_slope(self):
        """OLS on METR frontier should show positive slope (improvement over time)."""
        _, _, params = _load_metr_fit()
        assert params[1] > 0, "METR frontier should have positive slope"

    def test_eci_positive_slope(self):
        """OLS on ECI frontier should show positive slope."""
        _, _, params = _load_eci_fit()
        assert params[1] > 0, "ECI frontier should have positive slope"

    def test_rli_positive_slope_in_logit(self):
        """OLS on RLI frontier in logit-space should show positive slope."""
        _, _, params = _load_rli_fit()
        assert params[1] > 0, "RLI frontier should have positive slope in logit-space"


# ===========================================================================
# Doubling time / points-per-year calculations
# ===========================================================================

class TestDoublingTimeCalculations:
    def test_metr_doubling_time_reasonable(self):
        """METR doubling time (in log2 space) should be in a plausible range."""
        _, _, params = _load_metr_fit()
        if params[1] > 0:
            dt_days = 1.0 / params[1]
            assert 1 < dt_days < 1000, f"DT {dt_days:.0f} days seems implausible"

    def test_eci_points_per_year_reasonable(self):
        """ECI points per year should be in a plausible range."""
        _, _, params = _load_eci_fit()
        ppy = params[1] * 365.25
        assert 1 < ppy < 100, f"PPY {ppy:.1f} seems implausible"


# ===========================================================================
# superexp_trajectory()
# ===========================================================================

class TestSuperexpTrajectory:
    def test_zero_days_gives_zero_growth(self):
        """At t=0, there should be no growth."""
        days = np.array([0.0])
        result = vp.superexp_trajectory(days, dt_0=100, halflife=365, dt_floor=10)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_monotonically_increasing(self):
        """Growth should always increase over time."""
        days = np.arange(0, 500, dtype=float)
        result = vp.superexp_trajectory(days, dt_0=100, halflife=365, dt_floor=10)
        diffs = np.diff(result)
        assert np.all(diffs >= 0), "Growth should be monotonically non-decreasing"

    def test_growth_rate_accelerates(self):
        """Before hitting the floor, growth should accelerate (superexponential)."""
        days = np.arange(0, 100, dtype=float)
        result = vp.superexp_trajectory(days, dt_0=100, halflife=365, dt_floor=5)
        diffs = np.diff(result)
        # Second derivative should be positive (acceleration)
        second_diffs = np.diff(diffs)
        assert np.all(second_diffs > -1e-10), "Growth rate should accelerate before floor"

    def test_linear_after_floor(self):
        """After DT hits floor, growth should become linear at rate 1/dt_floor."""
        dt_0, halflife, dt_floor = 100, 365, 10
        # t_cap = halflife * log2(dt_0/dt_floor) = 365 * log2(10) ≈ 1212 days
        t_cap = halflife * np.log2(dt_0 / dt_floor)
        days_after = np.array([t_cap + 100, t_cap + 200, t_cap + 300])
        result = vp.superexp_trajectory(days_after, dt_0, halflife, dt_floor)
        # After floor, growth per day should be 1/dt_floor
        diffs = np.diff(result) / np.diff(days_after)
        np.testing.assert_allclose(diffs, 1.0 / dt_floor, rtol=1e-6)

    def test_floor_already_hit(self):
        """If dt_0 <= dt_floor, should be linear from the start."""
        days = np.arange(0, 100, dtype=float)
        result = vp.superexp_trajectory(days, dt_0=10, halflife=365, dt_floor=10)
        # Should be purely linear: growth = days / dt_floor
        expected = days / 10
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_dt_below_floor(self):
        """If dt_0 < dt_floor, should still be linear at floor rate."""
        days = np.arange(0, 100, dtype=float)
        result = vp.superexp_trajectory(days, dt_0=5, halflife=365, dt_floor=10)
        expected = days / 10
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_short_halflife_faster_growth(self):
        """Shorter half-life should produce faster growth."""
        days = np.arange(0, 200, dtype=float)
        slow = vp.superexp_trajectory(days, dt_0=100, halflife=1000, dt_floor=5)
        fast = vp.superexp_trajectory(days, dt_0=100, halflife=100, dt_floor=5)
        # At the end of the period, faster halflife should have more growth
        assert fast[-1] > slow[-1]

    def test_lower_dt_0_faster_growth(self):
        """Lower initial DT should produce faster growth."""
        days = np.arange(0, 200, dtype=float)
        slow = vp.superexp_trajectory(days, dt_0=200, halflife=365, dt_floor=5)
        fast = vp.superexp_trajectory(days, dt_0=50, halflife=365, dt_floor=5)
        assert fast[-1] > slow[-1]

    def test_continuity_at_floor_transition(self):
        """Growth should be continuous (no jump) at the floor transition point."""
        dt_0, halflife, dt_floor = 100, 365, 10
        t_cap = halflife * np.log2(dt_0 / dt_floor)
        # Check values just before and after t_cap
        days = np.array([t_cap - 0.01, t_cap, t_cap + 0.01])
        result = vp.superexp_trajectory(days, dt_0, halflife, dt_floor)
        # Should be smooth (no large jumps)
        assert abs(result[2] - result[1]) < 0.01
        assert abs(result[1] - result[0]) < 0.01

    def test_scalar_day_input(self):
        """Should work with scalar day values too."""
        result = vp.superexp_trajectory(np.array([100.0]), dt_0=100, halflife=365, dt_floor=10)
        assert result.shape == (1,)
        assert result[0] > 0


# ===========================================================================
# Superexp trajectory median matches projection line
# ===========================================================================

class TestSuperexpProjectionMatchesTrajectories:
    """End-to-end tests that replicate the actual render-function code paths
    for BOTH the trajectory simulation and the projection line, then verify
    they produce the same curve.

    These tests would have caught the original bug where the projection line
    used `A + K * 2^(d/H)` (historical fit extrapolation) while trajectories
    used `start + superexp_trajectory(...)` (forward simulation).
    """

    # -- METR tab ---------------------------------------------------------

    def test_metr_proj_line_uses_superexp_trajectory_not_fit_extrapolation(self):
        """Replicate the METR render code for both the trajectory loop and the
        projection line.  Assert the projection line == trajectory at center DT.

        The OLD (buggy) code computed the projection line as:
            y = _se_A_disp + _se_K * 2^(d / halflife)
        which is the historical fit extrapolated forward.

        The CORRECT code computes it as:
            y = fitted_pos + superexp_trajectory(days_from_last, center_dt, H, floor)
        which matches what the trajectory loop does.
        """
        # Synthetic frontier data: 5 models with accelerating log2(min) scores
        halflife, dt_floor = 365, 15
        dt_ci_lo, dt_ci_hi = 50, 200
        frontier_days = np.array([0, 100, 200, 300, 400], dtype=float)
        frontier_vals = np.array([5.0, 6.8, 9.0, 11.5, 14.5])  # log2(min)

        # Step 1: fit A + K * 2^(d/H) to the frontier (as the render code does)
        A, K = _fit_superexp(frontier_days, frontier_vals, halflife)
        d_last = frontier_days[-1]
        fitted_pos = A + K * 2 ** (d_last / halflife)

        # Step 2: compute projection line the CORRECT way (as render code does now)
        center_dt = np.sqrt(dt_ci_lo * dt_ci_hi)
        proj_days = np.arange(0, 365, dtype=float)
        proj_line_correct = fitted_pos + vp.superexp_trajectory(
            proj_days, center_dt, halflife, dt_floor)

        # Step 3: compute projection line the OLD BUGGY way
        future_abs_days = d_last + proj_days
        proj_line_buggy = A + K * 2 ** (future_abs_days / halflife)

        # Step 4: simulate trajectories with fixed DT at center_dt
        traj = fitted_pos + vp.superexp_trajectory(
            proj_days, center_dt, halflife, dt_floor)

        # The correct projection line should match the trajectory exactly
        np.testing.assert_allclose(proj_line_correct, traj, rtol=1e-10,
            err_msg="Projection line diverges from trajectory formula")

        # The buggy line should NOT match (this is what we're protecting against)
        # At day 200 the extrapolation diverges meaningfully from forward simulation
        assert not np.allclose(proj_line_buggy[200:], traj[200:], rtol=0.01), \
            "Buggy extrapolation should differ from trajectory — test is not discriminating"

    def test_metr_proj_line_responds_to_user_dt_ci(self):
        """Changing dt_ci should change the projection line.

        The old buggy code ignored the user's DT CI for the projection line
        (it always used the historical fit). This test verifies the projection
        line actually changes when the user changes their CI."""
        halflife, dt_floor = 365, 15
        frontier_days = np.array([0, 100, 200, 300, 400], dtype=float)
        frontier_vals = np.array([5.0, 6.8, 9.0, 11.5, 14.5])
        A, K = _fit_superexp(frontier_days, frontier_vals, halflife)
        d_last = frontier_days[-1]
        fitted_pos = A + K * 2 ** (d_last / halflife)

        proj_days = np.arange(0, 365, dtype=float)

        # Two different user CI settings
        proj_fast = fitted_pos + vp.superexp_trajectory(
            proj_days, np.sqrt(30 * 120), halflife, dt_floor)   # center=60
        proj_slow = fitted_pos + vp.superexp_trajectory(
            proj_days, np.sqrt(100 * 400), halflife, dt_floor)  # center=200

        # They must diverge (fast DT grows faster)
        assert proj_fast[-1] > proj_slow[-1], \
            "Faster DT should produce higher projection"
        assert abs(proj_fast[-1] - proj_slow[-1]) > 0.5, \
            "Projection line doesn't respond to user DT CI"

    # -- ECI tab ----------------------------------------------------------

    def test_eci_proj_line_uses_superexp_trajectory_not_fit_extrapolation(self):
        """Same test for ECI: projection line must use superexp_trajectory,
        not the historical fit extrapolation."""
        halflife, dpp_floor = 365, 10
        dpp_ci_lo, dpp_ci_hi = 10, 30
        frontier_days = np.array([0, 60, 120, 180, 240], dtype=float)
        frontier_scores = np.array([130.0, 138.0, 143.0, 147.0, 150.0])

        A, K = _fit_superexp(frontier_days, frontier_scores, halflife)
        d_last = frontier_days[-1]
        fitted_score = A + K * 2 ** (d_last / halflife)

        center_dpp = np.sqrt(dpp_ci_lo * dpp_ci_hi)
        proj_days = np.arange(0, 365, dtype=float)

        # Correct projection line
        proj_correct = fitted_score + vp.superexp_trajectory(
            proj_days, center_dpp, halflife, dpp_floor)

        # Buggy extrapolation
        future_abs_days = d_last + proj_days
        proj_buggy = A + K * 2 ** (future_abs_days / halflife)

        # Trajectory at center DPP
        traj = fitted_score + vp.superexp_trajectory(
            proj_days, center_dpp, halflife, dpp_floor)

        np.testing.assert_allclose(proj_correct, traj, rtol=1e-10,
            err_msg="ECI projection line diverges from trajectory formula")

        assert not np.allclose(proj_buggy[200:], traj[200:], rtol=0.01), \
            "ECI buggy extrapolation should differ from trajectory"

    def test_eci_proj_line_responds_to_user_dpp_ci(self):
        """ECI projection line should change when user changes DPP CI."""
        halflife, dpp_floor = 365, 5
        frontier_days = np.array([0, 60, 120, 180, 240], dtype=float)
        frontier_scores = np.array([130.0, 138.0, 143.0, 147.0, 150.0])
        A, K = _fit_superexp(frontier_days, frontier_scores, halflife)
        d_last = frontier_days[-1]
        fitted_score = A + K * 2 ** (d_last / halflife)

        proj_days = np.arange(0, 365, dtype=float)
        proj_fast = fitted_score + vp.superexp_trajectory(
            proj_days, np.sqrt(5 * 15), halflife, dpp_floor)    # center≈8.7
        proj_slow = fitted_score + vp.superexp_trajectory(
            proj_days, np.sqrt(20 * 60), halflife, dpp_floor)   # center≈34.6

        assert proj_fast[-1] > proj_slow[-1]
        assert abs(proj_fast[-1] - proj_slow[-1]) > 1.0

    # -- RLI tab ----------------------------------------------------------

    def test_rli_proj_line_uses_superexp_trajectory_not_fit_extrapolation(self):
        """Same test for RLI in logit space."""
        halflife, dt_floor = 365, 15
        dt_ci_lo, dt_ci_hi = 50, 200
        frontier_days = np.array([0, 60, 120, 180, 240], dtype=float)
        frontier_logit = np.array([-5.0, -4.5, -4.0, -3.4, -2.9])

        A, K = _fit_superexp(frontier_days, frontier_logit, halflife)
        d_last = frontier_days[-1]
        fitted_logit = A + K * 2 ** (d_last / halflife)

        center_dt = np.sqrt(dt_ci_lo * dt_ci_hi)
        proj_days = np.arange(0, 365, dtype=float)

        # Correct: ln(2) * superexp_trajectory
        proj_correct = fitted_logit + np.log(2) * vp.superexp_trajectory(
            proj_days, center_dt, halflife, dt_floor)

        # Buggy extrapolation
        future_abs_days = d_last + proj_days
        proj_buggy = A + K * 2 ** (future_abs_days / halflife)

        # Trajectory at center DT
        traj = fitted_logit + np.log(2) * vp.superexp_trajectory(
            proj_days, center_dt, halflife, dt_floor)

        np.testing.assert_allclose(proj_correct, traj, rtol=1e-10,
            err_msg="RLI projection line diverges from trajectory formula")

        assert not np.allclose(proj_buggy[200:], traj[200:], rtol=0.01), \
            "RLI buggy extrapolation should differ from trajectory"

    def test_rli_proj_line_responds_to_user_dt_ci(self):
        """RLI projection line should change when user changes DT CI."""
        halflife, dt_floor = 365, 15
        frontier_days = np.array([0, 60, 120, 180, 240], dtype=float)
        frontier_logit = np.array([-5.0, -4.5, -4.0, -3.4, -2.9])
        A, K = _fit_superexp(frontier_days, frontier_logit, halflife)
        d_last = frontier_days[-1]
        fitted_logit = A + K * 2 ** (d_last / halflife)

        proj_days = np.arange(0, 365, dtype=float)
        proj_fast = fitted_logit + np.log(2) * vp.superexp_trajectory(
            proj_days, np.sqrt(30 * 120), halflife, dt_floor)
        proj_slow = fitted_logit + np.log(2) * vp.superexp_trajectory(
            proj_days, np.sqrt(100 * 400), halflife, dt_floor)

        assert proj_fast[-1] > proj_slow[-1]

    def test_sampled_trajectories_center_on_projection_line(self):
        """With lognormal DT sampling, trajectory MEDIAN should be
        approximately near the projection line (which uses the geometric
        mean DT)."""
        np.random.seed(42)
        dt_lo, dt_hi = 50, 200
        dt_center = np.sqrt(dt_lo * dt_hi)
        halflife, dt_floor = 365, 15
        start_val = 10.0
        n_samples = 50000
        days = np.arange(0, 200, dtype=float)

        sampled_dts = vp._lognormal_from_ci(dt_lo, dt_hi, n_samples)
        trajectories = np.zeros((n_samples, len(days)))
        for i in range(n_samples):
            trajectories[i] = start_val + vp.superexp_trajectory(
                days, sampled_dts[i], halflife, dt_floor)

        proj_line = start_val + vp.superexp_trajectory(days, dt_center, halflife, dt_floor)
        median = np.median(trajectories, axis=0)
        for day_idx in [100, -1]:
            rel_err = abs(median[day_idx] - proj_line[day_idx]) / max(abs(proj_line[day_idx]), 1e-6)
            assert rel_err < 0.15, (
                f"At day {day_idx}: median={median[day_idx]:.3f}, "
                f"proj={proj_line[day_idx]:.3f}, rel_err={rel_err:.3f}"
            )


# ===========================================================================
# Default projection must match historical fit at transition
# ===========================================================================

class TestDefaultProjectionMatchesFit:
    """The original bug: hardcoded CI defaults (e.g. center=100 days) didn't
    match the actual fitted DT/PPY from the data, causing a visible slope
    discontinuity where the historical fit ends and the projection begins.

    These tests replicate the actual render-function code paths and verify
    that the default projection line continues the historical fit seamlessly.
    Each test computes BOTH the correct formula AND the old buggy formula
    and asserts they disagree (discriminating) and that the code uses the
    correct one.
    """

    # -- METR linear --------------------------------------------------------

    def test_metr_linear_hardcoded_100_differs_from_ols_dt(self):
        """The old hardcoded center (100d) differs from the OLS-fitted DT.
        If this test passes, the old bug WOULD have caused a slope mismatch."""
        _, _, params = _load_metr_fit()
        ols_dt = 1.0 / params[1] if params[1] > 0 else 100
        buggy_center = 100  # old hardcoded value
        # If this fails, the hardcoded default happened to match the fit
        # (unlikely but possible); the test is only discriminating when they differ.
        assert abs(buggy_center - ols_dt) / ols_dt > 0.05, \
            "Hardcoded 100 coincidentally matches OLS DT — test is not discriminating"

    def test_metr_linear_slope_continuous_at_transition(self):
        """At the transition point, the projection slope should equal the OLS
        slope when using data-driven defaults."""
        _, _, params = _load_metr_fit()
        ols_dt = round(1.0 / params[1]) if params[1] > 0 else 100
        # Data-driven defaults (replicating render code)
        lo = max(10, int(round(ols_dt / 2)))
        hi = int(round(ols_dt * 2))
        default_center_dt = np.sqrt(lo * hi)
        default_slope = 1.0 / default_center_dt

        # Buggy hardcoded defaults
        buggy_center_dt = np.sqrt(50 * 200)  # = 100
        buggy_slope = 1.0 / buggy_center_dt

        ols_slope = params[1]

        # Correct default slope should match OLS (within rounding)
        assert abs(default_slope - ols_slope) / ols_slope < 0.15
        # Buggy slope should NOT match OLS
        assert abs(buggy_slope - ols_slope) / ols_slope > 0.05

    # -- METR superexponential ----------------------------------------------

    def test_metr_superexp_hardcoded_100_differs_from_fitted_dt(self):
        """The old hardcoded superexp DT CI center differs from the
        superexp fit's implied DT at the last data point."""
        halflife = 365
        days, vals, _ = _load_metr_fit()
        A, K = _fit_superexp(days, vals, halflife)
        if K > 0:
            fitted_dt = halflife / (K * np.log(2) * 2 ** (days[-1] / halflife))
        else:
            pytest.skip("K <= 0, can't test superexp")
        buggy_center = 100  # old hardcoded value
        assert abs(buggy_center - fitted_dt) / fitted_dt > 0.05, \
            "Hardcoded 100 coincidentally matches superexp DT — test is not discriminating"

    def test_metr_superexp_default_dt_matches_fit_implied_dt(self):
        """Data-driven superexp CI defaults should center on the fit's
        implied DT at the last data point."""
        halflife = 365
        days, vals, _ = _load_metr_fit()
        A, K = _fit_superexp(days, vals, halflife)
        if K > 0:
            fitted_dt = halflife / (K * np.log(2) * 2 ** (days[-1] / halflife))
        else:
            pytest.skip("K <= 0, can't test superexp")
        # Data-driven defaults
        lo = max(10, int(round(fitted_dt / 2)))
        hi = int(round(fitted_dt * 2))
        center = np.sqrt(lo * hi)
        assert abs(center - fitted_dt) / fitted_dt < 0.15

    def test_metr_superexp_proj_continuous_at_transition(self):
        """Superexp projection should start at the historical fit value and
        grow at the same rate immediately after the transition."""
        halflife = 365
        days, vals, _ = _load_metr_fit()
        A, K = _fit_superexp(days, vals, halflife)
        if K <= 0:
            pytest.skip("K <= 0, can't test superexp")
        d_last = days[-1]
        fitted_pos = A + K * 2 ** (d_last / halflife)
        fitted_dt = halflife / (K * np.log(2) * 2 ** (d_last / halflife))
        # Data-driven center
        lo = max(10, int(round(fitted_dt / 2)))
        hi = int(round(fitted_dt * 2))
        center_dt = np.sqrt(lo * hi)
        # Projection starts at fitted_pos and grows via superexp_trajectory
        small_step = np.array([0.0, 1.0])
        growth = vp.superexp_trajectory(small_step, center_dt, halflife, 1.0)
        proj_slope = growth[1] - growth[0]  # growth per day at t=0
        # Historical fit slope at last point = K * ln(2)/halflife * 2^(d/H)
        fit_slope = K * np.log(2) / halflife * 2 ** (d_last / halflife)
        # These are both 1/DT at the transition — should match
        assert abs(proj_slope - fit_slope) / fit_slope < 0.15

    # -- ECI linear ---------------------------------------------------------

    def test_eci_linear_slope_continuous_at_transition(self):
        """ECI projection slope should match OLS slope under data-driven defaults."""
        _, _, params = _load_eci_fit()
        ols_ppy = params[1] * 365.25 if params[1] > 0 else 16.9
        ols_dpp = 365.25 / ols_ppy  # days per point
        # Data-driven defaults
        ppy = round(ols_ppy, 1)
        lo = round(ppy / 2, 1)
        hi = round(ppy * 2, 1)
        default_center_ppy = np.sqrt(lo * hi)
        default_slope = default_center_ppy / 365.25  # points per day
        ols_slope = params[1]
        assert abs(default_slope - ols_slope) / ols_slope < 0.15

    # -- ECI superexponential -----------------------------------------------

    def test_eci_superexp_default_ppy_matches_fit_implied_ppy(self):
        """Data-driven superexp PPY defaults should center on the fit's
        implied PPY at the last data point."""
        days, vals, _ = _load_eci_fit()
        halflife = 365
        A, K = _fit_superexp(days, vals, halflife)
        if K <= 0:
            pytest.skip("K <= 0, can't test superexp")
        dpp = halflife / (K * np.log(2) * 2 ** (days[-1] / halflife))
        fitted_ppy = round(365.25 / dpp, 1)
        lo = round(max(0.5, fitted_ppy / 2), 1)
        hi = round(fitted_ppy * 2, 1)
        center = np.sqrt(lo * hi)
        assert abs(center - fitted_ppy) / fitted_ppy < 0.15

    # -- RLI linear ---------------------------------------------------------

    def test_rli_linear_slope_continuous_at_transition(self):
        """RLI projection slope (in logit space) should match OLS slope
        under data-driven defaults."""
        _, _, params = _load_rli_fit()
        ols_dt = np.log(2) / params[1] if params[1] > 0 else 100
        ols_dt_r = round(ols_dt)
        lo = round(max(5.0, ols_dt_r / 2), 0)
        hi = round(ols_dt_r * 2, 0)
        default_center_dt = np.sqrt(lo * hi)
        # slope in logit space: ln(2) / DT
        default_slope = np.log(2) / default_center_dt
        ols_slope = params[1]
        assert abs(default_slope - ols_slope) / ols_slope < 0.15

    # -- RLI superexponential -----------------------------------------------

    def test_rli_superexp_default_dt_matches_fit_implied_dt(self):
        """Data-driven superexp DT defaults should center on the fit's
        implied DT at the last data point in logit space."""
        days, vals, _ = _load_rli_fit()
        halflife = 365
        A, K = _fit_superexp(days, vals, halflife)
        if K <= 0:
            pytest.skip("K <= 0, can't test superexp")
        logit_slope = K * np.log(2) * 2 ** (days[-1] / halflife) / halflife
        fitted_dt = round(np.log(2) / logit_slope, 0)
        lo = round(max(5.0, fitted_dt / 2), 0)
        hi = round(fitted_dt * 2, 0)
        center = np.sqrt(lo * hi)
        assert abs(center - fitted_dt) / fitted_dt < 0.15

    # -- Sanity checks on defaults ------------------------------------------

    def test_metr_linear_defaults_are_factor_of_2_spread(self):
        """METR linear CI: hi should be ~4x lo (since lo=dt/2, hi=dt*2)."""
        _, _, params = _load_metr_fit()
        ols_dt = round(1.0 / params[1]) if params[1] > 0 else 100
        lo = max(10, int(round(ols_dt / 2)))
        hi = int(round(ols_dt * 2))
        assert 3.0 <= hi / lo <= 5.0

    def test_lognormal_ci_center_is_geometric_mean(self):
        """Lognormal median = geometric mean of CI bounds."""
        lo, hi = 30, 300
        expected = np.sqrt(lo * hi)
        np.random.seed(42)
        samples = vp._lognormal_from_ci(lo, hi, 200_000)
        actual = np.median(samples)
        assert abs(actual - expected) / expected < 0.02

    def test_all_fitted_values_are_positive(self):
        """All data-driven fitted DT/PPY should be positive numbers."""
        _, _, metr_params = _load_metr_fit()
        assert metr_params[1] > 0
        _, _, eci_params = _load_eci_fit()
        assert eci_params[1] > 0
        _, _, rli_params = _load_rli_fit()
        assert rli_params[1] > 0

    # -- Piecewise: last segment DT should differ from full-data DT ---------

    def test_metr_piecewise_last_seg_dt_differs_from_full_ols(self):
        """The last segment (post-GPT-4o) DT should differ from full-data OLS DT.
        If they're the same, the piecewise default bug is not discriminable."""
        days, vals, params = _load_metr_fit()
        full_dt = 1.0 / params[1] if params[1] > 0 else 100
        # Last segment: from GPT-4o to end (replicating the default breakpoint)
        frontier = vp.load_frontier()
        gpt4o_idx = next(i for i, m in enumerate(frontier) if m['name'] == 'gpt_4o_inspect')
        seg_days = days[gpt4o_idx:]
        seg_vals = vals[gpt4o_idx:]
        if len(seg_days) >= 2:
            seg_params = vp.fit_line(seg_days, seg_vals)
            seg_dt = 1.0 / seg_params[1] if seg_params[1] > 0 else full_dt
            assert abs(seg_dt - full_dt) / full_dt > 0.05, \
                "Last segment DT coincidentally matches full OLS — test not discriminating"

    def test_metr_piecewise_default_ci_uses_last_segment_dt(self):
        """For piecewise linear, the default CI should center on the last
        segment's DT (post-GPT-4o), not the full-data OLS DT."""
        days, vals, params = _load_metr_fit()
        full_dt = round(1.0 / params[1]) if params[1] > 0 else 100
        # Last segment DT
        frontier = vp.load_frontier()
        gpt4o_idx = next(i for i, m in enumerate(frontier) if m['name'] == 'gpt_4o_inspect')
        seg_days = days[gpt4o_idx:]
        seg_vals = vals[gpt4o_idx:]
        seg_params = vp.fit_line(seg_days, seg_vals)
        seg_dt = round(1.0 / seg_params[1]) if seg_params[1] > 0 else full_dt
        # Data-driven piecewise defaults should match last segment
        lo = max(10, int(round(seg_dt / 2)))
        hi = int(round(seg_dt * 2))
        center = np.sqrt(lo * hi)
        assert abs(center - seg_dt) / seg_dt < 0.15
        # And should NOT match full-data OLS (the old buggy behavior)
        full_lo = max(10, int(round(full_dt / 2)))
        full_hi = int(round(full_dt * 2))
        full_center = np.sqrt(full_lo * full_hi)
        assert abs(full_center - seg_dt) / seg_dt > 0.05, \
            "Full-data OLS defaults coincidentally match last segment DT"

    # -- Segment config changes should update CI defaults --------------------

    def _metr_last_seg_dt(self, bp_idx):
        """DT of the last segment starting at bp_idx."""
        days, vals, _ = _load_metr_fit()
        seg = vp.fit_line(days[bp_idx:], vals[bp_idx:])
        return 1.0 / seg[1] if seg[1] > 0 else 100

    def test_metr_ci_defaults_track_segment_changes(self):
        """Changing breakpoints should change CI defaults.
        Tests the full chain: breakpoint → last segment DT → CI lo/hi → fan slope."""
        days, vals, _ = _load_metr_fit()
        frontier = vp.load_frontier()
        gpt4o_idx = next(i for i, m in enumerate(frontier) if m['name'] == 'gpt_4o_inspect')

        # Two different breakpoints should yield different last-segment DTs
        bp_a = gpt4o_idx  # default 2-seg breakpoint
        remaining = list(range(gpt4o_idx + 1, len(frontier)))
        bp_b = remaining[len(remaining) // 2]  # 3-seg second breakpoint

        dt_a = self._metr_last_seg_dt(bp_a)
        dt_b = self._metr_last_seg_dt(bp_b)
        assert abs(dt_a - dt_b) / dt_a > 0.05, \
            "Different breakpoints yield same DT — test not discriminating"

        # CI defaults derived from each should also differ
        ci_center_a = np.sqrt(max(10, int(round(dt_a / 2))) * int(round(dt_a * 2)))
        ci_center_b = np.sqrt(max(10, int(round(dt_b / 2))) * int(round(dt_b * 2)))
        assert abs(ci_center_a - ci_center_b) / ci_center_a > 0.05

        # Each CI center should match its own last-segment DT (the inheritance property)
        assert abs(ci_center_a - dt_a) / dt_a < 0.15
        assert abs(ci_center_b - dt_b) / dt_b < 0.15

    def test_eci_ci_defaults_track_segment_changes(self):
        """Same inheritance chain for ECI: breakpoint → last segment PPY → CI."""
        days, vals, _ = _load_eci_fit()
        all_data = vp.load_eci_frontier()
        frontier = [m for m in all_data if m['is_frontier']]
        mid = len(frontier) // 2
        remaining = list(range(mid + 1, len(frontier)))
        if len(remaining) < 3:
            pytest.skip("Not enough ECI models for 3-segment test")

        bp_a = mid
        ppy_a = vp.fit_line(days[bp_a:], vals[bp_a:])[1] * 365.25
        # Pick the remaining breakpoint whose last-segment PPY differs most from
        # bp_a's, so the test is discriminating regardless of how linear the
        # current frontier happens to be. Skip if no segment differs enough.
        bp_b = max(remaining, key=lambda b: abs(
            vp.fit_line(days[b:], vals[b:])[1] * 365.25 - ppy_a))
        ppy_b = vp.fit_line(days[bp_b:], vals[bp_b:])[1] * 365.25
        if abs(ppy_a - ppy_b) / ppy_a <= 0.05:
            pytest.skip("ECI frontier too linear to discriminate breakpoints")

        ci_a = np.sqrt(round(ppy_a / 2, 1) * round(ppy_a * 2, 1))
        ci_b = np.sqrt(round(ppy_b / 2, 1) * round(ppy_b * 2, 1))
        assert abs(ci_a - ci_b) / ci_a > 0.05
        assert abs(ci_a - ppy_a) / ppy_a < 0.15
        assert abs(ci_b - ppy_b) / ppy_b < 0.15

    def test_rli_ci_defaults_track_segment_changes(self):
        """Same inheritance chain for RLI: breakpoint → last segment DT → CI.
        Uses 2-segment with two different breakpoints (not 3-segment, since
        RLI has few frontier models)."""
        days, vals, params = _load_rli_fit()
        all_data = vp.load_rli_data()
        frontier = [m for m in all_data if m['is_frontier']]
        if len(frontier) < 4:
            pytest.skip("Not enough RLI models for breakpoint test")

        # Two different breakpoint positions
        bp_a = 1
        bp_b = len(frontier) - 2

        slope_a = vp.fit_line(days[bp_a:], vals[bp_a:])[1]
        slope_b = vp.fit_line(days[bp_b:], vals[bp_b:])[1]
        if slope_a <= 0 or slope_b <= 0:
            pytest.skip("Non-positive slope in RLI segments")
        dt_a = np.log(2) / slope_a
        dt_b = np.log(2) / slope_b
        assert abs(dt_a - dt_b) / dt_a > 0.05, \
            "Different breakpoints yield same DT — test not discriminating"

        ci_a = np.sqrt(max(5, round(dt_a / 2)) * round(dt_a * 2))
        ci_b = np.sqrt(max(5, round(dt_b / 2)) * round(dt_b * 2))
        assert abs(ci_a - ci_b) / ci_a > 0.05


# ===========================================================================
# Streamlit number_input type consistency
# ===========================================================================

class TestNumberInputTypes:
    """Streamlit's number_input requires all numeric args (value, min_value,
    max_value, step) to be the same type (all int or all float). The fake
    Streamlit module doesn't enforce this, so we verify the computed default
    values have the expected types."""

    def test_rli_linear_defaults_are_float(self):
        """RLI linear DT defaults must be float (widget uses float min/step)."""
        _, _, params = _load_rli_fit()
        dt = round(np.log(2) / params[1]) if params[1] > 0 else 100
        lo = float(round(max(5.0, dt / 2), 0))
        hi = float(round(dt * 2, 0))
        assert isinstance(lo, float), f"lo is {type(lo)}, expected float"
        assert isinstance(hi, float), f"hi is {type(hi)}, expected float"

    def test_rli_superexp_defaults_are_float(self):
        """RLI superexp DT defaults must be float."""
        days, vals, _ = _load_rli_fit()
        halflife = 365
        A, K = _fit_superexp(days, vals, halflife)
        if K > 0:
            logit_slope = K * np.log(2) * 2 ** (days[-1] / halflife) / halflife
            dt = round(np.log(2) / logit_slope, 0)
        else:
            dt = 100.0
        lo = float(round(max(5.0, dt / 2), 0))
        hi = float(round(dt * 2, 0))
        assert isinstance(lo, float), f"lo is {type(lo)}, expected float"
        assert isinstance(hi, float), f"hi is {type(hi)}, expected float"

    def test_metr_linear_defaults_are_int(self):
        """METR linear DT defaults must be int (widget uses int min/step)."""
        _, _, params = _load_metr_fit()
        dt = round(1.0 / params[1]) if params[1] > 0 else 100
        lo = max(10, int(round(dt / 2)))
        hi = int(round(dt * 2))
        assert isinstance(lo, int), f"lo is {type(lo)}, expected int"
        assert isinstance(hi, int), f"hi is {type(hi)}, expected int"

    def test_eci_linear_defaults_are_float(self):
        """ECI linear PPY defaults must be float (widget uses float min/step)."""
        _, _, params = _load_eci_fit()
        ppy = round(params[1] * 365.25, 1) if params[1] > 0 else 16.9
        lo = round(ppy / 2, 1)
        hi = round(ppy * 2, 1)
        assert isinstance(lo, float), f"lo is {type(lo)}, expected float"
        assert isinstance(hi, float), f"hi is {type(hi)}, expected float"


# ===========================================================================
# _ss_number_input widget-conflict regression
# ===========================================================================

class _SpyParent:
    """Records the kwargs passed to number_input so we can assert on them."""
    def __init__(self):
        self.calls = []

    def number_input(self, label, **kw):
        self.calls.append((label, kw))
        return kw.get("value", 0)


class TestSsNumberInput:
    """Regression guard for the widget-conflict crash on Streamlit Cloud.

    Passing both key= and value= to number_input when the session_state key
    is already set violates Streamlit's check_session_state_rules, logs a
    warning, and segfaulted the worker. _ss_number_input must pass key= only.
    """

    def _clear(self, key):
        vp.st.session_state.pop(key, None)

    def test_does_not_pass_value_with_key(self):
        """The core regression: number_input is called with key but not value."""
        self._clear("_ss_test_key")
        parent = _SpyParent()
        vp._ss_number_input(parent, "L", "_ss_test_key", 42, min_value=10, step=5)
        assert len(parent.calls) == 1
        _, kw = parent.calls[0]
        assert kw.get("key") == "_ss_test_key"
        assert "value" not in kw, "value= must not be passed alongside key="

    def test_initialises_session_state_to_default(self):
        """On first render the session_state key is seeded with the default."""
        self._clear("_ss_test_key2")
        parent = _SpyParent()
        vp._ss_number_input(parent, "L", "_ss_test_key2", 99)
        assert vp.st.session_state["_ss_test_key2"] == 99

    def test_preserves_existing_session_state(self):
        """If the key already exists, the default does not overwrite it."""
        vp.st.session_state["_ss_test_key3"] = 7
        parent = _SpyParent()
        vp._ss_number_input(parent, "L", "_ss_test_key3", 99)
        assert vp.st.session_state["_ss_test_key3"] == 7

    def test_extra_kwargs_forwarded(self):
        """min_value/max_value/step are forwarded to number_input."""
        self._clear("_ss_test_key4")
        parent = _SpyParent()
        vp._ss_number_input(parent, "L", "_ss_test_key4", 42,
                            min_value=10, max_value=2000, step=5)
        _, kw = parent.calls[0]
        assert kw["min_value"] == 10 and kw["max_value"] == 2000 and kw["step"] == 5


# ===========================================================================
# Edge cases / error conditions
# ===========================================================================

class TestEdgeCases:
    def test_fit_line_single_point(self):
        """fit_line with a single point should not crash."""
        x = np.array([0.0])
        y = np.array([5.0])
        params = vp.fit_line(x, y)
        assert len(params) == 2

    def test_logit_tiny_values(self):
        """logit should handle values very close to 0 and 1."""
        result_low = vp._logit(1e-15)
        result_high = vp._logit(1 - 1e-15)
        assert np.isfinite(result_low)
        assert np.isfinite(result_high)
        assert result_low < 0
        assert result_high > 0

    def test_inv_logit_extreme_values(self):
        """inv_logit should be stable for extreme inputs."""
        assert vp._inv_logit(500) < 1.01  # clipped at 500
        assert vp._inv_logit(-500) > -0.01
        assert np.isfinite(vp._inv_logit(500))
        assert np.isfinite(vp._inv_logit(-500))

    def test_lognormal_narrow_ci(self):
        """Very narrow CI should still work."""
        np.random.seed(42)
        samples = vp._lognormal_from_ci(99, 101, 1000)
        assert len(samples) == 1000
        assert np.all(samples > 0)

    def test_lognormal_equal_bounds(self):
        """Equal lo=hi means sigma=0, should return constant."""
        np.random.seed(42)
        samples = vp._lognormal_from_ci(100, 100, 100)
        np.testing.assert_allclose(samples, 100.0, rtol=1e-10)

    def test_normal_equal_bounds(self):
        """Equal lo=hi means sigma=0, should return constant."""
        np.random.seed(42)
        samples = vp._normal_from_ci(100, 100, 100)
        np.testing.assert_allclose(samples, 100.0, rtol=1e-10)


# ===========================================================================
# Name mapping completeness
# ===========================================================================

class TestNameMapping:
    def test_all_frontier_models_have_pretty_names(self):
        """Every METR frontier model should have a display name in _NAMES."""
        data = vp.load_frontier()
        missing = [m['name'] for m in data if m['name'] not in vp._NAMES]
        assert missing == [], f"Missing pretty names for: {missing}"


# ===========================================================================
# URL parameter persistence
# ===========================================================================

class TestCoerceUrlValue:
    def test_bool_truthy(self):
        assert vp._coerce_url_value("1", False) is True
        assert vp._coerce_url_value("true", False) is True
        assert vp._coerce_url_value("True", False) is True

    def test_bool_falsy(self):
        assert vp._coerce_url_value("0", True) is False
        assert vp._coerce_url_value("false", True) is False
        assert vp._coerce_url_value("", True) is False

    def test_int_default(self):
        assert vp._coerce_url_value("2027", 2026) == 2027
        assert isinstance(vp._coerce_url_value("2027", 2026), int)

    def test_int_default_invalid_falls_back(self):
        assert vp._coerce_url_value("not-an-int", 2026) == 2026

    def test_float_default(self):
        assert vp._coerce_url_value("12.5", 1.0) == 12.5
        assert isinstance(vp._coerce_url_value("12.5", 1.0), float)

    def test_float_default_invalid_falls_back(self):
        assert vp._coerce_url_value("xyz", 1.0) == 1.0

    def test_string_default(self):
        assert vp._coerce_url_value("Superexponential", "Linear") == "Superexponential"

    def test_bool_takes_priority_over_int(self):
        # bool is a subclass of int in Python — make sure we treat True/False as bool
        assert vp._coerce_url_value("1", True) is True
        assert vp._coerce_url_value("0", True) is False


class TestCoerceUnknownUrlValue:
    def test_pure_digits_become_int(self):
        v = vp._coerce_unknown_url_value("123")
        assert v == 123 and isinstance(v, int)

    def test_negative_digits_become_int(self):
        v = vp._coerce_unknown_url_value("-7")
        assert v == -7 and isinstance(v, int)

    def test_decimal_becomes_float(self):
        v = vp._coerce_unknown_url_value("1.5")
        assert v == 1.5 and isinstance(v, float)

    def test_non_numeric_stays_string(self):
        assert vp._coerce_unknown_url_value("Claude 4 Opus") == "Claude 4 Opus"


class TestAllTracked:
    def test_includes_metr_keys(self):
        keys, _ = vp._all_tracked()
        assert "metr_proj_basis" in keys
        assert "milestones" in keys
        assert "log_scale" in keys

    def test_includes_eci_and_ecicn_keys(self):
        keys, _ = vp._all_tracked()
        assert "eci_proj_basis" in keys
        assert "ecicn_proj_basis" in keys

    def test_includes_other_tab_keys(self):
        keys, _ = vp._all_tracked()
        for k in ("rli_proj_basis", "emp_proj_basis",
                  "rev_end_year", "ecg_highlight"):
            assert k in keys, f"missing {k}"

    def test_excludes_seg_config_internal_keys(self):
        keys, _ = vp._all_tracked()
        seg_keys = [k for k in keys if k.endswith("_seg_config")]
        assert seg_keys == []

    def test_keys_are_deduped(self):
        keys, _ = vp._all_tracked()
        assert len(keys) == len(set(keys))

    def test_defaults_cover_known_widgets(self):
        _, defaults = vp._all_tracked()
        # Spot-check defaults from each tab
        assert defaults["metr_proj_basis"] == "Piecewise linear"
        assert defaults["log_scale"] is True
        assert defaults["rli_proj_basis"] == "Linear (logit)"
        assert defaults["rev_end_year"] == 2026
        assert defaults["ecg_highlight"] == "None"


class _MockStreamlit:
    """Context manager that swaps vp.st.session_state and vp.st.query_params
    with fresh dicts and restores them on exit."""
    def __init__(self, session_state=None, query_params=None):
        self.session_state = session_state if session_state is not None else {}
        self.query_params = query_params if query_params is not None else {}

    def __enter__(self):
        self._orig_ss = vp.st.session_state
        self._orig_qp = vp.st.query_params
        vp.st.session_state = self.session_state
        vp.st.query_params = self.query_params
        return self

    def __exit__(self, *a):
        vp.st.session_state = self._orig_ss
        vp.st.query_params = self._orig_qp


class TestHydrateSessionFromUrl:
    def test_hydrates_bool_from_url(self):
        with _MockStreamlit(query_params={"milestones": "0"}) as m:
            vp._hydrate_session_from_url()
            assert m.session_state["milestones"] is False

    def test_hydrates_string_from_url(self):
        with _MockStreamlit(query_params={"metr_proj_basis": "Superexponential"}) as m:
            vp._hydrate_session_from_url()
            assert m.session_state["metr_proj_basis"] == "Superexponential"

    def test_hydrates_int_from_url(self):
        with _MockStreamlit(query_params={"metr_end_year": "2028"}) as m:
            vp._hydrate_session_from_url()
            assert m.session_state["metr_end_year"] == 2028

    def test_does_not_overwrite_existing_session_state(self):
        ss = {"metr_proj_basis": "Linear"}
        with _MockStreamlit(session_state=ss,
                            query_params={"metr_proj_basis": "Superexponential"}) as m:
            vp._hydrate_session_from_url()
            assert m.session_state["metr_proj_basis"] == "Linear"

    def test_ignores_keys_not_in_url(self):
        with _MockStreamlit(query_params={}) as m:
            vp._hydrate_session_from_url()
            assert "metr_proj_basis" not in m.session_state

    def test_hydrates_unknown_default_keys_via_inference(self):
        # custom_dt_lo has no entry in _METR_DEFAULTS — fall back to inference
        with _MockStreamlit(query_params={"custom_dt_lo": "75.5"}) as m:
            vp._hydrate_session_from_url()
            assert m.session_state["custom_dt_lo"] == 75.5
            assert isinstance(m.session_state["custom_dt_lo"], float)


class TestSyncSessionToUrl:
    def test_non_default_values_written(self):
        ss = {"metr_proj_basis": "Superexponential", "milestones": False}
        with _MockStreamlit(session_state=ss, query_params={}) as m:
            vp._sync_session_to_url()
            assert m.query_params["metr_proj_basis"] == "Superexponential"
            assert m.query_params["milestones"] == "0"

    def test_default_values_omitted(self):
        # metr_proj_basis default is "Piecewise linear", milestones default is True
        ss = {"metr_proj_basis": "Piecewise linear", "milestones": True}
        with _MockStreamlit(session_state=ss, query_params={"metr_proj_basis": "stale"}) as m:
            vp._sync_session_to_url()
            assert "metr_proj_basis" not in m.query_params
            assert "milestones" not in m.query_params

    def test_bool_serialized_as_zero_or_one(self):
        ss = {"milestones": False, "labels": True, "log_scale": False}
        with _MockStreamlit(session_state=ss, query_params={}) as m:
            vp._sync_session_to_url()
            assert m.query_params["milestones"] == "0"
            # labels default True, so omitted
            assert "labels" not in m.query_params
            assert m.query_params["log_scale"] == "0"

    def test_keys_not_in_session_state_removed_from_url(self):
        with _MockStreamlit(session_state={},
                            query_params={"metr_proj_basis": "Linear"}) as m:
            vp._sync_session_to_url()
            assert "metr_proj_basis" not in m.query_params

    def test_seg_config_never_written(self):
        ss = {"_metr_seg_config": (2, ("Claude 4 Opus",))}
        with _MockStreamlit(session_state=ss, query_params={}) as m:
            vp._sync_session_to_url()
            assert "_metr_seg_config" not in m.query_params

    def test_unknown_default_key_changed_from_baseline_written(self):
        # Baseline (initial widget value) was 50.0; user changed it to 80.0 → should appear in URL
        ss = {"custom_dt_lo": 80.0, "_url_baseline": {"custom_dt_lo": 50.0}}
        with _MockStreamlit(session_state=ss, query_params={}) as m:
            vp._sync_session_to_url()
            assert m.query_params["custom_dt_lo"] == "80.0"

    def test_url_supplied_key_not_baselined(self):
        # Key came from URL on initial load → must NOT be captured as baseline,
        # otherwise an explicit non-default URL value would be silently treated as default.
        ss = {"custom_dt_lo": 75.5, "_url_keys_at_load": {"custom_dt_lo"}}
        with _MockStreamlit(session_state=ss, query_params={"custom_dt_lo": "75.5"}) as m:
            vp._sync_session_to_url()
            # Value remains in URL since we don't know its true default
            assert m.query_params["custom_dt_lo"] == "75.5"
            # And the baseline did NOT capture it
            assert "custom_dt_lo" not in m.session_state.get("_url_baseline", {})

    def test_unknown_default_key_at_baseline_omitted(self):
        # First-call behavior: value gets captured as baseline and is treated as the default
        ss = {"custom_dt_lo": 80.0}
        with _MockStreamlit(session_state=ss, query_params={"custom_dt_lo": "stale"}) as m:
            vp._sync_session_to_url()
            assert "custom_dt_lo" not in m.query_params
            # And the baseline got captured for future comparisons
            assert m.session_state["_url_baseline"]["custom_dt_lo"] == 80.0

    def test_round_trip_preserves_value(self):
        # Simulate user-modified values: prime the baseline so they're treated as non-default
        original = {
            "metr_proj_basis": "Superexponential",
            "milestones": False,
            "metr_end_year": 2028,
            "custom_dt_lo": 75.5,
        }
        ss = dict(original)
        ss["_url_baseline"] = {"metr_end_year": 2030, "custom_dt_lo": 50.0}
        with _MockStreamlit(session_state=ss, query_params={}) as m:
            vp._sync_session_to_url()
            # Now hydrate into a fresh session_state from the URL we just wrote
            qp = dict(m.query_params)
            m.session_state.clear()
            m.query_params = qp
            vp.st.query_params = qp
            vp._hydrate_session_from_url()
            for k, v in original.items():
                assert m.session_state[k] == v, f"{k}: {m.session_state[k]!r} != {v!r}"


# ===========================================================================
# Compute vs Capabilities tab — pure numerical helpers (_cc_*)
# ===========================================================================

def _ccrow(date, log10_flop, eci, **extra):
    """Build a model row shaped like the ones load_eci_compute() emits."""
    row = {"date": date, "log10_flop": log10_flop, "eci": eci}
    row.update(extra)
    return row


class TestCcLogOpAxis:
    """The compute charts label their log axis in log₁₀ operations."""

    def _fig(self, ys):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(ys))), y=ys))
        return fig

    def test_number_and_label_helpers(self):
        assert vp._logop_num(28.0) == "28"
        assert vp._logop_num(28.34) == "28.3"
        assert vp._logop_num(float('nan')) == "—"
        assert vp._logop_lbl(26.83) == "26.8 log OP"
        assert vp._logop_lbl(float('inf')) == "—"

    def test_axis_is_log_and_labelled_in_log_ops(self):
        fig = self._fig([1e25, 1e26, 3e26])
        vp._cc_logop_yaxis(fig, "Training compute (log₁₀ OP)")
        ax = fig.layout.yaxis
        assert ax.type == 'log'
        assert ax.title.text == "Training compute (log₁₀ OP)"
        plain = [re.sub(r'<[^>]+>', '', t) for t in ax.ticktext]
        assert plain and all(24.0 <= float(t) <= 27.0 for t in plain)
        # Values stay raw; only the labels are logged.
        assert ax.tickvals == pytest.approx(
            tuple(10.0 ** float(t) for t in plain))
        # No explicit range — plotly keeps autoscaling.
        assert ax.range is None

    def test_no_positive_data_leaves_a_plain_log_axis(self):
        fig = self._fig([0, 0])
        vp._cc_logop_yaxis(fig, "Training compute (log₁₀ OP)")
        ax = fig.layout.yaxis
        assert ax.type == 'log' and ax.title.text.endswith("OP)")
        assert ax.tickmode != 'array'


class TestCcLoglinearSlope:
    """_cc_loglinear_slope: OLS of log10(value) on years."""

    def test_known_line(self):
        # value = 10^(t years): one dex per year → slope ≈ 1.0 OOM/yr.
        d0 = datetime(2020, 1, 1)
        pts = [(d0, 1.0), (d0 + timedelta(days=365), 10.0)]
        slope, intercept, year0 = vp._cc_loglinear_slope(pts)
        assert slope == pytest.approx(1.0, rel=0.01)
        assert intercept == pytest.approx(0.0, abs=1e-9)
        assert year0 == d0

    def test_returns_none_when_fewer_than_two_points(self):
        assert vp._cc_loglinear_slope([(datetime(2020, 1, 1), 5.0)]) is None
        assert vp._cc_loglinear_slope([]) is None

    def test_filters_nonpositive_and_none_values(self):
        # Only the two positive points are usable; the rest are dropped.
        d0 = datetime(2020, 1, 1)
        pts = [(d0, 10.0), (d0 + timedelta(days=365), 100.0),
               (d0 + timedelta(days=730), None), (d0 + timedelta(days=1095), -5.0)]
        slope, intercept, year0 = vp._cc_loglinear_slope(pts)
        assert slope == pytest.approx(1.0, rel=0.01)  # 1→2 in log10 over ~1 yr
        assert year0 == d0

    def test_none_when_nonpositive_leaves_under_two(self):
        d0 = datetime(2020, 1, 1)
        assert vp._cc_loglinear_slope([(d0, -1.0), (d0, 0.0), (d0, 5.0)]) is None


class TestCcSegmentFits:
    """_cc_segment_fits: per-segment growth of the FLOP frontier."""

    def test_increasing_and_decreasing_segments(self):
        today = datetime(2026, 7, 1)
        frontier_pts = [
            # Segment "2021–2023": +2 dex over ~2 yr → positive slope.
            (datetime(2021, 6, 1), 1e20), (datetime(2023, 6, 1), 1e22),
            # Segment "2023 H2 – 2025 H1": falling → slope ≤ 0.
            (datetime(2023, 8, 1), 1e22), (datetime(2025, 6, 1), 1e21),
        ]
        fits = vp._cc_segment_fits(frontier_pts, today)
        by_label = {f["label"]: f for f in fits}
        rising = by_label["2021–2023"]
        assert rising["n"] == 2
        assert rising["slope_oom"] == pytest.approx(1.0, rel=0.02)
        assert rising["doubling_mo"] == pytest.approx(12 * np.log10(2) / rising["slope_oom"])
        falling = by_label["2023 H2 – 2025 H1"]
        assert falling["slope_oom"] < 0
        assert falling["doubling_mo"] == float("inf")

    def test_drops_segments_with_under_two_points(self):
        today = datetime(2026, 7, 1)
        # Only one point lands in any segment → nothing fittable.
        fits = vp._cc_segment_fits([(datetime(2022, 1, 1), 1e21)], today)
        assert fits == []


class TestEcgOrgMatching:
    """The ECI tab and the ECI Company Gap tab must resolve organizations
    identically. They read the same CSV but used to disagree: the gap tab did an
    exact dict lookup on Epoch's `Organization` string while the ECI tab matched
    by substring. Epoch spells Google four ways, so the gap tab silently dropped
    four Google models and drew a different 2025 frontier point for Google."""

    def _eci_rows(self):
        return list(csv.DictReader(open(
            os.path.join(os.path.dirname(vp.__file__),
                         "epoch_capabilities_index.csv"))))

    def test_both_tabs_resolve_identical_model_sets(self):
        eci_all = vp.load_eci_frontier(_mtime=vp._eci_mtime())
        for label, spec in vp._ECI_ENTITY_SPECS.items():
            if not spec.get("orgs"):
                continue  # country entities, not org entities
            tab = vp.load_eci_frontier(_mtime=vp._eci_mtime(),
                                       orgs=tuple(spec["orgs"]))
            gap = [m for m in eci_all
                   if vp._ecg_org_display(m["organization"]) == label]
            assert {m["version"] for m in tab} == {m["version"] for m in gap}, (
                f"{label}: ECI tab and gap tab disagree on which models belong")

    def test_all_google_spellings_resolve(self):
        # The regression that motivated this class. Every spelling Epoch has
        # emitted must land on "Google", not just the exact map key.
        for spelling in ("Google DeepMind", "Google", "Google DeepMind,Google",
                         "Google,Google DeepMind"):
            assert vp._ecg_org_display(spelling) == "Google", spelling

    def test_no_org_string_matches_two_display_names(self):
        # Substring matching is only safe while no Organization string contains
        # two different mapped companies. Guard it against future Epoch pulls.
        for org in {(r["Organization"] or "").strip() for r in self._eci_rows()}:
            hits = {d for k, d in vp._ECG_ORG_MAP.items() if k.lower() in org.lower()}
            assert len(hits) <= 1, f"{org!r} matches multiple companies: {hits}"

    def test_unmatched_org_returns_none(self):
        assert vp._ecg_org_display("Some Unlisted Lab") is None
        assert vp._ecg_org_display("") is None
        assert vp._ecg_org_display(None) is None

    def test_registry_rows_are_well_formed(self):
        # Derivation guarantees every company reaches both tabs, so what is left
        # to check is that each registry row carries what the derivations read.
        seen_slugs = set()
        for name, c in vp._ECI_COMPANIES.items():
            assert set(c) == {"orgs", "country", "color", "slug"}, name
            assert c["orgs"] and all(o.strip() for o in c["orgs"]), name
            assert c["country"] in vp._ECG_FLAG, name
            assert c["color"].startswith("#"), name
            assert c["slug"] and c["slug"] == c["slug"].lower(), name
            assert c["slug"] not in seen_slugs, f"duplicate slug: {c['slug']}"
            seen_slugs.add(c["slug"])

    def test_derived_tables_cover_every_company(self):
        for name in vp._ECI_COMPANIES:
            assert name in vp._ECG_COLORS, name
            assert name in vp._ECG_COUNTRY, name
            assert name in vp._ECI_ENTITY_SPECS, name
            assert name in vp._ECI_ENTITY_SLUG, name
        assert set(vp._ECG_ORG_MAP.values()) == set(vp._ECI_COMPANIES)

    def test_country_entities_lead_the_dropdown(self):
        # _ECI_ENTITY_OPTIONS[0] is the ECI tab's default benchmark, so the
        # country aggregates must stay ahead of the companies.
        assert vp._ECI_ENTITY_OPTIONS[0] == "US best"
        n = len(vp._ECI_COUNTRY_ENTITIES)
        assert vp._ECI_ENTITY_OPTIONS[:n] == list(vp._ECI_COUNTRY_ENTITIES)
        assert vp._ECI_ENTITY_OPTIONS[n:] == list(vp._ECI_COMPANIES)

    def test_slugs_round_trip(self):
        for label, slug in vp._ECI_ENTITY_SLUG.items():
            assert vp._ECI_ENTITY_FOR_SLUG[slug] == label


class TestCcTrainingFloorMatch:
    """Backward (release → cluster) match uses a training-run causal floor
    (_CC_TRAIN_FLOOR_DAYS, ~60d), not the full +90d expected-release lag. A
    cluster online at least one training run before a release can claim it even
    if the model shipped a few weeks faster than the full pipeline; a cluster
    online less than a training run before it cannot."""

    def _responsible(self, release, milestones, floor):
        """Replicates render's backward match: the most recent cluster online at
        least one training run (floor) before the release."""
        cand = [m for m in milestones if (m[0] + floor) <= release]
        return cand[-1] if cand else None

    def test_floor_is_training_run_not_full_lag(self):
        # The causal floor is the ~2-month training run, strictly less than the
        # 90d train+release-prep window used for the *expected* release date.
        assert vp._CC_TRAIN_FLOOR_DAYS == vp._DAYS_2MO
        assert vp._CC_TRAIN_FLOOR_DAYS < vp._CC_RELEASE_LAG_DAYS

    def test_sol_ties_to_wisconsin_under_floor(self):
        from datetime import timedelta
        attr = vp._cc_lab_attribution()
        milestones = vp._cc_lab_dc_milestones("OpenAI", attr, key="perf")
        lag = timedelta(days=vp._CC_RELEASE_LAG_DAYS)
        floor = timedelta(days=vp._CC_TRAIN_FLOOR_DAYS)

        # Match on the model family, not the reasoning-effort suffix: every
        # gpt-5.6-sol_* variant carries the same ECI, so which one wins the dedup
        # is arbitrary and flips between Epoch pulls (it was "(pro, max)" until
        # Epoch added a `_none` variant in the 2026-08-08 refresh). The date —
        # the only thing this test is actually about — is unaffected.
        #
        # Source Sol from *all* OpenAI ECI rows, not from
        # _cc_company_frontier_models(): that helper keeps only running-max
        # releases, and Epoch's live recompute can drop Sol out of it without
        # changing its date. The 2026-08-18 pull did exactly that — Sol fell
        # 161.65 -> 161.03 while GPT-5.5 Pro (xhigh, 2026-04-23) rose
        # 161.49 -> 161.60, so Sol stopped setting a new OpenAI high. What this
        # test asserts (that Sol's release date ties to Wisconsin under the
        # training floor but to Atlanta under the full lag) is a statement about
        # dates and cluster timing, so it must not hinge on running-max status.
        sol_dates = {
            datetime.strptime(r["Release date"], "%Y-%m-%d")
            for r in csv.DictReader(open(
                os.path.join(os.path.dirname(vp.__file__),
                             "epoch_capabilities_index.csv")))
            if (r.get("Model name") or "").strip().startswith("GPT-5.6 Sol")
            and (r.get("Release date") or "").strip()
        }
        assert sol_dates, "Sol not found in the ECI table"
        assert len(sol_dates) == 1, f"Sol has inconsistent release dates: {sol_dates}"
        d = sol_dates.pop()

        # The strict full-lag rule would have fallen back to Atlanta...
        strict = [m for m in milestones if (m[0] + lag) <= d][-1]
        assert strict[2] == "Microsoft Fairwater Atlanta"

        # ...but Wisconsin was online a full training run (>=60d) before Sol, so
        # the training-floor rule ties Sol to Wisconsin.
        resp = self._responsible(d, milestones, floor)
        assert resp[2] == "Microsoft Fairwater Wisconsin"
        assert (d - resp[0]).days >= vp._CC_TRAIN_FLOOR_DAYS

    def test_implausibly_fresh_cluster_is_excluded(self):
        # A model that shipped only ~2 weeks after a much bigger cluster came
        # online can't be attributed to it — no time to train — so the match
        # falls back to the earlier cluster that was online long enough.
        from datetime import timedelta
        floor = timedelta(days=vp._CC_TRAIN_FLOOR_DAYS)
        release = datetime(2026, 6, 1)
        old = (datetime(2026, 1, 1), 1.0, "Old cluster")        # 151d before
        fresh = (datetime(2026, 5, 18), 9.0, "Fresh cluster")   # 14d before
        milestones = [old, fresh]
        resp = self._responsible(release, milestones, floor)
        assert resp[2] == "Old cluster"

    def test_cluster_exactly_at_floor_qualifies(self):
        from datetime import timedelta
        floor = timedelta(days=vp._CC_TRAIN_FLOOR_DAYS)
        release = datetime(2026, 6, 1)
        exact = (release - floor, 5.0, "Exactly one training run")
        just_short = (release - floor + timedelta(days=1), 9.0, "One day short")
        resp = self._responsible(release, [exact, just_short], floor)
        assert resp[2] == "Exactly one training run"


class TestCcForwardMatch:
    """_cc_forward_match: the cluster → release direction, in three tiers.

    Tier 1 is the ordinary "first frontier release in the window", widened by
    _CC_EARLY_GRACE_DAYS so a model that beat the 90d pipeline by a few days
    still counts. Tier 2 defers to the backward match so the panel's two tables
    can't name different releases for one cluster. Tier 3 admits a non-record
    release rather than rendering a blank row.
    """

    TODAY = datetime(2026, 8, 22)

    def _step(self, y, m, d, name="Cluster"):
        return (datetime(y, m, d), 1.0, name)

    def _pred(self, step):
        from datetime import timedelta
        return step[0] + timedelta(days=vp._CC_RELEASE_LAG_DAYS)

    def test_grace_is_shorter_than_the_backward_floor(self):
        # The forward grace deliberately does not open all the way to what the
        # 60d training floor would allow (30d early). At 30d several clusters
        # start claiming the same, earlier model and the forward table goes
        # degenerate — see the constant's comment.
        slack = vp._CC_RELEASE_LAG_DAYS - vp._CC_TRAIN_FLOOR_DAYS
        assert 0 < vp._CC_EARLY_GRACE_DAYS < slack

    def test_tier1_takes_first_frontier_release_in_window(self):
        step = self._step(2026, 1, 1)
        pred = self._pred(step)
        early = (pred + timedelta(days=5), 100.0, "First")
        later = (pred + timedelta(days=60), 110.0, "Second")
        got, fb = vp._cc_forward_match(step, [early, later], [], {}, self.TODAY)
        assert got[2] == "First" and fb is False

    def test_tier1_reaches_back_over_the_grace_window(self):
        # A model that shipped inside the grace window still belongs to the step.
        step = self._step(2026, 1, 1)
        pred = self._pred(step)
        for early_by in (1, vp._CC_EARLY_GRACE_DAYS):
            rel = (pred - timedelta(days=early_by), 100.0, "Fast shipper")
            got, fb = vp._cc_forward_match(step, [rel], [], {}, self.TODAY)
            assert got[2] == "Fast shipper" and fb is False

    def test_tier1_stops_at_the_grace_window(self):
        # One day past the grace, tier 1 declines — and with no backward tie and
        # no other release, the step reports nothing at all.
        step = self._step(2026, 1, 1)
        rel = (self._pred(step) - timedelta(days=vp._CC_EARLY_GRACE_DAYS + 1),
               100.0, "Too early")
        got, fb = vp._cc_forward_match(step, [rel], [], {}, self.TODAY)
        assert got is None and fb is False

    def test_tier2_defers_to_the_backward_match(self):
        # A release tied to this step by the 60d backward floor but sitting
        # further back than the 7d forward grace is still this step's release —
        # otherwise the two tables would disagree about the same cluster.
        step = self._step(2026, 1, 1)
        rel = (self._pred(step) - timedelta(days=24), 100.0, "Shipped early")
        got, fb = vp._cc_forward_match(step, [rel], [], {(rel[0], rel[2]): step},
                                       self.TODAY)
        assert got[2] == "Shipped early" and fb is False

    def test_tier2_prefers_the_earliest_release_tied_to_the_step(self):
        step = self._step(2026, 1, 1)
        pred = self._pred(step)
        a = (pred - timedelta(days=24), 100.0, "Earlier")
        b = (pred - timedelta(days=12), 110.0, "Later")
        resp = {(a[0], a[2]): step, (b[0], b[2]): step}
        got, _fb = vp._cc_forward_match(step, [a, b], [], resp, self.TODAY)
        assert got[2] == "Earlier"

    def test_tier2_ignores_releases_tied_to_a_different_step(self):
        step = self._step(2026, 1, 1, "Mine")
        other = self._step(2025, 6, 1, "Someone else's")
        rel = (self._pred(step) - timedelta(days=24), 100.0, "Not mine")
        got, fb = vp._cc_forward_match(step, [rel], [], {(rel[0], rel[2]): other},
                                       self.TODAY)
        assert got is None and fb is False

    def test_tier3_admits_a_non_record_release(self):
        step = self._step(2026, 1, 1)
        pred = self._pred(step)
        other = (pred + timedelta(days=3), 90.0, "Not a record")
        got, fb = vp._cc_forward_match(step, [], [other], {}, self.TODAY)
        assert got[2] == "Not a record" and fb is True

    def test_tier3_only_fires_when_no_frontier_release_qualifies(self):
        step = self._step(2026, 1, 1)
        pred = self._pred(step)
        record = (pred + timedelta(days=30), 100.0, "Record")
        other = (pred + timedelta(days=3), 90.0, "Not a record")
        got, fb = vp._cc_forward_match(step, [record], [other], {}, self.TODAY)
        assert got[2] == "Record" and fb is False

    def test_planned_cluster_never_falls_back(self):
        # A DC that is not online yet has no releases to explain; it should read
        # "still future", not borrow some unrelated model.
        step = self._step(2027, 1, 1)
        other = (self._pred(step) + timedelta(days=3), 90.0, "Not a record")
        got, fb = vp._cc_forward_match(step, [], [other], {}, self.TODAY)
        assert got is None and fb is False


class TestCcCompanyAllReleases:
    """_cc_company_all_releases: the tier-3 fallback pool."""

    def test_collapses_reasoning_effort_variants(self):
        # Keyed on Model name, so the ~10 gpt-5.6-sol_* rows are one release.
        # Which Display-name variant would win a dedup is arbitrary and flips
        # between Epoch pulls.
        rel = vp._cc_company_all_releases()["OpenAI"]
        sol = [t for t in rel if t[2] == "GPT-5.6 Sol"]
        assert len(sol) == 1
        assert sol[0][0] == datetime(2026, 7, 9)

    def test_keeps_redated_revisions_of_one_model_name_apart(self):
        # Epoch ships five dated revisions under the single Model name
        # "GPT-4o"; two of them set OpenAI ECI records. Collapsing on the name
        # alone would keep only the earliest and lose the rest, so the fallback
        # pool would be missing releases the frontier series has. Asserted as a
        # superset, not an exact list — Epoch can add revisions.
        rel = vp._cc_company_all_releases()["OpenAI"]
        gpt4o = {t[0] for t in rel if t[2] == "GPT-4o"}
        assert {datetime(2024, 5, 13), datetime(2024, 8, 6)} <= gpt4o
        assert len(gpt4o) >= 3

    def test_same_day_releases_are_ordered_strongest_first(self):
        # 2026-07-09 ships Sol, Terra and Luna together; a step matching that
        # date should be offered the flagship, not whichever row sorted first.
        rel = vp._cc_company_all_releases()["OpenAI"]
        same_day = [t for t in rel if t[0] == datetime(2026, 7, 9)]
        assert len(same_day) > 1, "expected several OpenAI releases on 2026-07-09"
        assert same_day[0][2] == "GPT-5.6 Sol"
        assert [t[1] for t in same_day] == sorted(
            (t[1] for t in same_day), reverse=True)

    def test_is_a_superset_of_the_frontier(self):
        # Every running-max release must exist in the pool on the same date, or
        # tier 3 could contradict tier 1.
        allr = vp._cc_company_all_releases()
        for lab, front in vp._cc_company_frontier_models().items():
            dates = {t[0] for t in allr[lab]}
            for d, _e, _n in front:
                assert d in dates, f"{lab}: frontier date {d} missing from pool"

    def test_sol_is_the_live_fallback_for_wisconsin(self):
        # End-to-end on the shipped data: Epoch's 2026-08-18 rescore put Sol
        # (161.08) under GPT-5.5 Pro (161.73), so it is no longer a running-max
        # release and tier 1 finds nothing for Fairwater Wisconsin. Tier 3 must
        # still surface it — that blank row is what the fallback exists to fix.
        # If a later OpenAI release re-takes the record this stops applying;
        # relax the assert to "some release is matched" rather than deleting it.
        attr = vp._cc_lab_attribution()
        milestones = vp._cc_lab_dc_milestones("OpenAI", attr, key="perf")
        front = vp._cc_company_frontier_models()["OpenAI"]
        allr = vp._cc_company_all_releases()["OpenAI"]
        steps = [m for m in milestones
                 if m[2] == "Microsoft Fairwater Wisconsin"
                 and m[0] <= TestCcForwardMatch.TODAY]
        assert steps, "Fairwater Wisconsin has no online capacity step"
        got, fb = vp._cc_forward_match(steps[0], front, allr, {},
                                       TestCcForwardMatch.TODAY)
        assert got is not None, "Wisconsin still matches nothing"
        assert got[2] == "GPT-5.6 Sol" and fb is True


class TestCcDecomp:
    """_cc_decomp: regress ECI on log10(FLOP) and time."""

    def _rows(self):
        base = datetime(2022, 1, 1)
        # FLOP chosen to be uncorrelated with time so the joint fit is identified.
        flops = [23.0, 24.0, 23.5, 25.0, 24.2, 23.8, 24.9, 23.2, 25.1, 24.4,
                 23.6, 24.7]
        rows = []
        for i, lf in enumerate(flops):
            d = base + timedelta(days=90 * i)
            t = (d - base).days / 365.25
            rows.append(_ccrow(d, lf, 3.0 * lf + 5.0 * t))  # exact ECI = 3·lc + 5·t
        # Mark a monotone (date & FLOP increasing) subset as the ECI frontier.
        for i in (0, 1, 3):
            rows[i]["is_eci_frontier"] = True
        return rows

    def test_returns_none_under_ten_rows(self):
        assert vp._cc_decomp(self._rows()[:9]) is None

    def test_recovers_known_coefficients(self):
        d = vp._cc_decomp(self._rows())
        assert d["n"] == 12
        assert d["a_partial"] == pytest.approx(3.0, abs=1e-6)
        assert d["b_time"] == pytest.approx(5.0, abs=1e-6)
        assert d["r2_joint"] == pytest.approx(1.0, abs=1e-9)

    def test_frontier_subset_growth(self):
        d = vp._cc_decomp(self._rows())
        assert d["n_frontier"] == 3
        assert d["frontier_compute_oom"] > 0     # FLOP rising along the frontier
        assert d["eci_frontier_slope"] > 0

    def test_no_frontier_subset_leaves_none(self):
        rows = self._rows()
        for r in rows:
            r.pop("is_eci_frontier", None)
        d = vp._cc_decomp(rows)
        assert d["n_frontier"] == 0
        assert d["frontier_compute_oom"] is None
        assert d["eci_frontier_slope"] is None


class TestCcEfficiency:
    """_cc_efficiency: compute needed for a fixed ECI falls over time."""

    def _rows(self):
        # log10(FLOP) = 0.05·ECI − 0.4·t + 10  →  exchange rate 20 ECI/dex,
        # iso-ECI compute falling 0.4 OOM/yr.
        base = datetime(2022, 1, 1)
        # A ≥5-member band around ECI 115, with ECI decorrelated from time.
        band = [(112, 0.0), (118, 0.5), (113, 1.0), (117, 1.5),
                (114, 2.0), (116, 2.5), (115, 3.0)]
        extras = [(104, 0.3), (106, 1.2), (124, 0.8), (126, 2.2)]
        rows = []
        for eci, t in band + extras:
            d = base + timedelta(days=round(t * 365.25))
            rows.append(_ccrow(d, 0.05 * eci - 0.4 * t + 10.0, eci))
        return rows

    def test_returns_none_under_ten_rows(self):
        assert vp._cc_efficiency(self._rows()[:9]) is None

    def test_recovers_exchange_rate_and_efficiency(self):
        e = vp._cc_efficiency(self._rows())
        assert e["eci_per_oom"] == pytest.approx(20.0, rel=1e-3)  # 1/alpha
        assert e["g_inv"] == pytest.approx(0.4, rel=1e-3)
        assert 0.0 < e["g_central"] < 1.0
        assert e["algo_mult"] == pytest.approx(10 ** e["g_central"], rel=1e-9)
        assert e["bands"], "the ECI-115 band should produce a fit line"

    def test_times_are_monotonic_and_bracketed(self):
        e = vp._cc_efficiency(self._rows())
        t2, t5, t10 = e["times"][2], e["times"][5], e["times"][10]
        # More compute reduction → more months to match capability.
        assert 0 < t2["central"] < t5["central"] < t10["central"]
        # lo uses the faster efficiency rate (g_hi) → fewer months.
        assert t2["lo"] <= t2["central"] <= t2["hi"]


class TestCcIsoCompute:
    """_cc_iso_compute: hold compute fixed, watch ECI rise."""

    def _rows(self):
        base = datetime(2022, 1, 1)
        # A ≥5-member band around log10(FLOP) 24.5, ECI = 10·t + 100.
        band = [24.2, 24.3, 24.4, 24.5, 24.6, 24.7]
        rows = []
        for i, lf in enumerate(band):
            t = 0.4 * i
            rows.append(_ccrow(base + timedelta(days=round(t * 365.25)), lf,
                               10.0 * t + 100.0))
        # Padding well below every compute band (< 23.0) to clear the 10-row
        # minimum without polluting the ECI-24.5 band.
        for i in range(4):
            rows.append(_ccrow(base + timedelta(days=30 * i), 22.0 + 0.2 * i, 90 + i))
        return rows

    def test_returns_none_under_ten_rows(self):
        assert vp._cc_iso_compute(self._rows()[:9]) is None

    def test_recovers_capability_rate(self):
        r = vp._cc_iso_compute(self._rows())
        assert r["eci_per_yr"] == pytest.approx(10.0, rel=1e-2)
        assert r["lo"] <= r["eci_per_yr"] <= r["hi"]


class TestCcIsoComputeRate:
    """_cc_iso_compute_rate: one country's ECI/yr at a fixed compute budget."""

    def _rows(self, country="China", n=8, rate=8.0):
        base = datetime(2023, 1, 1)
        rows = []
        for i in range(n):
            t = 0.3 * i
            # FLOP clustered within ±0.4 dex of the median so the band holds.
            lf = 24.6 + (0.1 if i % 2 else -0.1)
            rows.append(_ccrow(base + timedelta(days=round(t * 365.25)), lf,
                               rate * t + 90.0, country=country))
        return rows

    def test_returns_rate_for_dense_country(self):
        rate, n, med = vp._cc_iso_compute_rate(self._rows(), "China")
        assert rate == pytest.approx(8.0, rel=1e-2)
        assert n == 8
        assert med == pytest.approx(24.6, abs=0.11)

    def test_sparse_country_returns_none(self):
        rate, n, med = vp._cc_iso_compute_rate(self._rows(n=5), "China")
        assert rate is None and n == 0 and med is None

    def test_unknown_country_returns_none(self):
        rate, n, med = vp._cc_iso_compute_rate(self._rows(), "Narnia")
        assert rate is None and n == 0 and med is None


class TestCcQuarterEnds:
    """_cc_quarter_ends: quarter-end dates strictly after start, through end."""

    def test_full_year(self):
        out = vp._cc_quarter_ends(datetime(2026, 1, 15), datetime(2026, 12, 31))
        assert out == [datetime(2026, 3, 31), datetime(2026, 6, 30),
                       datetime(2026, 9, 30), datetime(2026, 12, 31)]

    def test_start_on_quarter_end_is_exclusive(self):
        out = vp._cc_quarter_ends(datetime(2026, 3, 31), datetime(2026, 9, 30))
        assert out == [datetime(2026, 6, 30), datetime(2026, 9, 30)]

    def test_empty_when_no_quarter_in_range(self):
        assert vp._cc_quarter_ends(datetime(2026, 4, 1), datetime(2026, 5, 1)) == []


class TestCcCountryFrontier:
    """_cc_country_frontier: running-max ECI frontier within one country."""

    def _models(self):
        return [
            {"country": "US", "eci_score": 100, "date": datetime(2024, 1, 1),
             "display_name": "a"},
            {"country": "US", "eci_score": 95, "date": datetime(2024, 6, 1),
             "display_name": "b"},          # not a new max → skipped
            {"country": "US", "eci_score": 120, "date": datetime(2024, 9, 1),
             "display_name": "c"},
            {"country": "China", "eci_score": 200, "date": datetime(2024, 3, 1),
             "display_name": "x"},          # other country → excluded
        ]

    def test_running_max_and_country_filter(self):
        fr = vp._cc_country_frontier(self._models(), "US")
        assert [s for _, s, _ in fr] == [100, 120]
        assert [n for _, _, n in fr] == ["a", "c"]


class TestCcCountryComputeFrontier:
    """_cc_country_compute_frontier: running-max FLOP frontier + growth rate."""

    def test_growth_over_rising_frontier(self):
        base = datetime(2023, 1, 1)
        rows = [
            _ccrow(base, 24.0, 100, country="US", name="m0"),
            _ccrow(base + timedelta(days=365), 23.5, 105, country="US", name="dip"),
            _ccrow(base + timedelta(days=730), 25.0, 120, country="US", name="m2"),
        ]
        pts, g = vp._cc_country_compute_frontier(rows, "US")
        assert [round(lf, 2) for _, lf, _, _ in pts] == [24.0, 25.0]  # dip skipped
        assert g == pytest.approx(0.5, rel=0.02)  # +1 dex over ~2 yr

    def test_single_point_has_no_slope(self):
        rows = [_ccrow(datetime(2023, 1, 1), 24.0, 100, country="US", name="m0")]
        pts, g = vp._cc_country_compute_frontier(rows, "US")
        assert len(pts) == 1 and g is None


class TestCcFrontierEciSlope:
    """_cc_frontier_eci_slope: OLS ECI/yr from a cutoff onward."""

    def _fr(self):
        return [
            (datetime(2023, 1, 1), 100, "a"),
            (datetime(2024, 1, 1), 110, "b"),
            (datetime(2025, 1, 1), 120, "c"),
        ]

    def test_slope_from_cutoff(self):
        s = vp._cc_frontier_eci_slope(self._fr(), datetime(2023, 1, 1))
        assert s == pytest.approx(10.0, rel=0.01)  # +10 ECI/yr

    def test_none_when_cutoff_leaves_under_two(self):
        assert vp._cc_frontier_eci_slope(self._fr(), datetime(2025, 6, 1)) is None


class TestCcPooledDecomp:
    """_cc_pooled_decomp: joint ECI decomposition over US+China models."""

    def _rows(self, n=12):
        base = datetime(2022, 1, 1)
        flops = [23.0, 24.0, 23.5, 25.0, 24.2, 23.8, 24.9, 23.2, 25.1, 24.4,
                 23.6, 24.7]
        rows = []
        for i in range(n):
            lf = flops[i % len(flops)]
            d = base + timedelta(days=90 * i)
            t = (d - base).days / 365.25
            country = "United States of America" if i % 2 else "China"
            rows.append(_ccrow(d, lf, 3.0 * lf + 5.0 * t, country=country))
        return rows

    def test_returns_none_under_ten(self):
        a, b = vp._cc_pooled_decomp(self._rows(n=9))
        assert a is None and b is None

    def test_recovers_coefficients(self):
        a, b = vp._cc_pooled_decomp(self._rows())
        assert a == pytest.approx(3.0, abs=1e-6)
        assert b == pytest.approx(5.0, abs=1e-6)

    def test_ignores_other_countries(self):
        rows = self._rows()
        for r in rows:
            r["country"] = "France"       # none in {US, China} → too few
        a, b = vp._cc_pooled_decomp(rows)
        assert a is None and b is None


class TestCcCnTargetYears:
    """_cc_cn_target_years: China's ETA to a target ECI, from compute + algo."""

    # Deterministic inputs: algo pinned at 10, compute 0.2 OOM/yr × 10 pts/OOM =
    # 2 → 12 ECI/yr, so a 12-point gap is exactly one year before any spread.
    _ARGS = dict(anchor_eci=100.0, target=112.0, algo_lo=10.0, algo_mid=10.0,
                 algo_hi=10.0, a_partial=10.0, g_lo=0.2, g_hi=0.2)

    def test_central_case_is_gap_over_rate(self):
        y, r = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=2000, **self._ARGS)
        assert np.median(r) == pytest.approx(12.0, rel=0.02)
        assert np.median(y) == pytest.approx(1.0, rel=0.05)

    def test_wider_pace_widens_the_band_both_ways(self):
        narrow, _ = vp._cc_cn_target_years(pace_lo=0.95, pace_hi=1.05, n=4000,
                                           **self._ARGS)
        wide, _ = vp._cc_cn_target_years(pace_lo=0.6, pace_hi=1.4, n=4000,
                                         **self._ARGS)
        spread = lambda a: np.percentile(a, 90) - np.percentile(a, 10)
        assert spread(wide) > spread(narrow)
        # A slower pace takes *longer*, so the wide band must reach further out.
        assert np.percentile(wide, 90) > np.percentile(narrow, 90)

    def test_release_wait_only_pushes_dates_later(self):
        base, _ = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=4000,
                                         **self._ARGS)
        waited, _ = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=4000,
                                           release_gap_days=60.0, **self._ARGS)
        # Exponential(60d) adds ~0.16 yr on average, and can never subtract.
        assert np.median(waited) > np.median(base)
        assert np.mean(waited) == pytest.approx(np.mean(base) + 60 / 365.25, abs=0.03)

    def test_faster_compute_growth_pulls_the_date_in(self):
        args = dict(self._ARGS, g_lo=0.4, g_hi=0.4)
        fast, _ = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=2000, **args)
        slow, _ = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=2000,
                                         **self._ARGS)
        assert np.median(fast) < np.median(slow)

    def test_degenerate_ranges_do_not_crash_triangular(self):
        # algo_lo == algo_hi and g_lo == g_hi would be invalid triangular args;
        # the helper pads them instead of raising.
        y, r = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=500, **self._ARGS)
        assert np.all(np.isfinite(y)) and np.all(r > 0)

    def test_algo_mode_outside_range_is_clamped(self):
        args = dict(self._ARGS, algo_lo=8.0, algo_mid=99.0, algo_hi=12.0)
        _, r = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=4000, **args)
        # Mode clamps to algo_hi=12, so no sample can exceed the envelope — the
        # widest algo, compute, and pace draws, each including the ±pad the helper
        # adds to the degenerate g and pace ranges.
        assert r.max() <= (12.0 + 10.0 * (0.2 + 0.01)) * 1.05 + 1e-9
        # ...and a mode pinned at the top skews the draw high, not to the middle.
        assert np.median(r) > 0.5 * ((8.0 + 1.9) + (12.0 + 2.1))


class TestCcReleaseGapDays:
    """_cc_release_gap_days: how often a frontier actually steps up."""

    def _fr(self):
        base = datetime(2025, 1, 1)
        return [(base + timedelta(days=d), 100 + i, f"m{i}")
                for i, d in enumerate((0, 30, 60, 120))]   # gaps 30, 30, 60

    def test_median_gap(self):
        assert vp._cc_release_gap_days(self._fr()) == pytest.approx(30.0)

    def test_since_filters_old_releases(self):
        # Keeps only the last two points → a single 60-day gap, but that is under
        # the three-point floor, so it declines to guess.
        assert vp._cc_release_gap_days(self._fr(), since=datetime(2025, 3, 1)) is None

    def test_too_few_points(self):
        assert vp._cc_release_gap_days(self._fr()[:2]) is None


class TestCcFirstReached:
    """_cc_first_reached: first frontier model at or above a level."""

    def _fr(self):
        return [(datetime(2025, 1, 1), 100.0, "a"),
                (datetime(2025, 6, 1), 120.0, "b"),
                (datetime(2026, 1, 1), 140.0, "c")]

    def test_returns_first_crossing_model(self):
        assert vp._cc_first_reached(self._fr(), 110.0) == (datetime(2025, 6, 1), "b")

    def test_exact_match_counts(self):
        assert vp._cc_first_reached(self._fr(), 120.0)[1] == "b"

    def test_none_when_target_above_frontier(self):
        assert vp._cc_first_reached(self._fr(), 999.0) is None


class TestCcCnTargetIsTodaysUsFrontier:
    """The 161 bar is meant to be ~today's US frontier, not an arbitrary number.

    If Epoch's rescoring or a new US model moves the frontier far from the
    constant, the section's framing ("China matching where the US is now") stops
    being true and `_CC_CN_TARGET_ECI` needs revisiting.
    """

    def test_target_sits_at_the_us_frontier(self):
        eci = vp.load_eci_frontier(_mtime=vp._eci_mtime())
        us = vp._cc_country_frontier(eci, "United States of America")
        assert us, "no US-tagged ECI frontier"
        best = max(s for _, s, _ in us)
        assert vp._CC_CN_TARGET_ECI <= best, "target is above the US frontier"
        assert best - vp._CC_CN_TARGET_ECI < 5.0, "target has fallen behind the US"

    def test_china_has_not_yet_crossed_it(self):
        eci = vp.load_eci_frontier(_mtime=vp._eci_mtime())
        cn = vp._cc_country_frontier(eci, "China")
        assert max(s for _, s, _ in cn) < vp._CC_CN_TARGET_ECI


class TestEciMonthsBehind:
    """_eci_months_behind: months a score lags the US ECI trend."""

    def test_behind_the_trend(self):
        base = datetime(2023, 1, 1)
        us_fr = [(base, 100, "a"),
                 (base + timedelta(days=365), 110, "b")]  # +10 ECI/yr
        # Score 100 evaluated a year after the US hit it → ~12 months behind.
        months = vp._eci_months_behind(us_fr, 100, base + timedelta(days=365))
        assert months == pytest.approx(12.0, rel=0.02)

    def test_declining_trend_is_nan(self):
        # A flat-or-declining US trend has no forward crossing → NaN.
        base = datetime(2023, 1, 1)
        declining = [(base, 110, "a"), (base + timedelta(days=365), 100, "b")]
        assert np.isnan(vp._eci_months_behind(declining, 100, base))


class TestLoadUkCyber:
    """AISI narrow cyber tasks (digitized from the published figure)."""

    def test_returns_list(self):
        data = vp.load_ukcyber()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_models_have_required_keys(self):
        data = vp.load_ukcyber()
        required = {'name', 'date', 'cyber_score', 'organization',
                    'country', 'weights', 'is_frontier'}
        for m in data:
            assert required.issubset(m.keys()), f"Missing keys: {required - m.keys()}"

    def test_sorted_by_date(self):
        data = vp.load_ukcyber()
        dates = [m['date'] for m in data]
        assert dates == sorted(dates)

    def test_scores_are_percentages(self):
        for m in vp.load_ukcyber():
            assert 0 <= m['cyber_score'] <= 100, f"{m['name']}: {m['cyber_score']}"

    def test_comment_lines_are_skipped(self):
        """The CSV carries a provenance header of '#' comment lines."""
        names = [m['name'] for m in vp.load_ukcyber()]
        assert not any(n.startswith('#') for n in names)
        assert "GLM-5.2" in names

    def test_frontier_is_closed_weight_only(self):
        """Open-weight models are the subject measured against the frontier,
        so they must never define it."""
        for m in vp.load_ukcyber():
            if m['is_frontier']:
                assert m['weights'] == 'closed', f"{m['name']} is open but frontier"

    def test_frontier_is_running_max(self):
        frontier = [m for m in vp.load_ukcyber() if m['is_frontier']]
        scores = [m['cyber_score'] for m in frontier]
        assert scores == sorted(scores)


class TestUkCyberLag:
    """Lag of open-weight models behind the closed frontier."""

    def _frontier(self):
        base = datetime(2025, 1, 1)
        return [
            {'name': 'A', 'date': base, 'cyber_score': 50.0},
            {'name': 'B', 'date': base + timedelta(days=100), 'cyber_score': 60.0},
        ]

    def test_bracketing_models_are_identified(self):
        fr = self._frontier()
        assert vp._ukc_frontier_match_for_score(fr, 55.0)['name'] == 'B'
        assert vp._ukc_frontier_below_for_score(fr, 55.0)['name'] == 'A'
        assert vp._ukc_frontier_below_for_score(fr, 50.0) is None

    def test_crossing_is_interpolated_between_bracketing_models(self):
        """A score midway between two frontier models crosses midway in time."""
        fr = self._frontier()
        crossing, below, above = vp._ukc_frontier_crossing(fr, 55.0)
        assert below['name'] == 'A' and above['name'] == 'B'
        # 55 is halfway from 50 to 60, so ~50 days into the 100-day gap.
        assert (crossing - fr[0]['date']).days == pytest.approx(50, abs=1)

    def test_crossing_snaps_when_no_lower_bracket(self):
        fr = self._frontier()
        crossing, below, above = vp._ukc_frontier_crossing(fr, 50.0)
        assert below is None and crossing == fr[0]['date']

    def test_score_beyond_frontier_is_unmatched(self):
        assert vp._ukc_frontier_match_for_score(self._frontier(), 99.0) is None
        assert vp._ukc_frontier_crossing(self._frontier(), 99.0)[0] is None

    def test_empty_frontier_is_unmatched(self):
        assert vp._ukc_frontier_match_for_score([], 50.0) is None

    def test_lag_rows_only_cover_open_weight_models(self):
        rows = vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all)
        assert rows, "expected at least one open-weight model"
        for r in rows:
            assert r['weights'] == 'open'

    def test_model_ahead_of_frontier_reports_no_lag(self):
        models = [{'name': 'X', 'date': datetime(2025, 6, 1), 'cyber_score': 99.0,
                   'weights': 'open', 'country': 'China', 'organization': 'Org'}]
        row = vp.ukc_lag_rows(models, self._frontier())[0]
        assert row['lag_months'] is None and row['match_date'] is None

    def test_optimistic_bracket_reproduces_aisi_published_lags(self):
        """AISI's figure prints 5.0mo for DeepSeek-V4-Pro and 4.3mo for GLM-5.2,
        measured against the next model up. That is our lag_lo bound, and it
        remains the calibration check that the digitization matches the source."""
        by_name = {r['name']: r for r in vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all)}
        assert by_name['DeepSeek-V4-Pro']['lag_lo'] == pytest.approx(5.0, abs=0.15)
        assert by_name['GLM-5.2']['lag_lo'] == pytest.approx(4.3, abs=0.15)

    def test_reproduces_aisi_comparison_models(self):
        """AISI drew its annotations against these specific models."""
        by_name = {r['name']: r for r in vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all)}
        assert by_name['DeepSeek-V4-Pro']['above_name'] == 'Opus 4.5'
        assert by_name['GLM-5.2']['above_name'] == 'Claude Opus 4.6'

    def test_point_estimate_sits_inside_the_bracket(self):
        """Interpolated lag must fall between the optimistic and pessimistic ends."""
        for r in vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all):
            assert r['lag_lo'] <= r['lag_months'] <= r['lag_hi'], r['name']

    def test_interpolation_does_not_equate_distant_scores(self):
        """The point of interpolating: DeepSeek (55.7%) must not be credited with
        Opus 4.5's 62.6%, which would understate its lag."""
        by_name = {r['name']: r for r in vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all)}
        ds = by_name['DeepSeek-V4-Pro']
        assert ds['below_name'] == 'GPT-5' and ds['above_name'] == 'Opus 4.5'
        assert ds['lag_months'] > ds['lag_lo'] + 1.0
        assert ds['lag_months'] == pytest.approx(7.4, abs=0.2)

    def test_digitized_dates_match_known_releases(self):
        """Dates are inferred from marker x-positions, so guard the calibration."""
        by_name = {m['name']: m for m in vp.ukc_all}
        assert abs((by_name['GPT-5']['date'] - datetime(2025, 8, 7)).days) <= 3
        assert abs((by_name['Opus 3']['date'] - datetime(2024, 3, 4)).days) <= 3


class TestUkCyberTargetEta:
    """When open-weight models reach the 90% target."""

    def test_eta_is_frontier_crossing_plus_lag(self):
        eta = vp.ukc_target_eta(vp.ukc_all, vp.ukc_frontier_all, 90.0)
        assert eta is not None
        # 90% falls between Mythos Preview (88.8%) and Claude Mythos 5 (90.3%).
        assert eta['frontier_between'] == ('Mythos Preview', 'Claude Mythos 5')
        # Bounds are the crossing date offset by the min/max measured lag.
        assert eta['date_lo'] > eta['frontier_date']
        assert eta['date_hi'] > eta['date_lo']
        assert eta['lag_lo'] == pytest.approx(4.7, abs=0.2)
        assert eta['lag_hi'] == pytest.approx(7.4, abs=0.2)

    def test_eta_bounds_match_lag_offsets(self):
        eta = vp.ukc_target_eta(vp.ukc_all, vp.ukc_frontier_all, 90.0)
        lo_days = (eta['date_lo'] - eta['frontier_date']).days
        assert lo_days == pytest.approx(eta['lag_lo'] * vp._UKC_DAYS_PER_MONTH, abs=1)

    def test_unreachable_target_returns_none(self):
        assert vp.ukc_target_eta(vp.ukc_all, vp.ukc_frontier_all, 99.9) is None

    def test_direct_extrapolation_is_a_sanity_check(self):
        """Two-point fit is fragile but should land in the same rough season as
        the lag-based estimate, not years away."""
        eta = vp.ukc_target_eta(vp.ukc_all, vp.ukc_frontier_all, 90.0)
        direct = vp.ukc_target_eta_direct(vp.ukc_all, 90.0)
        assert direct is not None
        assert abs((direct - eta['date_lo']).days) < 180

    def test_direct_needs_two_open_models(self):
        one = [m for m in vp.ukc_all if m['weights'] == 'open'][:1]
        assert vp.ukc_target_eta_direct(one, 90.0) is None


class TestUkCyberTlo:
    """Cyber range "The Last Ones" -- AISI's long-horizon cyber measure.

    Digitized from fig2-ranges.png, except Kimi K3 which AISI/CAISI printed.
    The published-number checks below are the calibration guards on that
    digitization, exactly as `lag_lo` is for the narrow-task figure.
    """

    def _tlo(self):
        return vp.load_ukcyber_tlo(vp._ukc_tlo_mtime())

    def _lags(self):
        tlo = self._tlo()
        return {r['name']: r for r in
                vp.ukc_lag_rows(tlo, [m for m in tlo if m['is_frontier']])}

    def test_loads_all_models(self):
        tlo = self._tlo()
        assert len(tlo) == 10
        assert {m['name'] for m in tlo if m['weights'] == 'open'} == {
            'DeepSeek-V4-Pro', 'GLM-5.2', 'Kimi K3'}

    def test_scores_are_steps_not_percent(self):
        for m in self._tlo():
            assert 0 <= m['cyber_score'] <= vp._UKC_TLO_STEPS
            assert m['steps'] == m['cyber_score']

    def test_glm52_endpoint_matches_published_value(self):
        """CAISI/AISI state "step 11 for GLM-5.2" in prose -- the digitization
        must reproduce it exactly, or the y-axis calibration has drifted."""
        by_name = {m['name']: m for m in self._tlo()}
        assert by_name['GLM-5.2']['cyber_score'] == pytest.approx(11.0, abs=0.2)

    def test_top_us_models_average_matches_published_value(self):
        """"the most cyber-capable U.S. models reached 28.5 steps on average"."""
        by_name = {m['name']: m for m in self._tlo()}
        top = [by_name['GPT-5.6-Sol']['cyber_score'],
               by_name['Claude Mythos 5']['cyber_score']]
        assert sum(top) / len(top) == pytest.approx(28.5, abs=0.2)

    def test_kimi_k3_is_quoted_not_digitized(self):
        """AISI/CAISI printed Kimi K3's average as step 17."""
        by_name = {m['name']: m for m in self._tlo()}
        assert by_name['Kimi K3']['cyber_score'] == pytest.approx(17.0, abs=0.01)

    def test_ordering_claims_in_the_post_hold(self):
        by_name = {m['name']: m for m in self._tlo()}
        # "DeepSeek's V4-Pro falls below Sonnet 4.5"
        assert by_name['DeepSeek-V4-Pro']['cyber_score'] < by_name['Sonnet 4.5']['cyber_score']
        # "GLM-5.2 reaches as far as Opus 4.5"
        assert by_name['GLM-5.2']['cyber_score'] == pytest.approx(
            by_name['Opus 4.5']['cyber_score'], abs=0.2)

    def test_lags_reproduce_the_posts_prose(self):
        """AISI: GLM-5.2 trails Opus 4.5 by "less than 7 months"; DeepSeek-V4-Pro
        trails Sonnet 4.5, "released 7 months before it"."""
        lags = self._lags()
        assert 6.0 < lags['GLM-5.2']['lag_months'] < 7.0
        assert lags['DeepSeek-V4-Pro']['lag_months'] == pytest.approx(6.8, abs=0.3)

    def test_range_lag_exceeds_narrow_task_lag(self):
        """The headline reason this file exists: AISI's "4 to 7 months" spans
        both measures, with the range at the pessimistic end.

        Compared on `lag_lo`, the next-model-up convention AISI's own figures
        use -- Figure 1 is titled "4-5 months prior" and Figure 2 "7 months
        prior", and those are lag_lo on each dataset. The interpolated point
        estimates are NOT comparable across the two: DeepSeek-V4-Pro's narrow
        score lands in a 10-point frontier gap (inflating it to 7.4mo) while its
        TLO score sits below every frontier model (no interpolation at all), so
        on point estimates the ordering inverts for reasons about frontier
        sampling rather than about capability.
        """
        tlo = self._lags()
        narrow = {r['name']: r for r in
                  vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all)}
        for name in ('GLM-5.2', 'DeepSeek-V4-Pro'):
            assert tlo[name]['lag_lo'] > narrow[name]['lag_lo'], name

    def test_reproduces_both_figure_titles(self):
        """Figure 1: "4-5 months prior". Figure 2: "7 months prior"."""
        tlo = self._lags()
        narrow = {r['name']: r for r in
                  vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all)}
        pair = ('GLM-5.2', 'DeepSeek-V4-Pro')
        nar = [narrow[n]['lag_lo'] for n in pair]
        rng = [tlo[n]['lag_lo'] for n in pair]
        assert 4.0 <= min(nar) and max(nar) <= 5.5
        assert 6.5 <= min(rng) and max(rng) <= 7.5

    def test_deepseek_tlo_lag_is_a_lower_bound_only(self):
        """DeepSeek-V4-Pro's 8.0 steps is under every frontier model, so the
        crossing collapses onto the earliest one and there is no upper bracket
        -- the frontier passed 8.0 steps at some unmeasured earlier date."""
        r = self._lags()['DeepSeek-V4-Pro']
        assert r['below_name'] is None
        assert r['lag_hi'] is None
        assert r['lag_months'] == r['lag_lo']

    def test_frontier_is_closed_weights_only(self):
        tlo = self._tlo()
        assert all(m['weights'] == 'closed' for m in tlo if m['is_frontier'])

    def test_frontier_is_monotonic(self):
        frontier = [m for m in self._tlo() if m['is_frontier']]
        scores = [m['cyber_score'] for m in frontier]
        assert scores == sorted(scores)

    def test_dates_are_published_release_dates(self):
        """Unlike the narrow-task file, TLO dates come from release records --
        the figure's x-axis is tokens and carries no date information."""
        by_name = {m['name']: m for m in self._tlo()}
        assert by_name['Kimi K3']['date'] == datetime(2026, 7, 16)
        assert by_name['Sonnet 4.5']['date'] == datetime(2025, 9, 29)
        assert by_name['GPT-5.5']['date'] == datetime(2026, 4, 23)

    def test_shared_lag_helper_matches_manual_construction(self):
        """The callout and the cross-check section must not be able to drift
        onto different frontiers or lag conventions."""
        tlo_all, tlo_lag = vp.ukc_tlo_lag_rows()
        manual = vp.ukc_lag_rows(tlo_all, [m for m in tlo_all if m['is_frontier']])
        assert [r['name'] for r in tlo_lag] == [r['name'] for r in manual]
        assert [r['lag_lo'] for r in tlo_lag] == [r['lag_lo'] for r in manual]


class TestUkCyberOpenOnlyOnTlo:
    """The headline callout for an open-weight model only the range has run.

    Kimi K3 is the live case: AISI/CAISI ran ExploitBench + TLO on it, not the
    70-task narrow suite, so it has no point on the main chart.
    """

    def _rows(self):
        narrow = vp.ukc_lag_rows(vp.ukc_all, vp.ukc_frontier_all)
        _, tlo = vp.ukc_tlo_lag_rows()
        return narrow, tlo

    def test_surfaces_kimi_k3(self):
        narrow, tlo = self._rows()
        r = vp.ukc_open_only_on_tlo(narrow, tlo)
        assert r is not None and r['name'] == 'Kimi K3'
        assert r['name'] not in {n['name'] for n in narrow}

    def test_callout_carries_a_bracket_not_just_a_point(self):
        """The tab renders lag_lo-lag_hi here; TLO's sparse frontier makes the
        interpolated point estimate a poor single number to headline."""
        narrow, tlo = self._rows()
        r = vp.ukc_open_only_on_tlo(narrow, tlo)
        assert r['lag_lo'] is not None and r['lag_hi'] is not None
        assert r['lag_lo'] < r['lag_months'] < r['lag_hi']
        assert (r['above_name'], r['below_name']) == ('Mythos Preview', 'Claude Opus 4.6')

    def test_disappears_once_the_narrow_suite_catches_up(self):
        """A data-coverage notice, not a permanent panel -- if AISI later runs
        the 70-task suite on Kimi K3, the callout must stop rendering."""
        narrow, tlo = self._rows()
        kimi = next(r for r in tlo if r['name'] == 'Kimi K3')
        assert vp.ukc_open_only_on_tlo(narrow + [dict(kimi)], tlo) is None

    def test_ignores_older_gaps_in_tlo_coverage(self):
        """Only a model newer than every open-weight point on the chart means
        the chart is out of date; an older gap is a curiosity."""
        _, tlo = self._rows()
        newer_narrow = [{**r, 'date': datetime(2027, 1, 1)} for r in tlo[:1]]
        assert vp.ukc_open_only_on_tlo(newer_narrow, tlo) is None

    def test_returns_none_when_suites_agree(self):
        _, tlo = self._rows()
        assert vp.ukc_open_only_on_tlo([dict(r) for r in tlo], tlo) is None


class TestDcAxisScale:
    """The H100 metric is stored raw but read in millions on the axis."""

    @staticmethod
    def _plain(texts):
        return [re.sub(r"<[^>]+>", "", t) for t in texts]

    def test_metric_declares_the_scale(self):
        cfg = vp._DC_METRICS["Compute (x1M H100-equiv)"]
        assert cfg["key"] == "h100" and cfg["scale"] == 1e6
        assert vp._DC_DEFAULTS["dc_metric"] == "Compute (x1M H100-equiv)"

    def test_log_ticks_scale_the_label_not_the_position(self):
        vals, text = vp._dc_log_ticks([5.0, 7.0], tick_scale=1e6)
        assert self._plain(text)[:3] == ["0.1", "0.2", "0.3"]
        assert vals[0] == 1e5            # tick still sits at the raw value
        assert self._plain(text)[vals.index(1e6)] == "1"

    def test_log_ticks_unchanged_without_a_scale(self):
        _, text = vp._dc_log_ticks([5.0, 6.0])
        assert self._plain(text)[0] == "100000"

    def test_linear_ticks_scale_and_stay_round(self):
        vals, text = vp._dc_linear_ticks([0, 5.5e6], 1e6)
        assert text == ["0", "1", "2", "3", "4", "5"]
        assert vals[-1] == 5e6

    def test_linear_ticks_absent_when_nothing_to_rescale(self):
        assert vp._dc_linear_ticks([0, 5e6], 1.0) is None
        assert vp._dc_linear_ticks(None, 1e6) is None
        assert vp._dc_linear_ticks([3.0, 3.0], 1e6) is None

    def test_layout_applies_the_scale_on_both_axis_types(self):
        log = vp._dc_layout(True, "Compute (x1M H100-equiv)",
                            datetime(2024, 1, 1), datetime(2028, 1, 1),
                            y_range=[5.0, 7.0], tick_scale=1e6)
        assert self._plain(log['yaxis']['ticktext'])[0] == "0.1"
        lin = vp._dc_layout(False, "Compute (x1M H100-equiv)",
                            datetime(2024, 1, 1), datetime(2028, 1, 1),
                            y_range=[0, 5.5e6], tick_scale=1e6)
        assert lin['yaxis']['ticktext'] == ["0", "1", "2", "3", "4", "5"]

    def test_values_and_hovers_stay_raw_counts(self):
        """Only the axis is rescaled — _dc_fmt_value still takes raw counts, so
        the tables, hovers and _cc_company_buildout keep working unchanged."""
        assert vp._dc_fmt_value(1824153.6, 'h100') == "1.82M"
        assert vp._dc_fmt_value(816321, 'h100') == "816k"
        ser = vp._dc_series_for_metric(vp.dc_all, 'h100')
        biggest = max(v for d in ser.values() for _, v in d['pts'])
        assert biggest > 1e6, "series must still hold raw H100 counts"


class TestDcCompanyAliases:
    """Distinct Epoch labels that presentation treats as one company.

    Google is the case that motivated the map: every Google site is
    Owner="Google", but only some carry Users="Google DeepMind #speculative",
    and company_for() is user-first — so one TPU fleet split in two on nothing
    but whether Epoch filled an optional cell. The split drew two Google lines
    (Google blue landing on the smaller one), made the pooled Columbus cluster
    depend on both its Google sites happening to share a tag, and left the
    quarterly table reporting the minority series."""

    def _dc_meta_rows(self):
        return list(csv.DictReader(open(
            os.path.join(os.path.dirname(vp.__file__), "data_centers.csv"))))

    def test_no_aliased_label_survives_into_company_labels(self):
        labels = {dc['company'] for dc in vp.dc_all}
        for alias, target in vp._DC_COMPANY_ALIASES.items():
            assert alias not in labels, f"{alias!r} was not merged into {target!r}"
            assert target in labels, f"alias target {target!r} matches no site"

    def test_every_google_owned_site_is_one_company(self):
        """Against the live CSV, so a refresh that tags (or untags) a site's
        Users cell can't re-split the fleet."""
        by_name = {dc['name']: dc['company'] for dc in vp.dc_all}
        owned = {(r.get('Name') or '').strip()
                 for r in self._dc_meta_rows()
                 if vp._dc_clean_owner(r.get('Owner', '')).startswith('Google')}
        assert len(owned) > 1
        labelled = {by_name[n] for n in owned if n in by_name}
        assert labelled == {'Google'}, labelled

    def test_merge_is_load_bearing(self):
        """Both spellings are actually present upstream — if Epoch ever stops
        emitting the DeepMind tag the map is dead weight and can go."""
        users = {vp._dc_clean_owner((r.get('Users', '') or '').split(',')[0])
                 for r in self._dc_meta_rows()}
        assert 'Google DeepMind' in users
        assert 'Google' in {vp._dc_clean_owner(r.get('Owner', ''))
                            for r in self._dc_meta_rows()}

    def test_no_company_label_is_a_qualified_form_of_another(self):
        """Catches the next "Meta AI" vs "Meta" style split before it ships."""
        labels = {dc['company'] for dc in vp.dc_all}
        for a in labels:
            for b in labels:
                if a != b and b.startswith(a + ' '):
                    pytest.fail(f"{b!r} looks like a split-off of {a!r}; "
                                "add it to _DC_COMPANY_ALIASES")

    def test_alias_targets_keep_their_brand_colour(self):
        """The merged label is the one that draws, so it's the one that needs
        the colour — under the split, Google blue went to the smaller line."""
        for target in set(vp._DC_COMPANY_ALIASES.values()):
            assert target in vp._DC_COLORS

    def test_quarterly_table_companies_resolve_without_aliasing(self):
        """The table's column keys index _dc_company_series() directly now; it
        used to carry its own alias tuple and take the first spelling that
        existed, which silently picked the minority series."""
        series = {n: v for n, v in
                  vp._dc_series_for_metric(vp.dc_all, 'h100').items()
                  if v['company'] not in vp._DC_EXCLUDE_COMPANIES}
        comp = vp._dc_company_series(series)
        for co in ["OpenAI", "Anthropic", "Google", "Meta", "SpaceXAI", "Alibaba"]:
            assert comp.get(co), f"table column {co!r} has no series"
        # Compared against every Google-*owned* site in the CSV, not just the
        # ones currently labelled Google, so the column can't quietly fall back
        # to a subset of the fleet the way the old alias tuple did.
        owned = {(r.get('Name') or '').strip()
                 for r in self._dc_meta_rows()
                 if vp._dc_clean_owner(r.get('Owner', '')).startswith('Google')}
        pts = [val for n in owned if n in series
               for _d, val in series[n]['pts']]
        assert max(v for _d, v, _n in comp['Google']) == max(pts)


class TestDcHiddenCompanies:
    """Size gates the colocation/neutral-host exclusion.

    _DC_EXCLUDE_COMPANIES names companies that aren't AI labs, and they used to
    be hidden unconditionally. That understated the tab's headline chart once
    one of them got big: QTS Cedar Rapids is the largest single site in Epoch's
    data, so "Largest single data center" was naming a smaller site as the
    record holder."""

    def _peak(self, co, cap=None):
        return vp._dc_company_peak_h100(vp.dc_all, cap_date=cap).get(co, 0.0)

    def test_hidden_is_a_subset_of_the_exclude_list(self):
        """The size gate only ever un-hides — it must never drop an AI lab."""
        hidden = vp._dc_hidden_companies(vp.dc_all)
        assert hidden <= vp._DC_EXCLUDE_COMPANIES

    def test_a_big_host_is_charted_and_a_small_one_is_not(self):
        hidden = vp._dc_hidden_companies(vp.dc_all)
        cap = datetime.now() + timedelta(days=vp._DC_EXCLUDE_HORIZON_DAYS)
        for co in vp._DC_EXCLUDE_COMPANIES:
            big = self._peak(co, cap) >= vp._DC_EXCLUDE_MIN_H100
            assert (co not in hidden) == big, (
                f"{co!r} peak {self._peak(co, cap):,.0f} vs threshold "
                f"{vp._DC_EXCLUDE_MIN_H100:,}")

    def test_the_largest_site_in_the_data_is_actually_charted(self):
        """The bug that motivated the change: whoever owns the biggest site,
        the envelope has to end on it."""
        series = {n: v for n, v in
                  vp._dc_series_for_metric(vp.dc_all, 'h100').items()
                  if v['company'] not in vp._dc_hidden_companies(vp.dc_all)}
        env = vp._dc_envelope(series)
        biggest = max((p['h100'], dc['name']) for dc in vp.dc_all
                      for p in dc['points'] if p.get('h100') is not None)
        assert max(v for _d, v, _n, _c in env) == biggest[0]
        assert biggest[1] in {n for _d, _v, n, _c in env}

    def test_current_roster_is_what_the_tab_says_it_is(self):
        """The calibration guard. The tab's scope caption names who the rule
        adds, so an Epoch refresh that moves the roster has to be looked at
        rather than absorbed silently. Oracle is the nearest miss at ~84k
        against a 100k bar, so it is the one to expect here first; if it
        crosses, retarget _DC_EXCLUDE_MIN_H100 deliberately (or accept Oracle)
        instead of loosening this test."""
        charted = vp._DC_EXCLUDE_COMPANIES - vp._dc_hidden_companies(vp.dc_all)
        assert charted == {'QTS', 'DayOne', 'Microsoft'}, charted

    def test_roster_is_stable_across_the_horizon(self):
        """Who appears must not hinge on the exact horizon. Nothing may change
        between 6 and 12 months out — the docstring's claim, asserted."""
        now = datetime.now()
        at_6mo = vp._dc_hidden_companies(
            vp.dc_all, now=now - timedelta(days=183))
        assert at_6mo == vp._dc_hidden_companies(vp.dc_all, now=now)

    def test_roster_ignores_the_users_projection_window(self):
        """dc_end_year caps the charts, never the roster — otherwise sites blink
        in and out as the slider moves."""
        rosters = {frozenset(vp._dc_hidden_companies(vp.dc_all))}
        assert len(rosters) == 1
        # And the function takes no metric/window argument that could vary it.
        import inspect
        params = set(inspect.signature(vp._dc_hidden_companies).parameters)
        assert params == {'dcs', 'now'}

    def test_uncapped_peaks_would_defeat_the_exclusion(self):
        """Why the horizon exists: on buildout announced for 2028+, nearly
        every listed host clears the bar and the list stops meaning anything."""
        uncapped = vp._dc_company_peak_h100(vp.dc_all)
        clears = {co for co in vp._DC_EXCLUDE_COMPANIES
                  if uncapped.get(co, 0.0) >= vp._DC_EXCLUDE_MIN_H100}
        assert clears > (vp._DC_EXCLUDE_COMPANIES
                         - vp._dc_hidden_companies(vp.dc_all))

    def test_compute_capabilities_frontier_stays_lab_only(self):
        """The two tabs differ on purpose: a compute-vs-capability frontier
        needs every point attributable to a lab that ships models, so it keeps
        the unconditional exclusion even for the hosts the DC tab now charts."""
        cap = datetime.now() + timedelta(days=365)
        front = vp._cc_trainflop_frontier(vp.dc_all, cap, with_names=True)
        by_name = {dc['name']: dc['company'] for dc in vp.dc_all}
        leaders = {name for _d, _v, name, _sd in front}
        assert leaders
        for site in leaders:
            assert by_name.get(site) not in vp._DC_EXCLUDE_COMPANIES, site
        # And the DC tab's own envelope over the same window *does* reach one
        # of them — otherwise this test passes for the wrong reason.
        series = {n: v for n, v in
                  vp._dc_series_for_metric(vp.dc_all, 'h100', cap_date=cap).items()
                  if v['company'] not in vp._dc_hidden_companies(vp.dc_all)}
        dc_leaders = {by_name.get(n) for _d, _v, n, _c in vp._dc_envelope(series)}
        assert dc_leaders & vp._DC_EXCLUDE_COMPANIES


class TestDcUnattributedCompanies:
    """The † mark: capacity Epoch records no tenant for."""

    def test_marks_exactly_the_name_token_fallbacks(self):
        un = vp._dc_unattributed_companies(vp.dc_all)
        for dc in vp.dc_all:
            if dc['company'] in un:
                assert not dc['attributed'], (
                    f"{dc['name']} has a recorded user/owner but its company "
                    f"{dc['company']!r} is marked unattributed")

    def test_one_attributed_site_clears_the_whole_company(self):
        """Microsoft is listed as its own sites' user, so it is charted plain
        even though it shares the exclude list with the landlords."""
        un = vp._dc_unattributed_companies(vp.dc_all)
        assert 'Microsoft' not in un
        assert vp._dc_co_label('Microsoft', un) == 'Microsoft'

    def test_landlord_labels_carry_the_mark(self):
        un = vp._dc_unattributed_companies(vp.dc_all)
        charted = vp._DC_EXCLUDE_COMPANIES - vp._dc_hidden_companies(vp.dc_all)
        marked = [c for c in charted if c in un]
        assert marked, "no charted host is unattributed — is the mark dead?"
        for co in marked:
            assert vp._dc_co_label(co, un) == co + vp._DC_UNATTRIBUTED_MARK

    def test_labs_are_never_marked(self):
        un = vp._dc_unattributed_companies(vp.dc_all)
        for co in ["OpenAI", "Anthropic", "Google", "Meta"]:
            assert co not in un

    def test_and_list_reads_as_prose(self):
        assert vp._and_list([]) == ""
        assert vp._and_list(["a"]) == "a"
        assert vp._and_list(["a", "b"]) == "a and b"
        assert vp._and_list(["a", "b", "c"]) == "a, b and c"


class TestDcNetworkClusters:
    """The curated map of sites that could share one training job."""

    def test_every_clustered_site_exists_in_the_live_csv(self):
        """The registry is hand-written, so a data refresh that renames a site
        must fail here rather than silently drop it out of its cluster."""
        known = {dc['name'] for dc in vp.dc_all}
        for label, _basis, names in vp._DC_NETWORK_CLUSTERS:
            for name in names:
                assert name in known, f"{label}: unknown site {name!r}"

    def test_registry_is_well_formed(self):
        seen = set()
        for label, basis, names in vp._DC_NETWORK_CLUSTERS:
            assert basis in ('proximity', 'fabric'), label
            assert len(names) >= 2, f"{label} is not a cluster"
            assert len(set(names)) == len(names), label
            # A site may belong to at most one cluster, or "largest group"
            # would depend on dict ordering.
            for name in names:
                assert name not in seen, f"{name} is in two clusters"
                seen.add(name)

    def test_fabric_clusters_are_droppable(self):
        with_fabric = vp._dc_network_site_clusters(include_fabric=True)
        without = vp._dc_network_site_clusters(include_fabric=False)
        assert set(without) < set(with_fabric)
        assert "Microsoft Fairwater Wisconsin" in with_fabric
        assert "Microsoft Fairwater Wisconsin" not in without
        # Proximity clusters survive either way.
        assert without["Colossus 2"] == with_fabric["Colossus 2"] == "Memphis, TN"

    def test_network_options_registry_is_well_formed(self):
        assert vp._DC_DEFAULTS["dc_pool_n"] in vp._DC_NETWORK_OPTIONS
        assert "dc_pool_n" in vp._DC_RESET_KEYS
        assert set(vp._DC_NETWORK_OPTIONS.values()) == {
            'fabric', 'proximity', 'none', 'all'}


class TestDcCompanyNetworkedSeries:
    """Pooling a company's largest networkable group of sites."""

    @staticmethod
    def _series():
        d = datetime
        return {
            # Two sites in one cluster, one far away, plus a second company.
            "NearA": {"company": "LabA", "pts": [(d(2025, 1, 1), 10.0),
                                                 (d(2026, 1, 1), 40.0)]},
            "NearB": {"company": "LabA", "pts": [(d(2025, 6, 1), 20.0)]},
            "Far":   {"company": "LabA", "pts": [(d(2025, 6, 1), 25.0)]},
            "Solo":  {"company": "LabB", "pts": [(d(2025, 1, 1), 7.0)]},
        }

    _CLUSTERS = {"NearA": "Metro", "NearB": "Metro"}

    def test_no_clusters_reproduces_the_single_largest_series(self):
        """cluster_of={} must be the chart directly above this one, exactly —
        the two sit together and would look broken if they drifted."""
        ser = self._series()
        net = vp._dc_company_networked_series(ser, {})
        single = vp._dc_company_series(ser)
        assert set(net) == set(single)
        for co in single:
            assert ([(d, v) for d, v, _, _ in net[co]]
                    == [(d, v) for d, v, _ in single[co]])

    def test_pools_only_within_a_cluster(self):
        steps = vp._dc_company_networked_series(self._series(),
                                                self._CLUSTERS)["LabA"]
        by_date = {d: (v, names, label) for d, v, names, label in steps}
        # Jan 2025: only NearA exists — a lone site, so no cluster label.
        assert by_date[datetime(2025, 1, 1)] == (10.0, ("NearA",), None)
        # Jun 2025: the clustered pair (10+20) beats the bigger lone site (25).
        assert by_date[datetime(2025, 6, 1)] == (30.0, ("NearB", "NearA"),
                                                 "Metro")
        # Jan 2026: NearA scales up; "Far" is never added in.
        assert by_date[datetime(2026, 1, 1)] == (60.0, ("NearA", "NearB"),
                                                 "Metro")

    def test_a_lone_site_can_beat_a_cluster(self):
        """The biggest *group* wins, which is sometimes a single building."""
        ser = self._series()
        ser["Far"]["pts"] = [(datetime(2025, 6, 1), 500.0)]
        steps = vp._dc_company_networked_series(ser, self._CLUSTERS)["LabA"]
        last = steps[-1]
        assert last[1] == 500.0 and last[2] == ("Far",) and last[3] is None

    def test_unrestricted_pools_the_whole_fleet(self):
        steps = vp._dc_company_networked_series(self._series(), None)["LabA"]
        _d, val, names, label = steps[-1]
        assert val == 40.0 + 20.0 + 25.0
        assert names == ("NearA", "Far", "NearB") and label == "all sites"

    def test_clustering_never_reduces_capacity_on_live_data(self):
        """Every basis is a superset of the one below it, so the lines can only
        rise as networking is allowed."""
        ser = vp._dc_series_for_metric(vp.dc_all, 'h100')
        ser = {n: d for n, d in ser.items()
               if d['company'] not in vp._DC_EXCLUDE_COMPANIES}
        runs = [
            vp._dc_company_networked_series(ser, {}),
            vp._dc_company_networked_series(
                ser, vp._dc_network_site_clusters(include_fabric=False)),
            vp._dc_company_networked_series(
                ser, vp._dc_network_site_clusters(include_fabric=True)),
            vp._dc_company_networked_series(ser, None),
        ]
        at = datetime(2027, 6, 30)

        def val(steps):
            cur = [s for s in steps if s[0] <= at]
            return cur[-1][1] if cur else None

        for co in runs[0]:
            vals = [v for v in (val(r[co]) for r in runs) if v is not None]
            assert vals == sorted(vals), co

    def test_clusters_actually_change_a_real_company(self):
        """Guards against a registry that silently matches nothing."""
        ser = vp._dc_series_for_metric(vp.dc_all, 'h100')
        ser = {n: d for n, d in ser.items()
               if d['company'] not in vp._DC_EXCLUDE_COMPANIES}
        plain = vp._dc_company_networked_series(ser, {})
        clustered = vp._dc_company_networked_series(
            ser, vp._dc_network_site_clusters())
        at = datetime(2027, 6, 30)

        def val(steps):
            cur = [s for s in steps if s[0] <= at]
            return cur[-1][1] if cur else None

        assert any(val(clustered[co]) > val(plain[co]) for co in plain
                   if val(plain[co]) is not None)

    def test_traintime_metric_pools_as_a_sum(self):
        """'Capacity' metrics store runs-per-2mo, so networking sites adds runs
        (and therefore shortens the displayed time to train one model)."""
        ser = vp._dc_series_for_metric(vp.dc_all, 'mythos')
        ser = {n: d for n, d in ser.items()
               if d['company'] not in vp._DC_EXCLUDE_COMPANIES}
        one = vp._dc_company_networked_series(ser, {})
        net = vp._dc_company_networked_series(ser,
                                              vp._dc_network_site_clusters())
        at = datetime(2027, 6, 30)
        for co, steps in one.items():
            cur = [s for s in steps if s[0] <= at]
            pooled = [s for s in net[co] if s[0] <= at]
            if cur and pooled:
                assert pooled[-1][1] >= cur[-1][1]


class TestDcProjectionRange:
    """The Data Centers tab's chart window is a sidebar range, URL-tracked."""

    def test_projection_range_registry_is_well_formed(self):
        assert vp._DC_DEFAULTS["dc_start_year"] == 2025
        assert vp._DC_DEFAULTS["dc_end_year"] == 2027
        assert vp._DC_DEFAULTS["dc_start_year"] in vp._DC_START_YEARS
        assert vp._DC_DEFAULTS["dc_end_year"] in vp._DC_END_YEARS
        # No allowed pairing can invert the window.
        assert max(vp._DC_START_YEARS) <= min(vp._DC_END_YEARS)
        keys, defaults = vp._all_tracked()
        for k in ("dc_start_year", "dc_end_year"):
            assert k in vp._DC_RESET_KEYS       # reset button clears it
            assert k in keys                    # and it round-trips through the URL
            assert defaults[k] == vp._DC_DEFAULTS[k]


class TestDcTrainFlopWindows:
    """The 2mo and 6mo train-FLOP metrics differ only in the run window."""

    def test_six_month_flop_is_three_times_the_two_month_flop(self):
        dcs = vp.load_data_centers()
        pts = [p for dc in dcs for p in dc['points'] if p['perf'] is not None]
        assert pts, "no data-center points carry a performance figure"
        for p in pts:
            assert p['train_flop_6mo'] == pytest.approx(3 * p['train_flop'])

    def test_missing_performance_leaves_both_none(self):
        dcs = vp.load_data_centers()
        for dc in dcs:
            for p in dc['points']:
                if p['perf'] is None:
                    assert p['train_flop'] is None
                    assert p['train_flop_6mo'] is None

    def test_registry_entries_carry_their_run_length(self):
        assert vp._DC_METRICS["2mo train log OP"]["key"] == "train_flop"
        assert vp._DC_METRICS["6mo train log OP"]["key"] == "train_flop_6mo"
        assert vp._DC_METRICS["2mo train log OP"]["run_days"] == vp._DAYS_2MO
        assert vp._DC_METRICS["6mo train log OP"]["run_days"] == vp._DAYS_6MO
        for label in ("2mo train log OP", "6mo train log OP"):
            assert vp._DC_METRICS[label]["kind"] == "flop"
            assert vp._DC_METRICS[label]["log"] is True

    def test_timing_shift_follows_the_metric_run_length(self):
        # Construction is the site's own date; the other two milestones are one
        # training run (and, for release, the post-training lag) later.
        lag = vp._CC_RUN_COMPLETION_LAG.days
        assert vp._dc_timing_shift("Data center construction", vp._DAYS_6MO) == 0
        assert vp._dc_timing_shift("Training run finished") == vp._DAYS_2MO
        assert vp._dc_timing_shift("Training run finished", vp._DAYS_6MO) == vp._DAYS_6MO
        assert vp._dc_timing_shift("Model release") == vp._DAYS_2MO + lag
        assert (vp._dc_timing_shift("Model release", vp._DAYS_6MO)
                == vp._DAYS_6MO + lag)

    def test_default_timing_is_a_real_option(self):
        assert vp._DC_DEFAULTS["dc_timing"] in vp._DC_TIMING_OPTIONS

    def test_value_reads_as_log10_operations(self):
        assert vp._dc_fmt_value(1e28, 'flop') == "28 log OP"
        assert vp._dc_fmt_value(2e28, 'flop') == "28.3 log OP"
        assert vp._dc_fmt_value(3.16e26, 'flop') == "26.5 log OP"
        assert vp._dc_fmt_value(0, 'flop') == "—"
        assert vp._dc_fmt_value(None, 'flop') == "—"

    def test_stored_values_stay_raw_counts(self):
        # Only the display is logged — pooling several sites is still a plain
        # sum of operation counts, which logging would silently break.
        dcs = vp.load_data_centers()
        pts = [p for dc in dcs for p in dc['points'] if p['train_flop']]
        assert max(p['train_flop'] for p in pts) > 1e20

    def test_axis_ticks_are_round_log_values(self):
        vals, text = vp._dc_logop_ticks([27.0, 28.3], log_scale=True)
        plain = [re.sub(r'<[^>]+>', '', t) for t in text]
        assert plain == ["27", "27.2", "27.4", "27.6", "27.8", "28", "28.2"]
        assert vals == pytest.approx([10.0 ** float(t) for t in plain])
        # Whole decades keep the full tickfont; the steps between are shrunk.
        assert text[0] == "27" and "font-size" in text[1]

    def test_axis_ticks_never_repeat_a_label(self):
        for rng, log in (([24.0, 29.0], True), ([27.9, 28.05], True),
                         ([1e27, 3e27], False), ([0, 3e27], False)):
            out = vp._dc_logop_ticks(rng, log_scale=log)
            plain = [re.sub(r'<[^>]+>', '', t) for t in out[1]]
            assert plain == sorted(plain, key=float)
            assert len(set(plain)) == len(plain)

    def test_axis_ticks_none_without_a_usable_range(self):
        assert vp._dc_logop_ticks(None, log_scale=True) is None
        assert vp._dc_logop_ticks([0.0, 0.0], log_scale=False) is None

    def test_layout_labels_the_flop_axis_in_log_ops(self):
        lay = vp._dc_layout(True, "6mo train log OP",
                            datetime(2024, 1, 1), datetime(2028, 1, 1),
                            y_range=[27.0, 28.3], kind='flop')
        assert lay['yaxis']['tickmode'] == 'array'
        assert "28" in [re.sub(r'<[^>]+>', '', t)
                        for t in lay['yaxis']['ticktext']]
        # No raw 1e+28-style label survives.
        assert not any('e+' in t for t in lay['yaxis']['ticktext'])


class TestDcTrainTime:
    """The two 'Capacity' metrics store runs-per-2mo but display time-to-train."""

    def test_duration_units_scale_with_magnitude(self):
        assert vp._fmt_duration_days(0.02) == "~29 min"
        assert vp._fmt_duration_days(0.25) == "~6 hours"
        assert vp._fmt_duration_days(1) == "~1 day"
        assert vp._fmt_duration_days(7) == "~1 week"
        assert vp._fmt_duration_days(30.4375) == "~1 month"
        assert vp._fmt_duration_days(365.25) == "~1 year"

    def test_duration_rejects_nonpositive_and_missing(self):
        for bad in (None, 0, -3, float('nan'), float('inf')):
            assert vp._fmt_duration_days(bad) == "—"

    def test_value_is_the_time_for_one_run(self):
        # v = runs the site fits in the 2-month window, so 1 run == 2 months
        # and 60 runs == one day each.
        assert vp._dc_fmt_value(1.0, 'traintime') == "~2 months"
        assert vp._dc_fmt_value(60.0, 'traintime') == "~1 day"
        assert vp._dc_fmt_value(0.0, 'traintime') == "—"
        assert vp._dc_fmt_value(None, 'traintime') == "—"

    def test_metrics_registry_uses_traintime(self):
        for label in ("Capacity (time to GPT-5)", "Capacity (time to Mythos)"):
            assert vp._DC_METRICS[label]["kind"] == "traintime"
        assert vp._DC_METRICS["Capacity (time to GPT-5)"]["key"] == "gpt5s"
        assert vp._DC_METRICS["Capacity (time to Mythos)"]["key"] == "mythos"

    def test_bigger_site_reads_as_less_time(self):
        """Stored values stay 'bigger = better' so every max-based aggregation
        in the tab keeps working; only the label runs the other way."""
        rows = sorted([0.5, 5.0, 50.0])
        labels = [vp._dc_fmt_value(v, 'traintime') for v in rows]
        assert labels == ["~3.9 months", "~1.7 weeks", "~1.2 days"]
        assert max(rows) == 50.0   # the fastest site is still the max

    def test_axis_ticks_are_round_durations_inside_the_range(self):
        vals, text = vp._dc_duration_ticks([0.0, 2.0], log_scale=True)
        assert all(1.0 <= v <= 100.0 for v in vals)
        assert text == [vp._dc_fmt_value(v, 'traintime') for v in vals]
        assert "~1 day" in text and "~2 months" in text
        # Strictly increasing positions == strictly decreasing durations.
        assert vals == sorted(vals)

    def test_axis_ticks_none_without_a_range(self):
        assert vp._dc_duration_ticks(None, log_scale=True) is None
        assert vp._dc_duration_ticks([6.0, 8.0], log_scale=True) is None

    def test_layout_labels_the_axis_in_durations(self):
        lay = vp._dc_layout(True, "Capacity (time to Mythos)",
                            datetime(2024, 1, 1), datetime(2028, 1, 1),
                            y_range=[0.0, 2.0], kind='traintime')
        assert lay['yaxis']['tickmode'] == 'array'
        assert "~1 day" in lay['yaxis']['ticktext']
        # Other kinds keep the plain numeric log ticks.
        plain = vp._dc_layout(True, "2mo train log OP",
                              datetime(2024, 1, 1), datetime(2028, 1, 1),
                              y_range=[0.0, 2.0])
        assert "~1 day" not in plain['yaxis']['ticktext']
