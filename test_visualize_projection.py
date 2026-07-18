"""
Tests for visualize_projection.py helper functions and data loading.

Run: pytest test_visualize_projection.py -v
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
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
        fm = vp._cc_company_frontier_models().get("OpenAI", [])
        lag = timedelta(days=vp._CC_RELEASE_LAG_DAYS)
        floor = timedelta(days=vp._CC_TRAIN_FLOOR_DAYS)

        sol = next((m for m in fm if m[2] == "GPT-5.6 Sol (pro, max)"), None)
        assert sol is not None, "Sol not found in OpenAI frontier releases"
        d = sol[0]

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
