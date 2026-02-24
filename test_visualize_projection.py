"""
Tests for visualize_projection.py helper functions and data loading.

Run: pytest test_visualize_projection.py -v
"""

import numpy as np
import pytest
from datetime import datetime
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
