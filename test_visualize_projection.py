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
# pretty()
# ===========================================================================

class TestCacheKeys:
    def test_cached_functions_hash_their_file_mtime(self):
        """Streamlit skips arguments named with a leading underscore when it
        hashes a cache key, so an `_mtime` argument never invalidates the
        cache and an edited data file is ignored until restart."""
        import ast
        tree = ast.parse(open(vp.__file__).read())
        bad = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            cached = any("cache_data" in ast.unparse(d)
                         for d in fn.decorator_list)
            args = fn.args.args + fn.args.kwonlyargs
            if cached and any(a.arg.startswith("_") and "mtime" in a.arg
                              for a in args):
                bad.append(fn.name)
        assert not bad


class TestPretty:
    @pytest.mark.parametrize("name", ["some_unknown_model", ""])
    def test_unknown_name_passthrough(self, name):
        assert vp.pretty(name) == name


# ===========================================================================
# log2min_to_label() / fmt_hrs()
# ===========================================================================

class TestLog2MinToLabel:
    @pytest.mark.parametrize("minutes,want", [
        (0.5, "30s"), (1, "1m"), (30, "30m"), (60, "1h"), (120, "2h"),
        (800, "13h20m"),   # under the 100h threshold, so hours, not days
    ])
    def test_label(self, minutes, want):
        assert vp.log2min_to_label(np.log2(minutes)) == want


class TestFmtHrs:
    """Work-time units: 8h/d, 40h/w, 176h/mo, 2000h/y; hours below 100h."""

    @pytest.mark.parametrize("h,want", [
        (0, "0m"), (0.5, "30m"),
        (3, "3h"), (1.5, "1h30m"), (16, "16h"), (99, "99h"),
        (100, "2w2d"),                     # 100h crosses into work units
        (160, "4w"), (168, "4w1d"),        # under 4.4w stays in weeks
        (176, "1mo"),
        (2000, "1y"), (2176, "1y1mo"), (4000, "2y"),
    ])
    def test_format(self, h, want):
        assert vp.fmt_hrs(h) == want

    @pytest.mark.parametrize("hours_only", [False, True])
    def test_minutes_that_round_to_an_hour_read_as_hours(self, hours_only):
        assert vp.fmt_hrs(59.9 / 60, hours_only=hours_only) == "1h"
        assert vp.fmt_hrs(59.4 / 60, hours_only=hours_only) == "59m"


# ===========================================================================
# fit_line()
# ===========================================================================

class TestFitLine:
    """fit_line returns [intercept, slope]; every caller indexes it that way."""

    @pytest.mark.parametrize("x,y,a,b", [
        ([0, 1, 2, 3, 4], [2, 5, 8, 11, 14], 2.0, 3.0),
        ([0, 1, 2, 3], [5, 5, 5, 5], 5.0, 0.0),
        ([0, 1, 2, 3], [10, 8, 6, 4], 10.0, -2.0),
        ([0, 10], [0, 30], 0.0, 3.0),
    ])
    def test_exact_line(self, x, y, a, b):
        params = vp.fit_line(np.array(x, dtype=float), np.array(y, dtype=float))
        np.testing.assert_allclose(params, [a, b], atol=1e-10)


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

    def test_equal_bounds(self):
        """Equal lo=hi means sigma=0, should return constant."""
        np.random.seed(42)
        samples = vp._lognormal_from_ci(100, 100, 100)
        np.testing.assert_allclose(samples, 100.0, rtol=1e-10)


class TestNormalFromCi:
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

    def test_equal_bounds(self):
        """Equal lo=hi means sigma=0, should return constant."""
        np.random.seed(42)
        samples = vp._normal_from_ci(100, 100, 100)
        np.testing.assert_allclose(samples, 100.0, rtol=1e-10)


class TestLogLognormalFromCi:
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
    @pytest.mark.parametrize("p,want", [
        (0.5, 0.0), (0.9, np.log(9)), (0.1, -np.log(9)),
    ])
    def test_logit(self, p, want):
        np.testing.assert_allclose(vp._logit(p), want, atol=1e-10)

    @pytest.mark.parametrize("x,want", [(0, 0.5), (100, 1.0), (-100, 0.0)])
    def test_inv_logit(self, x, want):
        np.testing.assert_allclose(vp._inv_logit(x), want, atol=1e-10)

    def test_roundtrip(self):
        """logit and inv_logit should be inverses."""
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            np.testing.assert_allclose(vp._inv_logit(vp._logit(p)), p, atol=1e-10)

    def test_arrays_elementwise(self):
        np.testing.assert_allclose(vp._logit(np.array([0.1, 0.5, 0.9])),
                                   [-np.log(9), 0.0, np.log(9)], atol=1e-10)
        np.testing.assert_allclose(vp._inv_logit(np.array([-2, 0, 2]))[1], 0.5)

    def test_logit_clips_at_boundaries(self):
        """Scores of 0% or 100% clip to 1e-10 from the edge instead of ±inf,
        so a logit-space fit stays finite."""
        lo, hi = vp._logit(1e-10), vp._logit(1 - 1e-10)
        assert np.isfinite(lo) and lo < 0 and np.isfinite(hi) and hi > 0
        for p in (0, 0.0, 1e-15):
            assert vp._logit(p) == lo
        for p in (1, 1.0, 1 - 1e-15):
            assert vp._logit(p) == hi

    def test_inv_logit_clips_before_exp(self):
        """The input clips at ±500 before exp, so extreme logits don't overflow."""
        with np.errstate(over='raise'):
            assert vp._inv_logit(1000) == 1.0
            assert 0.0 <= vp._inv_logit(-1000) < 1e-200


# ===========================================================================
# Data loading
# ===========================================================================

class TestLoadFrontier:
    def test_frontier_models_carry_p80(self):
        """The loader admits on p50 only; the p80 milestone and chart need p80
        on every frontier model too."""
        missing = [m['name'] for m in vp.load_frontier() if m['p80_min'] is None]
        assert missing == []


def _assert_running_max_flags(data, score):
    """Every model is flagged exactly when it is its date's best and beats
    every earlier date's best."""
    for m in data:
        same_day = [o for o in data if o['date'] == m['date']]
        earlier = [o[score] for o in data if o['date'] < m['date']]
        want = (m is max(same_day, key=lambda o: o[score])
                and (not earlier or m[score] > max(earlier)))
        assert m['is_frontier'] == want, f"{m['name']}: flag {m['is_frontier']}"


class TestLoadEciFrontier:
    def test_frontier_is_running_max(self):
        data = vp.load_eci_frontier()
        assert any(m['is_frontier'] for m in data)
        _assert_running_max_flags(data, 'eci_score')

    def test_one_frontier_model_per_date(self):
        """Same-day releases are one release: only that date's best can take
        the frontier, so GPT-5.4 Pro is on it and GPT-5.4 is not."""
        for kw in ({}, {'full_window': True}, {'country': 'United States of America'}):
            data = vp.load_eci_frontier(**kw)
            frontier = [m for m in data if m['is_frontier']]
            dates = [m['date'] for m in frontier]
            assert len(dates) == len(set(dates)), f"two frontier models share a date: {kw}"
            for m in frontier:
                same_day = [o for o in data if o['date'] == m['date']]
                assert m['eci_score'] == max(o['eci_score'] for o in same_day), \
                    f"{m['name']} is not the best model of its date"

    def test_frontier_does_not_depend_on_row_order(self):
        """The old rule admitted both members of a same-date pair, and which
        one came first was CSV order."""
        d0, d1 = datetime(2025, 1, 1), datetime(2025, 6, 1)
        pair = [{'date': d1, 'eci_score': 150.0, 'name': 'lo'},
                {'date': d1, 'eci_score': 155.0, 'name': 'hi'}]
        for order in (pair, pair[::-1]):
            rows = [{'date': d0, 'eci_score': 140.0, 'name': 'old'}] + [dict(m) for m in order]
            flagged = vp._mark_eci_frontier(rows, 'eci_score')
            assert [m['name'] for m in flagged if m['is_frontier']] == ['old', 'hi']

    def test_full_window_extends_before_the_cutoff(self):
        """full_window=True skips the Feb-2024 cutoff and only that: the
        cutoff view must be exactly the full view's post-cutoff tail."""
        full = vp.load_eci_frontier(full_window=True)
        cut = vp.load_eci_frontier()
        cutoff = datetime(2024, 2, 29)
        assert min(m['date'] for m in full) < cutoff
        assert ({m['name'] for m in full if m['date'] >= cutoff}
                == {m['name'] for m in cut})



class TestLoadRliData:
    def test_frontier_is_running_max(self):
        data = vp.load_rli_data()
        assert any(m['is_frontier'] for m in data)
        _assert_running_max_flags(data, 'rli_score')

    def test_scores_in_reasonable_range(self):
        """RLI scores are percentages, should be 0-100."""
        data = vp.load_rli_data()
        for m in data:
            assert 0 <= m['rli_score'] <= 100, f"{m['name']}: score {m['rli_score']} out of range"


# ===========================================================================
# Backtest helpers
# ===========================================================================

class TestBacktestStats:
    def test_band_membership(self):
        """Models placed at the fan's centre, 1.45σ out (between the 80% and
        90% edges) and 6σ out land in the 50%, 90% and no band."""
        np.random.seed(42)
        start = datetime(2025, 1, 1)
        t = np.arange(60)
        trajs = 10 + 0.1 * t[np.newaxis, :] + np.random.normal(0, 0.5, (20000, 60))
        models = [
            {'date': start + timedelta(days=14), 'val': 11.4, 'name': 'centre'},
            {'date': start + timedelta(days=31), 'val': 13.1 + 1.45 * 0.5, 'name': 'tail'},
            {'date': start + timedelta(days=40), 'val': 14.0 + 3.0, 'name': 'out'},
        ]
        results = vp._backtest_stats(
            models, trajs, start, datetime(2025, 3, 1),
            get_value=lambda m: m['val'], get_name=lambda m: m['name'])
        by = {r['name']: r for r in results}
        assert by['centre']['within_50'] and abs(by['centre']['percentile'] - 50) < 3
        assert (not by['tail']['within_80']) and by['tail']['within_90']
        assert not by['out']['within_90'] and by['out']['percentile'] == 100.0

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
    """The innermost band a result sits in sets its colour."""

    @pytest.mark.parametrize("flags,color", [
        ((True, True, True), '#27ae60'),
        ((False, True, True), '#f1c40f'),
        ((False, False, True), '#e67e22'),
        ((False, False, False), '#e74c3c'),
    ])
    def test_color(self, flags, color):
        r = dict(zip(('within_50', 'within_80', 'within_90'), flags))
        assert vp._bt_color_for(r) == color


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
        """At the floor, the pre- and post-floor rates both equal 1/dt_floor,
        so a 0.01-day step either side grows by 0.01/dt_floor."""
        dt_0, halflife, dt_floor = 100, 365, 10
        t_cap = halflife * np.log2(dt_0 / dt_floor)
        days = np.array([t_cap - 0.01, t_cap, t_cap + 0.01])
        result = vp.superexp_trajectory(days, dt_0, halflife, dt_floor)
        np.testing.assert_allclose(np.diff(result), 0.01 / dt_floor, rtol=1e-4)

    def test_scalar_day_input(self):
        """A scalar day gives the same value as a one-element array."""
        scalar = vp.superexp_trajectory(100.0, dt_0=100, halflife=365, dt_floor=10)
        arr = vp.superexp_trajectory(np.array([100.0]), dt_0=100, halflife=365, dt_floor=10)
        assert np.ndim(scalar) == 0 and scalar == arr[0] > 0


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
    @pytest.mark.parametrize("key", [
        "metr_proj_basis", "milestones", "log_scale", "eci_proj_basis",
        "ecicn_proj_basis", "rli_proj_basis", "emp_proj_basis",
        "rev_end_year", "ecg_highlight",
    ])
    def test_includes_each_tabs_keys(self, key):
        keys, _ = vp._all_tracked()
        assert key in keys
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
        ss["_url_baseline"] = {"custom_dt_lo": 50.0}
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
        eci_all = vp.load_eci_frontier(mtime=vp._eci_mtime())
        for label, spec in vp._ECI_ENTITY_SPECS.items():
            if not spec.get("orgs"):
                continue  # country entities, not org entities
            tab = vp.load_eci_frontier(mtime=vp._eci_mtime(),
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

    def test_country_entities_lead_the_dropdown(self):
        # _ECI_ENTITY_OPTIONS[0] is the ECI tab's default benchmark, so the
        # country aggregates must stay ahead of the companies.
        assert vp._ECI_ENTITY_OPTIONS[0] == "US best"
        n = len(vp._ECI_COUNTRY_ENTITIES)
        assert vp._ECI_ENTITY_OPTIONS[:n] == list(vp._ECI_COUNTRY_ENTITIES)
        assert vp._ECI_ENTITY_OPTIONS[n:] == list(vp._ECI_COMPANIES)

    def test_slugs_round_trip(self):
        # The registry check dedupes company slugs among themselves; this one
        # also catches a company slug colliding with a country's ("us", "china").
        for label, slug in vp._ECI_ENTITY_SLUG.items():
            assert vp._ECI_ENTITY_FOR_SLUG[slug] == label


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


class TestTodayForward:
    """Today-forward mode: `_rel_date_label`, `_relativize_dates` and the
    figure post-process `_today_forward`."""

    T = datetime(2026, 9, 3)

    def _d(self, days):
        return self.T + timedelta(days=days)

    def test_labels_count_days_to_thirty_then_calendar_units(self):
        lab = lambda n: vp._rel_date_label(self._d(n), self.T)
        assert lab(0) == "today"
        assert lab(1) == "1 day from today"
        assert lab(30) == "30 days from today"
        assert lab(33) == "1mo 3d from today"
        assert lab(-3) == "3 days ago"
        assert lab(-45) == "1mo 14d ago"

    def test_labels_keep_two_units_and_use_calendar_months(self):
        # Sep 3 → Nov 11 two years on is 2y 2mo 8d; the days drop.
        assert vp._rel_date_label(datetime(2028, 11, 11), self.T) == "2y 2mo from today"
        # Calendar units: Oct 3 is 30 days (still counted in days), Nov 3 is
        # two whole months, and the day before it is one month plus October's
        # leftover.
        assert vp._rel_date_label(datetime(2026, 10, 3), self.T) == "30 days from today"
        assert vp._rel_date_label(datetime(2026, 11, 3), self.T) == "2mo from today"
        assert vp._rel_date_label(datetime(2026, 11, 2), self.T) == "1mo 30d from today"
        assert vp._rel_date_label(datetime(2027, 9, 3), self.T) == "1y from today"

    def test_relativize_rewrites_every_date_form_and_nothing_else(self):
        got = vp._relativize_dates(
            "Sep 03, 2026 · 2026-10-06 · Mar 2028 · ECI 5.4 · 2027-13-40", self.T)
        assert got == ("today · 1mo 3d from today · ~1y 6mo from today · ECI 5.4"
                       " · 2027-13-40")

    def _fig(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[self._d(-30), self._d(60)], y=[1, 2],
                                 hovertemplate='%{x|%b %Y}<br>v %{y}'))
        fig.add_trace(go.Scatter(x=[self._d(10)], y=[1], mode='markers',
                                 hovertext=[f"{self._d(10):%b %d, %Y}"]))
        fig.add_trace(go.Scatter(x=[self._d(10)], y=[1], mode='markers',
                                 hoverinfo='text', text=["Sep 13, 2026"]))
        fig.add_trace(go.Scatter(x=[self._d(10)], y=[1], mode='markers+text',
                                 hoverinfo='text', text=["Sep 13, 2026"]))
        fig.update_layout(hovermode='x unified',
                          xaxis=dict(range=[self._d(-400), self._d(500)]))
        return fig

    def test_figure_opens_at_today_and_keeps_its_end(self):
        fig = vp._today_forward(self._fig(), self.T)
        assert fig.layout.xaxis.range[0] == self.T
        assert fig.layout.xaxis.range[1] == self._d(500)
        # A unified hover would print the calendar date as its header.
        assert fig.layout.hovermode == 'closest'

    def test_figure_without_a_range_ends_at_its_data(self):
        fig = self._fig()
        fig.update_layout(xaxis=dict(range=None))
        vp._today_forward(fig, self.T)
        assert fig.layout.xaxis.range[0] == self.T
        assert fig.layout.xaxis.range[1] == self._d(60)

    def test_hovers_go_relative(self):
        fig = vp._today_forward(self._fig(), self.T)
        t0, t1, t2, t3 = fig.data
        assert t0.hovertemplate == '%{customdata}<br>v %{y}'
        assert list(t0.customdata) == ["30 days ago", "1mo 30d from today"]
        assert list(t1.hovertext) == ["10 days from today"]
        # `text` is rewritten only when it serves as hover text, never when it
        # is drawn on the chart.
        assert list(t2.text) == ["10 days from today"]
        assert list(t3.text) == ["Sep 13, 2026"]

    def test_existing_customdata_is_left_alone(self):
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=[self._d(5)], y=[1], customdata=["k"],
                                   hovertemplate='%{x|%b %Y} %{customdata}'))
        vp._today_forward(fig, self.T)
        assert list(fig.data[0].customdata) == ["k"]
        assert fig.data[0].hovertemplate == '%{x|%b %Y} %{customdata}'

    def _ramp(self, ys, **layout):
        import plotly.graph_objects as go
        xs = [self._d(d) for d in range(-300, 300, 10)]
        fig = go.Figure(go.Scatter(x=xs, y=ys(len(xs))))
        fig.add_trace(go.Scatter(x=[xs[0], xs[-1]], y=[50, 50], hoverinfo='skip'))
        fig.update_layout(xaxis=dict(range=[xs[0], xs[-1]]), **layout)
        return fig

    def test_y_axis_refits_to_the_forward_data(self):
        # History sized the axis; forward data sits in a sliver, so it refits —
        # and the two-point reference level at 50 is not data.
        fig = self._ramp(lambda n: [0.1 + 0.01 * i for i in range(n)],
                         yaxis=dict(range=[-10, 60]))
        vp._today_forward(fig, self.T)
        lo, hi = fig.layout.yaxis.range
        assert 0.35 < lo < 0.4 and 0.69 < hi < 0.75

    def test_log_axis_refits_in_log10(self):
        fig = self._ramp(lambda n: [10 ** (i / 20) for i in range(n)],
                         yaxis=dict(type='log', range=[-1, 4]))
        vp._today_forward(fig, self.T)
        lo, hi = fig.layout.yaxis.range
        assert 1.4 < lo < 1.5 and 3.0 < hi < 3.1

    def test_a_categorical_y_axis_is_left_alone(self):
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=[self._d(5), self._d(9)], y=["US", "China"],
                                   mode='markers'))
        vp._today_forward(fig, self.T)
        assert fig.layout.yaxis.range is None

    def test_a_range_the_data_mostly_spans_is_kept(self):
        fig = self._ramp(lambda n: list(np.linspace(-70, 90, n)),
                         yaxis=dict(range=[0, 100]))
        vp._today_forward(fig, self.T)
        assert tuple(fig.layout.yaxis.range) == (0, 100)

    def test_toggle_is_url_tracked_and_reset_by_no_tab(self):
        keys, defaults = vp._all_tracked()
        assert vp._TODAY_FWD_KEY in keys and defaults[vp._TODAY_FWD_KEY] is False
        for name in dir(vp):
            if name.endswith("_RESET_KEYS"):
                assert vp._TODAY_FWD_KEY not in getattr(vp, name), name


class TestCcResponsibleCluster:
    """_cc_responsible_cluster: the release → cluster direction, with pins.

    The rule takes the latest record step online a training run before the
    release; a `_CC_TRAINED_ON` pin restricts that to one site and falls back
    to the rule when the site has no eligible step.
    """

    FLOOR = timedelta(days=vp._CC_TRAIN_FLOOR_DAYS)

    def _steps(self):
        return [(datetime(2026, 1, 1), 1.0, "Small"),
                (datetime(2026, 3, 1), 2.0, "Big")]

    def test_default_is_the_latest_eligible_step(self):
        rel = datetime(2026, 3, 1) + self.FLOOR
        got, pinned = vp._cc_responsible_cluster(rel, "Unpinned", "Lab",
                                                 self._steps())
        assert got[2] == "Big" and pinned is False

    def test_a_step_exactly_one_training_run_before_qualifies(self):
        # The floor is inclusive: online exactly _CC_TRAIN_FLOOR_DAYS before
        # the release counts, one day later does not.
        rel = datetime(2026, 6, 1)
        exact = (rel - self.FLOOR, 5.0, "Exactly one training run")
        short = (rel - self.FLOOR + timedelta(days=1), 9.0, "One day short")
        got, pinned = vp._cc_responsible_cluster(rel, "X", "Lab", [exact, short])
        assert got[2] == "Exactly one training run" and pinned is False

    def test_nothing_predates_the_release(self):
        got, pinned = vp._cc_responsible_cluster(datetime(2025, 1, 1), "X",
                                                 "Lab", self._steps())
        assert got is None and pinned is False

    def test_pin_takes_the_named_site(self, monkeypatch):
        monkeypatch.setattr(vp, "_CC_TRAINED_ON", {("Lab", "Model"): "Small"})
        rel = datetime(2026, 3, 1) + self.FLOOR
        got, pinned = vp._cc_responsible_cluster(rel, "Model (high)", "Lab",
                                                 self._steps())
        assert got[2] == "Small" and pinned is True

    def test_pin_falls_back_when_the_site_has_no_eligible_step(self, monkeypatch):
        # Pinned to a site whose only step came online after the release.
        monkeypatch.setattr(vp, "_CC_TRAINED_ON", {("Lab", "Model"): "Big"})
        rel = datetime(2026, 1, 1) + self.FLOOR
        got, pinned = vp._cc_responsible_cluster(rel, "Model", "Lab",
                                                 self._steps())
        assert got[2] == "Small" and pinned is False

    def test_pin_is_per_lab(self, monkeypatch):
        monkeypatch.setattr(vp, "_CC_TRAINED_ON", {("Other", "Model"): "Small"})
        rel = datetime(2026, 3, 1) + self.FLOOR
        got, pinned = vp._cc_responsible_cluster(rel, "Model", "Lab",
                                                 self._steps())
        assert got[2] == "Big" and pinned is False

    def test_every_pin_names_a_live_model_and_site(self):
        # A pin naming a model Epoch no longer lists, or a site outside the
        # lab's own milestone series, is dead weight — surface it on refresh.
        attr = vp._cc_lab_attribution(mtime=vp._dc_meta_mtime())
        rel = vp._cc_company_all_releases(mtime=vp._eci_mtime())
        assert vp._CC_TRAINED_ON, "the pin table exists to hold pins"
        for (lab, model), site in vp._CC_TRAINED_ON.items():
            assert lab in vp._CC_PANEL_LABS
            hits = [t for t in rel.get(lab, []) if t[2] == model]
            assert hits, f"{model!r} is not an Epoch {lab} release"
            ms = vp._cc_lab_dc_milestones(lab, attr, key='perf')
            assert any(m[2] == site for m in ms), f"{site!r} sets no {lab} record"
            got, pinned = vp._cc_responsible_cluster(hits[0][0], model, lab, ms)
            assert pinned and got[2] == site, f"pin for {model!r} does not bind"


class TestCcCompanyAllReleases:
    """_cc_company_all_releases: the tier-3 fallback pool."""

    def test_keys_on_model_name_and_release_date(self, monkeypatch):
        # A model's reasoning-effort rows (one Model name, one date, several
        # Display names) are one release carrying the best score; the same
        # Model name on another date is a separate release.
        def row(display, name, date, eci):
            return {"Organization": "OpenAI", "Display name": display,
                    "Model name": name, "Release date": date, "ECI Score": eci}
        rows = [row("GPT-X (low)", "GPT-X", "2026-01-01", "140"),
                row("GPT-X (high)", "GPT-X", "2026-01-01", "150"),
                row("GPT-X (high)", "GPT-X", "2026-03-01", "152"),
                row("Other", "Other", "2026-01-01", "")]
        monkeypatch.setattr(vp, "csv",
                            types.SimpleNamespace(DictReader=lambda _f: rows))
        # Under real Streamlit (an AppTest earlier on this worker) the loader is
        # cached: bypass it so the fixture is read and never cached.
        fn = getattr(vp._cc_company_all_releases, "__wrapped__",
                     vp._cc_company_all_releases)
        assert fn()["OpenAI"] == [
            (datetime(2026, 1, 1), 150.0, "GPT-X"),
            (datetime(2026, 1, 1), None, "Other"),
            (datetime(2026, 3, 1), 152.0, "GPT-X")]

    def test_keeps_redated_revisions_of_one_model_name_apart(self):
        # GPT-4o shipped dated revisions; collapsing them would lose releases
        # the frontier series has. Epoch has spelled them both as one Model
        # name and as dated names ("GPT-4o (May 2024)"), so match either.
        # Asserted as a superset — Epoch can add revisions.
        rel = vp._cc_company_all_releases()["OpenAI"]
        gpt4o = {t[0] for t in rel
                 if t[2] == "GPT-4o" or t[2].startswith("GPT-4o (")}
        assert {datetime(2024, 5, 13), datetime(2024, 8, 6)} <= gpt4o
        assert len(gpt4o) >= 3

    def test_same_day_releases_are_ordered_strongest_first(self):
        # A step matching a shared release date should be offered the
        # flagship, not whichever row sorted first.
        groups = 0
        for lab, rel in vp._cc_company_all_releases().items():
            by_date = {}
            for t in rel:
                by_date.setdefault(t[0], []).append(t[1] or 0)
            for d, ecis in by_date.items():
                if len(ecis) > 1:
                    groups += 1
                    assert ecis == sorted(ecis, reverse=True), (lab, d)
        assert groups, "no same-day releases left to order"

    def test_is_a_superset_of_the_frontier(self):
        # Every running-max release must exist in the pool on the same date, or
        # tier 3 could contradict tier 1.
        allr = vp._cc_company_all_releases()
        for lab, front in vp._cc_company_frontier_models().items():
            dates = {t[0] for t in allr[lab]}
            for d, _e, _n in front:
                assert d in dates, f"{lab}: frontier date {d} missing from pool"


class TestCcDcDerivation:
    """The CC tab derives its compute inputs from the DC tab's machinery."""

    def test_lab_attribution_reads_the_loaders_fields(self):
        """Every site→lab mapping is explainable by the loader's own operator
        or aliased-users fields, and a fallback-labelled site never maps to a
        lab — so a spelling change Epoch makes is fixed in the loader once."""
        attr = vp._cc_lab_attribution()
        by_name = {dc['name']: dc for dc in vp.dc_all}
        assert set(attr) == set(by_name)
        mapped = 0
        for name, lab in attr.items():
            dc = by_name[name]
            if lab is None:
                continue
            mapped += 1
            assert dc['attributed'], name
            cands = {vp._CC_LAB_ALIASES.get(dc['operator'], dc['operator'])}
            cands |= {vp._CC_LAB_ALIASES.get(u, u) for u in dc['users']}
            assert any(c == lab or (lab == 'Google' and c.startswith('Google'))
                       for c in cands), (name, lab, cands)
        assert mapped > 0

    def test_colossus_is_xais_buildout_not_its_tenants(self):
        """Operator outranks tenants here (unlike the DC/Pacing shared-tenancy
        rule): Colossus 2 lists Anthropic first among its users, but the
        release-timing panel credits the buildout to its builder, xAI — which
        also pins the SpaceXAI→xAI alias against the live CSV."""
        attr = vp._cc_lab_attribution()
        spacex = [dc['name'] for dc in vp.dc_all if dc['operator'] == 'SpaceXAI']
        assert spacex, "no SpaceXAI-operated site in the live CSV — retire the alias?"
        for name in spacex:
            assert attr[name] == 'xAI', name

    def test_milestones_match_the_shared_envelope(self):
        """_cc_lab_dc_milestones is the DC envelope filtered to record steps:
        one milestone per date, values strictly increasing."""
        attr = vp._cc_lab_attribution()
        for lab in vp._CC_PANEL_LABS:
            ms = vp._cc_lab_dc_milestones(lab, attr, key='train_flop')
            dates = [d for d, _v, _n in ms]
            vals = [v for _d, v, _n in ms]
            assert dates == sorted(dates)
            assert len(set(dates)) == len(dates), f"{lab}: duplicate-date record"
            assert all(b > a for a, b in zip(vals, vals[1:])), lab

    def test_run_window_scales_levels_only(self):
        """The 6-month frontier is the 2-month one scaled by the window ratio
        and slid one window later — same leaders, same structure — so the
        sidebar's Run length toggle changes levels and dates, never which
        sites make the frontier."""
        cap = datetime(2028, 12, 31)
        fr2 = vp._cc_trainflop_frontier(vp.dc_all, cap, key='train_flop',
                                        run_days=vp._DAYS_2MO)
        fr6 = vp._cc_trainflop_frontier(vp.dc_all, cap, key='train_flop_6mo',
                                        run_days=vp._DAYS_6MO)
        assert len(fr2) == len(fr6) > 0
        ratio = vp._SECONDS_6MO / vp._SECONDS_2MO
        dshift = timedelta(days=vp._DAYS_6MO - vp._DAYS_2MO)
        for (d2, v2), (d6, v6) in zip(fr2, fr6):
            assert d6 == d2 + dshift
            assert abs(v6 / v2 - ratio) < 1e-9

    def test_us_pace_cross_check_agrees_across_tabs(self):
        """The 'recent built' segment fit (this tab) and the by-country engine's
        US fit (DC tab) measure the same buildout two ways; they may differ —
        lab-only vs any-landlord, event points vs monthly samples — but a gap
        beyond twice the country fit's own pace uncertainty means one of the
        two engines drifted."""
        today = datetime.now()
        cap = datetime(2028, 12, 31) + timedelta(days=vp._DAYS_2MO)
        fits = vp._cc_segment_fits(vp._cc_trainflop_frontier(vp.dc_all, cap),
                                   today)
        # The recent segment is the one that ends today.
        recent = next(lab for lab, start, end in vp._CC_SEGMENTS
                      if start is not None and end is None)
        g_recent = next(f['slope_oom'] for f in fits if f['label'] == recent)
        chk = vp._cc_country_pace_check(today)[vp._DC_CTY_US]
        assert chk is not None
        assert abs(g_recent - chk['g']) < 2 * chk['sigma_g'], \
            (g_recent, chk['g'], chk['sigma_g'])


class TestCcCnComputeBand:
    """The hand-set China compute band vs the DC tab's catalogued paces."""

    def test_band_sits_at_or_below_the_catalogue_paces(self):
        """The band claims coherent-single-run growth, a strictly harder thing
        than building halls, so it must not exceed what the catalogue shows:
        HI ≤ the China-accessible ramp, LO ≤ the domestic-sites pace. If an
        Epoch refresh breaks this, retarget the constants deliberately (the
        catalogue slowing below the band means the judgment is now optimistic)
        rather than loosening the test."""
        chk = vp._cc_country_pace_check(datetime.now())
        ca, cd = chk[vp._DC_CTY_CN_ACCESS], chk[vp._DC_CTY_CN_DOMESTIC]
        assert ca is not None and cd is not None, (
            "Epoch no longer catalogues enough Chinese sites to fit — the "
            "cross-check (and the captions quoting it) are dead weight")
        assert vp._CC_CN_COMPUTE_HI <= ca['g'], (vp._CC_CN_COMPUTE_HI, ca['g'])
        assert vp._CC_CN_COMPUTE_LO <= cd['g'], (vp._CC_CN_COMPUTE_LO, cd['g'])


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

    def test_recovers_exchange_rate_and_efficiency(self):
        e = vp._cc_efficiency(self._rows())
        assert e["eci_per_oom"] == pytest.approx(20.0, rel=1e-3)  # 1/alpha
        assert e["g_inv"] == pytest.approx(0.4, rel=1e-3)
        assert 0.0 < e["g_central"] < 1.0
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

    def test_recovers_capability_rate(self):
        r = vp._cc_iso_compute(self._rows())
        assert r["eci_per_yr"] == pytest.approx(10.0, rel=1e-2)
        assert r["lo"] <= r["eci_per_yr"] <= r["hi"]


@pytest.mark.parametrize("fn, rows", [
    (lambda r: vp._cc_decomp(r), lambda: TestCcDecomp()._rows()),
    (lambda r: vp._cc_efficiency(r), lambda: TestCcEfficiency()._rows()),
    (lambda r: vp._cc_iso_compute(r), lambda: TestCcIsoCompute()._rows()),
    (lambda r: vp._cc_pooled_decomp(r)[0], lambda: TestCcPooledDecomp()._rows()),
], ids=["decomp", "efficiency", "iso_compute", "pooled_decomp"])
def test_cc_fits_decline_under_ten_rows(fn, rows):
    """Each model-level CC fit returns nothing on fewer than ten rows."""
    assert fn(rows()[:9]) is None


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

    @pytest.mark.parametrize("n, country", [(5, "China"), (8, "Narnia")],
                             ids=["sparse", "unknown"])
    def test_too_few_rows_for_the_country_returns_none(self, n, country):
        rate, got_n, med = vp._cc_iso_compute_rate(self._rows(n=n), country)
        assert rate is None and got_n == 0 and med is None


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

    @pytest.fixture(autouse=True)
    def _seeded(self):
        """The samplers draw from the global np.random: seed it, then restore."""
        state = np.random.get_state()
        np.random.seed(0)
        yield
        np.random.set_state(state)

    def test_central_case_is_gap_over_rate(self):
        # Every range is degenerate here, so this also covers the padding that
        # keeps np.random.triangular from raising on lo == hi.
        y, r = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=2000, **self._ARGS)
        assert np.all(np.isfinite(y)) and np.all(r > 0)
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

    def test_algo_mode_outside_range_is_clamped(self):
        args = dict(self._ARGS, algo_lo=8.0, algo_mid=99.0, algo_hi=12.0)
        _, r = vp._cc_cn_target_years(pace_lo=1.0, pace_hi=1.0, n=4000, **args)
        # Mode clamps to algo_hi=12, so no sample can exceed the envelope — the
        # widest algo, compute, and pace draws, each including the ±pad the helper
        # adds to the degenerate g and pace ranges.
        assert r.max() <= (12.0 + 10.0 * (0.2 + 0.01)) * 1.05 + 1e-9
        # ...and a mode pinned at the top skews the draw high, not to the middle.
        assert np.median(r) > 0.5 * ((8.0 + 1.9) + (12.0 + 2.1))


class TestCcInnovationAlgoBand:
    def test_band_is_ordered_and_below_the_measured_central(self):
        cc = vp.load_eci_compute()
        band = vp._cc_innovation_algo_band(cc)
        assert band is not None
        lo, hi = band
        assert 0 < lo < hi
        # The indigenous band sits at or below the distillation-inclusive
        # central rate — the premise of the correction.
        assert hi <= vp._cc_iso_compute(cc)['eci_per_yr'] + 1.0


class TestCcFrontierGradeAlgo:
    def test_time_coefficient_slows_at_the_frontier(self):
        """The surviving distillation fingerprint is one-way: frontier-grade
        models (near-frontier ECI at frontier-scale compute) gain ECI at
        fixed compute well below the pooled fit's rate — followers ride a
        teacher, the frontier cannot. No assertion on a_partial rising: the
        old two-way gradient was an artifact of the ECI tab's Feb-2024
        cutoff auto-admitting every earlier model as near-frontier."""
        cc = vp.load_eci_compute()
        eci = vp.load_eci_frontier(full_window=True)
        dec = vp._cc_decomp(cc)
        r = vp._cc_frontier_grade_algo(cc, eci, margin=5.0)
        assert r is not None, "margin 5 too thin to fit"
        assert r['b_time'] < dec['b_time']
        pure = vp._cc_pure_innovation_band(cc, eci)
        assert pure is not None and pure[0] < pure[1], "pure-innovation band collapsed"

    def test_flop_screen_drops_the_low_compute_distillers(self):
        """Without the compute screen the 'can't-distill' subset admits the
        models nearest the frontier at the least compute — the heaviest
        distillers — and its time coefficient drifts up toward the pooled
        fit's. The screen must both admit extra rows and slow the fit."""
        cc = vp.load_eci_compute()
        eci = vp.load_eci_frontier(full_window=True)
        tight = vp._cc_frontier_grade_algo(cc, eci, margin=5.0)
        loose = vp._cc_frontier_grade_algo(cc, eci, margin=5.0, flop_margin=99.0)
        assert tight is not None and loose is not None
        assert loose['n'] > tight['n']
        assert loose['b_time'] > tight['b_time']

    def test_rows_before_the_frontier_series_are_dropped_not_admitted(self):
        """A cc row predating eci_all's coverage has no frontier to compare
        against; it must be excluded from the refit, never auto-admitted as
        near-frontier (the old behavior, via the frontier-of-zero fallback)."""
        base = datetime(2025, 1, 1)
        eci = [{'date': base + timedelta(days=30 * i), 'eci_score': 100.0 + i}
               for i in range(12)]
        good = [{'date': base + timedelta(days=30 * i), 'eci': 100.0 + i,
                 'log10_flop': 25.0 + 0.05 * i} for i in range(12)]
        early = [{'date': base - timedelta(days=600 - 30 * i), 'eci': 50.0,
                  'log10_flop': 24.0} for i in range(12)]
        r = vp._cc_frontier_grade_algo(early + good, eci, margin=5.0)
        assert r is not None and r['n'] == len(good)


class TestCcRateCalibration:
    def test_scales_a_build_up_rate_to_the_observed_slope(self):
        """A calibrated rate reproduces the frontier's own since-2024 slope,
        so a projected line fits its history; no data leaves it unscaled."""
        base = vp._CC_CALIB_SINCE
        fr = [(base + timedelta(days=90 * i), 140.0 + 15.0 * 90 * i / 365.25, "m")
              for i in range(8)]
        k = vp._cc_rate_calibration(fr, 12.0)
        assert k * 12.0 == pytest.approx(15.0, rel=1e-6)
        assert vp._cc_rate_calibration(fr[:1], 12.0) == 1.0
        assert vp._cc_rate_calibration(fr, 0.0) == 1.0

    def test_live_us_build_up_under_runs_its_frontier(self):
        """The case the calibration exists for: on live data the US build-up
        rate falls short of the observed slope. Retarget if a refresh flips it."""
        eci_all = vp.load_eci_frontier(mtime=vp._eci_mtime())
        us_fr = vp._cc_country_frontier(eci_all, 'United States of America')
        rows = vp.load_eci_compute(mtime=vp._eci_mtime())
        fgm = vp._cc_frontier_grade_algo(
            rows, vp.load_eci_frontier(mtime=vp._eci_mtime(), full_window=True))
        _, lo, hi = vp._cc_us_compute_band(datetime.now(), datetime(2029, 12, 31))
        k = vp._cc_rate_calibration(
            us_fr, fgm['b_time'] + fgm['a_partial'] * 0.5 * (lo + hi))
        assert k > 1.0


class TestCcGapEndRange:
    def test_range_spans_the_window_choices(self):
        """The end-year gap's hover range covers every alternative fit start,
        and on live data the since-2024 central sits inside it."""
        eci_all = vp.load_eci_frontier(mtime=vp._eci_mtime())
        us_fr = vp._cc_country_frontier(eci_all, 'United States of America')
        cn_fr = vp._cc_country_frontier(eci_all, 'China')
        horizon = datetime(2029, 12, 31)
        (lo, hi), us_r, cn_r = vp._cc_gap_end_range(us_fr, cn_fr, horizon)
        us_d, us_v = max(((d, s) for d, s, n in us_fr), key=lambda p: p[1])
        cn_d, cn_v = max(((d, s) for d, s, n in cn_fr), key=lambda p: p[1])
        central = (
            us_v + vp._cc_frontier_eci_slope(us_fr, vp._CC_CALIB_SINCE)
            * (horizon - us_d).days / 365.25
            - cn_v - vp._cc_frontier_eci_slope(cn_fr, vp._CC_CALIB_SINCE)
            * (horizon - cn_d).days / 365.25)
        assert lo <= central <= hi
        assert hi - lo > 2.0
        assert us_r[1] - us_r[0] < cn_r[1] - cn_r[0]


class TestCcCnChannels:
    def test_channels_sum_to_the_observed_slope_and_feed_the_sim(self):
        """The China tab and the Pacing sim share one split of China's
        growth: the channels add to its observed slope, and the sim's
        bands, re-centred on them, start China at that slope."""
        rows = vp.load_eci_compute(mtime=vp._eci_mtime())
        eci_all = vp.load_eci_frontier(mtime=vp._eci_mtime())
        cn_fr = vp._cc_country_frontier(eci_all, 'China')
        fgm = vp._cc_frontier_grade_algo(
            rows, vp.load_eci_frontier(mtime=vp._eci_mtime(), full_window=True))
        ch = vp._cc_cn_channels(rows, cn_fr, fgm['a_partial'])
        parts = (ch['compute'] + ch['innovation'] + ch['diffusion']
                 + ch['distillation'])
        assert parts == pytest.approx(ch['obs'], abs=1e-9)
        assert ch['distillation'] > 0 and ch['diffusion'] > 0
        np.random.seed(0)
        _, grid, traj = vp._cc_cn_crossing_sim(
            150.0, 1e9, us_anchor=1e6, us_rate=0.0, a_partial=fgm['a_partial'],
            g_lo=vp._CC_CN_COMPUTE_LO, g_hi=vp._CC_CN_COMPUTE_HI,
            algo_lo=ch['algo'][0], algo_mid=ch['algo'][1], algo_hi=ch['algo'][2],
            inno_lo=ch['inno'][0], inno_hi=ch['inno'][1],
            pure_lo=ch['pure'][0], pure_hi=ch['pure'][1],
            pace_lo=1.0, pace_hi=1.0, horizon_yrs=1.0, n=4000)
        first_month = (traj[:, 1] - 150.0).mean() / grid[1]
        assert first_month == pytest.approx(ch['obs'], abs=0.3)


class TestCcCrossingBacktest:
    def test_level_noise_brings_the_80pct_band_near_80pct(self):
        """Rolling origins over China's 2025-26 releases: the crossing sim's
        paths cover about 80% of later releases inside their 80% band with
        `_cc_frontier_level_sd`, and far fewer from the rate spread alone."""
        rows = vp.load_eci_compute(mtime=vp._eci_mtime())
        eci_all = vp.load_eci_frontier(mtime=vp._eci_mtime())
        cn_fr = vp._cc_country_frontier(eci_all, 'China')
        us_fr = vp._cc_country_frontier(eci_all, 'United States of America')
        fgm = vp._cc_frontier_grade_algo(
            rows, vp.load_eci_frontier(mtime=vp._eci_mtime(), full_window=True))
        us_algo, _, _ = vp._cc_iso_compute_rate(rows, 'United States of America')
        cn_algo, _, _ = vp._cc_iso_compute_rate(rows, 'China')
        inno = vp._cc_innovation_algo_band(rows)
        a, g_lo, g_hi = fgm['a_partial'], vp._CC_CN_COMPUTE_LO, vp._CC_CN_COMPUTE_HI
        _, ug_lo, ug_hi = vp._cc_us_compute_band(datetime.now(), datetime(2029, 12, 31))
        us_rate = (vp._cc_rate_calibration(us_fr, fgm['b_time'] + a * 0.5 * (ug_lo + ug_hi))
                   * (fgm['b_time'] + a * 0.5 * (ug_lo + ug_hi)))
        pace_lo, pace_hi, _ = vp._cc_cn_pace_band(cn_fr, cn_algo + a * 0.5 * (g_lo + g_hi))
        gap_d = vp._cc_release_gap_days(cn_fr, since=datetime.now() - timedelta(days=730))

        def best_at(fr, d):
            return max(s for dd, s, n in fr if dd <= d)

        def coverage(level_sd):
            inside = total = 0
            origins = [datetime(2025, 1, 1)] + [d for d, s, n in cn_fr
                                                if d.year >= 2025][:-2]
            for d0 in origins:
                np.random.seed(0)
                _, grid, traj = vp._cc_cn_crossing_sim(
                    best_at(cn_fr, d0), 1e9, us_anchor=best_at(us_fr, d0),
                    us_rate=us_rate, a_partial=a, g_lo=g_lo, g_hi=g_hi,
                    algo_lo=min(us_algo, cn_algo), algo_mid=cn_algo,
                    algo_hi=max(us_algo, cn_algo), inno_lo=inno[0],
                    inno_hi=inno[1], pace_lo=pace_lo, pace_hi=pace_hi,
                    horizon_yrs=2.0, n=2000, level_sd=level_sd,
                    level_ramp_yrs=gap_d / 365.25)
                p10, p90 = (np.percentile(traj, q, axis=0) for q in (10, 90))
                for d, s, n in cn_fr:
                    if d > d0:
                        t = (d - d0).days / 365.25
                        total += 1
                        inside += np.interp(t, grid, p10) <= s <= np.interp(t, grid, p90)
            return inside / total

        assert coverage(0.0) < 0.4
        assert 0.65 <= coverage(vp._cc_frontier_level_sd(cn_fr)) <= 0.95

        # Dates, as the catch-up card reports them: from each origin, the
        # date China first reached each later release's level should land
        # inside the 80% crossing range about 80% of the time, with misses
        # on both sides. A release wait on top skews it late.
        def date_coverage(wait):
            inside = early = late = 0
            origins = [datetime(2025, 1, 1)] + [d for d, s, n in cn_fr
                                                if d.year >= 2025][:-1]
            for d0 in origins:
                c0 = best_at(cn_fr, d0)
                for d_hit, target in [(d, s) for d, s, n in cn_fr
                                      if d > d0 and s > c0]:
                    np.random.seed(0)
                    yrs, _, _ = vp._cc_cn_crossing_sim(
                        c0, target, us_anchor=best_at(us_fr, d0),
                        us_rate=us_rate, a_partial=a, g_lo=g_lo, g_hi=g_hi,
                        algo_lo=min(us_algo, cn_algo), algo_mid=cn_algo,
                        algo_hi=max(us_algo, cn_algo), inno_lo=inno[0],
                        inno_hi=inno[1], pace_lo=pace_lo, pace_hi=pace_hi,
                        horizon_yrs=3.0, n=800,
                        release_gap_days=gap_d if wait else None,
                        level_sd=vp._cc_frontier_level_sd(cn_fr),
                        level_ramp_yrs=gap_d / 365.25)
                    lo, hi = np.percentile(yrs[np.isfinite(yrs)], [10, 90])
                    t = (d_hit - d0).days / 365.25
                    early += t < lo
                    late += t > hi
                    inside += lo <= t <= hi
            return inside / (inside + early + late), early, late

        cov, early, late = date_coverage(wait=False)
        assert 0.65 <= cov <= 0.95 and early > 0 and late > 0
        _, early_w, late_w = date_coverage(wait=True)
        assert early_w > 2 * late_w


class TestCcCnLevelOffset:
    def test_recovers_a_constructed_level_premium(self):
        base = datetime(2024, 6, 1)
        rows = []
        for i in range(12):
            d = base + timedelta(days=60 * i)
            fl = 24.3 if i % 2 else 24.7
            rows.append({'date': d, 'country': 'United States of America',
                         'eci': 100.0 + 12.0 * (d - base).days / 365.25,
                         'log10_flop': fl})
            rows.append({'date': d, 'country': 'China',
                         'eci': 106.0 + 12.0 * (d - base).days / 365.25,
                         'log10_flop': fl})
        off, n = vp._cc_cn_level_offset(rows)
        assert n == 24
        assert off == pytest.approx(6.0, abs=0.2)

    def test_live_offset_is_a_positive_level_in_points(self):
        """The measured distillation edge quoted in the CC control caption
        and the Pacing distillation checkbox: China sits a few ECI above US
        models at matched compute and date. If an Epoch refresh pushes it
        out of this range, re-check the captions before loosening."""
        lvl = vp._cc_cn_level_offset(vp.load_eci_compute())
        assert lvl is not None
        off, n = lvl
        assert n >= 20
        assert 0.0 < off < 12.0


class TestCcCnPaceBand:
    def test_degenerate_input_falls_back(self):
        lo, hi, obs = vp._cc_cn_pace_band([], 14.0)
        assert (lo, hi, obs) == (0.85, 1.15, None)


class TestCcCnCrossingSim:
    KW = dict(a_partial=8.0, g_lo=0.15, g_hi=0.30, algo_lo=11.0,
              algo_mid=12.5, algo_hi=13.0, pace_lo=0.9, pace_hi=1.1, n=4000)

    @pytest.fixture(autouse=True)
    def _seeded(self):
        """The samplers draw from the global np.random: seed it, then restore."""
        state = np.random.get_state()
        np.random.seed(0)
        yield
        np.random.set_state(state)

    def test_distil_saturated_matches_constant_rate(self):
        # With the US pulling away (gap/gap0 >= 1 throughout) the decay never
        # binds and the sim reduces to the constant-rate sampler.
        y_sim, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=165.0, us_rate=25.0,
            inno_lo=3.0, inno_hi=4.0, **self.KW)
        y_const, _ = vp._cc_cn_target_years(
            150.0, 160.0, 11.0, 12.5, 13.0, 8.0, 0.15, 0.30,
            pace_lo=0.9, pace_hi=1.1, n=4000)
        assert abs(np.nanmedian(y_sim) - np.nanmedian(y_const)) < 0.15

    def test_pause_is_slower_for_the_same_bar(self):
        # Frozen US: the gap dries up en route, distillation decays to the
        # indigenous rate, and the same bar takes longer to cross.
        y_pause, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=160.0, us_rate=0.0,
            inno_lo=3.0, inno_hi=4.0, **self.KW)
        y_move, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=160.0, us_rate=15.0,
            inno_lo=3.0, inno_hi=4.0, **self.KW)
        assert np.nanmedian(y_pause) > np.nanmedian(y_move)

    def test_already_there_crosses_immediately(self):
        y, _, _ = vp._cc_cn_crossing_sim(
            150.0, 149.0, us_anchor=160.0, us_rate=0.0,
            inno_lo=3.0, inno_hi=4.0, **self.KW)
        assert np.nanmedian(y) <= 1.0 / 12.0 + 1e-9

    def test_pause_to_frozen_bar_lands_after_161_with_us_moving(self):
        """The tab-level ordering on live data: the Pacing tab's pause panel (frozen
        US best, distillation drying up) must cross later than the CC section's
        161 bar (US moving, distillation active) — same anchor, same pace band."""
        cc = vp.load_eci_compute()
        eci = vp.load_eci_frontier()
        us_fr = vp._cc_country_frontier(eci, 'United States of America')
        cn_fr = vp._cc_country_frontier(eci, 'China')
        us_best = max(us_fr, key=lambda x: x[1])
        cn_best = max(cn_fr, key=lambda x: x[1])
        target = us_best[1] - 1.0     # about where the US frontier is now
        assert cn_best[1] < target, "China already past today's US frontier"
        a_partial, b_algo = vp._cc_pooled_decomp(cc)
        us_algo, _, _ = vp._cc_iso_compute_rate(cc, 'United States of America')
        cn_algo, _, _ = vp._cc_iso_compute_rate(cc, 'China')
        inno = vp._cc_innovation_algo_band(cc)
        a_lo, a_hi = min(us_algo, cn_algo), max(us_algo, cn_algo)
        g_mid = 0.5 * (vp._CC_CN_COMPUTE_LO + vp._CC_CN_COMPUTE_HI)
        pace_lo, pace_hi, _ = vp._cc_cn_pace_band(cn_fr, cn_algo + a_partial * g_mid)
        kw = dict(a_partial=a_partial, g_lo=vp._CC_CN_COMPUTE_LO,
                  g_hi=vp._CC_CN_COMPUTE_HI, algo_lo=a_lo, algo_mid=cn_algo,
                  algo_hi=a_hi, inno_lo=inno[0], inno_hi=inno[1],
                  pace_lo=pace_lo, pace_hi=pace_hi, n=4000)
        y_cc, _, _ = vp._cc_cn_crossing_sim(
            cn_best[1], target, us_anchor=us_best[1],
            us_rate=15.0, **kw)
        y_pause, _, _ = vp._cc_cn_crossing_sim(
            cn_best[1], us_best[1], us_anchor=us_best[1], us_rate=0.0, **kw)
        assert np.nanmedian(y_pause) >= np.nanmedian(y_cc)

    def test_diffusion_decay_slows_a_pause(self):
        """With a pause, splitting the no-distillation band into diffusion
        (drying up on the publication clock) and innovation makes the same
        crossing later than treating it all as never-decaying."""
        common = dict(self.KW, inno_lo=8.0, inno_hi=10.0)
        y_two, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=160.0, us_rate=0.0, **common)
        y_three, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=160.0, us_rate=0.0,
            pure_lo=3.0, pure_hi=4.0, t_pause=0.0, **common)
        assert np.nanmedian(y_three) > np.nanmedian(y_two)

    def test_diffusion_split_is_inert_without_a_pause(self):
        """t_pause=None keeps D(t)=1, so the three-channel law reduces to the
        two-channel one whatever the pure band says."""
        common = dict(self.KW, inno_lo=8.0, inno_hi=10.0)
        y_two, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=165.0, us_rate=15.0, **common)
        y_three, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=165.0, us_rate=15.0,
            pure_lo=3.0, pure_hi=4.0, **common)
        assert abs(np.nanmedian(y_three) - np.nanmedian(y_two)) < 0.06

    def test_crossing_is_later_for_a_higher_pause_bar(self):
        """us_pause_level: the US climbs to the bar then freezes; a higher
        bar means a longer climb and a later Chinese crossing."""
        y_lo, _, _ = vp._cc_cn_crossing_sim(
            150.0, 158.0, us_anchor=158.0, us_rate=15.0, us_pause_level=158.0,
            inno_lo=3.0, inno_hi=4.0, **self.KW)
        y_hi, _, _ = vp._cc_cn_crossing_sim(
            150.0, 165.0, us_anchor=158.0, us_rate=15.0, us_pause_level=165.0,
            inno_lo=3.0, inno_hi=4.0, **self.KW)
        assert np.nanmedian(y_hi) > np.nanmedian(y_lo)

    def test_traj_is_nondecreasing(self):
        _, grid, traj = vp._cc_cn_crossing_sim(
            150.0, 160.0, us_anchor=160.0, us_rate=0.0,
            inno_lo=3.0, inno_hi=4.0, **self.KW)
        assert traj.shape == (self.KW['n'], len(grid))
        assert (np.diff(traj, axis=1) >= -1e-9).all()

    # With the US holding the gap wide (where gap decay never bites):
    # t_dist_stop cuts the teacher channel; comp_dead zeroes the compute term
    # inside its window (a level setback being regrown); a dist_teacher below
    # the bar dries distillation before the crossing.
    @pytest.mark.parametrize("lever", [
        dict(t_dist_stop=0.0), dict(comp_dead=(0.0, 50.0)),
        dict(dist_teacher=152.0)],
        ids=["dist_stop", "comp_dead", "teacher_withheld"])
    def test_lever_slows_the_crossing(self, lever):
        common = dict(self.KW, inno_lo=8.0, inno_hi=10.0,
                      us_anchor=165.0, us_rate=25.0)
        y_on, _, _ = vp._cc_cn_crossing_sim(150.0, 160.0, **common)
        y_off, _, _ = vp._cc_cn_crossing_sim(150.0, 160.0, **lever, **common)
        assert np.nanmedian(y_off) > np.nanmedian(y_on)

    # The same levers dated past the horizon, or a teacher above the US level.
    @pytest.mark.parametrize("lever", [
        dict(t_dist_stop=50.0), dict(comp_dead=(40.0, 50.0)),
        dict(dist_teacher=1e9)],
        ids=["dist_stop", "comp_dead", "teacher_withheld"])
    def test_lever_out_of_reach_is_inert(self, lever):
        common = dict(self.KW, inno_lo=8.0, inno_hi=10.0,
                      us_anchor=165.0, us_rate=25.0)
        y_none, _, _ = vp._cc_cn_crossing_sim(150.0, 160.0, **common)
        y_far, _, _ = vp._cc_cn_crossing_sim(150.0, 160.0, **lever, **common)
        assert abs(np.nanmedian(y_far) - np.nanmedian(y_none)) < 0.06

    def test_comp_slow_only_bites_after_its_cut(self):
        """comp_slow drops the compute term from t_cut on: cutting at t=0 is
        the same as sampling the slow band throughout, and a cut past the
        horizon is inert."""
        common = dict(self.KW, inno_lo=8.0, inno_hi=10.0,
                      us_anchor=165.0, us_rate=25.0)
        slow = np.full(self.KW['n'], self.KW['a_partial'] * 0.05)
        y_none, _, _ = vp._cc_cn_crossing_sim(150.0, 160.0, **common)
        y_now, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, comp_slow=(0.0, slow), **common)
        y_late, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, comp_slow=(50.0, slow), **common)
        assert np.nanmedian(y_now) > np.nanmedian(y_none)
        assert abs(np.nanmedian(y_late) - np.nanmedian(y_none)) < 0.06
        y_mid, _, _ = vp._cc_cn_crossing_sim(
            150.0, 160.0, comp_slow=(0.5, slow), **common)
        assert np.nanmedian(y_none) <= np.nanmedian(y_mid) + 0.06
        assert np.nanmedian(y_mid) <= np.nanmedian(y_now) + 0.06


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

    @pytest.mark.parametrize("load", [
        vp.load_ukcyber,
        lambda: vp.load_ukcyber_tlo(vp._ukc_tlo_mtime()),
    ], ids=["narrow", "tlo"])
    def test_frontier_is_the_closed_weight_running_max(self, load):
        """Open-weight models are the subject measured against the frontier,
        so they must never define it. Both loaders carry their own copy of
        the rule."""
        best = -float('inf')
        for m in load():
            new_max = m['weights'] == 'closed' and m['cyber_score'] > best
            assert m['is_frontier'] == new_max, m['name']
            if new_max:
                best = m['cyber_score']


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
        for edge in ('lo', 'hi'):
            days = (eta[f'date_{edge}'] - eta['frontier_date']).days
            assert days == pytest.approx(
                eta[f'lag_{edge}'] * vp._UKC_DAYS_PER_MONTH, abs=1)

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


class TestRsi:
    """CoBench, the RSI tab's only feed."""

    def test_slugs_round_trip(self):
        assert vp._TAB_OPTIONS[vp._TAB_SLUG["takeoff"]] == "Takeoff"
        # The RSI tab merged into Takeoff; its old links land there.
        assert "RSI" not in vp._TAB_OPTIONS
        assert vp._TAB_SLUG["rsi"] == vp._TAB_SLUG["takeoff"]

    def test_defaults_and_url_tracking_cover_the_widgets(self):
        keys, defaults = vp._all_tracked()
        for k in vp._RSI_RESET_KEYS:
            assert k in keys
        assert set(vp._RSI_DEFAULTS) <= set(vp._RSI_RESET_KEYS)
        for k, v in vp._RSI_DEFAULTS.items():
            assert defaults[k] == v

    def test_loader_sorts_and_flags_running_max(self):
        rows = vp.load_rsi_data()
        assert [m['name'] for m in rows] == [
            "Claude Opus 4.6", "Claude Mythos Preview",
            "Claude Mythos 5", "Model 2 (internal)",
            "Claude Opus 5", "Claude Mythos 5.1"]
        assert [m['date'] for m in rows] == sorted(m['date'] for m in rows)
        best = -float('inf')
        for m in rows:
            assert m['is_frontier'] == (m['cobench'] > best)
            best = max(best, m['cobench'])

    def test_unknown_dates_render_with_a_tilde(self):
        by_name = {m['name']: m for m in vp.load_rsi_data()}
        assert vp._rsi_date_label(by_name["Claude Opus 4.6"]) == "Feb 05, 2026"
        assert vp._rsi_date_label(by_name["Claude Mythos 5"]) == "Jun 09, 2026"
        assert vp._rsi_date_label(by_name["Claude Mythos Preview"]) == "~Apr 07, 2026"
        assert vp._rsi_date_label(by_name["Model 2 (internal)"], '%b %Y') == "~Jul 2026"

    def test_dt_ci_default_spans_both_segment_rates(self):
        """Three points whose segments disagree ~8x: the conventional
        fit/2..fit*2 interval must widen, never narrow."""
        fr = [m for m in vp.load_rsi_data() if m['is_frontier']]
        _, _, slope = vp._rsi_fit(fr)
        fit_dt = np.log(2) / slope
        lo, hi = vp._rsi_dt_ci(fr, fit_dt)
        assert lo <= max(5.0, fit_dt / 2) and hi >= fit_dt * 2
        for a, b in zip(fr, fr[1:]):
            seg = np.log(2) * (b['date'] - a['date']).days / (
                vp._logit(b['cobench'] / 100) - vp._logit(a['cobench'] / 100))
            assert lo <= round(seg) <= hi

    def test_dt_t_interval_reflects_sample_size(self):
        """A perfect line collapses to the fit, under 3 points there is no
        interval, and one residual dof with a flat segment reaches the cap."""
        days = np.arange(0, 500, 50, dtype=float)
        lo, hi = vp._dt_t_interval(days, 0.01 * days, 0.0, 0.01)
        assert abs(lo - np.log(2) / 0.01) < 1 and abs(hi - np.log(2) / 0.01) < 1
        assert vp._dt_t_interval([0.0, 50.0], [0.0, 0.5], 0.0, 0.01) is None
        days3 = np.array([0.0, 73.0, 134.0])
        ys3 = np.log(np.array([2.0, 2.0, 4.0]))
        icpt, slope = vp.fit_line(days3, ys3)
        lo3, hi3 = vp._dt_t_interval(days3, ys3, icpt, slope)
        assert hi3 == vp._DT_CAP_DAYS
        assert lo3 < np.log(2) / slope

    def test_small_sample_ci_widens_both_rsi_fits(self):
        """Live data: CoBench's four frontier points and the survey's four
        rounds (the estimated one included) each leave two residual dof, so
        both slow edges widen past the convention but stay under the cap.
        If a new round moves either, update this."""
        fr = [m for m in vp.load_rsi_data() if m['is_frontier']]
        _, _, slope = vp._rsi_fit(fr)
        dt = np.log(2) / slope
        assert round(dt * 2) < vp._rsi_dt_ci(fr, dt)[1] < vp._DT_CAP_DAYS
        rows = vp.load_rsi_survey()
        days = np.array([(r['date'] - rows[0]['date']).days for r in rows],
                        dtype=float)
        _, sslope = vp.fit_line(days, np.log([r['uplift'] for r in rows]))
        sdt = np.log(2) / sslope
        slo, shi = vp._rsi_survey_dt_ci(rows, sdt)
        assert round(sdt * 2) < shi < vp._DT_CAP_DAYS
        assert slo <= round(sdt / 2)

    def test_survey_eta_includes_the_estimated_round(self, monkeypatch):
        """The card fits and anchors on every round the tab fits — the
        carried-over Model 2 point included, whose flat reading is the
        reason the slope runs shallower than the surveyed points alone."""
        rows = vp.load_rsi_survey()
        assert rows[-1].get('estimated')
        seen = []
        real = vp._rsi_survey_dt_ci
        monkeypatch.setattr(vp, '_rsi_survey_dt_ci',
                            lambda r, dt: seen.append(dt) or real(r, dt))
        anchor, _days = vp._pc_rsi_survey_eta(rows, samples=True)
        assert anchor == rows[-1]['date']

        def fit_dt(rs):
            days = [(r['date'] - rs[0]['date']).days for r in rs]
            return np.log(2) / vp.fit_line(np.array(days, dtype=float),
                                           np.log([r['uplift'] for r in rs]))[1]
        assert seen == [pytest.approx(fit_dt(rows))]
        surveyed = fit_dt([r for r in rows if not r.get('estimated')])
        assert seen[0] > surveyed * 1.05


class TestRsiDirection:
    """Anthropic's Claude Code detour study, the RSI tab's third series."""

    def test_loader_sorts_and_flags_running_max(self):
        rows = vp.load_rsi_direction()
        assert [r['date'] for r in rows] == sorted(r['date'] for r in rows)
        best = -float('inf')
        for r in rows:
            assert r['is_frontier'] == (r['better'] > best)
            best = max(best, r['better'])
        # The published figure is not date-ordered (it prints Sonnet 4.6 above
        # Opus 4.6, which shipped twelve days earlier), so the dates decide the
        # frontier: Opus 4.7 ships after Mythos Preview and scores under it.
        assert [r['name'] for r in rows if r['is_frontier']] == [
            "Claude Haiku 3", "Claude Sonnet 4", "Claude Sonnet 4.5",
            "Claude Opus 4.5", "Claude Opus 4.6", "Claude Mythos Preview"]
        assert not next(r for r in rows
                        if r['name'] == "Claude Opus 4.7")['is_frontier']

    def test_dates_are_the_published_release_dates(self):
        """Every model the study scored is an Epoch-catalogued release, so the
        dates are checkable — the calibration guard on the digitization. Only
        Mythos Preview, which has no release record, is carried over."""
        eci = {m['display_name'].split(' (')[0]: m['date']
               for m in vp.load_eci_frontier(orgs=["Anthropic"],
                                             full_window=True)}
        checked = 0
        for r in vp.load_rsi_direction():
            if not r['date_known']:
                assert r['name'] == "Claude Mythos Preview"
                # Same unpublished date the CoBench series carries.
                assert r['date'] == next(m['date'] for m in vp.load_rsi_data()
                                         if m['name'] == r['name'])
                continue
            match = [d for name, d in eci.items() if name in r['name']
                     or r['name'] in name]
            if match:
                assert r['date'] in match, r['name']
                checked += 1
        assert checked >= 4

    def test_fit_is_logit_space_against_the_studys_own_ceiling(self):
        """A bounded win rate, so the trend runs on the log-odds like CoBench's
        — and the bar is the figure's own annotated practical ceiling, the only
        threshold the source names."""
        fr = vp.rsi_dir_frontier_all
        base, icpt, slope = vp._rsi_dir_fit(fr)
        assert slope > 0
        days = np.array([(r['date'] - base).days for r in fr], dtype=float)
        want = vp.fit_line(days,
                           vp._logit(np.array([r['better'] for r in fr]) / 100))
        assert (icpt, slope) == pytest.approx(tuple(want))
        # Every frontier point sits under the bar, so the crossing is ahead.
        assert max(r['better'] for r in fr) < vp._RSI_DIR_TARGET

class TestRsiExperiments:
    def test_research_pace_uses_2025_units_and_rebases_draws_before_quantiles(self):
        params = dict(years=1, **vp.takeoff.POINT_ESTIMATE)
        onset = np.ones(20)
        starts = np.linspace(np.log(1.5), np.log(2.5), 20)
        slopes = np.full(20, np.log(2))
        a = vp._tk_simulate(params, onset, "2025-test", starts, slopes)
        b = vp._tk_simulate(params, onset, "2025-test", starts, slopes, baseline_yield=2)
        h = a['workflow_progress']
        for key in ['pace_2025_quantiles', 'reference_2025_quantiles']:
            np.testing.assert_allclose(h[key][0], np.quantile(np.exp(starts), [.1, .5, .9]))
            assert h[key][0, 1] > 1
        for key in ['pace_2025_quantiles', 'reference_2025_quantiles', 'cumulative_2025_quantiles']:
            np.testing.assert_allclose(b['workflow_progress'][key], 2 * h[key])
        np.testing.assert_array_equal(h['cumulative_2025_quantiles'][0], 0)
        raw = vp.takeoff.simulate(params, n=20, onset_years=onset,
                                  experiment_slopes=slopes)['workflow_progress']
        for key, sample_key in [('pace_2025_quantiles', 'pace'),
                                ('reference_2025_quantiles', 'reference_pace'),
                                ('cumulative_2025_quantiles', 'validated_baseline_months')]:
            np.testing.assert_allclose(h[key], np.quantile(
                raw[sample_key] * np.exp(starts)[:, None], [.1, .5, .9], axis=0).T)
            assert sample_key not in h      # dropped from the cached result
        np.testing.assert_array_equal(a['software_log'], b['software_log'])

    def test_takeoff_inherits_the_same_activity_paths_without_changing_global_rng(self):
        rows = vp.load_rsi_experiments()
        before = np.random.get_state()
        starts, daily = vp._rsi_experiment_draws(rows, n=20)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
        today = datetime(2026, 9, 7)
        current, annual = vp._tk_rsi_activity(rows, today, 20)
        days = (today - rows[-1]['date']).days
        np.testing.assert_array_equal(current, starts + daily * days)
        np.testing.assert_array_equal(annual, daily * 365.25)
        params = dict(years=1, **vp.takeoff.POINT_ESTIMATE)
        onset = np.ones(20)
        a = vp._tk_simulate(params, onset, "test", current, annual)
        b = vp._tk_simulate(params, onset, "test", current + np.log(2), annual)
        # Relative growth informs dynamics; a different historical activity
        # normalization must not multiply today's productivity a second time.
        np.testing.assert_array_equal(a['software_log'], b['software_log'])
        expected = np.exp(starts[:, None] + daily[:, None] *
                          (days + a['workflow_progress']['years'] * 365.25))
        np.testing.assert_allclose(a['workflow_progress']['rsi_activity_quantiles'],
                                   np.quantile(expected, [.1, .5, .9], axis=0).T)

    def test_source_rows_are_consecutive_weeks(self):
        rows = vp.load_rsi_experiments()
        assert all((b['date'] - a['date']).days == 7 for a, b in zip(rows, rows[1:]))

    def test_threshold_dates_use_the_same_draws_as_the_fan(self, monkeypatch):
        rows = vp.load_rsi_experiments()
        # At a fixed 100-day doubling time, 1.25x reaches 10x in 300 days.
        monkeypatch.setattr(vp, '_rsi_experiment_draws',
                            lambda rows, n=None: (np.full(20, np.log(1.25)),
                                                   np.full(20, np.log(2) / 100)))
        anchor, days = vp._pc_rsi_experiment_eta(rows, samples=True)
        assert anchor == rows[-1]['date']
        assert days == pytest.approx(np.full(20, 300.0))

    def test_declining_series_has_no_future_crossing(self):
        rows = [{'date': datetime(2026, 1, 5), 'mult': 2.0},
                {'date': datetime(2026, 1, 12), 'mult': 1.0}]
        assert vp._pc_rsi_experiment_eta(rows) is None


class TestRsiCode:
    """Merged code per Anthropic contributor, the RSI tab's fourth series."""

    def test_loader_dates_each_quarter_at_its_midpoint(self):
        rows = vp.load_rsi_code()
        assert len(rows) == len(vp._RSI_CODE_RAW)
        assert [r['date'] for r in rows] == sorted(r['date'] for r in rows)
        assert vp._rsi_code_mid("2026Q2") == datetime(2026, 5, 16, 12)
        assert vp._rsi_code_mid("2025Q4") == datetime(2025, 11, 16)
        assert [r['q'] for r in rows if r['partial']] == [vp._RSI_CODE_PARTIAL]
        assert rows[-1]['partial'] and rows[-1]['mult'] == 8.0
        for r in rows:
            assert r['fitted'] == (r['date'] >= vp._RSI_CODE_FIT_FROM)

    def test_pre_2025_bars_average_to_the_baseline_they_define(self):
        """The calibration guard on the digitization: the unlabelled bars are
        the average the whole series is a multiple of, so they must come back
        out at ~1 — and they are charted, not fitted, for that reason."""
        pre = [r['mult'] for r in vp.load_rsi_code() if not r['fitted']]
        assert len(pre) == 15
        assert abs(np.mean(pre) - 1.0) < 0.05
        assert max(pre) < 1.5

    def test_chart_opens_in_2024_without_dropping_the_baseline(self):
        """The flat quarters before 2024 only stretch the axis, so the chart
        opens there — but every bar stays in the data: their mean is the
        calibration guard, and no fitted bar may fall off the left edge."""
        rows = vp.load_rsi_code()
        assert any(r['date'] < vp._RSI_CODE_CHART_FROM for r in rows)
        assert all(r['date'] >= vp._RSI_CODE_CHART_FROM
                   for r in rows if r['fitted'])

    def test_fit_runs_on_the_2025_bars_only(self):
        """A flat baseline in the fit would halve the slope, and the baseline
        carries no trend by construction."""
        rows = vp.load_rsi_code()
        fitted = [r for r in rows if r['fitted']]
        base, icpt, slope = vp._rsi_code_fit(rows)
        assert base == fitted[0]['date'] and slope > 0
        days = np.array([(r['date'] - base).days for r in fitted], dtype=float)
        want = vp.fit_line(days, np.log([r['mult'] for r in fitted]))
        assert (icpt, slope) == pytest.approx(tuple(want))
        all_days = np.array([(r['date'] - rows[0]['date']).days for r in rows],
                            dtype=float)
        pooled = vp.fit_line(all_days, np.log([r['mult'] for r in rows]))[1]
        assert pooled < slope / 2


class TestRsiAutomation:
    """Anthropic's R&D Automation Index — the AL4 ("AI leads") share.

    The six fitted values are transcribed from the figure's own printed
    labels; the 90% measurement intervals are digitized from it. The bracket
    checks below are the calibration guard on that digitization, exactly as
    the pre-2025 mean is for the merged-code bars.
    """

    def test_loader_dates_each_month_at_its_midpoint(self):
        rows = vp.load_rsi_automation()
        assert len(rows) == 13
        assert [r['date'] for r in rows] == sorted(r['date'] for r in rows)
        assert rows[0]['date'] == datetime(2025, 8, 15)
        assert rows[-1]['date'] == datetime(2026, 8, 15)
        assert rows[-1]['al4'] == 26.0 and rows[-1]['fitted']
        assert [r['al4'] for r in rows if r['fitted']] == [1, 3, 12, 14, 22, 26]

    def test_every_interval_brackets_its_printed_label(self):
        """The calibration guard: the labels are transcribed and the whiskers
        digitized, so a whisker that misses its own label is a bad read."""
        fitted = [r for r in vp.load_rsi_automation() if r['fitted']]
        assert len(fitted) == 6
        for r in fitted:
            assert r['lo'] < r['al4'] < r['hi'], r['month']
            # The figure's whiskers are a few points wide, never a whole band.
            assert r['hi'] - r['lo'] < 15.0, r['month']

    def test_unlabelled_months_are_charted_but_never_fitted(self):
        """A logit fit cannot take 0, and the figure prints no value for them
        — the post says only that February 2026 was "under 1%"."""
        rows = vp.load_rsi_automation()
        unfit = [r for r in rows if not r['fitted']]
        assert len(unfit) == 7
        assert all(r['al4'] == 0.0 for r in unfit)
        assert all(r['lo'] is None and r['hi'] is None for r in unfit)
        assert all(r['date'] < datetime(2026, 3, 1) for r in unfit)
        base, _icpt, _slope = vp._rsi_auto_fit(rows)
        assert base == datetime(2026, 3, 15)

    def test_fit_is_logit_space_over_the_labelled_months(self):
        """A bounded share, fitted on the log-odds like CoBench and the
        detour study — a score-space line runs through 100%."""
        rows = vp.load_rsi_automation()
        fitted = [r for r in rows if r['fitted']]
        base, icpt, slope = vp._rsi_auto_fit(rows)
        assert slope > 0
        days = np.array([(r['date'] - base).days for r in fitted], dtype=float)
        want = vp.fit_line(days, vp._logit(
            np.array([r['al4'] for r in fitted]) / 100))
        assert (icpt, slope) == pytest.approx(tuple(want))

    def test_rate_ci_is_the_shared_widener_stretched_slow(self):
        """The fast edge is the shared t-widened rule's; only the slow edge
        may move, and only outward."""
        rows = vp.load_rsi_automation()
        dt = np.log(2) / vp._rsi_auto_fit(rows)[2]
        lo, hi = vp._rsi_auto_dt_ci(rows, round(dt))
        fitted = [r for r in rows if r['fitted']]
        full = vp._dt_ci_t_widened(
            [(r['date'] - fitted[0]['date']).days for r in fitted],
            vp._logit(np.array([r['al4'] for r in fitted]) / 100), round(dt))
        assert lo == full[0] and hi >= full[1]

    def test_rate_ci_reaches_the_recent_months_slow_edge(self):
        """The slow edge holds the climb since the bend, not just the full
        fit's: it is at least the recent months' own fitted doubling time."""
        rows = vp.load_rsi_automation()
        recent = vp._rsi_auto_points(vp._rsi_auto_recent_rows(rows), 'al4')
        assert len(recent) == vp._RSI_AUTO_RECENT_N
        x = np.array([(r['date'] - recent[0]['date']).days for r in recent], float)
        _icpt, slope = vp.fit_line(x, vp._logit(
            np.array([r['al4'] for r in recent]) / 100))
        dt = np.log(2) / vp._rsi_auto_fit(rows)[2]
        assert vp._rsi_auto_dt_ci(rows, round(dt))[1] >= np.log(2) / slope

    def test_start_spans_the_fit_and_the_latest_reading(self):
        """The projection starts between the fitted line and the latest
        month, and its 80% band reaches both."""
        rows = vp.load_rsi_automation()
        cur, centre, sigma = vp._rsi_auto_start(rows)
        base, icpt, slope = vp._rsi_auto_fit(rows)
        fitted = icpt + slope * (cur['date'] - base).days
        observed = vp._logit(cur['al4'] / 100)
        assert centre == pytest.approx((fitted + observed) / 2)
        assert sigma >= vp._rsi_auto_pos_sigma(cur)
        assert 1.282 * sigma >= abs(fitted - observed) / 2

    def test_unlabelled_month_has_no_position_interval(self):
        """The figure prints no whisker for it, so there is nothing to read
        and nothing to crash on."""
        assert vp._rsi_auto_pos_sigma(
            {'al4': 0.0, 'lo': None, 'hi': None}) == 0.0

    def test_target_is_above_what_the_fit_already_reached(self):
        """A bar the trend has passed dates nothing — the low ECI card's
        lesson. The fitted share today is already a majority, which is why
        the bar is 90% and not 50%."""
        rows = vp.load_rsi_automation()
        base, icpt, slope = vp._rsi_auto_fit(rows)
        today = vp._inv_logit(
            icpt + slope * (datetime(2026, 10, 1) - base).days) * 100
        assert today > 50.0
        assert vp._RSI_AUTO_TARGET > today

    def test_declining_series_has_no_future_crossing(self):
        rows = [{'date': datetime(2026, 3, 15), 'al4': 20.0, 'lo': 15.0,
                 'hi': 25.0, 'fitted': True},
                {'date': datetime(2026, 4, 15), 'al4': 10.0, 'lo': 5.0,
                 'hi': 15.0, 'fitted': True}]
        assert vp._pc_rsi_auto_eta(rows) is None


class TestRsiAutomationLadder:
    """The AL2+/AL3+/AL4+ rungs and the lag between them.

    The rungs are cumulative shares digitized from the same stacked figure.
    AL4 is the only level the figure prints numbers for, so AL4-digitized vs
    AL4-printed is the calibration guard on the whole method.
    """

    def test_digitised_al4_reproduces_the_printed_labels(self):
        """The guard: the same pixel method that produced AL2+ and AL3+,
        applied to the one band whose truth is printed."""
        rows = [r for r in vp.load_rsi_automation() if r['fitted']]
        assert len(rows) == 6
        errs = [abs(r['al4_dig'] - r['al4']) for r in rows]
        assert max(errs) < 0.7, errs
        assert np.mean(errs) < 0.4, errs

    def test_levels_are_cumulative_and_nest(self):
        """al1 >= al2 >= al3 >= al4 at every month, all inside [0, 100].
        A cumulative share cannot be exceeded by a stricter one."""
        for r in vp.load_rsi_automation():
            assert 100.0 >= r['al1'] >= r['al2'] >= r['al3'] >= r['al4'] >= 0.0, r

    def test_every_level_climbs_monotonically(self):
        rows = vp.load_rsi_automation()
        for key in ('al1', 'al2', 'al3'):
            vals = [r[key] for r in rows]
            assert vals == sorted(vals), key

    def test_saturated_and_floor_months_are_dropped_before_fitting(self):
        """A share pinned against 100 carries no rate — one pixel moves its
        log-odds further than a month of progress does."""
        rows = vp.load_rsi_automation()
        for key in ('al1', 'al2', 'al3', 'al4'):
            for r in vp._rsi_auto_points(rows, key):
                assert vp._RSI_AUTO_FLOOR < r[key] < vp._RSI_AUTO_SAT_HI
        # AL2 saturates partway through; AL3 is the one full climb.
        assert len(vp._rsi_auto_points(rows, 'al2')) < len(rows)
        assert len(vp._rsi_auto_points(rows, 'al3')) == len(rows)
        # AL4 is additionally restricted to the months the figure labels.
        assert all(r['fitted'] for r in vp._rsi_auto_points(rows, 'al4'))

    def test_al1_is_not_a_rung(self):
        """It is already at 95% in the first rated month and pinned at 100
        from December, so it never shows a climb to measure."""
        assert 'al1' not in vp._RSI_AUTO_LADDER
        rows = vp.load_rsi_automation()
        assert all(r['al1'] > 95.0 for r in vp._rsi_auto_points(rows, 'al1'))

    def test_the_rungs_are_near_parallel_in_log_odds(self):
        """What licenses treating each level as the one below it shifted in
        time. If this stops holding the lag stops being well defined, and the
        AL5 projection loses its basis."""
        rows = vp.load_rsi_automation()
        dts = [np.log(2) / vp._rsi_auto_level_fit(rows, k)[2]
               for k in vp._RSI_AUTO_LADDER]
        assert all(d > 0 for d in dts)
        assert max(dts) / min(dts) < 1.5, dts

    def test_lag_is_positive_ordered_and_only_mildly_reference_dependent(self):
        rows = vp.load_rsi_automation()
        steps = vp._rsi_auto_ladder(rows)
        assert [s[:2] for s in steps] == [('al2', 'al3'), ('al3', 'al4')]
        assert all(s[2] > 0 for s in steps)
        for lo, hi, days in steps:
            spread = vp._rsi_auto_lag_spread(rows, lo, hi)
            assert spread[0] <= days <= spread[1]
            # Near-parallel, so where the gap is read moves it by well under
            # the gap itself.
            assert spread[1] - spread[0] < 0.5 * days, (lo, hi, spread)


class TestRsiAl5:
    """Full autonomy: projected, never measured."""

    def test_al5_has_no_column_at_all(self):
        """The whole point of the section's warning. If a refresh ever adds
        AL5 data this test should fail and the section be rebuilt on it."""
        rows = vp.load_rsi_automation()
        assert all('al5' not in r for r in rows)
        with open('anthropic_rd_automation.csv') as f:
            header = next(l for l in f if not l.startswith('#'))
        assert 'al5' not in header

    def test_ladder_extrapolation_continues_the_measured_step(self):
        rows = vp.load_rsi_automation()
        steps = [s[2] for s in vp._rsi_auto_ladder(rows)]
        lag = vp._rsi_al5_lag(rows)
        assert lag['ladder'] == pytest.approx(steps[-1] + (steps[-1] - steps[-2]))

    def test_the_absent_al5_band_censors_the_lag_from_below(self):
        """The zeros are evidence, not nothing: at a shorter lag AL5 would
        already be a visible band in the last rated month, and it is not. So
        the censoring bound binds, and the projection uses it."""
        rows = vp.load_rsi_automation()
        lag = vp._rsi_al5_lag(rows)
        assert lag['censor'] > lag['ladder']
        assert lag['lag'] == lag['censor']
        # At exactly the censoring lag, AL5 sits on the figure's floor.
        base, icpt, slope = vp._rsi_auto_fit(rows)
        at_last = ((rows[-1]['date'] - base).days - lag['censor'])
        assert vp._inv_logit(icpt + slope * at_last) * 100 == pytest.approx(
            vp._RSI_AUTO_FLOOR, rel=0.05)
        # And the extrapolated lag alone would have made it visible.
        at_ladder = ((rows[-1]['date'] - base).days - lag['ladder'])
        assert vp._inv_logit(icpt + slope * at_ladder) * 100 > vp._RSI_AUTO_FLOOR

    def test_al5_is_the_al4_curve_shifted(self):
        """Not a second fit: same rung, plus a lag. A zero lag must reproduce
        the AL4 card at the same bar."""
        rows = vp.load_rsi_automation()
        anchor4, d4 = vp._pc_rsi_auto_eta(rows, vp._RSI_AL5_TARGET, n=6000,
                                          samples=True)
        anchor5, d5 = vp._pc_rsi_al5_eta(rows, lag_days=1e-9, n=6000,
                                         samples=True)
        assert anchor5 == anchor4
        assert np.median(d5) == pytest.approx(np.median(d4), rel=0.1)

    def test_a_longer_assumed_lag_only_moves_it_later(self):
        rows = vp.load_rsi_automation()
        med = []
        for mo in (6, 12, 24):
            _a, d = vp._pc_rsi_al5_eta(rows, lag_days=mo * 30.44, n=6000,
                                       samples=True)
            med.append(np.median(d))
        assert med == sorted(med)
        # Doubling the lag moves the median by about the lag, not by more.
        assert 0.7 < (med[2] - med[1]) / (12 * 30.44) < 1.6

    def test_lag_slow_edge_reads_the_recent_al4_months(self):
        """The slower AL4 climb since the bend puts AL4 further behind AL3
        and its floor crossing earlier, so the lag read off those months is
        longer, and the band's slow edge is set off it."""
        lag = vp._rsi_al5_lag(vp.load_rsi_automation())
        assert lag['slow'] > lag['lag']
        draws = vp._rsi_al5_tau_draws(lag['lag'], 20000, lag['slow'])
        assert np.percentile(draws, 10) == pytest.approx(lag['lag'], rel=0.05)
        assert np.percentile(draws, 90) == pytest.approx(
            lag['slow'] * vp._RSI_AL5_TAU_HI_MULT, rel=0.05)

    def test_both_rungs_are_dated_against_the_same_bar(self):
        """One level, two crossings: the gap between them is the ladder's own
        step. If these diverge the chart's single bar starts lying."""
        assert vp._RSI_AL5_TARGET == vp._RSI_AUTO_TARGET


class TestRsiMilestoneCards:
    """Each RSI section's milestone card is that section's own fit, so the
    card and the fan cannot date the same bar differently."""

    @staticmethod
    def _case(which):
        if which == 'nextstep':
            rows = vp.rsi_dir_frontier_all
            return (rows[-1]['date'], vp._rsi_dir_fit(rows),
                    vp._logit(vp._RSI_DIR_TARGET / 100),
                    vp._pc_nextstep_eta(rows, n=4000, samples=True))
        if which == 'code':
            rows = vp.load_rsi_code()
            return (rows[-1]['date'], vp._rsi_code_fit(rows),
                    np.log(vp._RSI_CODE_TARGET),
                    vp._pc_rsi_code_eta(rows, n=4000, samples=True))
        rows = vp.load_rsi_automation()
        return ([r for r in rows if r['fitted']][-1]['date'],
                vp._rsi_auto_fit(rows), vp._logit(vp._RSI_AUTO_TARGET / 100),
                vp._pc_rsi_auto_eta(rows, n=4000, samples=True))

    @pytest.mark.parametrize("which", ["nextstep", "code", "rdauto"])
    def test_eta_reproduces_the_section_defaults(self, which):
        """Same anchor, and a median around the deterministic crossing of
        the section's fitted line."""
        want_anchor, (base, icpt, slope), target_y, (anchor, days) = \
            self._case(which)
        assert anchor == want_anchor
        deterministic = (target_y - (icpt + slope * (anchor - base).days)) / slope
        assert 0.6 * deterministic < np.median(days) < 1.6 * deterministic

    def test_milestone_cards_are_weighted_in_the_blend(self):
        """Weights are keyed by slugs built from the target constants, so a
        retargeted bar must still find its weight."""
        slugs = [f"nextstep_{vp._RSI_DIR_TARGET:.0f}",
                 f"code_{vp._RSI_CODE_TARGET:.0f}x",
                 f"rdauto_{vp._RSI_AUTO_TARGET:.0f}"]
        for slug in slugs:
            assert vp._PC_RSI_WEIGHTS.get(slug, 0) > 0, slug
        # AL5 is a card only: no weight, no row in the projection.
        al5 = f"al5_{vp._RSI_AL5_TARGET:.0f}"
        assert al5 in vp._PC_RSI_CARD_ONLY and al5 not in vp._PC_RSI_WEIGHTS


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


class TestDcAxisTicks:
    """Ticks and value labels have to read in the same units.

    They didn't: the H100 metric was called "Compute (x1M H100-equiv)" and its
    ticks were divided by a million, while `_dc_fmt_value` kept feeding the bar
    text, hovers and the quarterly table raw counts — so a bar labelled "1.11M"
    sat under a tick labelled "1". The divisor is gone; ticks now carry the same
    k/M suffixes the values do.
    """

    @staticmethod
    def _plain(texts):
        return [re.sub(r"<[^>]+>", "", t) for t in texts]

    def test_log_ticks_label_the_value_they_sit_at(self):
        vals, text = vp._dc_log_ticks([5.0, 7.0], 'h100')
        plain = self._plain(text)
        assert vals[0] == 1e5 and plain[0] == "100k"
        assert plain[vals.index(1e6)] == "1M"
        assert plain[vals.index(2e6)] == "2M"   # trimmed, not "2.00M"

    def test_log_ticks_stay_plain_for_other_kinds(self):
        _, text = vp._dc_log_ticks([5.0, 6.0])
        assert self._plain(text)[0] == "100000"

    def test_linear_ticks_are_round_and_suffixed(self):
        vals, text = vp._dc_linear_ticks([0, 5.5e6], 'h100')
        assert text == ["0", "1M", "2M", "3M", "4M", "5M"]
        assert vals[-1] == 5e6

    def test_linear_ticks_absent_where_plotly_reads_fine(self):
        assert vp._dc_linear_ticks([0, 5e6], 'mw') is None
        assert vp._dc_linear_ticks(None, 'h100') is None
        assert vp._dc_linear_ticks([3.0, 3.0], 'h100') is None

    def test_a_tick_and_a_value_agree_at_the_same_number(self):
        """The invariant the old divisor broke, stated directly."""
        for v in (11470.0, 768769.0, 1111673.0):
            tick = vp._dc_tick_label(v, 'h100')
            val = vp._dc_fmt_value(v, 'h100')
            # Same number at the precision each prints, so a bar's text
            # can't contradict its axis.
            assert tick[-1] in "kM" and val[-1] in "kM"
            scale = {'k': 1e3, 'M': 1e6}
            assert float(tick[:-1]) * scale[tick[-1]] == pytest.approx(v, rel=0.01)
            assert float(val[:-1]) * scale[val[-1]] == pytest.approx(v, rel=0.05)

    def test_layout_labels_both_axis_types(self):
        log = vp._dc_layout(True, "Compute (H100-equiv)",
                            datetime(2024, 1, 1), datetime(2028, 1, 1),
                            y_range=[5.0, 7.0], kind='h100')
        assert self._plain(log['yaxis']['ticktext'])[0] == "100k"
        lin = vp._dc_layout(False, "Compute (H100-equiv)",
                            datetime(2024, 1, 1), datetime(2028, 1, 1),
                            y_range=[0, 5.5e6], kind='h100')
        assert lin['yaxis']['ticktext'] == ["0", "1M", "2M", "3M", "4M", "5M"]

    def test_dispatcher_covers_every_metric_kind(self):
        """The snapshot bar chart builds its own axis through `_dc_axis_ticks`,
        so the dispatcher — not `_dc_layout` — is what keeps it in step. It used
        to hand-roll the same branching and had no 'flop' case, printing raw
        operation counts under bars labelled in log OP."""
        assert self._plain(vp._dc_axis_ticks([27.0, 28.5], True, 'flop')[1])[0] \
            == "27"
        assert "month" in vp._dc_axis_ticks([0.0, 2.0], True, 'traintime')[1][0]
        assert self._plain(vp._dc_axis_ticks([4.0, 7.0], True, 'h100')[1])[0] \
            == "10k"
        assert vp._dc_axis_ticks(None, False, 'h100') is None

    def test_values_and_hovers_stay_raw_counts(self):
        """Nothing downstream is rescaled — _dc_fmt_value still takes raw
        counts, so the hovers and _cc_company_buildout keep working."""
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

    def test_headline_companies_resolve_without_aliasing(self):
        """_dc_company_series() keys are the merged labels; the quarterly table
        that used to sit on the tab carried its own alias tuple and silently
        picked the minority Google series, which is the failure guarded here."""
        series = {n: v for n, v in
                  vp._dc_series_for_metric(vp.dc_all, 'h100').items()
                  if v['company'] not in vp._DC_EXCLUDE_COMPANIES}
        comp = vp._dc_company_series(series)
        assert comp.get("Google"), "Google has no series"
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
        adds, so a roster move has to be looked at rather than absorbed
        silently. A company leaves this set two ways: its largest site drops
        under _DC_EXCLUDE_MIN_H100 inside the rolling horizon, or Epoch
        re-attributes its capacity to a real tenant so the landlord label
        stops owning any site at all. DayOne left that way when Epoch filled
        Nusajaya's Owner, and rejoined on the 2026-09-29 pull, which blanked it
        again and added Kempas (also no Owner/Users). The horizon is pinned at
        _ROSTER_ASOF, so this fails only when the data changes; the rolling
        roster (STACK joins ~six months later) is
        test_roster_only_ever_grows_as_the_horizon_rolls's job. When this
        fails, retarget deliberately (accept the joiner, or move
        _DC_EXCLUDE_MIN_H100 / _DC_EXCLUDE_HORIZON_DAYS) rather than
        loosening it."""
        charted = vp._DC_EXCLUDE_COMPANIES - vp._dc_hidden_companies(
            vp.dc_all, now=self._ROSTER_ASOF)
        assert charted == {'QTS', 'Microsoft', 'Vantage', 'DayOne'}, charted

    _ROSTER_ASOF = datetime(2026, 10, 12)

    def test_roster_only_ever_grows_as_the_horizon_rolls(self):
        """Who appears may move as planned buildout comes inside the horizon —
        Vantage did — but a company already charted must never drop back out
        as time passes, which would read as a site ceasing to exist."""
        now = datetime.now()
        prev = None
        for off in (-365, -183, 0, 183, 365):
            charted = (vp._DC_EXCLUDE_COMPANIES
                       - vp._dc_hidden_companies(
                           vp.dc_all, now=now + timedelta(days=off)))
            if prev is not None:
                assert prev <= charted, (off, prev, charted)
            prev = charted

    def test_roster_ignores_the_users_projection_window(self):
        """dc_end_year caps the charts, never the roster — otherwise sites blink
        in and out as the slider moves. Cutting the catalogue at each offered
        end year past the horizon leaves the roster alone; an uncapped peak
        would not (STACK clears the bar only on its post-horizon plan)."""
        now = self._ROSTER_ASOF
        roster = vp._dc_hidden_companies(vp.dc_all, now=now)
        horizon = now + timedelta(days=vp._DC_EXCLUDE_HORIZON_DAYS)
        for year in vp._DC_END_YEARS:
            cut = max(datetime(year, 12, 31), horizon)
            dcs = [dict(dc, points=[p for p in dc['points'] if p['date'] <= cut])
                   for dc in vp.dc_all]
            assert vp._dc_hidden_companies(dcs, now=now) == roster, year
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
        # A host site ten times the biggest lab site leads the DC tab's
        # envelope and still never enters this frontier — otherwise the half
        # above could pass only because no host happens to lead.
        host = sorted(vp._DC_EXCLUDE_COMPANIES)[0]
        big = max(vp.dc_all, key=lambda dc: max(
            (p.get('h100') or 0) for p in dc['points'] if p['date'] <= cap))
        fake = dict(big, name='Synthetic host site', company=host,
                    tenant=host, operator=host, users=[], attributed=False,
                    points=[{k: (v * 10 if isinstance(v, float) else v)
                             for k, v in p.items()} for p in big['points']])
        dcs = vp.dc_all + [fake]
        front = vp._cc_trainflop_frontier(dcs, cap, with_names=True)
        assert fake['name'] not in {name for _d, _v, name, _sd in front}
        series = {n: v for n, v in
                  vp._dc_series_for_metric(dcs, 'h100', cap_date=cap).items()
                  if v['company'] not in vp._dc_hidden_companies(dcs)}
        assert fake['name'] in {n for _d, _v, n, _c in vp._dc_envelope(series)}


class TestDcUnattributedCompanies:
    """The † mark: capacity Epoch records no tenant for."""

    def test_marks_exactly_the_name_token_fallbacks(self):
        un = vp._dc_unattributed_companies(vp.dc_all)
        for dc in vp.dc_all:
            if dc['company'] in un:
                assert not dc['attributed'], (
                    f"{dc['name']} has a recorded user/owner but its company "
                    f"{dc['company']!r} is marked unattributed")
        # And the converse: a company with no attributed site is marked.
        attributed = {dc['company'] for dc in vp.dc_all if dc['attributed']}
        assert {dc['company'] for dc in vp.dc_all} - attributed == un

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
        labels = set()
        for label, basis, names in vp._DC_NETWORK_CLUSTERS:
            assert basis in vp._DC_NETWORK_LEVELS, label
            assert len(names) >= 2, f"{label} is not a cluster"
            assert len(set(names)) == len(names), label
            assert label not in labels, f"{label} is used twice"
            labels.add(label)
        # Within one basis a site may belong to at most one cluster, or
        # "largest group" would depend on dict ordering.
        for basis in vp._DC_NETWORK_LEVELS:
            seen = set()
            for label, b, names in vp._DC_NETWORK_CLUSTERS:
                if b != basis:
                    continue
                for name in names:
                    assert name not in seen, f"{name} is in two {basis} clusters"
                    seen.add(name)

    def test_a_wider_basis_subsumes_the_clusters_it_touches(self):
        """The levels are nested by overwriting, so a 'plausible' region that
        names *some* of a metro cluster's sites would re-cut that cluster rather
        than widen it — and the wider setting could then show less capacity."""
        by_basis = {}
        for label, basis, names in vp._DC_NETWORK_CLUSTERS:
            by_basis.setdefault(basis, []).append((label, set(names)))
        for i, basis in enumerate(vp._DC_NETWORK_LEVELS):
            for label, names in by_basis.get(basis, []):
                for lower in vp._DC_NETWORK_LEVELS[:i]:
                    for lo_label, lo_names in by_basis.get(lower, []):
                        if names & lo_names:
                            assert lo_names <= names, (
                                f"{label} splits {lo_label}: missing "
                                f"{sorted(lo_names - names)}")

    def test_each_level_is_a_superset_of_the_one_before(self):
        prox = vp._dc_network_site_clusters('proximity')
        fab = vp._dc_network_site_clusters('fabric')
        plaus = vp._dc_network_site_clusters('plausible')
        assert set(prox) < set(fab) < set(plaus)
        assert vp._dc_network_site_clusters() == fab, "default is 'fabric'"
        assert "Microsoft Fairwater Wisconsin" in fab
        assert "Microsoft Fairwater Wisconsin" not in prox
        # A proximity cluster keeps its own label until a wider basis absorbs
        # it whole. Every metro cluster is absorbed today (Memphis into the
        # Mid-South region); a cluster no region names would keep its label.
        assert prox["Colossus 2"] == fab["Colossus 2"] == "Memphis, TN"
        assert plaus["Colossus 2"] == "Mid-South"

    def test_every_plausible_region_pools_something(self):
        """A region where every site belongs to a different company pools
        nothing — the speculative level lists only groups that act."""
        companies = {dc['name']: dc['company'] for dc in vp.dc_all}
        for label, basis, names in vp._DC_NETWORK_CLUSTERS:
            if basis != 'plausible':
                continue
            cos = [companies[n] for n in names]
            assert len(cos) > len(set(cos)), f"{label} pools nothing"

    def test_network_options_registry_is_well_formed(self):
        assert vp._DC_DEFAULTS["dc_pool_n"] in vp._DC_NETWORK_OPTIONS
        assert vp._DC_DEFAULTS["dc_pool_n"] == "Nearby + announced fabric"
        assert "dc_pool_n" in vp._DC_RESET_KEYS
        assert set(vp._DC_NETWORK_OPTIONS.values()) == {
            'fabric', 'plausible', 'proximity', 'none', 'all'}
        assert set(vp._DC_NETWORK_LEVELS) <= set(vp._DC_NETWORK_OPTIONS.values())


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
            *[vp._dc_company_networked_series(
                ser, vp._dc_network_site_clusters(level))
              for level in vp._DC_NETWORK_LEVELS],
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
        (and therefore shortens the displayed time to train one model): a pooled
        step is exactly its sites' runs added, and beats the largest site."""
        ser = vp._dc_series_for_metric(vp.dc_all, 'mythos')
        ser = {n: d for n, d in ser.items()
               if d['company'] not in vp._DC_EXCLUDE_COMPANIES}
        one = vp._dc_company_networked_series(ser, {})
        net = vp._dc_company_networked_series(ser,
                                              vp._dc_network_site_clusters())
        at = datetime(2027, 6, 30)
        pooled_cos = set()
        for co, steps in net.items():
            cur = [s for s in steps if s[0] <= at]
            if not cur or len(cur[-1][2]) < 2:
                continue
            d, v, names, _label = cur[-1]
            assert v == pytest.approx(sum(vp._dc_val_at(ser[n]['pts'], d)
                                          for n in names)), co
            single = [s for s in one[co] if s[0] <= at][-1][1]
            assert v > single, co
            pooled_cos.add(co)
        assert pooled_cos, "no company pools two sites by mid-2027"


class TestDcProjectionRange:
    """The Data Centers tab's chart window is a sidebar range, URL-tracked."""

    def test_projection_range_registry_is_well_formed(self):
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
        """And a point with no performance figure carries neither."""
        pts = [p for dc in vp.load_data_centers() for p in dc['points']]
        assert any(p['perf'] is not None for p in pts)
        for p in pts:
            if p['perf'] is None:
                assert p['train_flop'] is None and p['train_flop_6mo'] is None
            else:
                assert p['train_flop_6mo'] == pytest.approx(3 * p['train_flop'])

    # label → (stored key, kind, run_days); every other metric has no run
    # length and is neither a FLOP nor a train-time metric.
    _RUN_METRICS = {
        "2mo train log OP": ("train_flop", "flop", vp._DAYS_2MO),
        "6mo train log OP": ("train_flop_6mo", "flop", vp._DAYS_6MO),
        "Capacity (time to GPT-5)": ("gpt5s", "traintime", None),
        "Capacity (time to Mythos)": ("mythos", "traintime", None),
    }

    @pytest.mark.parametrize("label", list(vp._DC_METRICS))
    def test_metric_registry_row(self, label):
        """Only the train-OP metrics carry a run length, which the hover
        milestones and timing shifts read."""
        cfg = vp._DC_METRICS[label]
        if label in self._RUN_METRICS:
            key, kind, run_days = self._RUN_METRICS[label]
            assert (cfg["key"], cfg["kind"]) == (key, kind)
            assert cfg.get("run_days") == run_days
        else:
            assert "run_days" not in cfg
            assert cfg["kind"] not in ("flop", "traintime")

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
        # A bigger stored value (more runs) reads as less time.
        assert [vp._dc_fmt_value(v, 'traintime') for v in (0.5, 5.0, 50.0)] \
            == ["~3.9 months", "~1.7 weeks", "~1.2 days"]

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


class TestDcByCountry:
    """The by-country panel at the bottom of the Data Centers tab."""

    def _series(self, key='h100', cap=datetime(2027, 12, 31)):
        return vp._dc_series_for_metric(vp.dc_all, key, cap_date=cap)

    def _country_of(self):
        return {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}

    def test_loader_carries_country(self):
        by = {dc['name']: dc['country'] for dc in vp.dc_all}
        assert by["VNET Bayin Ulanqab"] == "China"
        assert by["OpenAI Stargate Abilene"] == "United States"

    def test_country_fallback_only_names_sites_epoch_left_blank(self):
        """The fallback exists for timeline-only names with no metadata row. A
        refresh that gives them a row (or renames them) should retire it, and
        a new blank needs one: the two sets are equal (both empty today)."""
        blank = {dc['name'] for dc in vp.dc_all if not dc['country']}
        assert set(vp._DC_COUNTRY_FALLBACK) == blank
        assert all(vp._dc_site_country(dc) for dc in vp.dc_all), \
            "a site has no country even after the fallback"

    def test_china_abroad_sites_exist_and_are_not_in_china(self):
        by = {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}
        for name in vp._DC_CN_ACCESS_ABROAD:
            assert name in by, name
            assert by[name] != vp._DC_CTY_CN, name

    def test_abroad_scope_moves_sites_rather_than_copying_them(self):
        series = self._series()
        dom = vp._dc_country_groups(series, self._country_of(), 'domestic')
        abroad = vp._dc_country_groups(series, self._country_of(), 'abroad')
        assert set(dom[vp._DC_CTY_CN]) < set(abroad[vp._DC_CTY_CN_ACCESS])
        moved = set(abroad[vp._DC_CTY_CN_ACCESS]) - set(dom[vp._DC_CTY_CN])
        assert moved == set(vp._DC_CN_ACCESS_ABROAD)
        # Nothing is counted twice: every site lands in exactly one group.
        for groups in (dom, abroad):
            names = [n for g in groups.values() for n in g]
            assert len(names) == len(set(names))
            assert set(names) == set(series)
        # Only the tuple moves: every other Malaysian site stays in Malaysia.
        my = {n for n, c in self._country_of().items()
              if c == "Malaysia" and n in series}
        assert set(abroad.get("Malaysia", ())) == my - set(vp._DC_CN_ACCESS_ABROAD)
        assert vp._DC_CTY_CN not in abroad

    def test_sites_abroad_pool_as_one_party_only_on_a_hypothetical_link(self):
        """China-accessible pools its sites abroad as one tenant, but only a
        'plausible' cluster joins them, and the detail says the link is
        hypothetical."""
        series = self._series()
        names = vp._dc_country_groups(series, self._country_of(),
                                      'abroad')[vp._DC_CTY_CN_ACCESS]
        abroad = [n for n in vp._DC_CN_ACCESS_ABROAD if n in series]
        assert len(abroad) >= 2

        def last(level):
            return vp._dc_country_steps(series, names, 'company',
                                        vp._dc_network_site_clusters(level))[-1]
        _, v_fab, det_fab = last('fabric')
        _, v_pl, det_pl = last('plausible')
        assert "hypothetical" not in det_fab
        assert v_pl > v_fab
        assert det_pl.startswith(vp._DC_CN_ABROAD_PARTY)
        assert det_pl.endswith("(hypothetical link)")

    def test_pooling_modes_nest(self):
        """Single site ≤ per-company networked group ≤ whole country, at every
        date, for the US with the default clusters."""
        series = self._series()
        groups = vp._dc_country_groups(series, self._country_of(), 'abroad')
        cl = vp._dc_network_site_clusters('fabric')
        us = groups[vp._DC_CTY_US]
        site = vp._dc_country_steps(series, us, 'site', cl)
        comp = vp._dc_country_steps(series, us, 'company', cl)
        whole = vp._dc_country_steps(series, us, 'country', cl)
        assert site and comp and whole
        for d in (datetime(2025, 6, 1), datetime(2026, 6, 1), datetime(2027, 12, 1)):
            s = vp._dc_val_at([(x[0], x[1]) for x in site], d)
            c = vp._dc_val_at([(x[0], x[1]) for x in comp], d)
            w = vp._dc_val_at([(x[0], x[1]) for x in whole], d)
            assert s <= c <= w, (d, s, c, w)
        # Single site per company is the plain envelope, redrawn.
        comp_none = vp._dc_country_steps(series, us, 'company', {})
        assert [(x[0], x[1]) for x in comp_none] == [(x[0], x[1]) for x in site]

    def test_fit_anchors_on_last_recorded_step(self):
        series = self._series()
        groups = vp._dc_country_groups(series, self._country_of(), 'abroad')
        cl = vp._dc_network_site_clusters('fabric')
        steps = vp._dc_country_steps(series, groups[vp._DC_CTY_US], 'site', cl)
        fit = vp._dc_cty_fit(steps, since=2024)
        assert fit['t0'] == steps[-1][0] and fit['v0'] == steps[-1][1]
        assert 0.3 < fit['g'] < 0.7, "US largest-site pace left ×2–5/yr"
        assert fit['sigma_g'] >= vp._DC_CTY_SIGMA_G_FLOOR
        assert fit['sigma_g'] >= fit['se']

    def test_fit_refuses_a_short_history(self):
        steps = [(datetime(2027, 1, 1), 1e5, 'x'), (datetime(2027, 3, 1), 2e5, 'x')]
        assert vp._dc_cty_fit(steps) is None
        assert vp._dc_cty_fit([]) is None
        assert vp._dc_cty_fit([(datetime(2027, 1, 1), 0.0, 'x')]) is None

    def test_trajectories_reproduce_the_record_then_fan_out(self):
        steps = [(datetime(2024, 1, 1) + timedelta(days=30 * i), 1e4 * 2 ** i, 'x')
                 for i in range(24)]
        grid = vp._dc_cty_month_grid(datetime(2024, 1, 1), datetime(2028, 12, 1))
        fit = vp._dc_cty_fit(steps)
        traj = vp._dc_cty_trajectories(steps, fit, grid, 200)
        assert traj.shape == (200, len(grid))
        for j, d in enumerate(grid):
            if d <= fit['t0']:
                v = vp._dc_val_at([(s[0], s[1]) for s in steps], d)
                assert (traj[:, j] == v).all(), d
        last = traj[:, -1]
        assert last.std() > 0 and (last > fit['v0']).mean() > 0.9
        # The cone widens with horizon.
        j1 = grid.index(datetime(2026, 6, 1))
        spread = lambda col: np.log10(np.percentile(col, 90) / np.percentile(col, 10))
        assert spread(traj[:, -1]) > spread(traj[:, j1])
        # Without a fit there is no extrapolation, only the record.
        bare = vp._dc_cty_trajectories(steps, None, grid, 3)
        assert np.isnan(bare[:, -1]).all()

    def test_borrowed_pace_reanchors(self):
        steps = [(datetime(2024, 1, 1) + timedelta(days=30 * i), 1e4 * 2 ** i, 'x')
                 for i in range(24)]
        other = [(datetime(2026, 1, 1), 5e4, 'y'), (datetime(2027, 1, 1), 7e4, 'y')]
        grid = vp._dc_cty_month_grid(datetime(2024, 1, 1), datetime(2028, 12, 1))
        fit = vp._dc_cty_fit(steps)
        traj = vp._dc_cty_trajectories(other, dict(fit, t0=other[-1][0],
                                                   v0=other[-1][1]), grid, 100,
                                       pace=fit)
        j = grid.index(datetime(2027, 1, 1))
        assert (traj[:, j] == 7e4).all()
        assert np.median(traj[:, -1]) > 7e4

    def test_lag_months(self):
        grid = vp._dc_cty_month_grid(datetime(2025, 1, 1), datetime(2025, 12, 1))
        leader = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]],
                          dtype=float)
        follower = np.array([[np.nan, 10, 10, 20, 45, 200, 200, 200, 200, 200,
                              200, 200]], dtype=float)
        lag, unresolved = vp._dc_cty_lag_months(follower, leader, grid)
        assert np.isnan(lag[0, 0]) and not unresolved[0, 0]
        assert lag[0, 1] == 1 and lag[0, 2] == 2      # 10 reached at month 0
        assert lag[0, 3] == 2                          # 20 reached at month 1
        assert lag[0, 4] == 0                          # 45 → first ≥ is 50, month 4
        # 200 is never reached: floored one month past the grid, and flagged.
        assert lag[0, 5] == 5 - 12 and unresolved[0, 5]
        assert lag[0, 11] == -1 and unresolved[0, 11]

    def test_defaults_and_reset_keys(self):
        for k in ("dc_cty_pace", "dc_cty_since", "dc_cty_cones"):
            assert k in vp._DC_RESET_KEYS and k in vp._DC_DEFAULTS
        assert vp._DC_DEFAULTS["dc_cty_pace"] in vp._DC_CTY_PACE_OPTIONS
        assert vp._DC_DEFAULTS["dc_cty_since"] in vp._DC_CTY_SINCE_YEARS
        # The selectboxes have no index= so their first option must be the
        # default, and China-on-the-band is the default on purpose.
        assert list(vp._DC_CTY_PACE_OPTIONS)[0] == vp._DC_DEFAULTS["dc_cty_pace"]
        assert vp._DC_CTY_PACE_OPTIONS[vp._DC_DEFAULTS["dc_cty_pace"]] == 'band'

    def test_china_extrapolates_on_the_export_control_band(self):
        """Past its plans mainland China runs on _CC_CN_COMPUTE_LO/HI, the
        pace the China tab uses, and China-accessible on the
        offshore buildout's _DC_CTY_CN_ACCESS_PACE; each is widened to keep
        its own fit in the 80% cone, and every other country still borrows
        the US trend. The offshore band sits above the export-controlled one
        and below the US's own pace (CHINA_COMPUTE_JUDGMENTS.md)."""
        us = {'g': 0.49, 'sigma_g': 0.10, 't0': datetime(2028, 1, 1)}
        own = {'g': 0.40, 'sigma_g': 0.12}
        dom = (vp._CC_CN_COMPUTE_LO + vp._CC_CN_COMPUTE_HI) / 2
        acc = sum(vp._DC_CTY_CN_ACCESS_PACE) / 2
        assert vp._CC_CN_COMPUTE_LO < vp._DC_CTY_CN_ACCESS_PACE[0] < acc
        assert dom < acc < vp._DC_CTY_CN_ACCESS_PACE[1] < us['g']
        for label in vp._DC_CTY_CN_LABELS:
            mid = acc if label == vp._DC_CTY_CN_ACCESS else dom
            pace, src = vp._dc_cty_pace(label, own, us)
            assert src == 'band' and pace['g'] == pytest.approx(mid)
            assert mid + 1.282 * pace['sigma_g'] >= own['g'] - 1e-12
            assert vp._dc_cty_pace(label, None, us)[0]['g'] == \
                pytest.approx(mid)
            assert vp._dc_cty_pace(label, own, us, 'us')[1] == 'us'
            assert vp._dc_cty_pace(label, own, us, 'own') == (own, 'own')
        assert vp._dc_cty_pace("Norway", own, us) == \
            (dict(us, sigma_g=0.10), 'us')
        assert vp._dc_cty_pace(vp._DC_CTY_US, us, us) == (us, 'own')

    def test_china_largest_site_stays_inside_the_named_plans(self):
        """China's median largest site at end-2029, in facility MW, stays
        within the ~1–3 GW SemiAnalysis's China Datacenter Model (Sep 2026)
        names for its biggest campuses (ByteDance's Ulanqab and Horinger,
        ~1 GW planned each through 2029, 2–3 GW upside). If a refresh moves it
        out, re-check against newer reporting rather than loosening the range."""
        series = vp._dc_series_for_metric(vp.dc_all, 'power')
        country_of = {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}
        rows = []
        for label, country in ((vp._DC_CTY_US, vp._DC_CTY_US),
                               (vp._DC_CTY_CN_DOMESTIC, vp._DC_CTY_CN)):
            names = [n for n in series if country_of.get(n) == country]
            rows.append((label, 'country',
                         vp._dc_country_steps(series, names, 'site', {}), names))
        grid, traj = vp._pc_projection(
            rows, vp.dc_all, datetime(2026, 9, 27),
            since=vp._DC_DEFAULTS["dc_cty_since"],
            horizon=datetime(2029, 12, 31))
        j = grid.index(datetime(2029, 12, 1))
        assert 1000 < np.nanmedian(traj[vp._DC_CTY_CN_DOMESTIC][:, j]) < 3000

    def test_us_pace_is_above_the_offshore_band_on_live_data(self):
        """The offshore band is capped below the US's own largest-site pace:
        Chinese tenants abroad build on US-supplied chips in someone else's
        halls, so they should not outrun the leader's own buildout."""
        series = vp._dc_series_for_metric(vp.dc_all, 'train_flop')
        country_of = {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}
        us = [n for n in series if country_of.get(n) == vp._DC_CTY_US]
        fit = vp._dc_cty_fit(vp._dc_country_steps(series, us, 'site', {}),
                             since=vp._DC_DEFAULTS["dc_cty_since"],
                             t_end=datetime(2026, 9, 27) + timedelta(
                                 days=vp._DC_CTY_PLAN_HORIZON_DAYS))
        assert vp._DC_CTY_CN_ACCESS_PACE[1] < fit['g']


class TestDcRegionShare:
    """The region buckets (`_DC_REGIONS`) the world-share charts use."""

    def _series(self, key='h100', cap=datetime(2028, 12, 31)):
        return vp._dc_series_for_metric(vp.dc_all, key, cap_date=cap)

    def _country_of(self):
        return {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}

    def _grid(self):
        return vp._dc_cty_month_grid(datetime(2022, 1, 1), datetime(2028, 12, 31))

    def test_regions_match_the_labels_the_chart_promises(self):
        assert [lab for lab, _ in vp._DC_REGIONS] == \
            ["US domestic", "China domestic", "SEA", "Europe/UK", "UAE"]
        for country, region in (("United States", "US domestic"),
                                ("China", "China domestic"),
                                ("Malaysia", "SEA"), ("Indonesia", "SEA"),
                                ("United Kingdom", "Europe/UK"),
                                ("Portugal", "Europe/UK"),
                                # Europe's non-EU hosts belong with them.
                                ("Norway", "Europe/UK"),
                                ("United Arab Emirates", "UAE"),
                                ("Australia", "Other"), ("", "Other")):
            assert vp._dc_region_of(country) == region, country

    def test_every_catalogued_country_is_placed_deliberately(self):
        """A share chart drops nothing, so an unnamed country lands in Other.
        Other stays a sliver of catalogued capacity, today and at end-2028; a
        refresh that grows it past that names a country worth a region."""
        series, country_of = self._series(), self._country_of()
        for d in (datetime.now(), datetime(2028, 12, 31)):
            totals = vp._dc_region_totals(series, country_of, [d])
            whole = sum(v[0] for v in totals.values())
            assert whole > 0, d
            other = totals.get(vp._DC_REGION_OTHER, [0.0])[0]
            residual = sorted({c for c in country_of.values()
                               if vp._dc_region_of(c) == vp._DC_REGION_OTHER})
            assert other / whole < 0.05, (d, other / whole, residual)

    def test_no_country_falls_in_two_regions(self):
        seen = set()
        for _, members in vp._DC_REGIONS:
            assert not (members & seen)
            seen |= members

    def test_regions_partition_the_catalogue(self):
        """Every site's capacity is counted exactly once, so the region totals
        add up to the catalogue's own total at every date."""
        series, grid = self._series(), self._grid()
        totals = vp._dc_region_totals(series, self._country_of(), grid)
        for i, d in enumerate(grid):
            whole = sum(x for x in (vp._dc_val_at(v['pts'], d)
                                    for v in series.values()) if x)
            assert abs(sum(v[i] for v in totals.values()) - whole) < 1e-6, d

    def test_an_empty_region_is_left_off_the_chart(self):
        grid = self._grid()
        cty = dict(self._country_of())
        totals = vp._dc_region_totals(self._series(), cty, grid)
        assert "UAE" in totals
        stripped = {n: ('' if c == "United Arab Emirates" else c)
                    for n, c in cty.items()}
        assert "UAE" not in vp._dc_region_totals(self._series(), stripped, grid)

class TestNoDuplicateDefinitions:
    """One definition per name at module level.

    A scripted edit that splices on a marker appearing more than once
    duplicates a whole region of the file, and the later copy silently wins at
    import — the app then runs stale code while the tests still pass, because
    the stale copy is self-consistent. Cheap to check, and nothing else does.
    """

    def test_every_top_level_name_is_defined_once(self):
        import ast as _ast
        tree = _ast.parse(open(vp.__file__).read())
        names = [n.name for n in tree.body
                 if isinstance(n, (_ast.FunctionDef, _ast.ClassDef))]
        dupes = sorted({n for n in names if names.count(n) > 1})
        assert not dupes, dupes


class TestChinaComputeSources:
    """China's compute by source: flows add up to stocks, and the
    hatched-estimate rows are exactly the ones with no published number."""

    def _data(self):
        return vp.load_china_compute_sources(vp._cn_src_mtime())

    def test_every_period_has_every_source_and_a_valid_interval(self):
        d = self._data()
        keys = {k for k, *_ in vp._CN_SOURCES}
        assert list(d["flow"]) == list(d["stock"])
        for kind in ("flow", "stock"):
            for p, rows in d[kind].items():
                assert set(rows) == keys, (kind, p)
                for s, (v, lo, hi, basis) in rows.items():
                    assert lo <= v <= hi, (kind, p, s)
                    assert basis in vp._CN_SRC_BASIS

    def test_stocks_are_the_running_sum_of_flows(self):
        """Except smuggling, whose stock is Epoch's own cumulative row."""
        d = self._data()
        for s in ("legal", "domestic", "remote"):
            run = 0.0
            for p in d["flow"]:
                run += d["flow"][p][s][0]
                assert run == pytest.approx(d["stock"][p][s][0], rel=1e-3)

    def test_only_our_own_history_is_hatched(self):
        """Before the projection, only remote access is ours."""
        d = self._data()
        est = {(p, s) for p, rows in d["flow"].items()
               for s, r in rows.items() if r[3] == "estimate"}
        assert est == {(p, "remote") for p in ("2022", "2023", "2024", "2025")}
        assert {r[3] for p in ("2026", "2027")
                for r in d["flow"][p].values()} == {"ifp"}

    def test_projection_follows_its_stated_rules(self):
        d = self._data()
        p = vp._cn_src_projection(d)
        assert list(p["flow"]) == [str(y) for y in vp._CN_SRC_PROJ_YEARS]
        duv = vp.load_duv_production(vp._duv_mtime())
        us = {y: m for y, _l, m, _h in duv["us"]}
        cap = {y: m * 1e6 for y, _l, m, _h in duv["continue_trend"]}
        f27 = d["flow"]["2027"]
        run = {k: d["stock"]["2027"][k][0] for k in f27}
        for y in vp._CN_SRC_PROJ_YEARS:
            f = p["flow"][str(y)]
            assert f["legal"][0] == f27["legal"][0]
            assert f["smuggled"][0] == pytest.approx(
                f27["smuggled"][0] * us[y] / us[2027])
            assert f["domestic"][0] <= cap[y]
            assert f["domestic"][1] <= f["domestic"][0] <= f["domestic"][2]
            for k in f:
                run[k] += f[k][0]
                assert p["stock"][str(y)][k][0] == pytest.approx(run[k])
        # Offshore leasing flattens after 2027, so remote additions fall.
        assert p["flow"]["2029"]["remote"][0] < f27["remote"][0]

    def test_compute_payoff_default_is_the_mean_of_ours_and_ifps(self):
        """IFP's 10.8 months per 10x at the US's 15.7 ECI/yr is ~14 ECI;
        the default sits halfway between it and our fit."""
        f = vp._pc_compute_payoff
        assert f(6.3, 15.7, "fit") == 6.3
        assert f(6.3, 15.7, "ifp") == pytest.approx(14.13, abs=0.01)
        assert f(6.3, 15.7, "average") == pytest.approx((6.3 + 14.13) / 2,
                                                       abs=0.01)
        assert vp._PC_PAYOFF_OPTIONS[vp._PC_DEFAULTS["pc_payoff"]] == "average"

    def test_distillation_keeps_a_quarter_less_per_tenfold_data_cut(self):
        """IFP: ~25% of distilled capability per 10x less harvested data,
        so 90% keeps 75%, 99% keeps 50%, and only a total cut stops it."""
        k = vp._pc_dist_keep
        assert k(90) == pytest.approx(0.75)
        assert k(99) == pytest.approx(0.50)
        assert k(100) == 0.0
        opts = vp._PC_DIST_CUT_OPTIONS
        assert [k(c) for c in opts] == sorted((k(c) for c in opts),
                                              reverse=True)
        assert vp._pc_dist_cut_label(99) == "99% \u00b7 keeps 50% (30\u201360%)"

    def test_channel_cut_grows_with_the_cut_and_vanishes_without_one(self):
        """Stopping a channel removes a share of China's in-China compute
        growth: none for no cut or an empty window, more for a deeper cut,
        and stopping two channels removes more than either."""
        s, e = datetime(2027, 7, 1), datetime(2031, 12, 31)
        cut = vp._cn_src_channel_cut
        assert cut(s, e, {"smuggled": 0.0}) == 0.0
        assert cut(e, s, {"smuggled": 1.0}) == 0.0
        half, full = cut(s, e, {"smuggled": 0.5}), cut(s, e, {"smuggled": 1.0})
        assert 0 < half < full < 1
        legal = cut(s, e, {"legal": 1.0})
        both = cut(s, e, {"smuggled": 0.5, "legal": 1.0})
        assert both > max(half, legal)

    def test_policy_mix_only_removes_and_matches_baseline_without_levers(self):
        today, start = datetime(2026, 10, 19), datetime(2029, 1, 1)
        d = self._data()
        base = vp._cn_src_with_policy(today, data=d)
        proj = vp._cn_src_projection(d)
        for p, rows in {**d["stock"], **proj["stock"]}.items():
            for k, r in rows.items():
                assert base["stock"][p][k][0] == pytest.approx(r[0])
        pol = vp._cn_src_with_policy(today, smug_start=start, smug_cut=0.5,
                                     legal_start=start,
                                     stop_legal=True, remote_cut=today,
                                     dom_cut=0.4, data=d)
        for p in base["stock"]:
            for k in base["stock"][p]:
                assert pol["stock"][p][k][0] <= base["stock"][p][k][0] + 1e-6
        # Nothing changes before the pacing date but remote access, lost
        # from its own cut date; legal sales stop after it.
        assert pol["stock"]["2028"]["legal"] == base["stock"]["2028"]["legal"]
        assert pol["stock"]["2026"]["remote"][0] == 0.0
        assert (pol["flow"]["2030"]["legal"][0] == 0.0
                < base["flow"]["2030"]["legal"][0])

    def test_compute_ratio_is_one_without_levers_and_scales_with_power(self):
        today = datetime(2026, 10, 19)
        none = vp._cn_src_compute_ratio(today, today, {}, 0.5)
        assert none(0.0) == pytest.approx(1.0)
        assert none(3.0) == pytest.approx(1.0)
        mix = dict(remote_cut=today, dom_cut=0.4)
        half = vp._cn_src_compute_ratio(today, today, mix, 0.5)
        full = vp._cn_src_compute_ratio(today, today, mix, 1.0)
        assert half(0.0) == pytest.approx(1.0)
        assert 0 < full(3.0) < half(3.0) < 1
        assert half(3.0) == pytest.approx(full(3.0) ** 0.5)

    def test_hbm_controls_remove_their_share_of_domestic_output_only(self):
        today = datetime(2026, 10, 19)
        d = self._data()
        base = vp._cn_src_with_policy(today, data=d)
        hbm = vp._cn_src_with_policy(today, hbm_cut=vp._PC_HBM_CUT, data=d)
        assert hbm["flow"]["2029"]["domestic"][0] == pytest.approx(
            (1 - vp._PC_HBM_CUT) * base["flow"]["2029"]["domestic"][0])
        assert hbm["flow"]["2025"] == base["flow"]["2025"]
        for k in ("legal", "smuggled", "remote"):
            assert hbm["flow"]["2029"][k] == base["flow"]["2029"][k]

    def test_slower_domestic_growth_shrinks_only_domestic(self):
        d = self._data()
        hi, lo = vp._cn_src_projection(d), vp._cn_src_projection(d, growth=1.3)
        assert lo["flow"]["2030"]["domestic"][0] < hi["flow"]["2030"]["domestic"][0]
        for k in ("legal", "smuggled", "remote"):
            assert lo["flow"]["2030"][k] == hi["flow"]["2030"][k]


class TestDuvProduction:
    """The CTS lithography table and the ban ratio the Pacing lever reads."""

    def _data(self):
        return vp.load_duv_production(vp._duv_mtime())

    def test_transcription_matches_the_reports_own_2035_labels(self):
        """The figure's aria-labels state the 2035 values in prose; the
        table must reproduce them."""
        d = self._data()
        assert d["us"][-1] == (2035, 422, 974, 3566)
        assert d["continue_ceiling"][-1][1:] == (306, 532, 1071)
        assert d["continue_push"][-1][1:] == (204, 378, 784)
        assert d["continue_trend"][-1][1:] == (56, 114, 251)
        assert d["banned_ceiling"][-1][1:] == (87, 153, 321)
        assert d["banned_push"][-1][1:] == (58, 110, 235)
        assert d["banned_trend"][-1][1:] == (17, 33, 72)

    def test_table_is_complete_and_ordered(self):
        d = self._data()
        assert set(d) == {"us"} | {f"{i}_{a}" for i in ("continue", "banned")
                                   for a, _ in vp._DUV_ALLOCS}
        for key, rows in d.items():
            assert [r[0] for r in rows] == list(range(2026, 2036)), key
            for y, lo, mid, hi in rows:
                assert lo <= mid <= hi, (key, y)
        # A ban never raises the ceiling, and bites only from 2027.
        for a, _ in vp._DUV_ALLOCS:
            c, b = d[f"continue_{a}"], d[f"banned_{a}"]
            assert c[0] == b[0]
            assert all(bb[2] <= cc[2] for cc, bb in zip(c, b))

    def test_import_ban_waits_for_output_to_reach_the_ceiling(self):
        """China's real output is far under the banned ceiling, so an import
        ban alone barely moves growth this decade and bites later."""
        t0 = datetime(2026, 10, 1)
        near = vp._duv_policy_cut(t0, datetime(2031, 12, 31))
        far = vp._duv_policy_cut(t0, datetime(2035, 12, 31))
        assert 0.0 <= near < 0.08
        assert far > near + 0.04
        assert vp._duv_policy_cut(t0, t0) is None
        assert vp._duv_policy_cut(t0, datetime(2031, 12, 31),
                                  ban_imports=False) == 0.0

    def test_faster_output_growth_reaches_the_ceiling_sooner(self):
        """The import-ban cut is driven by how fast China's real output
        grows into the banned ceiling, so it rises with that rate."""
        t0, t1 = datetime(2026, 10, 1), datetime(2031, 12, 31)
        lo, hi = vp._DUV_CN_OUTPUT_GROWTH_RANGE
        cuts = [vp._duv_policy_cut(t0, t1, growth=g)
                for g in (lo, vp._DUV_CN_OUTPUT_GROWTH, hi)]
        assert cuts == sorted(cuts) and cuts[-1] > cuts[0] + 0.05

    def test_servicing_ban_bites_now_and_rises_with_the_decay(self):
        t0, t1 = datetime(2026, 10, 1), datetime(2028, 12, 31)
        cuts = [vp._duv_policy_cut(t0, t1, ban_imports=False, decay=d)
                for d in (0.05, 0.15, 0.30)]
        assert 0.05 < cuts[0] < cuts[1] < cuts[2] <= 1.0
        # Stacking the import ban on top never helps China.
        both = vp._duv_policy_cut(t0, datetime(2035, 12, 31), decay=0.15)
        svc = vp._duv_policy_cut(t0, datetime(2035, 12, 31),
                                 ban_imports=False, decay=0.15)
        assert both >= svc

    def test_realized_start_is_the_reports_ifp_range(self):
        """The 2026 grid centres on IFP's domestic output, the same 2026
        flow the compute-by-source chart draws."""
        lv = sorted({a for a, _b in vp._duv_realized_grid()})
        mid = 10 ** lv[len(lv) // 2]
        dom = vp.load_china_compute_sources(vp._cn_src_mtime())
        assert mid * 1e6 == pytest.approx(
            dom["flow"]["2026"]["domestic"][0], rel=0.05)


class TestCcWorldShares:
    """The global compute distribution at the bottom of the CC tab."""

    def test_priors_are_a_coherent_set(self):
        assert list(vp._WC_LABELS) == ["US domestic", "China domestic", "SEA",
                                       "UAE", "Europe/UK", "Other"]
        assert abs(sum(m for _, m, _, _ in vp._WC_REGIONS) - 100.0) < 1e-9
        for lab, mid, lo, hi in vp._WC_REGIONS:
            assert 0 < lo < mid < hi, lab
            assert lab in vp._WC_NOTES and vp._WC_NOTES[lab], lab
        # Every region is carried forward, and only the ratios matter.
        assert [lab for lab, *_ in vp._WC_GROWTH] == list(vp._WC_LABELS)
        for lab, mid, lo, hi in vp._WC_GROWTH:
            assert 1.0 < lo < mid < hi, lab

    def test_regions_are_the_data_centers_tab_s(self):
        """Both sections bucket a country the same way, so a reader comparing
        the two charts is comparing like with like."""
        for country in ("United States", "United Kingdom", "Norway", "China",
                        "Malaysia", "United Arab Emirates", "Australia", ""):
            assert vp._wc_region_of(country) == vp._dc_region_of(country)

    def test_samples_recover_their_centrals_and_spreads(self):
        sh = vp._wc_sample_shares(20000)
        assert set(sh) == set(vp._WC_LABELS)
        for lab, mid, lo, hi in vp._WC_REGIONS:
            med = float(np.percentile(sh[lab], 50))
            # Renormalizing pulls the medians a little off the inputs.
            assert abs(med - mid) < max(1.0, 0.1 * mid), (lab, med)
            p10, p90 = (float(np.percentile(sh[lab], q)) for q in (10, 90))
            assert p10 < med < p90 and p10 > lo * 0.7 and p90 < hi * 1.3, lab

    def test_paths_start_at_todays_shares_and_stay_shares(self):
        years = [0.0, 1.0, 2.5]
        paths = vp._wc_share_paths(years, 4000)
        assert set(paths) == set(vp._WC_LABELS)
        for i in range(len(years)):
            assert np.allclose(sum(p[i] for p in paths.values()), 100.0)
        for lab, mid, _, _ in vp._WC_REGIONS:
            assert abs(float(np.percentile(paths[lab][0], 50)) - mid) < \
                max(1.0, 0.1 * mid), lab

    def test_the_drawn_line_is_the_one_the_caption_describes(self):
        """The chart draws the central-rate scenario, which sums to 100 with
        no rescaling; the sampled paths only supply the range."""
        c = vp._wc_central_shares([0.0, 3.0])
        assert abs(sum(v[1] for v in c.values()) - 100.0) < 1e-6
        for lab, mid, _, _ in vp._WC_REGIONS:
            assert abs(c[lab][0] - mid) < 1e-6, lab
        assert c["China domestic"][1] < c["China domestic"][0]
        assert c["UAE"][1] > c["UAE"][0]

    def test_growth_shocks_are_shared_without_touching_the_marginals(self, monkeypatch):
        """A loading per region on one global shock. It must not change any
        region's own rate spread — only the joint — so every stated band
        stays as documented. Pinning "Other" to a fixed rate makes each
        sample's share growth relative to it that region's own rate."""
        assert set(vp._WC_COMMON_LOAD) == set(vp._WC_LABELS)
        assert all(0.0 <= x <= 1.0 for x in vp._WC_COMMON_LOAD.values())
        # China is the decoupled one: domestic silicon, controls already on.
        assert vp._WC_COMMON_LOAD["China domestic"] < \
            min(v for k, v in vp._WC_COMMON_LOAD.items() if k != "China domestic")
        ref = "Other"
        rows = {lab: (mid, lo, hi) for lab, mid, lo, hi in vp._WC_GROWTH}
        g_ref = rows[ref][0]
        monkeypatch.setattr(vp, "_WC_GROWTH", tuple(
            (lab, g_ref, g_ref, g_ref) if lab == ref else (lab, *rows[lab])
            for lab in vp._WC_LABELS))

        def rates(common):
            p = vp._wc_share_paths([0.0, 1.0], 40000, common=common)
            rel = lambda lab: p[lab][1] / p[lab][0]
            return {lab: g_ref * rel(lab) / rel(ref)
                    for lab in vp._WC_LABELS if lab != ref}
        indep, shared = rates(0.0), rates(None)
        for lab, g in shared.items():
            _mid, lo, hi = rows[lab]
            for q in (10, 90):
                assert np.percentile(g, q) == pytest.approx(
                    np.percentile(indep[lab], q), rel=0.02), (lab, q)
            # The 10–90 width is the stated band's, centred on the median.
            assert np.percentile(g, 90) / np.percentile(g, 10) == \
                pytest.approx(hi / lo, rel=0.03), lab

    def test_sharing_the_shock_damps_the_share_bands(self):
        """The shared part cancels in the renormalization, so correlated
        rates give tighter shares than independent ones."""
        wide = vp._wc_share_paths([3.3], 20000, common=0.0)
        tight = vp._wc_share_paths([3.3], 20000)
        for lab in ("US domestic", "UAE"):
            w = np.percentile(wide[lab][0], 90) - np.percentile(wide[lab][0], 10)
            t = np.percentile(tight[lab][0], 90) - np.percentile(tight[lab][0], 10)
            assert t < w, lab
        # The UAE's band is the one that ran away: it grows fastest from the
        # smallest base, so a lognormal rate compounds hardest there.
        assert np.percentile(tight["UAE"][0], 90) < 30

    def test_a_low_us_world_is_a_slow_world_not_a_handover(self):
        """Most non-US capacity is US firms on US-supplied chips, so a US
        slowdown drags them with it; China is the region that decouples. With
        the rates independent the model says the opposite."""
        def absorbed(paths):
            us = paths["US domestic"][0]
            m = us <= np.percentile(us, 10)
            gain = {l: paths[l][0][m].mean() - paths[l][0].mean()
                    for l in vp._WC_LABELS if l != "US domestic"}
            tot = sum(gain.values())
            return {l: g / tot for l, g in gain.items()}
        indep = absorbed(vp._wc_share_paths([3.3], 40000, common=0.0))
        shared = absorbed(vp._wc_share_paths([3.3], 40000))
        assert shared["China domestic"] > indep["China domestic"] + 0.10
        assert shared["Europe/UK"] < indep["Europe/UK"]

    def test_the_correction_is_the_point_of_the_section(self):
        """The tracked sites alone say the US holds nearly everything; the
        published estimates give the US less and China more."""
        cat = vp._wc_catalogued_shares(vp.dc_all, datetime.now())
        assert abs(sum(cat.values()) - 100.0) < 1e-6
        sh = vp._wc_sample_shares(4000)
        assert float(np.percentile(sh["US domestic"], 50)) < cat["US domestic"]
        assert float(np.percentile(sh["China domestic"], 50)) > \
            cat["China domestic"]


class TestDcMilestoneDates:
    """Hover milestones name a training run only for the train-OP metrics,
    and then the run length that metric assumes."""

    def test_no_run_without_a_train_metric(self):
        text = vp._dc_milestone_dates(datetime(2026, 3, 1))
        assert text == "DC online ~Mar 01, 2026"
        assert "Training done" not in text and "Model out" not in text

    def test_run_length_follows_the_metric(self):
        base = datetime(2026, 3, 1)
        two = vp._dc_milestone_dates(base, 0, vp._DAYS_2MO)
        six = vp._dc_milestone_dates(base, 0, vp._DAYS_6MO)
        assert "Training done (2mo run)" in two and "Training done (6mo run)" in six
        assert f"~{base + timedelta(days=vp._DAYS_2MO):%b %d, %Y}" in two
        assert f"~{base + timedelta(days=vp._DAYS_6MO):%b %d, %Y}" in six
        assert (f"Model out ~{base + timedelta(days=vp._DAYS_6MO) + vp._CC_RUN_COMPLETION_LAG:%b %d, %Y}"
                in six)

    def test_shift_is_undone_before_expanding(self):
        base = datetime(2026, 3, 1)
        shifted = base + timedelta(days=vp._DAYS_6MO)
        assert vp._dc_milestone_dates(shifted, vp._DAYS_6MO, vp._DAYS_6MO) == \
            vp._dc_milestone_dates(base, 0, vp._DAYS_6MO)


class TestDcCtyPlanSlip:
    """Planned steps carry symmetric timing noise; the trend takes over at the
    plan horizon with the plan as a floor."""

    def _plan(self):
        # Doubling every 6 months, recorded to 2028-06.
        return [(datetime(2024, 1, 1) + timedelta(days=182 * i), 1e4 * 2 ** i, 'x')
                for i in range(10)]

    def test_plan_centres_the_band_and_spread_grows_with_lead(self):
        steps = self._plan()
        today = datetime(2026, 6, 1)
        grid = vp._dc_cty_month_grid(datetime(2026, 1, 1), datetime(2028, 6, 1))
        traj = vp._dc_cty_trajectories(steps, None, grid, 2000, today=today,
                                       slip_sigma=0.3)
        pts = [(s[0], s[1]) for s in steps]
        plan = lambda d: vp._dc_val_at(pts, d)
        for j, d in enumerate(grid):
            if d <= today:
                assert (traj[:, j] == plan(d)).all()
            else:
                # Timing noise is symmetric, so the plan sits inside the
                # 25–75% band, not on its edge. (Monotonicity biases the path
                # up a little; the doubling plan makes the value spread wide.)
                lo, hi = np.percentile(traj[:, j], [25, 75])
                assert lo <= plan(d) * 1.05 and plan(d) <= hi * 1.05, d
        near, far = grid.index(datetime(2026, 9, 1)), grid.index(datetime(2028, 3, 1))
        spread = lambda j: np.log10(np.percentile(traj[:, j], 90)
                                    / np.percentile(traj[:, j], 10))
        assert spread(far) > spread(near) > 0
        # Every path is non-decreasing.
        assert (np.diff(traj, axis=1) >= -1e-9).all()

    def test_no_slip_without_today(self):
        steps = self._plan()
        grid = vp._dc_cty_month_grid(datetime(2026, 1, 1), datetime(2028, 6, 1))
        traj = vp._dc_cty_trajectories(steps, None, grid, 20)
        pts = [(s[0], s[1]) for s in steps]
        assert all((traj[:, j] == vp._dc_val_at(pts, d)).all()
                   for j, d in enumerate(grid))

    def test_fit_anchors_at_the_plan_horizon_and_plan_floors_the_trend(self):
        steps = self._plan()
        t_end = datetime(2027, 1, 1)
        fit = vp._dc_cty_fit(steps, since=2024, t_end=t_end)
        pts = [(s[0], s[1]) for s in steps]
        assert fit['t0'] == t_end and fit['v0'] == vp._dc_val_at(pts, t_end)
        grid = vp._dc_cty_month_grid(datetime(2026, 1, 1), datetime(2029, 6, 1))
        slow = dict(fit, g=0.01, sigma_g=0.0, sigma_res=0.0)
        traj = vp._dc_cty_trajectories(steps, slow, grid, 400, today=datetime(2026, 6, 1),
                                       slip_sigma=0.0001)
        # A near-flat trend cannot pull the line under the catalogued plan
        # (up to the plan's own level uncertainty, which is symmetric)…
        j = grid.index(datetime(2028, 6, 1))
        plan = vp._dc_val_at(pts, grid[j])
        assert np.median(traj[:, j]) >= plan * 0.93
        lead_yr = (grid[j] - datetime(2026, 6, 1)).days / 365.25
        assert (traj[:, j] >= plan * 10 ** (-4 * vp._DC_CTY_PLAN_LEVEL_SIGMA * lead_yr)).all()
        # …and past the catalogue the trend alone carries on.
        assert (traj[:, -1] >= traj[:, j]).all()

    def test_plan_quality_reads_construction_status(self):
        today = datetime(2026, 1, 1)
        dcs = [{'name': 'a', 'points': [
            {'date': datetime(2025, 1, 1), 'status': 'Building 1 is operational.'},
            {'date': datetime(2027, 1, 1), 'status': 'Per the [filing](http://x).'},
            {'date': datetime(2027, 6, 1), 'status': 'Estimated by analogy.'},
        ]}]
        assert vp._dc_plan_quality(dcs, ['a'], today) == 0.5
        assert vp._dc_plan_quality(dcs, ['b'], today) is None
        assert vp._dc_plan_quality(dcs, ['a'], datetime(2028, 1, 1)) is None
        lo, hi = vp._DC_CTY_SLIP_SIGMA_Q['sourced'], vp._DC_CTY_SLIP_SIGMA_Q['estimate']
        assert vp._dc_cty_slip_sigma(1.0) == lo
        assert vp._dc_cty_slip_sigma(0.0) == hi == vp._dc_cty_slip_sigma(None)
        assert lo < vp._dc_cty_slip_sigma(0.5) < hi

    def test_live_catalogue_has_both_kinds_of_plan(self):
        """The heuristic has to split Epoch's future rows, or it is dead weight."""
        q = vp._dc_plan_quality(vp.dc_all, [dc['name'] for dc in vp.dc_all],
                                datetime.now())
        assert q is not None and 0.1 < q < 0.9


class TestPacing:
    """The Pacing tab: threshold menu, crossing math, and the entity roster."""

    def test_race_bar_is_the_us_median_run_at_the_pause(self):
        caps = {vp._DC_CTY_US: (1e26, 3e26, 9e26),
                vp._DC_CTY_CN_ACCESS: (1e25, 2e25, 4e25)}
        assert vp._pc_race_threshold(caps) == 3e26
        assert vp._pc_race_threshold({vp._DC_CTY_CN_ACCESS: (1, 2, 3)}) == \
            vp._PC_FALLBACK_THRESHOLD

    def test_defaults_cover_reset_keys_and_url_tracking(self):
        """Every Pacing default is reset and URL-tracked, as are the RSI
        blend's weights and its not-yet conditioning. The pause is an int of
        months-from-today rather than a month label, so a bookmarked value
        can never go stale the way the cut-off sliders' options can."""
        assert set(vp._PC_DEFAULTS) == set(vp._PC_RESET_KEYS)
        assert vp._PC_DEFAULTS["pc_run"] in vp._PC_RUN_OPTIONS
        keys, defaults = vp._all_tracked()
        for k, v in vp._PC_DEFAULTS.items():
            assert k in keys and defaults[k] == v, k
        for slug, w in vp._PC_RSI_WEIGHTS.items():
            assert defaults[vp._PC_RSI_W_KEY + slug] == w, slug
        assert sum(vp._PC_RSI_WEIGHTS.values()) == 90       # nine research cards
        assert 0 <= defaults["pc_pause_mo"] <= vp._PC_PAUSE_MO_MAX
        assert isinstance(defaults["pc_pause_mo"], int)
        assert "rsi_notyet" in keys and defaults["rsi_notyet"] is True
        assert "rsi_notyet_ramp" in keys

    def test_world_stock_is_epochs_anchor_grown_at_the_regions_pace(self):
        """Installed compute is charted in H100e off Epoch's chip-sales total
        (31.7M by June 2026), grown at the share-weighted central growth
        (~2.2x a year), which lands end-2025 near the data's own 20.4M."""
        d0, h0 = vp._WC_WORLD_H100E
        assert vp._wc_world_h100e(d0) == pytest.approx(h0)
        yr = vp._wc_world_h100e(d0 + timedelta(days=365.25)) / h0
        assert 2.0 < yr < 2.5
        end25 = vp._wc_world_h100e(datetime(2025, 12, 31))
        assert 19e6 < end25 < 23e6

    def test_spans_past_a_year_read_in_years(self):
        assert vp._pc_span(11.6) == "11.6 mo"
        assert vp._pc_span(15.5) == "1.3 yr"
        assert vp._pc_span(-18, 0, sign=True) == "-1.5 yr"
        assert vp._pc_span(5, 0, sign=True) == "+5 mo"

    def test_paced_clock_runs_the_us_at_its_pace_from_the_start(self):
        """Before the pacing date the US runs on its own clock; after it,
        `pace` years of progress a year: 0 freezes it, 1 changes nothing."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        assert list(vp._pc_paced_clock(t, 1.0, 0.0)) == [0.0, 1.0, 1.0, 1.0]
        assert list(vp._pc_paced_clock(t, 1.0, 1.0)) == list(t)
        assert list(vp._pc_paced_clock(t, 1.0, 0.5)) == [0.0, 1.0, 1.5, 2.0]

    def test_pace_changes_list_each_new_pace_once(self):
        """The chart's rules: one per change in the effective pace, none
        where a window leaves the pace as it was."""
        d = [datetime(2027, m, 1) for m in range(1, 7)]
        assert vp._pc_pace_changes(d[0], 1.0, []) == []
        assert vp._pc_pace_changes(d[0], 0.5, [(d[1], d[3], 0.0)]) == [
            (d[0], 0.5), (d[1], 0.0), (d[3], 0.5)]
        assert vp._pc_pace_changes(d[0], 0.0, [(d[2], d[4], 0.5)]) == [
            (d[0], 0.0), (d[2], 0.5), (d[4], 0.0)]
        assert vp._pc_pace_changes(d[5], 1.0, [(d[1], d[2], 0.55),
                                               (d[2], d[3], 0.55)]) == [
            (d[1], 0.55), (d[3], 1.0)]

    def test_pace_rules_say_pause_or_how_many_times_slower(self):
        assert vp._pc_pace_rule_label(0.0) == "pause"
        assert vp._pc_pace_rule_label(1.0) == "no pacing"
        assert vp._pc_pace_rule_label(0.5) == "2\u00d7 slower"
        assert vp._pc_pace_rule_label(0.55) == "1.8\u00d7 slower"

    def test_pacing_windows_slow_the_us_then_return_it_to_normal(self):
        """A window runs the US at its own pace whatever the main lever says
        and hands it back after; where windows overlap, the slowest wins."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # A pause from 1 to 2, unpaced around it: one year lost for good.
        assert list(vp._pc_paced_clock(t, 9.0, 1.0, [(1.0, 2.0, 0.0)])) == \
            [0.0, 1.0, 1.0, 2.0, 3.0]
        # Half pace from 1 to 3 under a 0.5 main lever from 2: 0.5 throughout.
        assert list(vp._pc_paced_clock(t, 2.0, 0.5, [(1.0, 3.0, 0.5)])) == \
            [0.0, 1.0, 1.5, 2.0, 2.5]
        # A window restarts a paused US at half speed, then it pauses again.
        assert list(vp._pc_paced_clock(t, 1.0, 0.0, [(2.0, 3.0, 0.5)])) == \
            [0.0, 1.0, 1.0, 1.5, 1.5]
        # Overlapping windows: the pause wins where the two meet.
        assert list(vp._pc_paced_clock(
            t, 9.0, 1.0, [(1.0, 3.0, 0.5), (2.0, 3.0, 0.0)])) == \
            [0.0, 1.0, 1.5, 1.5, 2.5]

    def test_speedup_is_the_us_rate_at_that_eci_over_todays(self):
        """China's multiplier at an ECI is the Takeoff path's growth when it
        stood there, over today's trend: 1 on the trend, 3x once the path
        climbs three times as fast, and never below 1."""
        years = np.arange(0, 10, 1 / 12)
        path = np.where(years < 5, 100 + 10 * years, 150 + 30 * (years - 5))
        tk = {"years": years, "paths": np.vstack([path, 100 + 5 * years]),
              "ppy": 10.0}
        tab = vp._pc_speedup_table(tk, np.array([110.0, 200.0]))
        assert tab[0] == pytest.approx([1.0, 3.0], abs=0.01)
        assert tab[1] == pytest.approx([1.0, 1.0])      # floored at 1

    def test_speedup_multiplies_chinas_rate_and_is_its_own_channel(self):
        """A 2x speed-up on a fixed bar halves the time to cross, and the
        extra growth lands in the `speedup` channel so the channels still sum
        to the path."""
        kw = dict(us_anchor=150.0, us_rate=0.0, a_partial=0.0, g_lo=0.0,
                  g_hi=0.0, algo_lo=10.0, algo_mid=10.0, algo_hi=10.0,
                  inno_lo=10.0, inno_hi=10.0, pace_lo=1.0, pace_hi=1.0,
                  n=50, horizon_yrs=5.0)
        np.random.seed(0)
        base, _, _ = vp._cc_cn_crossing_sim(100.0, 120.0, **kw)
        chan = {}
        np.random.seed(0)
        fast, grid, traj = vp._cc_cn_crossing_sim(
            100.0, 120.0, channels=chan, speedup=lambda e: np.full(50, 2.0),
            **kw)
        assert np.nanmedian(fast) == pytest.approx(np.nanmedian(base) / 2, rel=0.1)
        total = sum(chan[k] for k in vp._CC_CHANNELS) + chan['speedup']
        assert total == pytest.approx(traj - 100.0)
        assert chan['speedup'][:, -1].min() > 0

    def test_less_compute_shrinks_only_the_speedup_channel(self):
        """`speedup_scale` multiplies the AI speed-up's extra growth and
        nothing else: at half, that channel halves."""
        kw = dict(us_anchor=150.0, us_rate=0.0, a_partial=0.0, g_lo=0.0,
                  g_hi=0.0, algo_lo=10.0, algo_mid=10.0, algo_hi=10.0,
                  inno_lo=10.0, inno_hi=10.0, pace_lo=1.0, pace_hi=1.0,
                  n=20, horizon_yrs=1.0)
        out = {}
        for scale in (None, lambda t: 0.5):
            chan = {}
            np.random.seed(0)
            vp._cc_cn_crossing_sim(100.0, 200.0, channels=chan,
                                   speedup=lambda e: np.full(20, 2.0),
                                   speedup_scale=scale, **kw)
            out[scale is None] = chan
        full, half = out[True], out[False]
        # Within 10%: a slower path shifts the gap-driven channels a little.
        assert half['speedup'][:, -1] == pytest.approx(
            0.5 * full['speedup'][:, -1], rel=0.1)
        for k in vp._CC_CHANNELS:
            assert half[k][:, -1] == pytest.approx(full[k][:, -1], rel=0.1)

    def test_full_tracking_gives_china_the_us_extra_in_eci_a_year(self):
        """With `speedup_follow` weight 1, China's extra growth is the US's
        AI-driven extra at that ECI, (s - 1) x the US trend rate, whatever
        China's own rate; at 0 it is (s - 1) x China's own rate."""
        kw = dict(us_anchor=150.0, us_rate=0.0, a_partial=0.0, g_lo=0.0,
                  g_hi=0.0, algo_lo=5.0, algo_mid=5.0, algo_hi=5.0,
                  inno_lo=5.0, inno_hi=5.0, pace_lo=1.0, pace_hi=1.0,
                  n=20, horizon_yrs=1.0, channels=None)
        for w, want in ((1.0, 5.0 + 20.0), (0.0, 5.0 * 2)):
            np.random.seed(0)
            _, grid, traj = vp._cc_cn_crossing_sim(
                100.0, 1e9, speedup=lambda e: np.full(20, 2.0),
                speedup_follow=(w, 20.0), **kw)
            assert ((traj[:, -1] - 100.0) / grid[-1]).mean() == pytest.approx(want, rel=0.02)

    def test_cross_years_reads_a_moving_bar(self):
        """A 2-D target is a per-sample path: the crossing is where the
        gap to it closes, interpolated inside the step."""
        grid = np.arange(5.0)
        traj = np.array([[0.0, 2.0, 4.0, 6.0, 8.0]])
        bar = np.array([[3.0, 4.0, 5.0, 6.0, 7.0]])   # gap closes at 3.0
        assert vp._pc_cross_years(traj, grid, bar)[0] == pytest.approx(3.0)
        assert np.isnan(vp._pc_cross_years(traj, grid, bar + 10)[0])

    def test_add_months_gives_one_distinct_label_per_step(self):
        """The pause slider labels each position with its date, so the steps
        must be whole calendar months — distinct, ordered, and clamped into
        short months rather than spilling into the next one."""
        start = datetime(2026, 8, 24)
        got = [vp._pc_add_months(start, m)
               for m in range(vp._PC_PAUSE_MO_MAX + 1)]
        assert got[0] == start
        assert got == sorted(got)
        labels = [f"{d:%b %Y}" for d in got]
        assert len(set(labels)) == len(labels)
        assert labels[1] == "Sep 2026" and labels[18] == "Feb 2028"
        # Day clamped to the target month, never rolled forward.
        assert vp._pc_add_months(datetime(2026, 1, 31), 1) == \
            datetime(2026, 2, 28)
        assert vp._pc_add_months(datetime(2028, 1, 31), 1) == \
            datetime(2028, 2, 29)          # leap year

    def test_capacity_at_reads_the_race_projection(self):
        """The state-of-play capacity is the threshold race's own projection
        read at one date, not a second fit: bracketed, ordered, and never
        smaller later."""
        today = datetime(2026, 8, 1)
        rows = [r for r in self._rows(party='country')
                if r[0] in (vp._DC_CTY_US, vp._DC_CTY_CN_ACCESS)]
        assert len(rows) == 2
        early = vp._pc_capacity_at(rows, vp.dc_all, today,
                                   datetime(2027, 6, 1), since=2024)
        late = vp._pc_capacity_at(rows, vp.dc_all, today,
                                  datetime(2029, 6, 1), since=2024)
        for label, (lo, mid, hi) in early.items():
            assert lo <= mid <= hi
            assert mid < late[label][1]      # built capacity does not shrink
        assert early[vp._DC_CTY_US][1] > early[vp._DC_CTY_CN_ACCESS][1]
        # An entity with nothing recorded is omitted, not zeroed.
        assert vp._pc_capacity_at([("X", 'country', [], ())], vp.dc_all,
                                  today, datetime(2028, 1, 1)) == {}

    def test_metr_eta_reproduces_the_metr_tab_defaults(self):
        """The METR cards are the METR tab's own milestone row, not a second
        model: same GPT-4o-broken segment, same DT and position priors."""
        fr = vp.frontier_all
        days = np.array([(m['date'] - fr[0]['date']).days
                         for m in fr[vp.gpt4o_idx:]], dtype=float)
        slope = vp.fit_line(
            days, np.array([np.log2(m['p50_min'])
                            for m in fr[vp.gpt4o_idx:]]))[1]
        dt = round(1.0 / slope)
        for _lab, key in vp._PC_METR_LEVELS:
            early, med, late = vp._pc_metr_eta(fr, key, n=20000)
            assert early < med < late
            # Median path: fitted position carried forward at the segment DT.
            fitted = 2 ** (np.mean(
                np.array([np.log2(m[key]) for m in fr[vp.gpt4o_idx:]])
                - slope * days) + slope * days[-1]) / 60
            want = fr[-1]['date'] + timedelta(
                days=np.log2(vp._PC_METR_TARGET_HRS / fitted) * dt)
            assert abs((med - want).days) < 45

    def test_metr_p80_crosses_the_bar_after_p50(self):
        """p80 is the more demanding bar, so it arrives later at every
        percentile."""
        p50 = vp._pc_metr_eta(vp.frontier_all, 'p50_min', n=20000)
        p80 = vp._pc_metr_eta(vp.frontier_all, 'p80_min', n=20000)
        assert all(b > a for a, b in zip(p50, p80))

    def test_rli_eta_reproduces_the_rli_tab_defaults(self):
        """The RLI 90% card is the RLI tab at its defaults: single OLS in
        logit space, odds-doubling time over [DT/2, DT*2]."""
        fr = vp.rli_frontier_all
        days = np.array([(m['date'] - fr[0]['date']).days for m in fr],
                        dtype=float)
        logit = vp._logit(np.array([m['rli_score'] / 100 for m in fr]))
        slope = vp.fit_line(days, logit)[1]
        fitted = np.mean(logit - slope * days) + slope * days[-1]
        dt = round(np.log(2) / slope)
        prev = None
        for target in (25.0, 50.0, vp._PC_RLI_TARGET_PCT):
            early, med, late = vp._pc_rli_eta(fr, target, n=20000)
            assert early < med < late
            want = fr[-1]['date'] + timedelta(
                days=(vp._logit(target / 100) - fitted) / (np.log(2) / dt))
            assert abs((med - want).days) < 45
            # A higher bar is never reached earlier.
            assert prev is None or med > prev
            prev = med

    def test_report_lag_moves_only_the_release_dated_milestones(self):
        """METR/ECI/RLI are dated off released, benchmarked models; CoBench and
        the staff survey are internal and already on the run-finished clock."""
        days = np.full(20000, 500.0)
        lo, hi = vp._PC_REPORT_LAG_DAYS
        shifted = vp._pc_report_lag(days, True, "Training run finished")
        assert (shifted <= 500 - lo).all() and (shifted >= 500 - hi).all()
        assert abs(shifted.mean() - (500 - (lo + hi) / 2)) < 1.0
        # Internal milestones never move, and nothing moves on the release clock.
        assert vp._pc_report_lag(days, False, "Training run finished") is days
        assert vp._pc_report_lag(days, True, vp._PC_TIMING_RELEASE) is days
        assert vp._PC_TIMING_RELEASE in vp._DC_TIMING_OPTIONS

    def test_rsi_blend_mixes_distributions_not_medians(self):
        """A mixture keeps each component's spread: shifting weight onto a
        later milestone moves the blend into its cluster, and the 80% band
        spans both components rather than collapsing to their mean."""
        origin = datetime(2026, 8, 24)
        comps = [("early", "early", origin, np.full(4000, 100.0)),
                 ("late", "late", origin, np.full(4000, 900.0))]
        _, med_early, _ = vp._pc_rsi_blend(
            comps, {"early": 80, "late": 20}, origin, n=8000)
        lo, med_late, hi = vp._pc_rsi_blend(
            comps, {"early": 20, "late": 80}, origin, n=8000)
        assert (med_early - origin).days == 100
        assert (med_late - origin).days == 900
        # p10 in one component, p90 in the other — not an average of medians.
        assert (lo - origin).days == 100 and (hi - origin).days == 900

    def test_rsi_blend_normalises_and_survives_all_zero(self):
        origin = datetime(2026, 8, 24)
        comps = [("a", "a", origin, np.full(2000, 100.0)),
                 ("b", "b", origin, np.full(2000, 300.0))]
        scaled = vp._pc_rsi_blend(comps, {"a": 300, "b": 100}, origin, n=6000)
        unscaled = vp._pc_rsi_blend(comps, {"a": 3, "b": 1}, origin, n=6000)
        assert scaled == unscaled            # only the ratio matters
        # All-zero weights on unknown slugs give no blend rather than
        # dividing by zero; on the real slugs they fall back to the defaults.
        assert vp._pc_rsi_blend(comps, {"a": 0, "b": 0}, origin, n=100) is None
        real = [(s, s, origin, np.full(500, 100.0)) for s in vp._PC_RSI_WEIGHTS]
        assert vp._pc_rsi_blend(real, {s: 0 for s in vp._PC_RSI_WEIGHTS},
                                origin, n=500) is not None

    def test_condition_on_today_truncates_and_reweights(self):
        """The reality check: samples dating a milestone before today are
        rejected (not clamped), the survival fraction is reported so the
        blend can down-weight the component, and a milestone with no future
        mass drops out entirely."""
        today = datetime(2026, 8, 24)
        early = today - timedelta(days=100)
        comps = [("half", "half", early, np.arange(1.0, 201.0)),
                 ("future", "future", today, np.full(50, 30.0)),
                 ("crossed", "crossed", early, np.full(50, 10.0))]
        out, surv = vp._pc_condition_on_today(comps, today)
        assert surv == {"half": 0.5, "future": 1.0, "crossed": 0.0}
        by = {c[0]: c[3] for c in out}
        assert set(by) == {"half", "future"}
        assert len(by["half"]) == 100 and (by["half"] > 100).all()
        assert len(by["future"]) == 50
        # Mixed with weight x survival — mixture-level rejection — the dead
        # component contributes nothing even at full prior weight.
        w = {"half": 50, "future": 25, "crossed": 100}
        days = vp._pc_rsi_blend_samples(
            out, {s: w[s] * surv[s] for s in w}, today, n=4000)
        assert days.min() > 0

    def test_condition_ramp_discounts_the_near_future(self):
        """The soft near-term likelihood: past mass still dies, mass beyond
        the window is untouched, and mass t days out survives w.p. t/N."""
        today = datetime(2026, 8, 24)
        comps = [("past", "past", today, np.full(400, -5.0)),
                 ("mid", "mid", today, np.full(4000, 15.0)),
                 ("far", "far", today, np.full(400, 60.0))]
        out, surv = vp._pc_condition_on_today(comps, today, ramp_days=30.0)
        assert surv["past"] == 0.0
        assert surv["far"] == 1.0
        assert 0.4 < surv["mid"] < 0.6
        assert {c[0] for c in out} == {"mid", "far"}

    def test_dist_fig_respects_the_horizon(self):
        """The CDF axis runs to the *Project through* year end when given
        one (98th-percentile cap otherwise), and a median past the horizon
        is not annotated — the curve ending below 50% already says so."""
        origin = datetime(2026, 8, 24)
        days = np.linspace(1, 2000, 4000)
        args = (days, origin, origin + timedelta(days=200),
                origin + timedelta(days=1000), origin + timedelta(days=1800))
        horizon = datetime(2027, 12, 31)
        capped = vp._pc_rsi_dist_fig(*args, horizon=horizon)
        assert max(capped.data[0].x) <= horizon
        assert not capped.layout.annotations
        free = vp._pc_rsi_dist_fig(*args)
        assert max(free.data[0].x) > horizon
        assert free.layout.annotations

    def test_rsi_subjective_penalty_calibration_and_ordering(self):
        days = np.arange(1.0, 10001.0)
        assert vp._pc_rsi_atc_samples(days, 0) is days
        adjusted = vp._pc_rsi_atc_samples(days, 50)
        stronger = vp._pc_rsi_atc_samples(days, 90)
        assert np.mean(adjusted <= 1000) == pytest.approx(0.05, abs=0.0001)
        assert np.mean(stronger <= 1000) == pytest.approx(0.01, abs=0.0001)
        assert np.all(stronger >= adjusted)
        assert np.all(adjusted >= days)
        assert adjusted.max() <= days.max()
        assert len(adjusted) == len(days)
        tied = np.full(100, 42.0)
        assert np.array_equal(vp._pc_rsi_atc_samples(tied, 50), tied)
        assert vp._RSI_DEFAULTS["rsi_atc_penalty"] == 0
        assert "rsi_atc_penalty" in vp._RSI_RESET_KEYS

    def test_rsi_penalty_comparison_curve(self):
        origin = datetime(2026, 9, 6)
        days = np.arange(1.0, 1001.0)
        adjusted = vp._pc_rsi_atc_samples(days, 50)
        dates = [origin + timedelta(days=float(x))
                 for x in np.percentile(adjusted, [10, 50, 90])]
        fig = vp._pc_rsi_dist_fig(adjusted, origin, *dates, model_days=days)
        assert len(fig.data) == 2
        assert np.all(np.array(fig.data[0].y) >= np.array(fig.data[1].y))
        assert "before subjective penalty" in fig.data[0].hovertext[0]

    def test_dist_fig_ghost_shows_the_removed_mass(self):
        """With `raw_days` the unconditioned blend draws as a ghost trace
        behind the main curve, nonzero where the reality check cut mass."""
        origin = datetime(2026, 8, 24)
        cond = np.linspace(1, 400, 2000)
        raw = np.concatenate([np.full(500, -50.0), np.linspace(1, 400, 1500)])
        fig = vp._pc_rsi_dist_fig(cond, origin, origin + timedelta(days=40),
                                  origin + timedelta(days=200),
                                  origin + timedelta(days=360), raw_days=raw)
        assert len(fig.data) == 2
        assert fig.data[0].y[0] > 10          # ghost carries the past mass
        assert fig.data[1].y[0] < fig.data[0].y[0]
        solo = vp._pc_rsi_dist_fig(cond, origin, origin + timedelta(days=40),
                                   origin + timedelta(days=200),
                                   origin + timedelta(days=360))
        assert len(solo.data) == 1

    def test_ramp_window_extends_on_the_release_clock(self):
        """Release dates trail run-finished by a report lag, so an active
        discount window grows by the lag's lower bound; a window turned off
        stays off."""
        assert vp._pc_ramp_for(vp._PC_TIMING_RELEASE, 90.0) == \
            90.0 + vp._PC_REPORT_LAG_DAYS[0]
        assert vp._pc_ramp_for("Training run finished", 90.0) == 90.0
        assert vp._pc_ramp_for(vp._PC_TIMING_RELEASE, 0.0) == 0.0

    def test_clock_note_splits_by_milestone_not_by_paragraph(self):
        """Which clock a card is on is a per-card fact, so it rides that
        card's hover: release-dated cards say they were pulled back,
        internal ones say they were not, and on the release clock the
        distinction is moot for both."""
        rel_dated = vp._pc_clock_note(True, "Training run finished")
        internal = vp._pc_clock_note(False, "Training run finished")
        assert "pulled back" in rel_dated and "training run finished" in rel_dated
        assert "internal" in internal.lower() and "pulled back" not in internal
        on_release = {vp._pc_clock_note(r, vp._PC_TIMING_RELEASE)
                      for r in (True, False)}
        assert len(on_release) == 1 and "releases" in on_release.pop()

    def test_rsi_eta_reproduces_the_rsi_tab_defaults(self):
        """The CoBench 85% card is the RSI tab at its defaults: single OLS in
        logit space, odds-doubling time over that tab's widened rate CI."""
        fr = vp.rsi_frontier_all
        base, intercept, slope = vp._rsi_fit(fr)
        fitted = intercept + slope * (fr[-1]['date'] - base).days
        dt_lo, dt_hi = vp._rsi_dt_ci(fr, np.log(2) / slope)
        dt = np.sqrt(dt_lo * dt_hi)           # lognormal CI's central value
        prev = None
        for target in (70.0, 80.0, vp._RSI_SUBSTITUTION_BAR):
            early, med, late = vp._pc_rsi_eta(fr, target, n=20000)
            assert early < med < late
            want = fr[-1]['date'] + timedelta(
                days=(vp._logit(target / 100) - fitted) / (np.log(2) / dt))
            assert abs((med - want).days) < 45
            # A higher bar is never reached earlier.
            assert prev is None or med > prev
            prev = med

    def test_each_card_hover_names_the_clock_its_dates_are_on(self):
        """The release_dated flag each card is registered with decides whether
        `_pc_report_lag` pulls it back, and the card's hover ends with the
        matching `_pc_clock_note`: internal evaluations already sit on the
        run-finished clock, while cards dated off shipped models (revenue
        included, since ARR is earned by them) are pulled back onto it."""
        today = datetime.now().replace(hour=0, minute=0, second=0,
                                       microsecond=0)
        _cap, _surv, _raw, notes = vp._pc_milestone_components(
            "Training run finished", today, False, 0.0, 200, "test")
        assert set(notes) == set(vp._PC_RSI_WEIGHTS) | set(vp._PC_RSI_CARD_ONLY)
        internal = ("cobench_", "rdauto_", "al5_", "staff_", "code_",
                    "experiments_")
        released = ("metr_", "rli_", "nextstep_")
        for slug, note in notes.items():
            if slug.startswith(internal):
                assert "internal evaluation" in note, slug
                assert "pulled back" not in note, slug
            else:
                assert slug.startswith(released), slug
                assert "pulled back" in note, slug

    def test_cutoff_slider_labels_round_trip(self):
        """The advanced cut-off sliders are month labels (URL- and
        reset-safe): first option is the constant default, every other
        option parses back to a first-of-month at or after today, and the
        list stops at the horizon."""
        today = datetime(2026, 8, 24)
        opts = vp._pc_when_options(today)
        assert opts[0] == vp._PC_WHEN_NOW == vp._PC_DEFAULTS["pc_dist_when"]
        assert opts[0] == vp._PC_DEFAULTS["pc_remote_when"]
        assert opts[1] == "Sep 2026" and opts[-1] == "Dec 2031"
        assert len(set(opts)) == len(opts)
        dates = [vp._pc_when_date(o, today) for o in opts]
        assert dates[0] == today
        assert dates == sorted(dates)
        assert all(d.day == 1 and d > today for d in dates[1:])
        assert dates[-1] <= vp._PC_HORIZON
        # A stale bookmarked label never parses into a date the sliders
        # can't show; unknown text falls back to today rather than raising.
        assert vp._pc_when_date("nonsense", today) == today
        assert vp._pc_when_date(None, today) == today

    def test_level_scatter_is_its_own_channel(self):
        """With level noise on, the offset is recorded as `level`, so the
        channels still sum to the path exactly."""
        kw = dict(a_partial=8.0, g_lo=0.15, g_hi=0.30, algo_lo=11.0,
                  algo_mid=12.0, algo_hi=13.0, pace_lo=0.8, pace_hi=1.1,
                  n=120, inno_lo=8.0, inno_hi=10.0, us_anchor=165.0,
                  us_rate=25.0, pure_lo=3.0, pure_hi=4.0, t_pause=0.5,
                  level_sd=2.0, level_ramp_yrs=0.2)
        chan = {}
        _, grid, traj = vp._cc_cn_crossing_sim(150.0, 160.0, channels=chan, **kw)
        assert set(chan) == set(vp._CC_CHANNELS) | {'level'}
        total = sum(chan[k] for k in chan)
        assert np.allclose(total, traj - 150.0)
        assert np.allclose(chan['level'][:, 0], 0.0)

    def test_channels_account_for_the_whole_climb(self):
        """The four channel cumulatives are an exact decomposition of the
        path — that identity is what makes the breakdown's shares real
        rather than a second, looser model."""
        kw = dict(a_partial=8.0, g_lo=0.15, g_hi=0.30, algo_lo=11.0,
                  algo_mid=12.0, algo_hi=13.0, pace_lo=0.8, pace_hi=1.1,
                  n=120, inno_lo=8.0, inno_hi=10.0, us_anchor=165.0,
                  us_rate=25.0, pure_lo=3.0, pure_hi=4.0, t_pause=0.5)
        chan = {}
        _, grid, traj = vp._cc_cn_crossing_sim(
            150.0, 160.0, channels=chan, **kw)
        assert set(chan) == set(vp._CC_CHANNELS)
        total = sum(chan[k] for k in vp._CC_CHANNELS)
        assert np.allclose(total, traj - 150.0)
        assert all((np.diff(chan[k], axis=1) >= -1e-9).all()
                   for k in vp._CC_CHANNELS)
        # Cutting a channel zeroes its column and nothing else's.
        chan2 = {}
        vp._cc_cn_crossing_sim(150.0, 160.0, channels=chan2,
                               t_dist_stop=0.0, **kw)
        assert chan2['distillation'][:, -1].max() < 1e-9
        assert chan2['innovation'][:, -1].min() > 0
        # Omitting the dict costs nothing and changes no result.
        assert vp._cc_cn_crossing_sim(150.0, 160.0, **kw)[2].shape == \
            traj.shape

    def test_comp_shadow_prices_access_to_compute_abroad(self):
        """comp_shadow adds a domestic-only compute column by capping each
        sample's own pace; the remainder is what access abroad bought. A cap
        above the band never binds, so the two coincide exactly and the row
        reads zero — which is what makes 'cut remote access' collapse it."""
        kw = dict(a_partial=8.0, g_lo=0.15, g_hi=0.30, algo_lo=11.0,
                  algo_mid=12.0, algo_hi=13.0, pace_lo=0.8, pace_hi=1.1,
                  n=120, inno_lo=8.0, inno_hi=10.0, us_anchor=165.0,
                  us_rate=25.0)
        loose, tight, setback = {}, {}, {}
        vp._cc_cn_crossing_sim(150.0, 160.0, channels=loose,
                               comp_shadow=(8.0 * 0.40, None), **kw)
        _, _, traj = vp._cc_cn_crossing_sim(
            150.0, 160.0, channels=tight,
            comp_shadow=(8.0 * 0.16, None), **kw)
        vp._cc_cn_crossing_sim(150.0, 160.0, channels=setback,
                               comp_shadow=(8.0 * 0.40, (0.0, 1.0)), **kw)
        assert np.allclose(loose['compute_domestic'], loose['compute'])
        assert (tight['compute'][:, -1] - tight['compute_domestic'][:, -1]
                > 0).mean() > 0.9      # the band barely reaches this cap
        # A dead window on the shadow alone also prices the level setback.
        assert (setback['compute'][:, -1]
                > setback['compute_domestic'][:, -1]).all()
        # The shadow never exceeds the term it shadows, and never disturbs
        # the four-channel identity.
        for ch in (loose, tight, setback):
            assert (ch['compute_domestic'] <= ch['compute'] + 1e-9).all()
        assert np.allclose(sum(tight[k] for k in vp._CC_CHANNELS),
                           traj - 150.0)

    def test_at_years_reads_a_cumulative_at_the_crossing(self):
        """_pc_at_years interpolates inside the step, like the crossing
        itself, and passes NaN through for samples that never cross."""
        grid = np.arange(5) * 0.25
        cum = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 2.0, 4.0, 6.0, 8.0]])
        got = vp._pc_at_years(cum, grid, np.array([0.375, np.nan]))
        assert got[0] == pytest.approx(1.5)
        assert np.isnan(got[1])
        # Clamped at both ends rather than wrapping.
        edges = vp._pc_at_years(cum, grid, np.array([0.0, 99.0]))
        assert edges[0] == pytest.approx(0.0)
        assert edges[1] == pytest.approx(8.0)   # sample 1's own last step

    def test_cross_years_reproduces_the_sims_own_crossings(self):
        """_pc_cross_years re-reads sampled paths against a second bar; on
        the bar the sim itself used it must return the sim's answer, else
        the run-length comparison isn't paired."""
        kw = dict(a_partial=8.0, g_lo=0.15, g_hi=0.30, algo_lo=11.0,
                  algo_mid=12.0, algo_hi=13.0, pace_lo=0.8, pace_hi=1.1,
                  n=200, inno_lo=8.0, inno_hi=10.0,
                  us_anchor=165.0, us_rate=25.0)
        years, grid, traj = vp._cc_cn_crossing_sim(150.0, 160.0, **kw)
        again = vp._pc_cross_years(traj, grid, 160.0)
        assert np.array_equal(np.isnan(years), np.isnan(again))
        assert np.allclose(years[~np.isnan(years)], again[~np.isnan(again)])
        # A lower bar is reached no later, per sample; a per-sample bar
        # works the same as the scalar it broadcasts from.
        lower = vp._pc_cross_years(traj, grid, 155.0)
        assert np.all(lower[~np.isnan(lower)] <=
                      np.nan_to_num(again, nan=np.inf)[~np.isnan(lower)])
        vec = vp._pc_cross_years(traj, grid, np.full(200, 160.0))
        assert np.allclose(np.nan_to_num(vec, nan=-1),
                           np.nan_to_num(again, nan=-1))
        # Never reached inside the grid → NaN, not a spurious crossing.
        assert np.isnan(vp._pc_cross_years(traj, grid, 1e6)).all()

    def test_plan_crossing_is_first_step_at_or_above(self):
        steps = [(datetime(2025, 1, 1), 5e26, "Site A"),
                 (datetime(2026, 1, 1), 2e27, "Cluster B (2 sites)"),
                 (datetime(2027, 1, 1), 8e27, "Cluster B (3 sites)")]
        assert vp._pc_plan_crossing(steps, 1e27) == \
            (datetime(2026, 1, 1), "Cluster B (2 sites)")
        assert vp._pc_plan_crossing(steps, 2e27) == \
            (datetime(2026, 1, 1), "Cluster B (2 sites)")
        assert vp._pc_plan_crossing(steps, 1e29) == (None, None)
        # Bare (date, value) steps still work, with no facility.
        assert vp._pc_plan_crossing([(datetime(2025, 1, 1), 5e26)], 1e26) == \
            (datetime(2025, 1, 1), None)

    def test_entity_steps_carry_the_facility(self):
        by = {l: s for l, _, s, _ in self._rows()}
        d, via = vp._pc_plan_crossing(by['Anthropic'], 1e27)
        assert d is not None and via
        by = {l: s for l, _, s, _ in self._rows(party='country')}
        d, via = vp._pc_plan_crossing(by[vp._DC_CTY_US], 1e27)
        assert d is not None and via

    def test_company_projection_borrows_the_us_reference_pace(self):
        """Company rosters have no US row, so the borrowed pace comes from
        ref_steps; without it a short-history company has no fit and its
        paths end in NaN, with it they extrapolate past its last step."""
        rows = self._rows()
        today = datetime(2026, 8, 1)
        plan_end = today + timedelta(days=vp._DC_CTY_PLAN_HORIZON_DAYS)
        short = [l for l, _, s, _ in rows
                 if s and vp._dc_cty_fit(s, since=2024, t_end=plan_end) is None]
        assert short
        grid, traj = vp._pc_projection(rows, vp.dc_all, today, since=2024,
                                       ref_steps=self._us_ref_steps())
        _, bare = vp._pc_projection(rows, vp.dc_all, today, since=2024)
        steps = {l: s for l, _, s, _ in rows}
        for label in short:
            assert traj[label].shape == (vp.N_SAMPLES, len(grid))
            assert np.isnan(bare[label][:, -1]).all(), label
            end = traj[label][:, -1]
            assert not np.isnan(end).any(), label
            assert np.median(end) > steps[label][-1][1], label

    def test_crossing_idx_sentinel_and_nan(self):
        traj = np.array([[np.nan, 1.0, 5.0, 5.0],     # crosses at 2
                         [1.0, 1.0, 1.0, 1.0],        # never → sentinel 4
                         [6.0, 6.0, 6.0, 6.0]])       # crosses at 0
        assert list(vp._pc_crossing_idx(traj, 5.0)) == [2.0, 4.0, 0.0]

    def test_crossing_idx_monotone_in_threshold(self):
        """Per sample the crossing index never moves earlier as the bar
        rises — which is what makes every percentile of it monotone too."""
        rng = np.random.default_rng(0)
        traj = np.fmax.accumulate(rng.random((200, 30)), axis=1)
        prev = np.zeros(200)
        for t in (0.2, 0.5, 0.8, 1.1):
            idx = vp._pc_crossing_idx(traj, t)
            assert (idx >= prev).all()
            prev = idx

    def test_idx_date_maps_percentiles_and_nones_past_grid(self):
        grid = [datetime(2026, m, 1) for m in range(1, 5)]
        idx = np.array([0.0, 1.0, 4.0, 4.0])
        assert vp._pc_idx_date(idx, grid, 25) == datetime(2026, 2, 1)
        assert vp._pc_idx_date(idx, grid, 90) is None

    # ── Live-data roster and projection ──

    def _rows(self, today=datetime(2026, 8, 1), party='tenant'):
        series_all = vp._dc_series_for_metric(vp.dc_all, 'train_flop_6mo',
                                              cap_date=None)
        hidden = vp._dc_hidden_companies(vp.dc_all, now=today)
        shown = {n: s for n, s in series_all.items()
                 if s['company'] not in hidden}
        cluster_of = vp._dc_network_site_clusters('fabric')
        country_of = {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}
        return vp._pc_entity_rows(
            shown, series_all, country_of, cluster_of,
            unattributed=vp._dc_unattributed_companies(vp.dc_all),
            party=party)

    def _us_ref_steps(self):
        series_all = vp._dc_series_for_metric(vp.dc_all, 'train_flop_6mo',
                                              cap_date=None)
        country_of = {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}
        cluster_of = vp._dc_network_site_clusters('fabric')
        names = vp._dc_country_groups(series_all, country_of,
                                      'abroad').get(vp._DC_CTY_US, [])
        return vp._dc_country_steps(series_all, names, 'company', cluster_of)

    def test_rosters_are_one_kind_per_party(self):
        """Tenant/operator rosters are companies only; the 'country' party
        races countries instead, with China differentiated into accessible
        and domestic-only."""
        company_rows = self._rows()
        assert all(k == 'company' for _, k, _, _ in company_rows)
        labels = [l for l, *_ in company_rows]
        for country in (vp._DC_CTY_US, vp._DC_CTY_CN_ACCESS,
                        vp._DC_CTY_CN_DOMESTIC):
            assert country not in labels
        unattr = vp._dc_unattributed_companies(vp.dc_all)
        for l, k, _, _ in company_rows:
            if l.endswith('†'):
                assert l[:-2].rstrip() in unattr, l
        country_rows = self._rows(party='country')
        assert all(k == 'country' for _, k, _, _ in country_rows)
        clabels = [l for l, *_ in country_rows]
        for want in (vp._DC_CTY_US, vp._DC_CTY_CN_ACCESS,
                     vp._DC_CTY_CN_DOMESTIC):
            assert want in clabels, want
        assert vp._DC_CTY_CN not in clabels, \
            "the plain-China group must be split into accessible/domestic"

    def test_projection_grid_runs_to_the_horizon(self):
        rows = self._rows(party='country')
        today = datetime(2026, 8, 1)
        grid, traj = vp._pc_projection(rows, vp.dc_all, today, since=2024)
        assert grid[0] == datetime(2026, 8, 1)
        assert grid[-1] == vp._PC_HORIZON
        us = traj[vp._DC_CTY_US]
        assert us.shape == (vp.N_SAMPLES, len(grid))
        # The US has recorded capacity today, so no sample starts NaN, and
        # a bar below today's capacity is crossed immediately in every sample.
        assert not np.isnan(us[:, 0]).any()
        idx = vp._pc_crossing_idx(us, 1e27)
        assert (idx == 0).all()

    @staticmethod
    def _shared_site(users=('Synthetic A', 'Synthetic B'), owner='Synthetic Owner'):
        """The biggest live site, ten times over, listed under several users
        and owned by someone else, so the shared-tenancy checks never depend on
        which sites Epoch happens to list as shared."""
        big = max(vp.dc_all, key=lambda dc: max(
            (p.get('train_flop_6mo') or 0) for p in dc['points']))
        return dict(big, name='Synthetic shared site', company=users[0],
                    tenant=users[0], operator=owner, users=list(users),
                    attributed=True,
                    points=[{k: (v * 10 if isinstance(v, float) else v)
                             for k, v in p.items()} for p in big['points']])

    def test_party_attribution_carried_by_loader(self):
        """Every site carries both attributions; 'tenant' is exactly the
        loader's company label (first-listed user, else the operator), and
        'operator' is the building's recorded owner."""
        by = {dc['name']: dc for dc in vp.dc_all}
        assert all(dc['tenant'] == dc['company'] for dc in vp.dc_all)
        for dc in vp.dc_all:
            want = dc['users'][0] if dc['users'] else dc['operator']
            assert dc['tenant'] == want, dc['name']
        owned = 0
        for r in csv.DictReader(open(os.path.join(
                os.path.dirname(vp.__file__), "data_centers.csv"))):
            name = (r.get('Name') or '').strip()
            owner = vp._dc_clean_owner(r.get('Owner', ''))
            if name in by and owner:
                assert by[name]['operator'] == \
                    vp._DC_COMPANY_ALIASES.get(owner, owner), name
                owned += 1
        assert owned
        assert set(vp._DC_PARTY_OPTIONS.values()) == {'tenant', 'operator'}
        assert set(vp._PC_PARTY_OPTIONS.values()) == \
            {'tenant', 'operator', 'country'}

    def test_with_party_sets_the_membership_list(self):
        """Tenant view: `company` stays the primary label but `companies` is
        every listed user, so a shared site counts under each tenant.
        Operator view: both collapse to the owner."""
        dcs = vp.dc_all + [self._shared_site()]
        tview = {v['name']: v for v in vp._dc_with_party(dcs, 'tenant')}
        for dc in dcs:
            v = tview[dc['name']]
            assert v['company'] == dc['company']
            assert v['companies'] == (dc['users'] or [dc['tenant']])
            assert v['points'] is dc['points']
        assert tview['Synthetic shared site']['companies'] == \
            ['Synthetic A', 'Synthetic B']
        oview = {v['name']: v for v in vp._dc_with_party(dcs, 'operator')}
        for dc in dcs:
            v = oview[dc['name']]
            assert v['company'] == dc['operator']
            assert v['companies'] == [dc['operator']]
        assert oview['Synthetic shared site']['companies'] == ['Synthetic Owner']

    def test_shared_site_counts_under_every_tenant(self):
        """The point of the membership list: every listed user of a shared
        site carries it in its per-company series — while the raw (viewless)
        site list keeps single membership for the CC tab."""
        dcs = vp.dc_all + [self._shared_site()]
        view = vp._dc_with_party(dcs, 'tenant')
        series = vp._dc_series_for_metric(view, 'train_flop_6mo',
                                          cap_date=None)
        per_co = vp._dc_company_networked_series(
            series, vp._dc_network_site_clusters('fabric'))
        shared = [dc for dc in dcs if len(dc['users']) > 1]
        assert any(dc['name'] == 'Synthetic shared site' for dc in shared)
        for dc in shared:
            peak = max(x for _, x in series[dc['name']]['pts'])
            for co in dc['users']:
                assert co in per_co, (dc['name'], co)
                assert max(t[1] for t in per_co[co]) >= peak * (1 - 1e-9), \
                    (dc['name'], co)
        raw = vp._dc_series_for_metric(dcs, 'train_flop_6mo', cap_date=None)
        for dc in shared:
            assert raw[dc['name']]['companies'] == [dc['company']], dc['name']

    def test_pool_menu_is_the_dc_tabs_and_nests(self):
        """The Pacing tab's pooling selector reuses _DC_NETWORK_OPTIONS, and a wider
        basis never shrinks an entity's final capacity."""
        assert vp._PC_DEFAULTS["pc_pool"] in vp._DC_NETWORK_OPTIONS
        assert vp._DC_NETWORK_OPTIONS[vp._PC_DEFAULTS["pc_pool"]] == 'fabric'
        series_all = vp._dc_series_for_metric(vp.dc_all, 'train_flop_6mo',
                                              cap_date=None)
        country_of = {dc['name']: vp._dc_site_country(dc) for dc in vp.dc_all}
        prev = {}
        for basis in ('none', 'proximity', 'fabric', 'plausible', 'all'):
            cluster_of = ({} if basis == 'none' else None if basis == 'all'
                          else vp._dc_network_site_clusters(basis))
            rows = vp._pc_entity_rows(series_all, series_all, country_of,
                                      cluster_of)
            finals = {l: s[-1][1] for l, _, s, _ in rows}
            for l, v in prev.items():
                assert finals.get(l, 0) >= v * (1 - 1e-9), (basis, l)
            prev = finals



class TestAnchorSlug:
    """`?to=` is the app's section deep link; the slug is inlined into an
    injected script, so its shape is whitelisted rather than escaped."""

    def test_accepts_a_streamlit_heading_anchor(self):
        assert (vp._anchor_slug("per-company-does-the-buildout-predict-release-timing")
                == "per-company-does-the-buildout-predict-release-timing")

    def test_normalizes_a_pasted_fragment(self):
        assert vp._anchor_slug(" #Buildout-By-Country-US-vs-China ") == \
            "buildout-by-country-us-vs-china"

    def test_rejects_anything_that_could_close_the_script_tag(self):
        for raw in ("</script><img src=x onerror=alert(1)>", 'a"b', "a b",
                    "a<b", "a/b", "-lead", "", "x" * 200):
            assert vp._anchor_slug(raw) == "", raw


class TestRevenueCombined:
    """The optional OpenAI + Anthropic sum line."""

    def _series(self):
        od, ov = vp._parse_revenue(vp._OPENAI_REVENUE)
        ad, av = vp._parse_revenue(vp._ANTHROPIC_REVENUE)
        return od, ov, ad, av

    def test_shared_dates_are_the_plain_sum(self):
        od, ov, ad, av = self._series()
        cd, cv = vp._rev_combined_series(od, ov, ad, av)
        shared = set(od) & set(ad)
        assert shared
        for d, v in zip(cd, cv):
            if d in shared:
                assert v == pytest.approx(ov[od.index(d)] + av[ad.index(d)])

    def test_dates_are_the_union_from_the_later_start(self):
        od, ov, ad, av = self._series()
        cd, _ = vp._rev_combined_series(od, ov, ad, av)
        assert cd == sorted(d for d in set(od) | set(ad) if d >= max(od[0], ad[0]))

    def test_a_one_sided_date_still_carries_the_other_side(self):
        od, ov, ad, av = self._series()
        cd, cv = vp._rev_combined_series(od, ov, ad, av)
        for d, v in zip(cd, cv):
            assert v >= max(vp._rev_value_at(od, ov, d), vp._rev_value_at(ad, av, d))

    def test_past_a_series_last_report_it_is_held_flat(self):
        """Never extrapolated: past one side's last report the sum moves only
        with the side still reporting."""
        a_d = [datetime(2025, 1, 1), datetime(2025, 2, 1)]
        b_d = [datetime(2025, 1, 1), datetime(2025, 3, 1)]
        cd, cv = vp._rev_combined_series(a_d, [1.0, 2.0], b_d, [1.0, 4.0])
        assert cd == [datetime(2025, 1, 1), datetime(2025, 2, 1), datetime(2025, 3, 1)]
        assert cv[-1] == pytest.approx(2.0 + 4.0)

    def test_interpolation_is_log_linear(self):
        dates = [datetime(2025, 1, 1), datetime(2025, 1, 3)]
        assert vp._rev_value_at(dates, [1.0, 4.0], datetime(2025, 1, 2)) == \
            pytest.approx(2.0)



class TestRevenueChina:
    """The Chinese-lab ARR series: one unit, sourced, in date order."""

    def test_every_point_is_dated_sourced_and_classed(self):
        for lab, spec in vp._CN_REVENUE.items():
            dates, vals = vp._parse_revenue([(d, v) for d, v, _, _ in spec["points"]])
            assert dates == sorted(dates), lab
            assert all(v > 0 for v in vals), lab
            for _, _, basis, note in spec["points"]:
                assert basis in vp._CN_REV_BASIS_LABEL, lab
                assert note, lab
            assert spec["scope"] and spec["comparable"], lab

    def test_series_sit_below_both_us_labs(self):
        """A value in RMB or millions rather than US$ billions would sit
        orders of magnitude off; every Chinese point is under OpenAI and
        Anthropic on its date."""
        for spec in vp._CN_REVENUE.values():
            for d, v, _, _ in spec["points"]:
                dt = datetime.strptime(d, "%Y-%m-%d")
                for ref in (vp._OPENAI_REVENUE, vp._ANTHROPIC_REVENUE):
                    assert v < vp._rev_value_at(*vp._parse_revenue(ref), dt)

    def test_private_labs_are_press_only(self):
        """Moonshot and DeepSeek have disclosed nothing; a point marked
        company-stated for either needs a source that says so."""
        for lab in ("Moonshot (Kimi)", "DeepSeek"):
            assert {b for _, _, b, _ in vp._CN_REVENUE[lab]["points"]} == {"press"}

    def test_summary_row_per_lab(self, monkeypatch):
        """One row per lab: its latest point, and the doubling time between
        its first and latest point (a dash when it did not grow)."""
        monkeypatch.setattr(vp, "_CN_REVENUE", {
            "Grower": {"scope": "API", "comparable": "No", "points": [
                ("2026-01-01", 1.0, "press", "a"),
                ("2026-04-11", 4.0, "company", "b")]},
            "Flat": {"scope": "MaaS", "comparable": "No", "points": [
                ("2026-01-01", 2.0, "press", "c"),
                ("2026-03-01", 2.0, "press", "d")]}})
        grower, flat = vp._cn_rev_rows()
        assert grower["Lab"] == "Grower" and flat["Lab"] == "Flat"
        assert grower["Doubling time"] == "50 days"     # 100 days, two doublings
        assert grower["Latest ARR"] == vp._fmt_revenue(4.0)
        assert grower["As of"] == "Apr 2026"
        assert grower["Basis"] == vp._CN_REV_BASIS_LABEL["company"]
        assert flat["Doubling time"] == "\u2014"


class TestRevenueUsChina:
    """US labs' summed ARR against the Chinese labs' sum."""

    def _cmp(self):
        us = [("OpenAI", *vp._parse_revenue(vp._OPENAI_REVENUE)),
              ("Anthropic", *vp._parse_revenue(vp._ANTHROPIC_REVENUE))]
        return us, vp._rev_us_china(us, vp._cn_rev_series())

    def test_two_series_sum_matches_the_combined_line(self):
        us, cmp = self._cmp()
        assert cmp["us"] == vp._rev_combined_series(us[0][1], us[0][2],
                                                    us[1][1], us[1][2])

    def test_china_total_starts_when_every_lab_reports(self):
        _, cmp = self._cmp()
        firsts = [d[0] for _, d, _ in vp._cn_rev_series()]
        assert cmp["start"] == max(firsts)
        assert cmp["cn_now"] == pytest.approx(sum(
            vp._rev_value_at(d, v, cmp["as_of"]) for _, d, v in vp._cn_rev_series()))

    def test_the_us_stood_at_chinas_total_on_the_lag_date(self):
        _, cmp = self._cmp()
        us_d, us_v = cmp["us"]
        assert cmp["us_reached"] < cmp["start"]
        assert vp._rev_value_at(us_d, us_v, cmp["us_reached"]) == \
            pytest.approx(cmp["cn_now"], rel=0.01)     # dates read to the day

    def test_nowcast_continues_the_last_interval_and_matches_inside(self):
        d = [datetime(2025, 1, 1), datetime(2025, 1, 11)]
        assert vp._rev_value_nowcast(d, [1.0, 2.0], datetime(2025, 1, 21)) == \
            pytest.approx(4.0)
        assert vp._rev_value_nowcast(d, [1.0, 2.0], datetime(2025, 1, 6)) == \
            vp._rev_value_at(d, [1.0, 2.0], datetime(2025, 1, 6))

    def test_ratio_path_matches_the_cards_and_top_n_drops_the_tail(self):
        us, cmp = self._cmp()
        cn = vp._cn_rev_series()
        grid, ratio = vp._rev_ratio_path(us, cn, cmp["start"], cmp["as_of"])
        assert grid[0] == cmp["start"] and grid[-1] == cmp["as_of"]
        assert ratio[0] == pytest.approx(cmp["ratio_start"])
        assert ratio[-1] == pytest.approx(cmp["ratio_now"])
        _, top = vp._rev_ratio_path(us, cn, cmp["start"], cmp["as_of"], top_n=2)
        assert all(t > r for t, r in zip(top, ratio))

    def test_first_reach_edges(self):
        d = [datetime(2025, 1, 1), datetime(2025, 1, 3)]
        assert vp._rev_first_reach(d, [1.0, 4.0], 2.0) == datetime(2025, 1, 2)
        assert vp._rev_first_reach(d, [1.0, 4.0], 0.5) == d[0]
        assert vp._rev_first_reach(d, [1.0, 4.0], 8.0) is None


class TestMilestoneCardLayout:
    """Thirteen cards have to stay legible: at most four per row, and the
    rows balanced rather than filled greedily."""

    def test_no_row_is_wider_than_the_cap(self):
        for n in range(1, 25):
            assert max(vp._pc_card_rows(n)) <= vp._PC_CARDS_PER_ROW, n

    def test_rows_are_balanced_and_account_for_every_card(self):
        for n in range(1, 25):
            sizes = vp._pc_card_rows(n)
            assert sum(sizes) == n, n
            # Balanced: no row carries two more cards than another.
            assert max(sizes) - min(sizes) <= 1, (n, sizes)

    def test_the_live_card_count_lays_out_four_three_three(self):
        """The shape the RSI tab actually renders today. If a milestone is
        added or dropped, retarget this deliberately."""
        assert len(vp._PC_RSI_WEIGHTS) + len(vp._PC_RSI_CARD_ONLY) == 10
        assert vp._pc_card_rows(10) == [4, 3, 3]


class TestTakeoffYearlyProgress:
    """Years of 2025-level ECI progress per calendar year: ECI gained over the
    points the frontier gained in 2025, +inf once a draw has run away; last
    year measured, this year measured to today then projected."""

    def test_measured_partly_measured_and_projected_years(self):
        today = datetime(2026, 10, 13)
        years = np.arange(0, 6, 1 / 12)
        ppy = 15.0
        frontier = [{"date": datetime(2024, 12, 1), "eci_score": 140.0},
                    {"date": datetime(2025, 12, 1), "eci_score": 152.0},
                    {"date": datetime(2026, 9, 1), "eci_score": 165.0}]
        paths = np.vstack([165 + ppy * years, 165 + 2 * ppy * years, 165 + ppy * years])
        result = {"workflow_progress": {"years": years},
                  "stopped_at": np.array([np.inf, np.inf, 2.5])}
        eci = {"paths": paths, "ppy": ppy, "frontier": frontier, "now": 165.0}
        rows = vp._tk_yearly_progress(result, eci, today, 2035)
        assert [(y, k) for y, _, k in rows][:3] == [
            (2025, "measured"), (2026, "partly"), (2027, "projected")]
        assert rows[-1][0] == 2031                       # the grid ends Sep 2032
        unit = 12.0                                      # 140 -> 152 over 2025
        np.testing.assert_allclose(rows[0][1], 1.0)      # 2025 is the unit
        # 2026: 152 -> 165 measured, then the rest of the year projected.
        rest = (datetime(2027, 1, 1) - today).days / 365.25
        assert rows[1][1][0] == pytest.approx((13 + ppy * rest) / unit, rel=1e-3)
        for year, made, kind in rows[2:]:
            # A calendar year is 365 or 366 days.
            assert made[0] == pytest.approx(ppy / unit, rel=0.003)
            assert made[1] == pytest.approx(2 * ppy / unit, rel=0.003)
        stopped = {y: v[2] for y, v, _ in rows}
        assert np.isfinite(stopped[2028]) and np.isinf(stopped[2029])


class TestTakeoffEciRates:
    """ECI reading: compute at the Compute/capabilities tab's frontier-grade
    exchange rate, algorithms at whatever fits the frontier's climb."""

    def test_the_reading_reproduces_the_frontiers_climb(self):
        import takeoff_model as tk
        hist = vp._tk_hist_compute_rate()
        _, _, ppy, _ = vp._tk_eci_scale(tk.DEFAULTS, hist)
        software = np.log(2) * 12 / tk.DEFAULTS["software_months"]
        r = vp._tk_eci_rates(tk.DEFAULTS, hist)
        assert r["compute"] * hist + r["software"] * software == pytest.approx(ppy)
        assert r["algo_ppy"] == pytest.approx(r["software"] * software)

    def test_compute_rate_is_the_frontier_grade_exchange_rate(self):
        import takeoff_model as tk
        rows = vp.load_eci_compute(mtime=vp._eci_mtime())
        full = vp.load_eci_frontier(mtime=vp._eci_mtime(), full_window=True)
        fit = (vp._cc_frontier_grade_algo(rows, full, margin=3.0)
               or vp._cc_frontier_grade_algo(rows, full))
        r = vp._tk_eci_rates(tk.DEFAULTS, vp._tk_hist_compute_rate())
        assert r["compute"] * np.log(10) == pytest.approx(fit["a_partial"])
        # Algorithms earn more ECI per 10x than compute: the split's whole point.
        assert r["software"] > r["compute"]
        # The leftover algorithm rate sits near the tab's own fixed-compute fits.
        lo, hi = sorted((r["fg_b_time"], r["pooled_b_time"]))
        assert lo - 2 <= r["algo_ppy"] <= hi + 2


class TestTakeoffHumanSpeedup:
    """Hours per automation level: the prior raised to one common power so
    the index's latest month gives the set human speed-up, ends fixed."""

    @pytest.mark.parametrize("accel", [1.3, 1.7, 2.5])
    def test_fitted_hours_reproduce_the_speedup(self, accel):
        row = vp.load_rsi_automation()[-1]
        hours = vp._tk_fit_al_hours(row, accel=accel)
        assert hours[0] == pytest.approx(1.0) and hours[-1] == 0.0
        assert vp._tk_ladder_human_share(row, hours) == pytest.approx(1 / accel, rel=1e-6)
        # The prior's order survives: each level takes less human time than the one below.
        assert all(a > b for a, b in zip(hours, hours[1:]))

    def test_a_faster_speedup_starts_people_lower(self):
        slow, fast = vp._tk_human_calibration(1.4), vp._tk_human_calibration(2.5)
        assert fast["tk_human_direction"] < slow["tk_human_direction"]

    def test_staff_survey_note_reads_the_ladder_at_the_surveys_month(self):
        # August is the ladder's latest month; the curve starts from fitted rungs,
        # so it lands within a couple of points of that month's reading.
        row = vp.load_rsi_automation()[-1]
        note = vp._tk_staff_survey_note(vp._TK_AL_HUMAN_HOURS)
        reading = 100 * (1 - vp._tk_ladder_human_share(row))
        m = re.search(r"51% vs (\d+)% in August 2026", note)
        assert m and abs(int(m.group(1)) - reading) <= 2
        assert "80% vs" in note and "95% vs" in note


class TestTakeoffDirectionCurve:
    """Research direction's human-hours curve averages the dates two sources
    give, the automation index's ladder and the next-step study."""

    def test_the_blend_crosses_any_bar_between_its_sources(self):
        t = np.linspace(0, 10, 241)
        a = np.vstack([np.exp(-t / 1.0), np.where(t < 2, 1.0, 0.01)])
        b = np.vstack([np.exp(-t / 3.0), np.exp(-t / 0.5)])
        blend = vp._tk_blend_in_time(t, a, b, 0.5)
        cross = lambda row, lvl: float(np.interp(-lvl, -row / row[0], t))
        for i in range(2):
            for lvl in (0.5, 0.1, 0.05):
                ca, cb, cm = cross(a[i], lvl), cross(b[i], lvl), cross(blend[i], lvl)
                assert min(ca, cb) - 0.05 <= cm <= max(ca, cb) + 0.05
                assert cm == pytest.approx((ca + cb) / 2, abs=0.06)

    def test_nextstep_curve_reaches_zero_at_the_studys_ceiling(self):
        t, curves = vp._tk_nextstep_curves(200, 42)
        assert curves.shape == (200, len(t))
        assert np.allclose(curves[:, 0], 1.0)
        assert (np.diff(curves, axis=1) <= 1e-12).all()
        assert (curves[:, -1] < 1e-3).mean() > 0.95      # the slowest draws may not get there


class TestTakeoffRsiCalibration:
    """Takeoff defaults read off RSI data rather than set by hand."""

    def test_stage_starts_match_the_ladders_latest_month(self):
        rows = vp.load_rsi_automation()
        starts = vp._tk_ladder_stage_starts(rows)
        keys = [f"tk_human_{k}" for k in vp.takeoff.WORKFLOW_KEYS]
        mean = np.array([starts[k] for k in keys]) @ vp.takeoff.WORKFLOW_WEIGHTS
        assert mean == pytest.approx(100 * vp._tk_ladder_human_share(rows[-1]), abs=0.1)

    def test_ladder_human_share_falls_as_automation_climbs(self):
        shares = [vp._tk_ladder_human_share(r) for r in vp.load_rsi_automation()]
        assert shares == sorted(shares, reverse=True)
        assert 0 < shares[-1] < shares[0] <= 1

    def test_taste_default_is_the_next_step_fit(self):
        args = (vp.rsi_dir_frontier_all, vp.load_rsi_experiments())
        cap = vp._TK_DEFAULTS["tk_cal_capability"]
        taste = vp._tk_nextstep_taste(*args, cap, 0.6)
        assert taste > 0
        # Faster capability growth means each unit of it buys less judgment.
        assert vp._tk_nextstep_taste(*args, 2 * cap, 0.6) < taste

    def test_halving_times_match_the_ladders_decline(self):
        rows = vp.load_rsi_automation()
        floor = vp.takeoff.DEFAULTS["human_floor"]
        halves = vp._tk_ladder_stage_halvings(rows, floor)
        keys = [f"tk_half_{k}" for k in vp.takeoff.WORKFLOW_KEYS]
        mean = np.array([halves[k] for k in keys]) @ vp.takeoff.WORKFLOW_WEIGHTS
        months = np.array([(r['date'] - rows[0]['date']).days / 30.44 for r in rows])
        share = np.array([100 * vp._tk_ladder_human_share(r) for r in rows])
        _icpt, slope = vp.fit_line(months, np.log(share - floor))
        assert mean == pytest.approx(-np.log(2) / slope, abs=0.1)

    def test_coding_default_is_the_survey_fit(self):
        args = (vp.load_rsi_survey(), vp.load_rsi_experiments())
        cap = vp._TK_DEFAULTS["tk_cal_capability"]
        coding = vp._tk_survey_coding(*args, cap, 0.6)
        assert coding > 0
        assert vp._tk_survey_coding(*args, 2 * cap, 0.6) < coding

    def test_judgment_reads_the_square_root_of_the_win_odds(self, monkeypatch):
        """Judgment grows as the next-step win odds raised to
        `_TK_ODDS_TO_JUDGMENT` (0.5), so the elasticity is the one a quantity
        growing at the square root of the odds' yearly growth would have —
        and setting the exponent to 1 reads the odds as productivity."""
        assert vp._TK_ODDS_TO_JUDGMENT == 0.5
        ev = vp.load_rsi_experiments()
        args = (vp.rsi_dir_frontier_all, ev, 3.0, 0.6)
        odds_growth = np.exp(vp._rsi_dir_fit(vp.rsi_dir_frontier_all)[2] * 365.25)
        assert odds_growth > 1
        fallback = vp.takeoff.DEFAULTS["taste_slope"]
        half = vp._tk_nextstep_taste(*args)
        assert half == vp._tk_elasticity(np.sqrt(odds_growth), ev, 3.0, 0.6,
                                         fallback)
        monkeypatch.setattr(vp, "_TK_ODDS_TO_JUDGMENT", 1.0)
        full = vp._tk_nextstep_taste(*args)
        assert full == vp._tk_elasticity(odds_growth, ev, 3.0, 0.6, fallback)
        assert full > half

    def test_experiment_fit_credits_measured_automation_first(self):
        """The automation index's falling human share explains part of the
        experiment growth, leaving less to judgment."""
        ev = vp.load_rsi_experiments()
        g = ev[-1]['mult'] / ev[0]['mult']
        aut = vp._tk_automation_growth(vp.load_rsi_automation(), ev, 30.0)
        assert aut > 1
        kw = dict(transfer=0.6, coding_slope=0.85, parallel=0.5, substitution=-0.5)
        with_aut = vp.takeoff.calibrate_feedback(g, 1.95, 3.38, automation_growth=aut, **kw)
        without = vp.takeoff.calibrate_feedback(g, 1.95, 3.38, **kw)
        assert with_aut["quality_slope"] < without["quality_slope"]


def test_coding_dates_pair_with_capability_without_changing_their_distribution():
    """Reassigning the RSI tab's dates keeps them exactly, and puts late dates
    on slow scenarios as strongly as the correlation says."""
    rng = np.random.default_rng(1)
    onset = np.r_[rng.uniform(0, 3, 1999), np.inf]
    capability = rng.standard_normal(2000)
    rank = lambda x: np.argsort(np.argsort(x))
    for corr, lo, hi in ((0.0, -0.1, 0.1), (0.7, -0.75, -0.55), (1.0, -1.0, -0.999)):
        paired = vp._tk_pair_onset(onset, capability, corr)
        np.testing.assert_array_equal(np.sort(paired), np.sort(onset))
        assert lo <= np.corrcoef(rank(paired), rank(capability))[0, 1] <= hi, corr


def test_takeoff_durations_read_in_the_coarsest_fitting_unit():
    end = datetime(2040, 12, 31)
    span = lambda y: vp._tk_span(y, end)
    assert span(3 / 365.25) == "3 d"
    assert span(-5 / 52) == "-5 wk"
    assert span(0.5) == "6 mo"
    assert span(1.83) == "1.8 yr"
    assert span(np.inf) == "not by 2040"
