"""
Integration tests for visualize_projection.py using Streamlit's AppTest.

These run the actual Streamlit app in a headless runtime, catching issues
that unit tests with a fake Streamlit module cannot (e.g., type mismatches
in number_input, missing session state keys, widget rendering errors).

Run: pytest test_integration.py -v
"""

import numpy as np
import pytest
from datetime import datetime
import json
import re
from visualize_projection import (_pc_add_months, _pc_pause_default_mo,
                                   _SLUG_FOR_TAB, N_SAMPLES)
from streamlit.testing.v1 import AppTest

SCRIPT = "visualize_projection.py"
TIMEOUT = 30


def _qp(at, key):
    """A query param's values as a list: AppTest returned a list before
    Streamlit 1.66 and returns a bare string from it."""
    v = at.query_params[key]
    return v if isinstance(v, list) else [v]


def _fresh_app():
    """Create a fresh AppTest instance."""
    return AppTest.from_file(SCRIPT, default_timeout=TIMEOUT)


def _assert_no_error(at, context):
    """Assert the app ran without exceptions."""
    excs = list(at.exception)
    assert not excs, f"{context}: {excs[0]}"


def _has_widget(at, widget_type, key):
    """Check if a widget with the given key exists (rendered on the page)."""
    try:
        getattr(at, widget_type)(key=key)
        return True
    except (KeyError, IndexError):
        return False


def _rsi_cdf_index(at):
    """Position of the RSI projection's CDF, the evidence half's last chart."""
    import json
    specs = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
    return next(i for i, sp in enumerate(specs)
                if ((sp["layout"].get("yaxis") or {}).get("title") or {}).get("text")
                == "Chance crossed by then")


def _tk_charts(at):
    """The takeoff model's own charts, after the RSI evidence above them."""
    return at.get("plotly_chart")[_rsi_cdf_index(at) + 1:]


def _switch_tab(at, tab_name):
    """Switch to a tab and run."""
    [r for r in at.radio if r.label == "Tab"][0].set_value(tab_name).run()
    _assert_no_error(at, f"switch to {tab_name}")


# ===========================================================================
# Non-default projection bases render without error
# ===========================================================================

class TestNonDefaultProjectionBases:
    """Switching to a non-default projection basis or entity renders OK.
    Default bases are covered by TestDefaultValues; Piecewise → Linear by
    TestModeSwitchingBehavior."""

    @pytest.mark.parametrize("tab,kind,key,value", [
        ("Epoch ECI", "radio", "eci_proj_basis", "Superexponential"),
        # [US best, China best]: project China against the US trend.
        ("Epoch ECI", "selectbox", "eci_entity_b", "China best"),
        # A non-default primary entity with no comparison renders alone.
        ("Epoch ECI", "selectbox", "eci_entity_a", "DeepSeek"),
        ("Remote Labor Index", "radio", "rli_proj_basis",
         "Superexponential (logit)"),
        ("Employment", "radio", "emp_proj_basis", "Piecewise linear (logit)"),
        ("Employment", "radio", "emp_proj_basis", "Superexponential (logit)"),
    ])
    def test_renders(self, tab, kind, key, value):
        at = _fresh_app()
        at.run()
        _switch_tab(at, tab)
        getattr(at, kind)(key=key).set_value(value).run()
        _assert_no_error(at, f"{tab} / {key}={value}")


# ===========================================================================
# Mode switching: Linear vs Piecewise produce different widgets and output
# ===========================================================================

class TestModeSwitchingBehavior:
    """Verify that switching projection basis actually changes the UI —
    not just that it doesn't crash."""

    @pytest.mark.parametrize("tab,basis_key,seg_key,piecewise,linear", [
        (None, "metr_proj_basis", "piecewise_n_seg",
         "Piecewise linear", "Linear"),
        ("Epoch ECI", "eci_proj_basis", "eci_piecewise_n_seg",
         "Piecewise linear", "Linear"),
        ("Remote Labor Index", "rli_proj_basis", "rli_piecewise_n_seg",
         "Piecewise linear (logit)", "Linear (logit)"),
    ])
    def test_piecewise_to_linear_clears_segments(self, tab, basis_key, seg_key,
                                                 piecewise, linear):
        """Piecewise draws a 2-segment radio; switching back to Linear removes
        both the radio and its session state."""
        at = _fresh_app()
        at.run()
        if tab:
            _switch_tab(at, tab)
        at.radio(key=basis_key).set_value(piecewise).run()
        _assert_no_error(at, f"{basis_key} / {piecewise}")
        assert at.radio(key=seg_key).value == 2
        at.radio(key=basis_key).set_value(linear).run()
        _assert_no_error(at, f"{basis_key} Piecewise→Linear")
        assert not _has_widget(at, "radio", seg_key), \
            "Segments radio should disappear after switching to Linear"
        assert seg_key not in at.session_state

    def test_metr_linear_to_piecewise_gets_segments(self):
        """Switching Linear → Piecewise should create the segments radio."""
        at = _fresh_app()
        at.run()
        # Switch to Linear first
        at.radio(key="metr_proj_basis").set_value("Linear").run()
        assert not _has_widget(at, "radio", "piecewise_n_seg")
        # Switch back to Piecewise
        at.radio(key="metr_proj_basis").set_value("Piecewise linear").run()
        _assert_no_error(at, "METR Linear→Piecewise")
        assert _has_widget(at, "radio", "piecewise_n_seg"), \
            "Segments radio should appear after switching to Piecewise"
        assert at.radio(key="piecewise_n_seg").value == 2

    def test_metr_superexp_has_no_segments_or_dt_keys(self):
        """Superexponential should not have linear/piecewise widgets."""
        at = _fresh_app()
        at.run()
        at.radio(key="metr_proj_basis").set_value("Superexponential").run()
        _assert_no_error(at, "METR / Superexponential")
        assert not _has_widget(at, "radio", "piecewise_n_seg")
        assert not _has_widget(at, "number_input", "custom_dt_lo"), \
            "Superexp should not have linear DT CI widget"
        assert _has_widget(at, "number_input", "superexp_dt_ci_lo"), \
            "Superexp should have its own DT CI widget"


# ===========================================================================
# (a) Default widget values are data-driven, not hardcoded
# ===========================================================================

class TestDefaultValues:
    """Verify that widget defaults are computed from data (not hardcoded)
    and that toggle/radio defaults are correct."""

    def test_metr_piecewise_defaults(self):
        """METR piecewise-linear CI defaults + toggles + segment count."""
        at = _fresh_app()
        at.run()
        _assert_no_error(at, "METR default")
        # DT CI values should be positive with ~4x spread (lo=dt/2, hi=dt*2)
        dt_lo = at.number_input(key="custom_dt_lo").value
        dt_hi = at.number_input(key="custom_dt_hi").value
        assert dt_lo > 0 and dt_hi > dt_lo
        assert 3.0 <= dt_hi / dt_lo <= 5.0, \
            f"Unexpected CI spread: lo={dt_lo}, hi={dt_hi}"
        # Segment default is 2 for piecewise
        assert at.radio(key="piecewise_n_seg").value == 2
        # Toggle defaults
        assert at.toggle(key="milestones").value is True
        assert at.toggle(key="labels").value is True
        assert at.toggle(key="p80").value is False
        assert at.toggle(key="log_scale").value is True

    def test_metr_superexp_defaults(self):
        """METR superexp CI defaults should be data-driven."""
        at = _fresh_app()
        at.run()
        proj = at.radio(key="metr_proj_basis")
        proj.set_value("Superexponential").run()
        _assert_no_error(at, "METR / Superexponential")
        dt_lo = at.number_input(key="superexp_dt_ci_lo").value
        dt_hi = at.number_input(key="superexp_dt_ci_hi").value
        assert dt_lo > 0 and dt_hi > dt_lo
        assert 3.0 <= dt_hi / dt_lo <= 5.0

    def test_eci_linear_defaults(self):
        """ECI linear PPY CI defaults + toggles."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        # ECI default basis is "Linear", so no segments radio
        ppy_lo = at.number_input(key="eci_custom_ppy_lo").value
        ppy_hi = at.number_input(key="eci_custom_ppy_hi").value
        assert ppy_lo > 0 and ppy_hi > ppy_lo
        assert 3.0 <= ppy_hi / ppy_lo <= 5.0
        assert at.toggle(key="eci_milestones").value is True
        assert at.toggle(key="eci_labels").value is True
        assert not _has_widget(at, "radio", "eci_piecewise_n_seg"), \
            "ECI Linear default should not have Segments radio"

    def test_rli_linear_defaults(self):
        """RLI linear DT CI defaults + toggles."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        dt_lo = at.number_input(key="rli_custom_dt_lo").value
        dt_hi = at.number_input(key="rli_custom_dt_hi").value
        assert dt_lo > 0 and dt_hi > dt_lo
        assert 3.0 <= dt_hi / dt_lo <= 5.0
        assert at.toggle(key="rli_milestones").value is True
        assert at.toggle(key="rli_labels").value is True
        assert not _has_widget(at, "radio", "rli_piecewise_n_seg"), \
            "RLI Linear default should not have Segments radio"


# ===========================================================================
# (b) Widget changes propagate to dependent values / don't crash
# ===========================================================================

class TestWidgetPropagation:
    """Changing upstream controls should update downstream defaults
    and render without errors."""

    def test_metr_segment_change_updates_ci(self):
        """Switching METR from 2-segment to 1-segment should change CI
        defaults (full OLS vs last-segment OLS)."""
        at = _fresh_app()
        at.run()
        dt_lo_2seg = at.number_input(key="custom_dt_lo").value
        # Switch to 1 segment (uses full OLS, not last-segment)
        at.radio(key="piecewise_n_seg").set_value(1).run()
        _assert_no_error(at, "1-segment")
        dt_lo_1seg = at.number_input(key="custom_dt_lo").value
        assert dt_lo_1seg != dt_lo_2seg, \
            f"CI didn't change when switching segments: {dt_lo_2seg} → {dt_lo_1seg}"

    def test_eci_segment_change_updates_ci(self):
        """Changing the ECI segment count re-derives the +Pts/Yr CI from the new
        last segment."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        ci_1seg = (at.number_input(key="eci_custom_ppy_lo").value,
                   at.number_input(key="eci_custom_ppy_hi").value)
        at.radio(key="eci_proj_basis").set_value("Piecewise linear").run()
        at.radio(key="eci_piecewise_n_seg").set_value(2).run()
        _assert_no_error(at, "ECI 2-segment")
        # 3, not 2: the CI is the last segment's slope halved and doubled, rounded
        # to the widget's 1dp, and the frontier is now near-log-linear over the whole
        # window -- splitting it at the default midpoint leaves the slope unmoved
        # (a 2026-09 Epoch pull put both at 7.9-31.6). The third segment is short
        # enough to still separate. Retarget the segment count if a later pull
        # collapses this one too; don't drop the inequality.
        at.radio(key="eci_piecewise_n_seg").set_value(3).run()
        _assert_no_error(at, "ECI 3-segment")
        ci_3seg = (at.number_input(key="eci_custom_ppy_lo").value,
                   at.number_input(key="eci_custom_ppy_hi").value)
        assert ci_3seg != ci_1seg, \
            f"ECI CI didn't change: {ci_1seg} → {ci_3seg}"

# ===========================================================================
# (c) Reset restores all defaults
# ===========================================================================

class TestReset:
    """Reset button should restore all widget values to their
    data-driven defaults."""

    def test_metr_linear_reset(self):
        """METR linear: reset restores CI values and toggles."""
        at = _fresh_app()
        at.run()
        # Record defaults
        dt_lo_default = at.number_input(key="custom_dt_lo").value
        dt_hi_default = at.number_input(key="custom_dt_hi").value
        # Modify to non-defaults
        at.number_input(key="custom_dt_lo").set_value(10).run()
        at.number_input(key="custom_dt_hi").set_value(999).run()
        at.toggle(key="milestones").set_value(False).run()
        at.toggle(key="log_scale").set_value(False).run()
        at.toggle(key="p80").set_value(True).run()
        _assert_no_error(at, "after modifications")
        # Verify they actually changed
        assert at.number_input(key="custom_dt_lo").value == 10
        assert at.toggle(key="milestones").value is False
        assert at.toggle(key="p80").value is True
        # Click reset
        at.button(key="reset_linear").click().run()
        _assert_no_error(at, "after reset")
        # All should be back to defaults
        assert at.number_input(key="custom_dt_lo").value == dt_lo_default, \
            f"DT lo not reset: {at.number_input(key='custom_dt_lo').value} != {dt_lo_default}"
        assert at.number_input(key="custom_dt_hi").value == dt_hi_default, \
            f"DT hi not reset: {at.number_input(key='custom_dt_hi').value} != {dt_hi_default}"
        assert at.toggle(key="milestones").value is True, "milestones not reset"
        assert at.toggle(key="log_scale").value is True, "log_scale not reset"
        assert at.toggle(key="p80").value is False, "p80 not reset"

    def test_metr_superexp_reset(self):
        """METR superexp reset reverts to default projection basis
        (Piecewise linear) with correct piecewise CI defaults."""
        at = _fresh_app()
        at.run()
        # Record default piecewise CI values
        pw_dt_lo_default = at.number_input(key="custom_dt_lo").value
        pw_dt_hi_default = at.number_input(key="custom_dt_hi").value
        # Switch to superexp and modify something
        proj = at.radio(key="metr_proj_basis")
        proj.set_value("Superexponential").run()
        at.number_input(key="superexp_dt_ci_lo").set_value(10).run()
        # Reset (clears ALL metr keys including proj basis → reverts to Piecewise)
        at.button(key="reset_superexp").click().run()
        _assert_no_error(at, "after superexp reset")
        # After reset, should be back on Piecewise linear with correct defaults
        assert at.radio(key="metr_proj_basis").value == "Piecewise linear", \
            f"Projection basis not reset: {at.radio(key='metr_proj_basis').value}"
        assert at.radio(key="piecewise_n_seg").value == 2, \
            f"Segments not reset to 2: {at.radio(key='piecewise_n_seg').value}"
        assert at.number_input(key="custom_dt_lo").value == pw_dt_lo_default
        assert at.number_input(key="custom_dt_hi").value == pw_dt_hi_default

    def test_eci_linear_reset(self):
        """ECI linear: reset restores CI and toggles."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        ppy_lo_default = at.number_input(key="eci_custom_ppy_lo").value
        ppy_hi_default = at.number_input(key="eci_custom_ppy_hi").value
        # Modify
        at.number_input(key="eci_custom_ppy_lo").set_value(1.0).run()
        at.number_input(key="eci_custom_ppy_hi").set_value(99.0).run()
        at.toggle(key="eci_milestones").set_value(False).run()
        # Reset
        at.button(key="reset_eci_linear").click().run()
        _assert_no_error(at, "after ECI reset")
        assert at.number_input(key="eci_custom_ppy_lo").value == ppy_lo_default, \
            f"ECI PPY lo not reset: {at.number_input(key='eci_custom_ppy_lo').value} != {ppy_lo_default}"
        assert at.number_input(key="eci_custom_ppy_hi").value == ppy_hi_default
        assert at.toggle(key="eci_milestones").value is True

    def test_rli_linear_reset(self):
        """RLI linear: reset restores CI values."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        dt_lo_default = at.number_input(key="rli_custom_dt_lo").value
        dt_hi_default = at.number_input(key="rli_custom_dt_hi").value
        # Modify
        at.number_input(key="rli_custom_dt_lo").set_value(5.0).run()
        at.number_input(key="rli_custom_dt_hi").set_value(999.0).run()
        # Reset
        at.button(key="reset_rli_linear").click().run()
        _assert_no_error(at, "after RLI reset")
        assert at.number_input(key="rli_custom_dt_lo").value == dt_lo_default, \
            f"RLI DT lo not reset: {at.number_input(key='rli_custom_dt_lo').value} != {dt_lo_default}"
        assert at.number_input(key="rli_custom_dt_hi").value == dt_hi_default


# ===========================================================================
# Employment tab tests
# ===========================================================================

def _emp_app():
    """Create a fresh app switched to the Employment tab."""
    at = _fresh_app()
    at.run()
    _switch_tab(at, "Employment")
    return at


class TestEmploymentDefaults:
    """Employment default values are reasonable and data-driven."""

    def test_emp_ci_defaults_are_data_driven(self):
        at = _emp_app()
        dt_lo = at.number_input(key="emp_custom_dt_lo").value
        dt_hi = at.number_input(key="emp_custom_dt_hi").value
        assert dt_lo > 0 and dt_hi > dt_lo, \
            f"DT CI should be positive with lo < hi: lo={dt_lo}, hi={dt_hi}"
        assert 3.0 <= dt_hi / dt_lo <= 5.0, \
            f"Unexpected CI spread: lo={dt_lo}, hi={dt_hi}"


class TestEmploymentSliderChanges:
    """The economic model renders at the ends of its sliders, in both
    display modes."""

    @pytest.mark.parametrize("mode,key,value", [
        (None, "emp_rli_coverage", 0.0),
        (None, "emp_rli_coverage", 10.0),
        (None, "emp_rli_coverage", 100.0),
        (None, "emp_jevons_recovery", 0.0),
        (None, "emp_adoption_lag", 0.0),
        (None, "emp_adoption_lag", 1460.0),
        (None, "emp_base_unemployment", 12.0),
        ("Jobs Lost Above Baseline", "emp_rli_coverage", 10.0),
    ])
    def test_renders_at_edge(self, mode, key, value):
        at = _emp_app()
        if mode:
            at.radio(key="emp_display_mode").set_value(mode).run()
        at.slider(key=key).set_value(value).run()
        _assert_no_error(at, f"Employment / {mode} / {key}={value}")


class TestEmploymentDisplayModes:
    """Jobs Lost mode renders and toggles correctly."""

    def test_emp_jobs_lost_mode_renders(self):
        at = _emp_app()
        at.radio(key="emp_display_mode").set_value("Jobs Lost Above Baseline").run()
        _assert_no_error(at, "Jobs Lost mode")
        # Labor force input should appear in Jobs Lost mode
        assert _has_widget(at, "number_input", "emp_labor_force"), \
            "Labor force input should appear in Jobs Lost mode"

    def test_emp_unemployment_mode_no_labor_force(self):
        at = _emp_app()
        assert not _has_widget(at, "number_input", "emp_labor_force"), \
            "Labor force input should not appear in Unemployment Rate mode"


class TestEmploymentReset:
    """Reset button restores employment defaults."""

    def test_emp_reset_restores_sliders(self):
        at = _emp_app()
        # Modify sliders
        at.slider(key="emp_rli_coverage").set_value(20.0).run()
        at.slider(key="emp_jevons_recovery").set_value(80.0).run()
        assert at.slider(key="emp_rli_coverage").value == 20.0
        # Reset
        at.button(key="reset_emp_all").click().run()
        _assert_no_error(at, "after emp reset")
        assert at.slider(key="emp_rli_coverage").value == 70.0, \
            "RLI coverage not reset"
        assert at.slider(key="emp_jevons_recovery").value == 30.0, \
            "Jevons not reset"

    def test_emp_reset_restores_ci(self):
        at = _emp_app()
        dt_lo_default = at.number_input(key="emp_custom_dt_lo").value
        # Modify
        at.number_input(key="emp_custom_dt_lo").set_value(10.0).run()
        assert at.number_input(key="emp_custom_dt_lo").value == 10.0
        # Reset
        at.button(key="reset_emp_linear").click().run()
        _assert_no_error(at, "after emp CI reset")
        assert at.number_input(key="emp_custom_dt_lo").value == dt_lo_default, \
            "emp DT lo not reset"


# ===========================================================================
# Data Centers tab
# ===========================================================================

class TestDataCenters:
    """The Data Centers tab renders and responds to its controls."""

    def _dc_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Data Centers")
        return at

    def test_renders(self):
        at = self._dc_app()
        _assert_no_error(at, "Data Centers default")
        # Headers for both requested views are present.
        text = " ".join(str(m.value) for m in at.markdown) + \
            " ".join(str(h.value) for h in at.subheader)
        assert "Current largest single data center by company" in text
        assert "Largest single data center by company over time" in text

    def test_every_networking_option_renders(self):
        """Each pooling level draws, and the caption follows the selection —
        it describes the curated groups only where the level uses them."""
        at = self._dc_app()
        labels = list(at.selectbox(key="dc_pool_n").options)
        assert labels[0] == "Nearby + announced fabric", "default moved"
        assert "Nearby + plausible fabric" in labels
        for label in labels:
            at = self._dc_app()
            at.selectbox(key="dc_pool_n").set_value(label).run()
            _assert_no_error(at, f"Data Centers / {label}")
            text = " ".join(str(c.value) for c in at.caption)
            assert ("By proximity" in text) == label.startswith("Nearby"), label
            assert ("By announced fabric" in text) == ("fabric" in label), label
            assert ("plausible fabric" in text) == label.startswith("Nearby + plausible"), \
                label
            if label.startswith("Nearby + plausible"):
                # The speculative regions are named, so the reader can see what
                # is being assumed rather than just a bigger number.
                assert "Mid-South" in text and "Texas & Oklahoma" in text

    @pytest.mark.parametrize("metric,timing,methodology,units", [
        ("2mo train log OP", None, "Methodology: *2mo train log OP*", False),
        ("6mo train log OP", "Model release",
         "Methodology: *6mo train log OP*", False),
        # Time-to-train captions are in time units, never a bare run count.
        ("Capacity (time to GPT-5)", None, "Methodology: time to train one", True),
        ("Capacity (time to Mythos)", None, "Methodology: time to train one", True),
    ])
    def test_metric_methodology_caption(self, metric, timing, methodology, units):
        """Each derived metric renders and its methodology caption names it."""
        at = self._dc_app()
        at.selectbox(key="dc_metric").set_value(metric).run()
        if timing:
            at.selectbox(key="dc_timing").set_value(timing).run()
        _assert_no_error(at, f"Data Centers / {metric} / {timing}")
        text = " ".join(str(m.value) for m in at.markdown) + \
            " ".join(str(c.value) for c in at.caption)
        assert methodology in text
        if units:
            assert any(u in text for u in ("hour", "day", "week", "month"))

    def test_projection_range_change(self):
        at = self._dc_app()
        at.radio(key="dc_end_year").set_value(2029).run()
        _assert_no_error(at, "Data Centers / project through 2029")
        at.radio(key="dc_start_year").set_value(2023).run()
        _assert_no_error(at, "Data Centers / start 2023")

    def test_reset(self):
        at = self._dc_app()
        at.selectbox(key="dc_metric").set_value("Capital cost ($B)").run()
        at.radio(key="dc_end_year").set_value(2029).run()
        at.button(key="dc_reset").click().run()
        _assert_no_error(at, "after dc reset")
        assert at.selectbox(key="dc_metric").value == "Compute (H100-equiv)"
        assert at.radio(key="dc_end_year").value == 2027


# ===========================================================================
# Revenue tab
# ===========================================================================

def _rev_trace_names(at):
    import json
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    return {t.get("name") or "" for t in spec["data"]}


class TestRevenueDefaults:
    """Revenue "Fit to last N points" sliders default to the maximum."""

    def _rev_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Revenue")
        return at

    def test_fit_to_last_n_defaults_to_max(self):
        import visualize_projection as vp
        at = self._rev_app()
        _, oai_vals = vp._parse_revenue(vp._OPENAI_REVENUE)
        _, ant_vals = vp._parse_revenue(vp._ANTHROPIC_REVENUE)
        oai = at.slider(key="oai_n_recent")
        ant = at.slider(key="ant_n_recent")
        # Default "project as of" is the latest date, so the fit window spans
        # every available data point.
        assert oai.value == len(oai_vals), \
            f"OpenAI fit-N should default to max ({len(oai_vals)}), got {oai.value}"
        assert ant.value == len(ant_vals), \
            f"Anthropic fit-N should default to max ({len(ant_vals)}), got {ant.value}"

    def test_combined_line_is_off_until_toggled(self):
        """Off by default; on, it adds a fitted sum series with its own
        controls and leaves the two company series in place."""
        import visualize_projection as vp
        at = self._rev_app()
        assert at.toggle(key="rev_combined").value is False
        names = _rev_trace_names(at)
        assert not any(vp._REV_COMBINED_NAME in n for n in names)

        at.toggle(key="rev_combined").set_value(True).run()
        _assert_no_error(at, "Revenue / combined line")
        names = _rev_trace_names(at)
        assert any((n or "").startswith(vp._REV_COMBINED_NAME + " trend") for n in names)
        assert "OpenAI" in names and "Anthropic" in names
        _, cv = vp._rev_combined_series(*vp._parse_revenue(vp._OPENAI_REVENUE),
                                        *vp._parse_revenue(vp._ANTHROPIC_REVENUE))
        assert at.slider(key="comb_n_recent").value == len(cv)


    def test_chinese_labs_chart_renders_every_lab(self):
        import json
        import visualize_projection as vp
        at = self._rev_app()
        charts = at.get("plotly_chart")
        assert len(charts) == 4
        names = {t.get("name") for t in json.loads(charts[1].proto.spec)["data"]}
        assert set(vp._CN_REVENUE) <= names
        assert len(at.dataframe[-1].value) == len(vp._CN_REVENUE)

    def test_us_vs_china_closes_the_tab(self):
        import json
        at = self._rev_app()
        assert at.subheader[-1].value == "US vs China"
        charts = at.get("plotly_chart")
        names = [t.get("name") for t in json.loads(charts[-2].proto.spec)["data"]]
        assert "OpenAI + Anthropic" in names and "Chinese labs" in names
        ratio = [t.get("name") for t in json.loads(charts[-1].proto.spec)["data"]]
        assert len(ratio) == 2 and all(n.startswith("Top 2 vs top 2") for n in ratio)
        assert [m.label for m in at.metric][-4:] == [
            "US labs' ARR", "Chinese labs' ARR", "US ÷ China", "China's lag"]


# ===========================================================================
# Compute vs Capabilities tab
# ===========================================================================

class TestDataCentersComputeEci:
    """Compute ⟷ ECI and ECI Forecasts, at the bottom of the Data Centers tab."""

    def test_renders_the_exchange_rate_and_forecast(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Data Centers")
        text = (" ".join(str(m.value) for m in at.markdown) +
                " ".join(str(h.value) for h in at.subheader)).lower()
        for s in ("exchange rate", "two engines", "eci forecasts",
                  "time to reach the same eci"):
            assert s in text, s
        labels = " ".join(str(m.label) for m in at.metric)
        for s in ("End 2029", "Physical compute", "Algorithmic efficiency",
                  "Share of growth of compute"):
            assert s in labels, s
        md = " ".join(str(m.value) for m in at.markdown)
        assert "2mo-capacity" in md
        # A longer chart range extends the forecast past 2029.
        at.radio(key="dc_end_year").set_value(2031).run()
        _assert_no_error(at, "Data Centers / through 2031")
        assert "End 2031" in " ".join(str(m.label) for m in at.metric)

    def test_not_on_the_china_tab(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "China")
        heads = [str(h.value) for h in at.subheader]
        assert "Compute ⟷ ECI" not in heads and "ECI Forecasts" not in heads
        assert "China" in [str(h.value) for h in at.header]

    def test_old_slug_lands_on_the_china_tab(self):
        at = _fresh_app()
        at.query_params["tab"] = "computecap"
        at.run()
        assert [r for r in at.radio if r.label == "Tab"][0].value == "China"


class TestComputeVsCapabilities:
    """The China tab (was Compute/capabilities/diffusion)."""

    def _cc_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "China")
        return at

    def test_renders_every_section(self):
        """US vs. China with the four-row distillation table and the
        three-channel decomposition with its consistency check render."""
        at = self._cc_app()
        text = (" ".join(str(m.value) for m in at.markdown) +
                " ".join(str(h.value) for h in at.subheader) +
                " ".join(str(w.value) for w in at.warning)).lower()
        for s in ("us vs. china", "distillation scenario", "no external distillation",
                  "indigenous only", "where frontier growth comes from",
                  "diffusion", "distillation"):
            assert s in text, s
        caps = " ".join(str(c.value) for c in at.caption)
        assert "consistency check" in caps

    def test_distillation_anchor_dropdown(self):
        """The scenario-lines anchor toggles between today and the Jan-2025
        backtest, and the caption follows."""
        at = self._cc_app()
        caps = " ".join(str(c.value) for c in at.caption)
        assert "From today's frontiers" in caps
        at.selectbox(key="cc_bd_anchor").set_value("Jan 2025 (backtest)").run()
        _assert_no_error(at, "Compute vs Capabilities / backdated scenario")
        caps = " ".join(str(c.value) for c in at.caption)
        assert "Backdated to" in caps

    def test_dc_derivation_is_stated(self):
        """The tab says where its compute inputs come from: the
        US-vs-China cross-check caption names the Data Centers tab, and the
        China band is quoted
        against the catalogued paces rather than asserted in a vacuum."""
        at = self._cc_app()
        caps = " ".join(str(c.value) for c in at.caption).lower()
        assert "data centers tab" in caps
        assert "cross-check" in caps
        assert "china-accessible" in caps

    def test_run_length_toggle(self):
        """Switching to the 6-month window (the Pacing tab's options,
        verbatim) renders clean and relabels the capacity captions with the
        chosen window."""
        at = self._cc_app()
        at.selectbox(key="cc_run").set_value("6-month run").run()
        _assert_no_error(at, "Compute vs Capabilities / 6-month run")
        caps = " ".join(str(c.value) for c in at.caption)
        assert "6-month run" in caps

    def test_project_through_horizon(self):
        """The Projection range expander's 'Project through' year moves the
        end-year gap metric, at both ends of the offered range."""
        at = self._cc_app()
        at.radio(key="cc_end_year").set_value(2031).run()
        _assert_no_error(at, "Compute vs Capabilities / through 2031")
        labels = " ".join(str(m.label) for m in at.metric)
        assert "ECI gap end-2031" in labels
        at.radio(key="cc_end_year").set_value(2027).run()
        _assert_no_error(at, "Compute vs Capabilities / through 2027")
        labels = " ".join(str(m.label) for m in at.metric)
        assert "ECI gap end-2027" in labels and "ECI gap end-2029" not in labels

    def test_reset(self):
        at = self._cc_app()
        at.radio(key="cc_end_year").set_value(2031).run()
        _assert_no_error(at, "Compute vs Capabilities / through 2031")
        at.button(key="cc_reset").click().run()
        _assert_no_error(at, "after cc reset")
        assert at.radio(key="cc_end_year").value == 2029


# ===========================================================================
# UK Cyber (AISI narrow cyber tasks)
# ===========================================================================

class TestCcWorldSharesTab:
    """The global compute distribution, second-to-last section of the CC tab."""

    def _app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "China")
        return at

    _HEAD = "Global compute distribution"

    def test_renders_last_with_both_charts(self):
        at = self._app()
        _assert_no_error(at, "CC / world shares")
        assert [str(h.value) for h in at.subheader][-2:] == [
            self._HEAD, TestCcDuvCeilingTab._HEAD]
        text = " ".join(str(m.value) for m in at.markdown)
        assert "of the world's AI compute is in the US" in text
        assert "Where it is heading" in text
        caps = " ".join(str(c.value) for c in at.caption)
        assert "epoch.ai/publications/chip-smuggling" in caps
        assert "Rates:" in caps and "80% CI" in caps

    def test_the_headline_states_both_readings(self):
        """The point of the section is the gap between the published
        estimates and what the tracked sites alone say."""
        at = self._app()
        [line] = [str(m.value) for m in at.markdown
                  if "of the world's AI compute is in the US" in str(m.value)]
        pct = [int(x) for x in re.findall(r"(\d+)%", line)]
        assert len(pct) == 4
        us_guess, cn_guess, us_tracked, cn_tracked = pct
        assert us_tracked > us_guess and cn_guess > cn_tracked


class TestCcDuvCeilingTab:
    """The CTS lithography ceiling, last section of the CC tab."""

    _HEAD = "Lithography ceiling on China's chip output"

    def test_renders_last_and_says_it_is_a_ceiling(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "China")
        _assert_no_error(at, "CC / DUV ceiling")
        assert [str(h.value) for h in at.subheader][-1] == self._HEAD
        text = " ".join(str(m.value) for m in at.markdown)
        assert "not a forecast" in text
        caps = " ".join(str(c.value) for c in at.caption)
        assert "techstatecraft.org/duv" in caps


class TestRsiTab:
    """CoBench score vs release date, plus the fitted trend."""

    def _rsi_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Takeoff")
        return at

    def test_the_sections_chart_and_the_cards_date(self):
        """Every bar is dated once, on its milestone card. The per-section
        ETA pairs and projected-value rows those cards duplicated are gone;
        that each card still reproduces its own section's fit is pinned
        unit-side, by each section's test_eta_reproduces_the_section_defaults."""
        at = self._rsi_app()
        labels = [str(m.label) for m in at.metric]
        assert [l for l in labels
                if "median" in l.lower()] == ["Blended median"]
        assert not [l for l in labels if "EOY" in l or "Projected" in l]
        assert not [m for m in at.markdown if "When does" in str(m.value)]

    def test_every_chart_marks_today(self):
        """Every RSI chart on a calendar x-axis carries the dashed "Today"
        divider the other tabs' projection charts have — without it the fan's
        start reads as the last data point."""
        import json
        at = self._rsi_app()
        # The blend's CDF is excluded: it starts at today, so a divider
        # would sit on its own left edge. Calendar charts are the ones whose
        # explicit x range is a pair of date strings.
        charts = [json.loads(el.proto.spec)["layout"]
                  for el in at.get("plotly_chart")[:_rsi_cdf_index(at) + 1]]
        def _calendar(L):
            x = L.get("xaxis") or {}
            rng = x.get("range") or []
            return (x.get("title")
                    and len(rng) == 2 and all(isinstance(v, str) for v in rng))
        series = [L for L in charts if _calendar(L)]
        assert series and len(series) == len(charts) - 1
        for L in series:
            notes = [a for a in L.get("annotations", [])
                     if a.get("text") == "Today"]
            assert len(notes) == 1, L["yaxis"]["title"]
            assert any(sh.get("line", {}).get("dash") == "dash"
                       and sh.get("x0") == sh.get("x1")
                       for sh in L.get("shapes", []))

    def test_experiment_velocity_chart_and_blend(self):
        import json
        at = self._rsi_app()
        assert "Experiment velocity at OpenAI" in [str(h.value) for h in at.subheader]
        label = "OpenAI experiment velocity reaches 10x"
        assert label in [str(m.label) for m in at.metric]
        table = next(x.value for x in at.table if "Milestone" in x.value.columns)
        row = table[table["Milestone"] == label]
        assert len(row) == 1
        figures = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
        fig = next(f for f in figures if any(t.get('name') == 'OpenAI observations'
                                             for t in f['data']))
        assert any(s.get('y0') == s.get('y1') == 10 for s in fig['layout']['shapes'])

    def test_rd_automation_section_and_blend_row(self):
        """The index's own section, card and weighted row — and the fitted
        markers carry the figure's published 90% intervals as error bars."""
        import json
        at = self._rsi_app()
        assert "R&D automation index" in [str(h.value) for h in at.subheader]
        label = "Claude leads 90% of Anthropic R&D"
        assert label in [str(m.label) for m in at.metric]
        t = next(x.value for x in at.table if "Milestone" in x.value.columns)
        row = t[t["Milestone"] == label]
        assert len(row) == 1
        figures = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
        fig = next(f for f in figures
                   if any(t.get("error_y") for t in f["data"]))
        obs = next(t for t in fig["data"] if t.get("error_y"))
        assert len(obs["x"]) == len(obs["error_y"]["array"])
        assert any(s.get("y0") == s.get("y1") == 90 for s in fig["layout"]["shapes"])
        # The rungs below AL4 ride the same chart as measured context.
        import visualize_projection as vp
        names = {t.get("name") for t in fig["data"]}
        assert vp._RSI_AUTO_LABELS["al2"] in names
        assert vp._RSI_AUTO_LABELS["al3"] in names

    def test_al5_rides_the_one_automation_chart(self):
        """AL5 has no section of its own: its fan, its flat-zero measurements
        and AL4's all share the single R&D automation chart, under one bar."""
        import json
        import visualize_projection as vp
        at = self._rsi_app()
        assert "Full autonomy (AL5)" not in [str(h.value) for h in at.subheader]
        label = f"AL5: {vp._RSI_AL5_TARGET:.0f}% of R&D fully autonomous"
        assert label in [str(m.label) for m in at.metric]
        t = next(x.value for x in at.table if "Milestone" in x.value.columns)
        # A card on the tab, but outside the RSI projection.
        assert not (t["Milestone"] == label).any()
        figures = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
        fig = next(f for f in figures if any(
            "AL5 measured" in (tr.get("name") or "") for tr in f["data"]))
        # ... the same figure that carries the AL4 error bars and both fans.
        assert any(tr.get("error_y") for tr in fig["data"])
        names = [tr.get("name") or "" for tr in fig["data"]]
        assert any("AL5 median" in n for n in names)
        assert any("AL4+ median" in n for n in names)
        zeros = next(tr for tr in fig["data"]
                     if "AL5 measured" in (tr.get("name") or ""))
        assert set(zeros["y"]) == {0.0}
        # One bar, shared: two crossings of the same level, not two thresholds.
        bars = [s for s in fig["layout"]["shapes"]
                if s.get("y0") == s.get("y1") == vp._RSI_AL5_TARGET]
        assert len(bars) == 1

    def test_the_rsi_tab_has_one_automation_chart(self):
        """The ladder diagnostic and the standalone AL5 chart were folded in;
        if a third reappears the tab is drifting back apart."""
        import json
        at = self._rsi_app()
        figures = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
        auto = [f for f in figures if any(
            "AL" in (tr.get("name") or "") for tr in f["data"])]
        assert len(auto) == 1

    def test_every_milestone_card_carries_its_caveats_on_hover(self):
        """Caveats ride the card they belong to rather than piling into one
        caption nobody reads: every card has a hover naming its fit and its
        clock, and the caption under them stays short."""
        at = self._rsi_app()
        helps = {str(m.label): (m.proto.help or "") for m in at.metric}
        t = next(x.value for x in at.table if "Milestone" in x.value.columns)
        milestones = list(t["Milestone"])
        assert len(milestones) == 9
        for lab in milestones:
            assert lab in helps, lab
            h = helps[lab]
            # Fitted cards name the tab defaults they reproduce; AL5 is no
            # fit of its own and names the AL4 fit it shifts.
            assert "defaults" in h or "fit" in h, lab
            assert "clock" in h or "releases" in h, lab
        cap = next(str(c.value) for c in at.caption if "hover a card" in str(c.value))
        assert len(cap) < 300
        # p50 and p80 differ: same fit, different reliability bar.
        assert helps["METR p50 horizon reaches 174h"] != \
            helps["METR p80 horizon reaches 174h"]

    def test_footnotes_anchor_to_their_phrase(self):
        """Footnotes hang off the words they qualify, not off markers parked
        at the end of the line: across every tab, no note falls back to a
        trailing `?`."""
        total = 0
        for tab, slug in _SLUG_FOR_TAB.items():
            at = _fresh_app()
            at.query_params["tab"] = slug
            at.run()
            _assert_no_error(at, f"footnotes / {tab}")
            assert at.session_state["_active_tab"] == tab
            for el in list(at.caption) + list(at.markdown):
                v = str(el.value)
                total += v.count('class="vp-fn-a')
                stray = re.findall(r'<span class="vp-fn(?: vp-fn-r)?">([^<]*)<',
                                   v)
                assert not stray, (tab, stray)
        assert total > 50            # the sweep is app-wide, not one tab

    def test_notyet_conditioning_toggle(self):
        """Default on: the blend table carries the reality-check column;
        toggling off removes it and still renders."""
        at = self._rsi_app()
        assert at.checkbox(key="rsi_notyet").value is True
        t = next(x.value for x in at.table if "Milestone" in x.value.columns)
        assert all("→" in w for w in t["Weight"])
        at.number_input(key="rsi_notyet_ramp").set_value(0.0).run()
        _assert_no_error(at, "RSI / ramp off")
        at.checkbox(key="rsi_notyet").set_value(False).run()
        _assert_no_error(at, "RSI / conditioning off")
        t = next(x.value for x in at.table if "Milestone" in x.value.columns)
        assert not any("→" in w for w in t["Weight"])

    def test_subjective_penalty_slider(self):
        at = self._rsi_app()
        assert at.slider(key="rsi_atc_penalty").value == 0
        at.slider(key="rsi_atc_penalty").set_value(50).run()
        _assert_no_error(at, "RSI / subjective penalty")
        assert any("becomes 5%" in c.value for c in at.caption)
        at.checkbox(key="rsi_notyet").set_value(False).run()
        _assert_no_error(at, "RSI / penalty without conditioning")
        assert at.slider(key="rsi_atc_penalty").value == 50
        at.slider(key="rsi_atc_penalty").set_value(0).run()
        _assert_no_error(at, "RSI / penalty off")
        assert not any("becomes 5%" in c.value for c in at.caption)

    def test_horizon_selector(self):
        """The tab's one *Project through* year sets the evidence charts' reach
        too, capped where their daily fans would outgrow memory."""
        at = self._rsi_app()
        assert at.number_input(key="tk_end_year").value == 2030

        def _al5_end(at):
            [trace] = [t for el in at.get("plotly_chart")
                       for t in json.loads(el.proto.spec)["data"]
                       if t.get("name") == "AL5 median (assumed lag)"]
            return max(trace["x"])[:4]

        assert _al5_end(at) == "2030"
        at.number_input(key="tk_end_year").set_value(2035).run()
        _assert_no_error(at, "Takeoff / project through 2035")
        import visualize_projection as vp
        assert _al5_end(at) == str(vp._RSI_MAX_END_YEAR)


class TestTakeoffTab:
    def _app(self):
        at = _fresh_app()
        at.query_params["tab"] = "takeoff"
        at.run()
        _assert_no_error(at, "Takeoff / deep link")
        return at

    def test_default_render_and_stable_rerun(self):
        at = self._app()
        assert at.session_state["_active_tab"] == "Takeoff"
        assert any(h.value == "Takeoff" for h in at.header)
        # RSI evidence, pace, pace factors, compute, people, ECI, yearly progress
        # and the distributions.
        assert len(_tk_charts(at)) == 10
        # The evidence first, then the takeoff model.
        # The takeoff model's loop and cards lead, then the evidence, then the
        # model's charts.
        assert [h.value for h in at.header] == ["Takeoff", "The takeoff model", "The evidence",
                                                "Inside the takeoff model"]
        subs = [s.value for s in at.subheader]
        assert subs.index("CoBench") < subs.index("Automated coder projection")
        assert subs[subs.index("Automated coder projection") + 1:][:6] == [
            "What the evidence feeds in", "1. Research speeds up", "2. People step out of the loop",
            "3. The research workforce", "4. What it does to capability (ECI)",
            "5. When it happens, and how fast"]
        import json
        dist = json.loads(_tk_charts(at)[9].proto.spec)
        # Density and cumulative share for RSI, ultraintelligence and the gap.
        assert len(dist["data"]) == 6
        import base64
        ys = lambda y: (np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
                        if isinstance(y, dict) else np.asarray(y, dtype=float))
        cdfs = [ys(d["y"]) for d in dist["data"] if d.get("yaxis") in ("y3", "y4")]
        assert len(cdfs) == 3
        assert all(((0 <= y) & (y <= 100)).all() and (np.diff(y) >= -1e-9).all() for y in cdfs)
        assert at.number_input(key="tk_end_year").value == 2030
        assert not _has_widget(at, "number_input", "tk_onset_low")
        evidence = json.loads(_tk_charts(at)[0].proto.spec)
        titles = " ".join(a["text"] for a in evidence["layout"]["annotations"])
        for name in ("Next-step wins", "Staff speedup", "Experiments/researcher"):
            assert name in titles
        yearly = json.loads(_tk_charts(at)[8].proto.spec)
        # Last year measured, this year partly, then each year to the sidebar's.
        assert yearly["data"][0]["x"] == list(range(datetime.now().year - 1, 2031))
        assert yearly["layout"]["yaxis"]["title"]["text"] == "Years of 2025-level ECI progress"
        compute = json.loads(_tk_charts(at)[3].proto.spec)
        assert {"Compute in the model", "Middle 80% of scenarios"} <= {
            d.get("name") for d in compute["data"]}
        assert compute["layout"]["yaxis"]["type"] == "log"
        assert {"Coding automated", "RSI", "Ultraintelligence"} <= {
            a.get("text") for a in compute["layout"]["annotations"]}
        eci = json.loads(_tk_charts(at)[7].proto.spec)
        [projected] = [d for d in eci["data"] if d.get("name") == "Projected ECI"]
        # The axis ends on the sidebar's year end; the line runs a month past it.
        assert eci["layout"]["xaxis"]["range"][1].startswith("2030-12-31")
        assert projected["x"][-1] > "2030-12-31"
        import base64

        def _ys(trace):
            y = trace["y"]
            return (np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
                    if isinstance(y, dict) else np.asarray(y, dtype=float))

        traces = {d.get("name"): d for d in eci["data"]}
        for name in ("Projected ECI", "Projected ECI: 10th percentile"):
            assert np.all(np.diff(_ys(traces[name])) >= -1e-9), name   # no step down
        first_year = slice(0, 13)        # monthly grid: today plus 12 months
        assert np.all(_ys(traces["Projected ECI: 10th percentile"])[first_year]
                      >= _ys(traces["Today's trend continued"])[first_year] - 1e-9)
        pace = json.loads(_tk_charts(at)[1].proto.spec)
        assert {"Modeled research pace", "If AI stopped improving today"} <= {
            d.get("name") for d in pace["data"]}
        # Today's pace is set, not inherited from the experiment proxy.
        [modeled] = [d for d in pace["data"] if d.get("name") == "Modeled research pace"]
        y0 = modeled["y"]
        if isinstance(y0, dict):          # plotly's packed binary arrays
            import base64
            y0 = np.frombuffer(base64.b64decode(y0["bdata"]), dtype=y0["dtype"])
        assert float(y0[0]) == pytest.approx(
            at.number_input(key="tk_pace_today").value, rel=0.05)
        # Draws run to the horizon, so there is no held-pace tail to label.
        assert not any("hold their last pace" in a.get("text", "")
                   for a in pace["layout"]["annotations"])
        factors = json.loads(_tk_charts(at)[2].proto.spec)
        assert [d["name"] for d in factors["data"]] == [
            "Human researchers", "AI capabilities", "Compute", "Research judgment", "Harder ideas (divides)",
            "Net research pace"]
        split = json.loads(_tk_charts(at)[4].proto.spec)
        assert [d["name"] for d in split["data"]] == [
            "Training", "Running AI researchers", "Experiments"]
        people = json.loads(_tk_charts(at)[5].proto.spec)
        workforce = json.loads(_tk_charts(at)[6].proto.spec)
        assert {"AI researcher copies", "Human researchers", "AI (FTE)", "Humans (FTE)", "Total (humans + AIs)"} <= {
            d.get("name") for d in workforce["data"]}
        # Three stages (band edges + median each), then the measured index history.
        assert len(people["data"]) == 10
        assert people["data"][-1]["name"] == "Measured: R&D automation index"
        assert [s["y0"] for s in people["layout"]["shapes"]
                if s.get("yref") == "y"] == [5]
        before = [(m.label, m.value) for m in at.metric]
        at.run()
        assert [(m.label, m.value) for m in at.metric] == before
        labels = [m.label for m in at.metric]
        labels = labels[labels.index("Full coding automation"):]
        assert labels[:6] == ["Full coding automation",
                              "RSI: AI directs research",
                              "Ultraintelligence: ECI 250",
                              "Coding automation \u2192 RSI",
                              "RSI \u2192 ultraintelligence",
                              "Progress in RSI's first year"]
        cis = [c.value for c in at.caption if c.value.startswith("80% CI: ")]
        assert len(cis) >= 5
        assert len([c for c in at.caption if c.value.startswith("90% CI: ")]) == 6
        assert any("RSI projections under construction" in w.value for w in at.warning)
        # The loading bars (the app's first-run one and Takeoff's) stay until
        # the page is drawn, then go; later reruns skip the first-run bar.
        assert not at.get("progress")
        assert at.session_state["_vp_booted"]

    def test_takeoff_refreshes_an_old_imported_model(self, monkeypatch):
        import takeoff_model as tk

        def old_simulator(*args, **kwargs):
            raise AssertionError("The stale imported simulator must be refreshed")

        monkeypatch.setattr(tk, "SOURCE_DIGEST", "older-model-without-workflow-progress")
        monkeypatch.setattr(tk, "simulate", old_simulator)
        at = self._app()
        assert tk.simulate is not old_simulator
        assert len(_tk_charts(at)) == 10

    def test_present_day_feedback_and_evidence_calibration(self, monkeypatch):
        import takeoff_model as tk
        captured = []
        original = tk.simulate

        def capture(params, **kwargs):
            result = original(params, **kwargs)
            captured.append(result)
            return result

        monkeypatch.setattr(tk, "simulate", capture)
        at = self._app()
        at.number_input(key="tk_seed").set_value(81219).run()
        r = captured[-1]
        assert r["params"]["taste_slope"] != at.number_input(key="tk_taste_slope").value
        at.number_input(key="tk_cal_weight").set_value(0.0).run()
        assert captured[-1]["params"]["taste_slope"] == at.number_input(key="tk_taste_slope").value
        at.number_input(key="tk_cal_weight").set_value(100.0)
        at.number_input(key="tk_cal_validated").set_value(1.0).run()
        assert captured[-1]["params"]["taste_slope"] == 0
        _assert_no_error(at, "Takeoff / measured progress calibration")
        at.button(key="tk_fit_verification").click().run()
        assert "positive matched-project hours" in at.session_state["_tk_fit_message"]
        at.number_input(key="tk_measure_baseline").set_value(10.0)
        at.number_input(key="tk_measure_before").set_value(5.0)
        at.number_input(key="tk_measure_now").set_value(2.5).run()
        at.button(key="tk_fit_verification").click().run()
        assert at.number_input(key="tk_human_verification").value == 25.0
        assert at.number_input(key="tk_half_verification").value < 12.0
        _assert_no_error(at, "Takeoff / matched intervention hours")
        fresh = _fresh_app()
        fresh.query_params.update(dict(at.query_params))
        fresh.run()
        assert fresh.number_input(key="tk_cal_validated").value == 1.0
        assert fresh.number_input(key="tk_human_verification").value == 25.0

    def test_grounded_thresholds_and_workflow_floor(self):
        """A human-work floor above the RSI bar rules RSI out, and reset restores it."""
        at = self._app()
        assert at.number_input(key="tk_full_human").value == 5.0
        assert any("2. People step out of the loop" in s.value for s in at.subheader)
        rsi_label = "RSI: AI directs research"
        assert next(m.value for m in at.metric if m.label == rsi_label) != "Beyond horizon"
        at.number_input(key="tk_human_floor").set_value(6.0).run()
        _assert_no_error(at, "Takeoff / persistent human bottleneck")
        assert next(m.value for m in at.metric if m.label == rsi_label) == "Beyond horizon"
        assert not any(m.label == "ECI when RSI arrives" for m in at.metric)
        assert _qp(at, "tk_human_floor") == ["6.0"]
        at.button(key="tk_reset").click().run()
        assert at.number_input(key="tk_human_floor").value == 0.0

    def test_human_speedup_resets_the_stage_settings_and_shows_its_table(self):
        """The speed-up input resets each stage's starting share and halving
        time, and section 2's table shows the mix it is read from."""
        at = self._app()
        before = at.number_input(key="tk_human_direction").value
        at.number_input(key="tk_human_accel").set_value(2.5).run()
        _assert_no_error(at, "Takeoff / human speed-up")
        assert at.number_input(key="tk_human_direction").value < before
        table = next(df.value for df in at.dataframe if "Fitted human time" in df.value.columns)
        assert table["Human share of work"].iloc[-1] == "40.0%"

    @pytest.mark.parametrize("bad,fix,message", [
        # A floor above a stage's starting share is inconsistent.
        ((("tk_human_experiments", 5.0), ("tk_human_floor", 10.0)),
         ("tk_human_experiments", 30.0), "starting human work"),
        ((("tk_training_share", 70),), ("tk_experiment_share", 20), "Reserve positive compute"),
    ])
    def test_invalid_input_is_recoverable(self, bad, fix, message):
        """An inconsistent setting shows the model's error, not a crash, and
        correcting a second setting clears it."""
        at = self._app()
        for key, value in bad:
            at.number_input(key=key).set_value(value)
        at.run()
        _assert_no_error(at, f"Takeoff / invalid {bad[-1][0]}")
        assert any(message in e.value for e in at.error)
        at.number_input(key=fix[0]).set_value(fix[1]).run()
        _assert_no_error(at, f"Takeoff / corrected {fix[0]}")
        assert not at.error

    @pytest.mark.parametrize("condition,timing", [
        ("1", "Training run finished"), ("0", "Model release")])
    def test_coding_automation_inherits_the_exact_final_rsi_cdf(self, monkeypatch,
                                                             condition, timing):
        import json
        import takeoff_model as tk
        captured = []
        original = tk.simulate

        def capture(*args, **kwargs):
            captured.append(kwargs["onset_years"].copy())
            return original(*args, **kwargs)

        monkeypatch.setattr(tk, "simulate", capture)
        at = _fresh_app()
        # Takeoff re-pairs every `_TK_DRAW_STRIDE`th of these dates with its
        # scenarios, so their CDF is the RSI tab's to within one full-sample draw.
        at.query_params.update(tab="rsi", rsi_atc_penalty="50",
                               rsi_timing=timing, rsi_notyet=condition, rsi_notyet_ramp="180",
                               pc_rsiw_rli_90="70", tk_seed="9182")
        at.run()
        _assert_no_error(at, "RSI / inherited configuration")
        curve = json.loads(at.get("plotly_chart")[_rsi_cdf_index(at)].proto.spec)["data"][-1]
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        grid = np.array([(datetime.fromisoformat(d) - today).total_seconds()
                         / (365.25 * 86400) for d in curve["x"]])
        _switch_tab(at, "Takeoff")
        assert captured
        np.testing.assert_allclose(100 * tk.cdf(captured[-1], grid), curve["y"],
                                   rtol=0, atol=100 / N_SAMPLES + 1e-9)
        assert at.session_state["rsi_atc_penalty"] == 50
        assert at.session_state["pc_rsiw_rli_90"] == 70
        _switch_tab(at, "Takeoff")
        assert at.slider(key="rsi_atc_penalty").value == 50
        second = json.loads(at.get("plotly_chart")[_rsi_cdf_index(at)].proto.spec)["data"][-1]
        assert second["y"] == curve["y"]

    def test_horizon_resets_and_rsi_settings_survive_reset(self):
        """*Reset to defaults* restores the Takeoff horizon but keeps the RSI
        settings Takeoff inherits."""
        at = _fresh_app()
        at.query_params.update(tab="takeoff", rsi_atc_penalty="50")
        at.run()
        _assert_no_error(at, "Takeoff / inherited RSI penalty")
        assert at.session_state["rsi_atc_penalty"] == 50
        cards = [(m.label, m.value) for m in at.metric]
        at.number_input(key="tk_end_year").set_value(2035).run()
        _assert_no_error(at, "Takeoff / year end")
        assert _qp(at, "tk_end_year") == ["2035"]
        # The year crops the charts; the cards come from scenarios run to 2040.
        assert [(m.label, m.value) for m in at.metric] == cards
        eci = json.loads(_tk_charts(at)[7].proto.spec)
        assert eci["layout"]["xaxis"]["range"][1].startswith("2035-12-31")
        at.button(key="tk_reset").click().run()
        assert at.number_input(key="tk_end_year").value == 2030
        assert "tk_end_year" not in at.query_params
        assert at.session_state["rsi_atc_penalty"] == 50

    def test_reset_and_url_roundtrip(self):
        at = self._app()
        assert not _has_widget(at, "button", "tk_preset_Central")
        at.number_input(key="tk_transfer").set_value(20).run()
        _assert_no_error(at, "Takeoff / low transfer")
        assert _qp(at, "tk_transfer") == ["20"]
        fresh = _fresh_app()
        fresh.query_params.update(dict(at.query_params))
        fresh.run()
        _assert_no_error(fresh, "Takeoff / shared scenario")
        assert fresh.number_input(key="tk_transfer").value == 20
        at.button(key="tk_reset").click().run()
        assert at.number_input(key="tk_transfer").value == 60
        assert "tk_transfer" not in at.query_params

    def test_compute_follows_the_data_center_projection_unless_toggled_off(self, monkeypatch):
        """On by default: each draw's compute growth comes from the US
        largest-run projection. Off: the flat sidebar rate, with its spread."""
        import takeoff_model as tk
        seen = []
        original = tk.simulate

        def capture(*args, **kwargs):
            seen.append(kwargs.get("compute_rates"))
            return original(*args, **kwargs)

        monkeypatch.setattr(tk, "simulate", capture)
        import streamlit as st
        st.cache_data.clear()             # an earlier case may have cached this run
        at = self._app()
        rates = seen[-1]
        assert rates is not None and len(set(np.round(rates, 6))) > 10
        assert 1.5 < np.exp(np.median(rates)) < 5
        at.checkbox(key="tk_dc_compute").uncheck().run()
        _assert_no_error(at, "Takeoff / flat compute")
        assert seen[-1] is None

    def test_gaps_between_milestones_and_a_capped_pace_view(self):
        """Under the three cards, coding to RSI and RSI to ultraintelligence
        each get a median and ordered 80%/90% intervals; spreads widen the
        takeoff's. The pace chart's view stays capped though runaways reach
        ~1e300."""
        import takeoff_model as tk
        at = self._app()
        labels = ["Coding automation \u2192 RSI", "RSI \u2192 ultraintelligence"]
        for label in labels:
            assert any(m.label == label for m in at.metric), label
        pace = json.loads(_tk_charts(at)[1].proto.spec)
        assert pace["layout"]["yaxis"]["range"][1] <= np.log10(1000) + 1e-9
        assert not any(m.label.startswith("FOOM") for m in at.metric)

        def ci80():
            # The fifth headline card, RSI -> ultraintelligence, leads the page.
            text = [c.value for c in at.caption if c.value.startswith("80% CI")][4]
            return text
        wide = ci80()
        for key in tk.POINT_ESTIMATE:
            at.number_input(key="tk_" + key).set_value(0)
        at.run()
        _assert_no_error(at, "Takeoff / point estimates")
        assert ci80() != wide

    def test_zero_transfer_runs_at_the_reference_pace(self, monkeypatch):
        """With no AI feedback (no transfer, and no post-RSI compute boost, a
        separate channel), research runs at the frozen-AI reference pace."""
        import takeoff_model as tk
        captured = []
        original = tk.simulate

        def capture(*args, **kwargs):
            captured.append(original(*args, **kwargs))
            return captured[-1]

        monkeypatch.setattr(tk, "simulate", capture)
        at = self._app()
        at.number_input(key="tk_transfer").set_value(0)
        at.number_input(key="tk_rsi_compute_boost").set_value(1.0).run()
        _assert_no_error(at, "Takeoff / zero feedback")
        progress = captured[-1]["workflow_progress"]
        # The weekly integration's error grows with difficulty, whose lognormal
        # tail reaches several times the setting.
        np.testing.assert_allclose(progress["pace"], progress["reference_pace"], rtol=2e-3)

    def test_parallelization_does_not_move_difficulty(self, monkeypatch):
        """Difficulty is set from r alone; parallelization moves only returns to inputs."""
        import takeoff_model as tk
        seen = []
        original = tk.simulate

        def capture(params, **kwargs):
            seen.append(dict(params))
            return original(params, **kwargs)

        monkeypatch.setattr(tk, "simulate", capture)
        import streamlit as st
        st.cache_data.clear()
        at = self._app()
        base = seen[-1]["difficulty"]
        at.number_input(key="tk_parallelization").set_value(0.8).run()
        _assert_no_error(at, "Takeoff / parallelization 0.8")
        assert seen[-1]["parallelization"] == 0.8
        assert seen[-1]["difficulty"] == base
        at.number_input(key="tk_returns").set_value(2.0).run()
        assert seen[-1]["difficulty"] == pytest.approx(base / 2)

    def test_malformed_url_values_and_tab_switching(self):
        at = _fresh_app()
        at.query_params.update(tab="takeoff", tk_compute_growth="nan", tk_seed="-2")
        at.run()
        _assert_no_error(at, "Takeoff / invalid URL")
        assert at.number_input(key="tk_compute_growth").value == 2.0
        at.number_input(key="tk_returns").set_value(2.0).run()
        _switch_tab(at, "Takeoff")
        _switch_tab(at, "Takeoff")
        assert at.number_input(key="tk_returns").value == 2.0


class TestUkCyberTab:
    """Frontier projection + open-weight lag."""

    def _ukc_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "UK Cyber")
        return at

    @pytest.mark.parametrize("key,kind,value", [
        ("ukc_proj_basis", "radio", "Piecewise linear (logit)"),
        ("ukc_proj_basis", "radio", "Superexponential (logit)"),
        ("_ukc_proj_as_of", "selectbox", "GPT-5"),
        ("ukc_show_open", "toggle", False),
        ("ukc_show_lag", "toggle", False),
    ])
    def test_each_control_renders(self, key, kind, value):
        at = self._ukc_app()
        getattr(at, kind)(key=key).set_value(value).run()
        _assert_no_error(at, f"UK Cyber / {key}={value}")

    def test_target_eta_is_shown(self):
        """The headline output is when open-weight models reach the target."""
        at = self._ukc_app()
        labels = [m.label for m in at.metric]
        assert "China reaches 90%" in labels
        assert "Measured open-weight lag" in labels
        # A bracket, lower edge first (the values move with the digitization).
        lag = next(m.value for m in at.metric if m.label == "Measured open-weight lag")
        lo, hi = re.fullmatch(r"([\d.]+)–([\d.]+) mo", lag).groups()
        assert float(lo) < float(hi)

    def test_confound_caveat_is_surfaced(self):
        """Country and openness are confounded; the UI must say so wherever it
        says "China"."""
        at = self._ukc_app()
        assert any("cannot be separated" in c.value for c in at.caption)

    def test_reset(self):
        at = self._ukc_app()
        at.toggle(key="ukc_show_lag").set_value(False).run()
        at.button(key="reset_ukc").click().run()
        _assert_no_error(at, "after UK Cyber reset")
        assert at.toggle(key="ukc_show_lag").value is True


class TestDataCentersByCountry:
    """The by-country panel: renders under every control, and its headline
    tracks the horizon and the China scope."""

    def _dc_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Data Centers")
        return at

    def _headline(self, at):
        return [str(m.value) for m in at.markdown
                if "training run by end-" in str(m.value)]

    def test_renders_with_headline_and_table(self):
        at = self._dc_app()
        _assert_no_error(at, "Data Centers / by country")
        assert "Buildout by country: US vs China" in \
            " ".join(str(h.value) for h in at.subheader)
        head = self._headline(at)
        assert len(head) == 1 and "China-accessible" in head[0] \
            and "end-2027" in head[0]
        table = at.table[-1].value
        # The panel runs from this year to the sidebar's "Project through" year.
        end = at.radio(key="dc_end_year").value
        assert list(table["Year end"]) == \
            [str(y) for y in range(datetime.now().year, end + 1)]
        assert "China-accessible" in table.columns
        assert "China (domestic only)" in table.columns
        # The mainland-only figure is a hover anchored on that phrase now.
        assert "Mainland China alone" in head[0]
        assert at.radio(key="dc_cty_since").value == 2024

    def test_sits_above_the_buildout_panel_and_shares_the_sidebar_selector(self):
        at = self._dc_app()
        heads = [str(h.value) for h in at.subheader]
        assert heads.index("Buildout by country: US vs China") < \
            heads.index("Per-company: does the buildout predict release timing?")
        assert heads.index("Buildout by country: US vs China") > heads.index(
            "Largest data center by company over time "
            "(including networking multiple data centers)")
        # One networking selector, in the sidebar, drives both sections.
        assert len([sb for sb in at.selectbox if sb.key == "dc_pool_n"]) == 1
        at.sidebar.selectbox(key="dc_pool_n")   # raises if not in the sidebar
        at.sidebar.radio(key="dc_cty_since")

    def test_every_control_renders(self):
        at = self._dc_app()
        for label in at.selectbox(key="dc_pool_n").options:
            at.selectbox(key="dc_pool_n").set_value(label).run()
            _assert_no_error(at, f"by country / {label}")
            assert self._headline(at), label
        for label in at.radio(key="dc_cty_pace").options:
            at.radio(key="dc_cty_pace").set_value(label).run()
            _assert_no_error(at, f"by country / {label}")
        for yr in at.radio(key="dc_cty_since").options:
            at.radio(key="dc_cty_since").set_value(yr).run()
            _assert_no_error(at, f"by country / since {yr}")

    def test_follows_the_projection_range(self):
        at = self._dc_app()
        at.radio(key="dc_end_year").set_value(2030).run()
        _assert_no_error(at, "by country / through 2030")
        head = self._headline(at)
        assert "training run by end-2030" in head[0]
        assert list(at.table[-1].value["Year end"]) == \
            [str(y) for y in range(datetime.now().year, 2031)]
        at.radio(key="dc_start_year").set_value(2023).run()
        _assert_no_error(at, "by country / from 2023")

    def test_cones_toggle(self):
        """Hiding the cones drops the shaded bands and keeps the numbers."""
        def _bands(at):
            return sum(1 for el in at.get("plotly_chart")
                       for d in json.loads(el.proto.spec)["data"] if d.get("fill"))

        def _cone_bands(at):
            # The ECI forecast at the bottom of the tab has fan bands of its
            # own, unaffected by this toggle.
            return sum(1 for el in at.get("plotly_chart")
                       for d in (lambda t: [] if any(
                           x.get("name") == "Median projection" for x in t)
                           else t)(json.loads(el.proto.spec)["data"])
                       if d.get("fill"))

        at = self._dc_app()
        assert at.checkbox(key="dc_cty_cones").value is True
        assert _cone_bands(at) > 0
        before = _bands(at)
        at.checkbox(key="dc_cty_cones").set_value(False).run()
        _assert_no_error(at, "by country / cones off")
        assert _cone_bands(at) == 0 and _bands(at) < before
        assert self._headline(at)

    def test_both_panels_render_with_planned_buildout_off(self):
        at = self._dc_app()
        at.checkbox(key="dc_future").set_value(False).run()
        _assert_no_error(at, "Data Centers / no planned buildout")
        assert self._headline(at)
        assert TestDataCentersRegionShare._HEAD in [str(h.value) for h in at.subheader]

    def test_both_panels_render_under_every_metric(self):
        at = self._dc_app()
        for label in at.selectbox(key="dc_metric").options:
            at.selectbox(key="dc_metric").set_value(label).run()
            _assert_no_error(at, f"Data Centers / {label}")
            assert self._headline(at), label
            assert TestDataCentersRegionShare._HEAD in \
                [str(h.value) for h in at.subheader], label

    def test_reset_restores_panel_defaults(self):
        at = self._dc_app()
        at.radio(key="dc_cty_since").set_value(2026).run()
        at.button(key="dc_reset").click().run()
        _assert_no_error(at, "by country / reset")
        assert at.radio(key="dc_cty_since").value == 2024


# ===========================================================================
# Data Centers: tenant vs operator attribution
# ===========================================================================

class TestDataCentersRegionShare:
    """The region-share chart at the bottom of the Data Centers tab."""

    def _app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Data Centers")
        return at

    _HEAD = "Share of catalogued capacity by region"

    def test_renders_last_on_the_tab(self):
        at = self._app()
        _assert_no_error(at, "Data Centers / region share")
        heads = [str(h.value) for h in at.subheader]
        # The Compute ⟷ ECI section follows it; ECI Forecasts closes the tab.
        assert heads[-3:] == [self._HEAD, "Compute ⟷ ECI", "ECI Forecasts"]
        # The coverage caveat is body text above the chart, not fine print.
        warn = [str(m.value) for m in at.markdown
                if "Tracked data centers only" in str(m.value)]
        assert len(warn) == 1 and "Non-US shares are floors" in warn[0]
        assert "Tracked data centers only" not in \
            " ".join(str(c.value) for c in at.caption)



class TestDataCentersParty:
    """The DC tab's attribution radio: defaults to tenant (shared sites count
    under every listed user), flips to operator, and resets."""

    def _app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Data Centers")
        return at

    @staticmethod
    def _names(at):
        return {d.get("name") for el in at.get("plotly_chart")
                for d in json.loads(el.proto.spec)["data"]}

    def test_operator_view_moves_sites_to_landlords_and_resets(self):
        """Leased clusters move to their landlords, so some owner the tenant
        view never names (Oracle, for Stargate Abilene) appears."""
        at = self._app()
        assert at.radio(key="dc_party").value == "Tenant (who trains there)"
        tenant = self._names(at)
        at.radio(key="dc_party").set_value(
            "Operator (who owns the building)").run()
        _assert_no_error(at, "Data Centers / operator")
        assert "Oracle" in self._names(at) - tenant
        [b for b in at.button if b.key == "dc_reset"][0].click().run()
        _assert_no_error(at, "Data Centers / reset")
        assert at.radio(key="dc_party").value == "Tenant (who trains there)"


# ===========================================================================
# Pacing tab
# ===========================================================================

class TestPacingTab:
    """The Pacing tab: renders with its defaults, follows the threshold, and
    resets."""

    def _app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Pacing")
        return at

    @staticmethod
    def _state(at):
        """The plan panel's state-of-play table, picked by its columns."""
        return next(t.value for t in at.table
                    if "At plan start" in t.value.columns)

    @staticmethod
    def _entities(at):
        """The race table, picked by its columns — the pause panel's own
        breakdown table renders after it, so position is not an address."""
        return next(t.value for t in at.table if "Entity" in t.value.columns)

    @staticmethod
    def _paired(el):
        """Run `el` (the app or a widget) on a fixed seed, so two Monte Carlo
        readings differ by the setting changed, not by a fresh draw."""
        np.random.seed(0)
        return el.run()

    def test_us_pause_panel_states_its_assumptions_in_the_body(self):
        """The US-pause counterfactual renders its crossing metrics, and its
        *Assumes* line (weight security first) is body text, not fine print."""
        at = self._app()
        subs = " ".join(str(s.value) for s in at.subheader)
        assert "If the US paced" in subs
        labels = " ".join(str(m.label) for m in at.metric)
        assert "paced US frontier" in labels
        assert "Time for China to surpass" in labels
        caps = " ".join(str(c.value) for c in at.caption)
        md = " ".join(str(m.value) for m in at.markdown)
        assert "**Assumes**" in md and "**Assumes**" not in caps
        # The weight-security assumption is a hover on its phrase.
        assert "never steal them" in md
        # The clock line follows the sidebar default (run finished).
        assert "Dates = training runs finish" in caps

    def test_pause_scenario_checkboxes(self):
        """Both scenario checkboxes render unchecked; cutting distillation
        pushes China's crossing later, and cutting remote compute (on top)
        renders and never pulls it earlier."""
        at = self._app()

        def _surpass_mo(at):
            m = next(m for m in at.metric
                     if "Time for China to surpass" in str(m.label))
            return float(str(m.value).strip("~ mo"))

        assert at.checkbox(key="pc_withhold").value is True
        assert at.checkbox(key="pc_stop_dist").value is False
        assert at.checkbox(key="pc_stop_remote").value is False
        base = _surpass_mo(self._paired(at))
        # Serving the paused frontier publicly restores the full teacher:
        # China can only cross sooner (or the same, within rounding).
        # Each channel's setting is the body of its hover on the *Assumes*
        # line; the markers capitalise it, so match case-insensitively.
        def _md(at):
            return " ".join(str(m.value) for m in at.markdown).lower()

        self._paired(at.checkbox(key="pc_withhold").uncheck())
        _assert_no_error(at, "Pacing / paused models served")
        assert _surpass_mo(at) <= base + 1
        assert "stays queryable" in _md(at)
        self._paired(at.checkbox(key="pc_withhold").check())
        assert "release freeze" in _md(at)
        self._paired(at.checkbox(key="pc_stop_dist").check())
        _assert_no_error(at, "Pacing / distillation stopped")
        dist_off = _surpass_mo(at)
        assert dist_off > base
        assert "cut today" in _md(at)
        self._paired(at.checkbox(key="pc_stop_remote").check())
        _assert_no_error(at, "Pacing / remote compute cut")
        assert _surpass_mo(at) >= dist_off - 1  # rounding guard
        assert ("largest domestic cluster" in _md(at)
                or "domestic pace only" in _md(at))

    def test_advanced_cutoff_dates_delay_the_controls(self):
        """The advanced sliders date each control. Default is 'Now' (= the
        checkbox alone); pushing a cut-off out gives China more time on
        that channel, so its crossing lands between 'no control' and
        'cut today' — never outside that bracket."""
        at = self._app()

        def _surpass_mo(at):
            m = next(m for m in at.metric
                     if "Time for China to surpass" in str(m.label))
            return float(str(m.value).strip("~ mo"))

        assert at.select_slider(key="pc_dist_when").value == "Now"
        assert at.select_slider(key="pc_remote_when").value == "Now"
        base = _surpass_mo(self._paired(at))

        self._paired(at.checkbox(key="pc_stop_dist").check())
        _assert_no_error(at, "Pacing / distillation cut today")
        now_d = _surpass_mo(at)
        late = at.select_slider(key="pc_dist_when").options[-1]
        self._paired(at.select_slider(key="pc_dist_when").set_value(late))
        _assert_no_error(at, "Pacing / distillation cut late")
        assert base - 1 <= _surpass_mo(at) <= now_d + 1
        md = " ".join(str(m.value) for m in at.markdown).lower()
        assert f"cut {late}".lower() in md
        at.checkbox(key="pc_stop_dist").uncheck().run()

        self._paired(at.checkbox(key="pc_stop_remote").check())
        _assert_no_error(at, "Pacing / remote cut today")
        now_r = _surpass_mo(at)
        self._paired(at.select_slider(key="pc_remote_when").set_value(late))
        _assert_no_error(at, "Pacing / remote cut late")
        assert base - 1 <= _surpass_mo(at) <= now_r + 1
        md = " ".join(str(m.value) for m in at.markdown)
        assert late in md

    def test_chinese_run_length_trades_compute_against_wall_clock(self):
        """A longer Chinese run lifts the path (more compute in one model)
        and shifts it right (the run takes longer), so the crossing is a
        U-curve in run length: a few extra months beat a year, which is
        worse than matching.
        Matching the bar's length is the default and changes nothing."""
        at = self._app()

        def _surpass_mo(at):
            m = next(m for m in at.metric
                     if "Time for China to surpass" in str(m.label))
            return float(str(m.value).strip("~ mo"))

        sl = at.slider(key="pc_cn_run")
        assert sl.value == 2 and sl.max == 12
        base = _surpass_mo(self._paired(at))
        assert not [c for c in at.caption if "Chinese run" in str(c.value)]

        mids = {}
        for L in (3, 4, 5, 6):
            self._paired(at.slider(key="pc_cn_run").set_value(L))
            _assert_no_error(at, f"Pacing / {L}-month Chinese run")
            mids[L] = _surpass_mo(at)
        cap = " ".join(str(c.value) for c in at.caption)
        assert "6-month Chinese run" in cap and "4 months of extra" in cap
        self._paired(at.slider(key="pc_cn_run").set_value(12))
        _assert_no_error(at, "Pacing / 12-month Chinese run")
        long_run = _surpass_mo(at)
        # A year of wall clock does not pay, and some moderate stretch beats it.
        assert long_run > base
        assert min(mids.values()) < long_run, (mids, long_run)

    def test_duvi_levers_compound_with_the_domestic_slider(self):
        """Each DUVi lever is named on the Assumes line even when its cut
        rounds to zero, and the cuts stack multiplicatively with the slider:
        1 - (1 - slider)(1 - duv)."""
        at = self._app()
        assert at.checkbox(key="pc_duv_ban").value is False
        assert at.checkbox(key="pc_duv_service").value is False
        assert at.slider(key="pc_duv_decay").disabled
        assert at.slider(key="pc_duv_growth").disabled

        def _slowed(at, names):
            md = " ".join(str(m.value) for m in at.markdown)
            m = re.search(r"domestic buildout slowed (\d+)% incl\. the DUVi "
                          + names, md)
            assert m, f"the Assumes line names the {names}"
            return int(m.group(1))

        at.checkbox(key="pc_duv_ban").check().run()
        _assert_no_error(at, "Pacing / DUVi import ban")
        imp = _slowed(at, "import ban")
        assert not at.slider(key="pc_duv_growth").disabled
        at.slider(key="pc_duv_growth").set_value(5.0).run()
        assert _slowed(at, "import ban") > imp
        at.slider(key="pc_duv_growth").set_value(2.4).run()

        at.checkbox(key="pc_duv_service").check().run()
        _assert_no_error(at, "Pacing / DUVi import and servicing bans")
        assert not at.slider(key="pc_duv_decay").disabled
        both = _slowed(at, "import and servicing bans")
        # The import ban binds mostly next decade; servicing bites from 2027.
        assert imp < both / 2 and both < 100
        at.slider(key="pc_duv_decay").set_value(40).run()
        assert _slowed(at, "import and servicing bans") > both

        at.slider(key="pc_duv_decay").set_value(15).run()
        at.slider(key="pc_dom_slow").set_value(50).run()
        stacked = _slowed(at, "import and servicing bans")
        assert abs(stacked - round(100 * (1 - 0.5 * (1 - both / 100)))) <= 1

    def test_why_breakdown_tracks_the_sliders(self):
        """The bottom breakdown decomposes the same crossing: shares sum to
        the whole gap, and cutting a channel collapses its row — which is
        the point of having it."""
        at = self._app()

        _KEYS = {"domestic": "domestic", "abroad": "abroad",
                 "innovation": "Indigenous", "diffusion": "Diffusion",
                 "distillation": "Distillation", "run": "Longer training",
                 "total": "Total"}

        def _rows(at):
            t = next(x.value for x in at.table if "Channel" in x.value.columns)
            out = {}
            for _, r in t.iterrows():
                key = next((k for k, sub in _KEYS.items()
                            if sub in r["Channel"]), r["Channel"])
                out[key] = (float(r["ECI closed"]), r["Share"],
                            r["Without it"])
            return out

        base = _rows(self._paired(at))
        assert set(base) == {"domestic", "abroad", "innovation", "diffusion",
                             "distillation", "total"}
        gap = base["total"][0]
        parts = sum(v[0] for k, v in base.items() if k != "total")
        # Loose on purpose: the columns are read at each sample's own crossing,
        # so at the suite's 400 samples the sum carries ~0.2 ECI of Monte Carlo
        # noise. `test_channels_account_for_the_whole_climb` is the exactness
        # guard; this one just checks the row is a decomposition.
        assert parts == pytest.approx(gap, abs=0.5)
        assert all(v[0] > 0 for v in base.values())
        # Every channel is load-bearing: removing it costs months.
        for k, v in base.items():
            if k == "total":
                continue
            assert v[2] == "not by 2031" or float(v[2].rstrip(" mo")) > 0, k

        self._paired(at.checkbox(key="pc_stop_dist").check())
        _assert_no_error(at, "Pacing / why with distillation cut")
        cut = _rows(at)
        assert cut["distillation"][0] < base["distillation"][0] / 2
        assert cut["total"][0] == pytest.approx(gap, abs=1.0)
        at.checkbox(key="pc_stop_dist").uncheck().run()

        # Cutting remote access collapses exactly the row that prices it:
        # the shadow the breakdown subtracts becomes the run itself.
        self._paired(at.checkbox(key="pc_stop_remote").check())
        _assert_no_error(at, "Pacing / why with remote access cut")
        rem = _rows(at)
        assert rem["abroad"][0] == pytest.approx(0.0, abs=0.05)
        assert rem["abroad"][2] == "+0.0 mo"
        assert rem["domestic"][0] == pytest.approx(base["domestic"][0],
                                                   abs=1.0)
        # Dating the cut later leaves part of it standing.
        when = at.select_slider(key="pc_remote_when").options
        self._paired(at.select_slider(key="pc_remote_when").set_value(
            when[len(when) // 2]))
        _assert_no_error(at, "Pacing / why with remote access cut later")
        later = _rows(at)
        # Between the two: more than a cut today leaves, no more than never
        # cutting. The upper bound carries a tolerance because the dated cut
        # re-reads each sample at its own crossing.
        assert later["abroad"][0] > rem["abroad"][0]
        assert later["abroad"][0] < base["abroad"][0] + 1.0
        at.select_slider(key="pc_remote_when").set_value("Now").run()
        at.checkbox(key="pc_stop_remote").uncheck().run()

        self._paired(at.slider(key="pc_cn_run").set_value(6))
        _assert_no_error(at, "Pacing / why with a 6-month run")
        run = _rows(at)
        assert run["run"][0] > 0

    def test_pause_dates_follow_timing_in_lockstep(self):
        """'Date points at' shifts both countries' pause-panel dates to the
        chosen milestone (construction → release ≈ run + 1 mo later), while
        the US–China gap stays put. The default is run-finished, so start
        from construction."""
        from datetime import datetime as _dt

        at = self._app()

        def _cross_date(at):
            m = next(m for m in at.metric
                     if "paced US frontier" in str(m.label))
            return _dt.strptime(str(m.value), "%b %Y")

        def _surpass_mo(at):
            m = next(m for m in at.metric
                     if "Time for China to surpass" in str(m.label))
            return float(str(m.value).strip("~ mo"))

        self._paired(at.selectbox(key="pc_timing").set_value(
            "Data center construction"))
        _assert_no_error(at, "Pacing / pause on construction clock")
        d_con, s_con = _cross_date(at), _surpass_mo(at)
        self._paired(at.selectbox(key="pc_timing").set_value("Model release"))
        _assert_no_error(at, "Pacing / pause on release clock")
        d_rel, s_rel = _cross_date(at), _surpass_mo(at)
        # 2-mo run + 30d prep ≈ 3 months later, ± a month's rounding.
        diff_mo = ((d_rel.year - d_con.year) * 12 + d_rel.month - d_con.month)
        assert 1 <= diff_mo <= 5
        assert abs(s_rel - s_con) <= 1
        caps = " ".join(str(c.value) for c in at.caption)
        assert "Dates = model releases" in caps

    def test_race_bar_is_the_state_of_play_us_run(self):
        """The race below runs to the US row's largest training run, at the
        default pause and at a later one."""
        import visualize_projection as vp
        at = self._app()
        for mo in (None, vp._PC_PAUSE_MO_MAX):
            if mo is not None:
                at.select_slider(key="pc_pause_mo").set_value(mo).run()
                _assert_no_error(at, f"Pacing / pause {mo}")
            us = self._state(at).iloc[0]["Largest training run"].split()[0]
            titles = [json.loads(el.proto.spec)["layout"].get("title", {})
                      .get("text", "") for el in at.get("plotly_chart")]
            race = next(t for t in titles if t.startswith("First \u2265"))
            assert race.startswith(f"First \u2265{us} log OP"), (us, race)

    def test_state_of_play_describes_both_sides_at_the_pause(self):
        """The panel opens with what each side has when the music stops:
        the US ahead of China on compute and on capability, China's lag in
        months, and a METR horizon read off the ECI bridge."""
        at = self._app()
        t = self._state(at)
        assert list(t["At plan start"]) == [
            "United States", "China-accessible", "China (domestic only)"]
        us, cn = (t.iloc[i] for i in (0, 1))
        assert float(us["Largest training run"].split()[0]) > \
            float(cn["Largest training run"].split()[0])
        assert float(us["Frontier ECI"].split()[0].lstrip("~")) > \
            float(cn["Frontier ECI"].split()[0].lstrip("~"))
        assert us["Behind the US"] == "—" and cn["Behind the US"].endswith("mo")
        # p80 is the more demanding bar, so its horizon is the shorter one.
        def _hrs(cell):
            units = {"y": 2000, "mo": 176, "w": 40, "d": 8, "h": 1, "m": 1 / 60}
            return sum(float(n) * units[u] for n, u in
                       re.findall(r"([\d.]+)(mo|y|w|d|h|m)", cell.replace(",", "")))
        for _, row in t.iterrows():
            assert _hrs(row["METR horizon (p80)"]) < _hrs(row["METR horizon (p50)"]), row
        # The bar the crossing metric quotes is the US row of this table.
        lab = next(str(m.label) for m in at.metric
                   if "paced US frontier" in str(m.label))
        assert abs(float(lab.split("ECI")[-1].strip(" ~)"))
                   - float(us["Frontier ECI"].split()[0].lstrip("~"))) <= 1

    def test_run_length_moves_the_compute_not_the_pause_date(self):
        """Run length is no longer what dates the pause — the slider is —
        but it still sets the window the capacity is quoted in: a 6-month
        run is ~3x the ops of a 2-month one."""
        at = self._app()

        def _pause_month(at):
            lab = next(str(m.label) for m in at.metric
                       if "paced US frontier" in str(m.label))
            return lab.split("plan start ")[1].split(",")[0]

        def _us_ops(at):
            return float(self._state(at).iloc[0][
                "Largest training run"].split()[0])

        month, ops_2mo = _pause_month(at), _us_ops(at)
        at.radio(key="pc_run").set_value("6-month run").run()
        _assert_no_error(at, "Pacing / pause with 6-month run")
        assert _pause_month(at) == month
        assert _us_ops(at) - ops_2mo == pytest.approx(np.log10(3), abs=0.1)

    def test_both_charts_end_at_the_projection_range(self):
        """Timeline and pause charts both run to *Project through*, like
        every other tab's projection chart."""
        import json

        def _ends(at):
            out = []
            for el in at.get("plotly_chart"):
                rng = json.loads(el.proto.spec)["layout"]["xaxis"]["range"]
                out.append(int(str(rng[1])[:4]))
            return out

        at = self._app()
        # A range ending before the default pause draws no pause chart; it
        # must still render.
        at.radio(key="pc_end_year").set_value(2027).run()
        _assert_no_error(at, "Pacing / through 2027")
        for yr in (2029, 2031):
            at.radio(key="pc_end_year").set_value(yr).run()
            _assert_no_error(at, f"Pacing / through {yr}")
            ends = _ends(at)
            assert len(ends) == 2, ends
            # The pause chart renders first and stops on the grid exactly;
            # the threshold timeline below it pads a few months past.
            assert ends[0] == yr, ends
            assert ends[1] in (yr, yr + 1), ends

    def test_horizon_too_narrow_still_describes_the_pause(self):
        """A pause past the projection range has no crossing to report, so
        the panel says so — but the state of play at the pause doesn't
        depend on the crossing and still renders."""
        at = self._app()
        at.radio(key="pc_end_year").set_value(2028).run()
        at.select_slider(key="pc_pause_mo").set_value(48).run()
        _assert_no_error(at, "Pacing / pause past the projection range")
        info = " ".join(str(i.value) for i in at.info)
        assert "does not reach the paced US frontier by 2028" in info
        assert list(self._state(at)["At plan start"]) == [
            "United States", "China-accessible", "China (domestic only)"]

    def test_us_pause_bar_follows_the_pause_slider(self):
        """The pause is a date the user names: pausing later freezes a
        better model, so the bar rises and the crossing lands later."""
        at = self._app()

        def _bar(at):
            lab = next(str(m.label) for m in at.metric
                       if "paced US frontier" in str(m.label))
            return float(lab.split("ECI")[-1].strip(" ~)"))

        def _cross(at):
            return datetime.strptime(str(next(
                m.value for m in at.metric
                if "paced US frontier" in str(m.label))), "%b %Y")

        sl = at.select_slider(key="pc_pause_mo")
        assert sl.value == _pc_pause_default_mo()   # months; labelled a date
        assert sl.options[18] == f"{_pc_add_months(datetime.now(), 18):%b %Y}"
        base, base_d = _bar(at), _cross(at)
        at.select_slider(key="pc_pause_mo").set_value(36).run()
        _assert_no_error(at, "Pacing / pause 36 months out")
        assert _bar(at) > base
        assert _cross(at) > base_d
        at.select_slider(key="pc_pause_mo").set_value(18).run()

    def test_renders_with_headline_and_table(self):
        at = self._app()
        _assert_no_error(at, "Pacing")
        assert "Pacing" in [str(h.value) for h in at.header]
        # The US-vs-China headline is a country line; the default roster is
        # companies, so it appears only under the 'Country' attribution.
        assert not [m for m in at.markdown if "China-accessible" in str(m.value)]
        assert at.radio(key="pc_run").value == "2-month run"
        table = self._entities(at)
        ents = list(table["Entity"])
        # The default roster is companies only; countries live under the
        # 'Country' attribution instead.
        assert "Anthropic" in ents and "OpenAI" in ents
        assert "United States" not in ents
        assert "Plan crosses" in table.columns

    def test_country_attribution_races_countries(self):
        at = self._app()
        at.radio(key="pc_party").set_value("Country").run()
        _assert_no_error(at, "Pacing / country")
        head = [str(m.value) for m in at.markdown
                if "China-accessible" in str(m.value)]
        assert len(head) == 1 and "United States" in head[0]
        ents = list(self._entities(at)["Entity"])
        assert "United States" in ents
        assert "China-accessible" in ents
        assert "China (domestic only)" in ents
        assert "Anthropic" not in ents

    def test_race_bar_is_the_us_run_at_the_pause(self):
        """No threshold control: the race runs to the *Largest training
        run* the state-of-play table gives the US, so the chart title and
        that cell quote one number — and a later plan start raises both."""
        at = self._app()
        import json

        def _title(at):
            return next(json.loads(el.proto.spec)["layout"]
                        .get("title", {}).get("text", "")
                        for el in at.get("plotly_chart")
                        if "by entity" in str(json.loads(el.proto.spec)
                                              ["layout"].get("title", {})
                                              .get("text", "")))

        assert not [b for b in at.selectbox if b.key == "pc_threshold"]
        us_run = self._state(at).iloc[0]["Largest training run"].split(
            " (")[0]
        assert us_run in _title(at)
        at.select_slider(key="pc_pause_mo").set_value(42).run()
        _assert_no_error(at, "Pacing / bar at a later pause")
        later = self._state(at).iloc[0]["Largest training run"].split(" (")[0]
        assert float(later.split()[0]) > float(us_run.split()[0])
        assert later in _title(at)

    def test_timing_toggle_shifts_the_dates(self):
        """The default dates a crossing at 'Training run finished';
        going back to 'Data center construction' takes off one run length
        (2mo at the default run), so a plan crossing moves earlier."""
        at = self._app()
        assert at.selectbox(key="pc_timing").value == \
            "Training run finished"
        # The bar is a Monte Carlo median and a crossing snaps to a
        # catalogued step, so a re-draw between the two runs can move a date
        # further than the shift does — the US bar currently lands inside a
        # run of near-equal OpenAI steps. Seed both renders so the bar is
        # identical and the shift is the only thing that moved.
        np.random.seed(0)
        at.run()
        base = dict(zip(self._entities(at)["Entity"],
                        self._entities(at)["Plan crosses"]))
        np.random.seed(0)
        at.selectbox(key="pc_timing").set_value(
            "Data center construction").run()
        _assert_no_error(at, "Pacing / construction")
        shifted = dict(zip(self._entities(at)["Entity"],
                           self._entities(at)["Plan crosses"]))
        import datetime as _dt
        moved = 0
        for ent, val in base.items():
            if val == "—" or ent not in shifted or shifted[ent] == "—":
                continue
            d0 = _dt.datetime.strptime(val, "%b %Y")
            d1 = _dt.datetime.strptime(shifted[ent], "%b %Y")
            assert d1 < d0, ent
            moved += 1
        assert moved > 0
        at.selectbox(key="pc_timing").set_value("Model release").run()
        _assert_no_error(at, "Pacing / model release")

    def test_every_pooling_option_renders(self):
        at = self._app()
        for label in at.selectbox(key="pc_pool").options:
            at.selectbox(key="pc_pool").set_value(label).run()
            _assert_no_error(at, f"Pacing / {label}")

    def test_operator_attribution_changes_the_roster(self):
        at = self._app()
        tenant_ents = set(self._entities(at)["Entity"])
        at.radio(key="pc_party").set_value(
            "Operator (who owns the building)").run()
        _assert_no_error(at, "Pacing / operator")
        op_ents = set(self._entities(at)["Entity"])
        # Leased clusters move to their landlords, so some operator the
        # tenant roster never names appears.
        assert op_ents - tenant_ents, (op_ents, tenant_ents)

    def test_reset_restores_defaults(self):
        at = self._app()
        at.select_slider(key="pc_pause_mo").set_value(42).run()
        at.radio(key="pc_party").set_value(
            "Operator (who owns the building)").run()
        [b for b in at.button if b.key == "pc_reset"][0].click().run()
        _assert_no_error(at, "Pacing / reset")
        assert at.select_slider(key="pc_pause_mo").value == \
            _pc_pause_default_mo()
        assert at.radio(key="pc_party").value == "Tenant (who trains there)"
        assert at.selectbox(key="pc_pool").value == \
            "Nearby + announced fabric"
        assert at.selectbox(key="pc_timing").value == \
            "Training run finished"


class TestTodayForwardMode:
    """The *Today forward mode* toggle on the six projection tabs: the
    projection charts open at today with relative hover dates, and the RSI
    tab's diagnostic charts are left alone."""

    TABS = ["METR Horizon", "Epoch ECI", "Remote Labor Index", "Takeoff",
            "Revenue", "Pacing"]
    _DATE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
                       r"(\d{1,2}, )?\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")

    @staticmethod
    def _specs(at):
        import json
        return [json.loads(el.proto.spec) for el in at.get("plotly_chart")]

    @staticmethod
    def _hover_strings(spec):
        out = []
        for tr in spec["data"]:
            for k in ("hovertext", "hovertemplate"):
                v = tr.get(k)
                if isinstance(v, str):
                    out.append(v)
                elif isinstance(v, list):
                    out.extend(x for x in v if isinstance(x, str))
            if "text" in str(tr.get("hoverinfo", "")) and "text" not in str(tr.get("mode", "")):
                v = tr.get("text")
                out.extend(v if isinstance(v, list) else [v] if isinstance(v, str) else [])
            # A template's dates land in the customdata column it now reads.
            if "customdata" in str(tr.get("hovertemplate", "")):
                out.extend(x for x in tr.get("customdata") or [] if isinstance(x, str))
        return out

    # Tabs where every chart is a projection, so every chart goes forward.
    _ALL_FORWARD = {"Pacing": 2}

    def _starts_today(self, spec):
        rng = spec["layout"].get("xaxis", {}).get("range")
        return bool(rng) and str(rng[0])[:10] == datetime.now().strftime("%Y-%m-%d")

    @pytest.mark.parametrize("tab", TABS)
    def test_projection_charts_open_at_today_with_relative_hovers(self, tab):
        at = _fresh_app()
        at.run()
        _switch_tab(at, tab)
        assert at.checkbox(key="today_fwd").value is False
        before = [self._starts_today(sp) for sp in self._specs(at)]
        assert not all(before)
        at.checkbox(key="today_fwd").set_value(True).run()
        _assert_no_error(at, f"today forward on {tab}")
        specs = self._specs(at)
        fwd = [sp for sp in specs if self._starts_today(sp)]
        assert fwd, tab
        if tab in self._ALL_FORWARD:
            assert len(fwd) == len(specs) == self._ALL_FORWARD[tab], tab
        for sp in fwd:
            hovers = self._hover_strings(sp)
            assert any("from today" in h or h == "today" for h in hovers), tab
            assert not any(self._DATE.search(h) for h in hovers), tab
            assert sp["layout"].get("hovermode") != "x unified"

    def test_metr_y_axis_zooms_to_the_projection(self):
        """The METR axis is sized to its history; from today the fan sits in
        a sliver of it, so the mode refits the y-range."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "METR Horizon")
        before = self._specs(at)[0]["layout"]["yaxis"]["range"]
        at.checkbox(key="today_fwd").set_value(True).run()
        after = self._specs(at)[0]["layout"]["yaxis"]["range"]
        assert after[1] - after[0] < 0.5 * (before[1] - before[0])
        assert after[0] > before[0]

    def test_toggle_sits_at_the_foot_of_the_sidebar(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "METR Horizon")
        sb = at.sidebar.checkbox
        assert sb[-1].key == "today_fwd"
        _switch_tab(at, "UK Cyber")
        assert not [c for c in at.checkbox if c.key == "today_fwd"]

    def test_takeoff_converts_its_projections_not_the_evidence(self):
        """The RSI projection's CDF and the eight takeoff projections go
        forward; the evidence charts above them keep their history."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Takeoff")
        at.checkbox(key="today_fwd").set_value(True).run()
        specs = self._specs(at)
        fwd = [sp for sp in specs if self._starts_today(sp)]
        assert len(fwd) == 9
        assert fwd[0]["layout"]["yaxis"]["title"]["text"] == "Chance crossed by then"
        assert len(specs) > len(fwd)

    def test_toggle_survives_a_tab_switch_and_its_reset(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "METR Horizon")
        at.checkbox(key="today_fwd").set_value(True).run()
        _switch_tab(at, "Revenue")
        assert at.checkbox(key="today_fwd").value is True
        # The projection converts; the Chinese-lab and US-vs-China charts have
        # no projection and keep their history, like the RSI tab's diagnostics.
        proj, *rest = self._specs(at)
        assert self._starts_today(proj) and len(rest) == 3
        assert not any(self._starts_today(sp) for sp in rest)
        # No tab's reset clears it.
        _switch_tab(at, "Pacing")
        at.button(key="pc_reset").click().run()
        assert at.checkbox(key="today_fwd").value is True


class TestSectionDeepLinks:
    """`?to=<anchor>` is how a section link survives a fresh load — a bare
    `#fragment` is dropped by the Community Cloud iframe and again by every
    `st.query_params` write."""

    _SLUGS = {"datacenters": "per-company-does-the-buildout-predict-release-timing",
              "pacing": "if-the-us-paced-when-does-china-catch-up",
              "rsi": "cobench"}

    def _script(self, at):
        [s] = [e.body for e in at.get("html") if e.body.lstrip().startswith("<script>")]
        return s

    def _app(self, **params):
        at = _fresh_app()
        for k, v in params.items():
            at.query_params[k] = v
        at.run()
        return at

    def test_the_script_rides_along_on_every_tab(self):
        for tab, slug in _SLUG_FOR_TAB.items():
            at = self._app(tab=slug)
            _assert_no_error(at, f"anchor script / {tab}")
            assert f'var TAB = "{slug}";' in self._script(at), tab

    def test_to_scrolls_and_is_consumed(self):
        for tab, slug in self._SLUGS.items():
            at = self._app(tab=tab, to=slug)
            _assert_no_error(at, f"?to= / {tab}")
            assert f'var TO = "{slug}";' in self._script(at), tab
            # Left in the URL it would re-jump the page on every later rerun.
            assert "to" not in at.query_params, tab

    def test_a_slug_that_is_not_one_never_reaches_the_script(self):
        at = self._app(tab="revenue", to="</script><img src=x onerror=alert(1)>")
        _assert_no_error(at, "?to= injection")
        assert 'var TO = "";' in self._script(at)
        assert "to" not in at.query_params

    def test_every_section_heading_stays_inside_the_whitelist(self):
        """A heading's own link icon offers `?to=<its id>`, so a heading the
        whitelist would then refuse hands out a dead link. Streamlit slugifies
        client-side; the two properties that survive it are checked here."""
        for tab in _SLUG_FOR_TAB.values():
            at = self._app(tab=tab)
            _assert_no_error(at, f"headings / {tab}")
            heads = list(at.get("header")) + list(at.get("subheader"))
            assert heads, tab
            for el in heads:
                assert len(el.body) <= 120, (tab, el.body)
                assert re.match(r"[A-Za-z0-9]", el.body), (tab, el.body)
