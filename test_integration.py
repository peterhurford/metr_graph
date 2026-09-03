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
import re
from visualize_projection import (_pc_add_months, _pc_pause_default_mo,
                                   _SLUG_FOR_TAB)
from streamlit.testing.v1 import AppTest

SCRIPT = "visualize_projection.py"
TIMEOUT = 30


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


def _switch_tab(at, tab_name):
    """Switch to a tab and run."""
    [r for r in at.radio if r.label == "Tab"][0].set_value(tab_name).run()
    _assert_no_error(at, f"switch to {tab_name}")


# ===========================================================================
# Non-default projection bases render without error
# ===========================================================================

class TestNonDefaultProjectionBases:
    """Test that switching to a non-default projection basis renders OK.
    Default bases are covered by TestDefaultValues."""

    def test_metr_linear(self):
        """METR defaults to Piecewise; verify Linear works."""
        at = _fresh_app()
        at.run()
        proj = at.radio(key="metr_proj_basis")
        proj.set_value("Linear").run()
        _assert_no_error(at, "METR / Linear")

    def test_eci_piecewise(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        at.radio(key="eci_proj_basis").set_value("Piecewise linear").run()
        _assert_no_error(at, "ECI / Piecewise linear")

    def test_eci_superexp(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        at.radio(key="eci_proj_basis").set_value("Superexponential").run()
        _assert_no_error(at, "ECI / Superexponential")

    def test_eci_entity_comparison(self):
        """A second entity (subject) is projected against the first (benchmark)."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        # [US best, China best]: project China against the US trend.
        at.selectbox(key="eci_entity_b").set_value("China best").run()
        _assert_no_error(at, "ECI / US best vs China best")
        assert at.selectbox(key="eci_entity_a").value == "US best"

    def test_eci_single_entity(self):
        """A non-default primary entity with no comparison renders alone."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        at.selectbox(key="eci_entity_a").set_value("DeepSeek").run()
        _assert_no_error(at, "ECI / DeepSeek solo")
        assert at.selectbox(key="eci_entity_b").value == "—"

    def test_rli_linear(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        at.radio(key="rli_proj_basis").set_value("Linear (logit)").run()
        _assert_no_error(at, "RLI / Linear (logit)")

    def test_rli_piecewise(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        at.radio(key="rli_proj_basis").set_value("Piecewise linear (logit)").run()
        _assert_no_error(at, "RLI / Piecewise linear (logit)")

    def test_rli_superexp(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        at.radio(key="rli_proj_basis").set_value("Superexponential (logit)").run()
        _assert_no_error(at, "RLI / Superexponential (logit)")


# ===========================================================================
# Mode switching: Linear vs Piecewise produce different widgets and output
# ===========================================================================

class TestModeSwitchingBehavior:
    """Verify that switching projection basis actually changes the UI —
    not just that it doesn't crash."""

    # -- METR --

    def test_metr_linear_has_no_segments_radio(self):
        """Linear mode should not render the Segments radio."""
        at = _fresh_app()
        at.run()
        at.radio(key="metr_proj_basis").set_value("Linear").run()
        _assert_no_error(at, "METR / Linear")
        assert not _has_widget(at, "radio", "piecewise_n_seg"), \
            "Linear mode should not have Segments radio"
        assert "piecewise_n_seg" not in at.session_state, \
            "Linear mode should not have piecewise_n_seg in session state"

    def test_metr_piecewise_has_segments_radio(self):
        """Piecewise mode should render the Segments radio with value 2."""
        at = _fresh_app()
        at.run()  # Default is Piecewise
        assert _has_widget(at, "radio", "piecewise_n_seg"), \
            "Piecewise mode should have Segments radio"
        assert at.radio(key="piecewise_n_seg").value == 2

    def test_metr_piecewise_to_linear_clears_segments(self):
        """Switching Piecewise → Linear should clear the segments state."""
        at = _fresh_app()
        at.run()
        # Starts in Piecewise with 2 segments
        assert at.radio(key="piecewise_n_seg").value == 2
        # Switch to Linear
        at.radio(key="metr_proj_basis").set_value("Linear").run()
        _assert_no_error(at, "METR Piecewise→Linear")
        assert not _has_widget(at, "radio", "piecewise_n_seg"), \
            "Segments radio should disappear after switching to Linear"
        assert "piecewise_n_seg" not in at.session_state

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

    def test_metr_linear_ci_differs_from_piecewise_ci(self):
        """Linear (full OLS) should have different CI defaults than
        Piecewise (last-segment OLS)."""
        at = _fresh_app()
        at.run()
        # Default is Piecewise — record its CI
        pw_dt_lo = at.number_input(key="custom_dt_lo").value
        # Switch to Linear
        at.radio(key="metr_proj_basis").set_value("Linear").run()
        _assert_no_error(at, "METR / Linear CI check")
        lin_dt_lo = at.number_input(key="custom_dt_lo").value
        assert lin_dt_lo != pw_dt_lo, \
            f"Linear and Piecewise should have different DT defaults: both={pw_dt_lo}"

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

    # -- ECI --

    def test_eci_linear_has_no_segments_radio(self):
        """ECI Linear mode should not render the Segments radio."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")  # Defaults to Linear
        assert not _has_widget(at, "radio", "eci_piecewise_n_seg"), \
            "ECI Linear mode should not have Segments radio"
        assert "eci_piecewise_n_seg" not in at.session_state

    def test_eci_piecewise_has_segments_radio(self):
        """ECI Piecewise mode should render the Segments radio."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        at.radio(key="eci_proj_basis").set_value("Piecewise linear").run()
        _assert_no_error(at, "ECI / Piecewise")
        assert _has_widget(at, "radio", "eci_piecewise_n_seg"), \
            "ECI Piecewise should have Segments radio"
        assert at.radio(key="eci_piecewise_n_seg").value == 2

    def test_eci_piecewise_to_linear_clears_segments(self):
        """ECI: switching Piecewise → Linear should clear segments state."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        # Switch to Piecewise first
        at.radio(key="eci_proj_basis").set_value("Piecewise linear").run()
        assert at.radio(key="eci_piecewise_n_seg").value == 2
        # Switch back to Linear
        at.radio(key="eci_proj_basis").set_value("Linear").run()
        _assert_no_error(at, "ECI Piecewise→Linear")
        assert not _has_widget(at, "radio", "eci_piecewise_n_seg"), \
            "ECI Segments radio should disappear after switching to Linear"
        assert "eci_piecewise_n_seg" not in at.session_state

    # -- RLI --

    def test_rli_linear_has_no_segments_radio(self):
        """RLI Linear mode should not render the Segments radio."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")  # Defaults to Linear (logit)
        assert not _has_widget(at, "radio", "rli_piecewise_n_seg"), \
            "RLI Linear mode should not have Segments radio"
        assert "rli_piecewise_n_seg" not in at.session_state

    def test_rli_piecewise_has_segments_radio(self):
        """RLI Piecewise mode should render the Segments radio."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        at.radio(key="rli_proj_basis").set_value("Piecewise linear (logit)").run()
        _assert_no_error(at, "RLI / Piecewise")
        assert _has_widget(at, "radio", "rli_piecewise_n_seg"), \
            "RLI Piecewise should have Segments radio"

    def test_rli_piecewise_to_linear_clears_segments(self):
        """RLI: switching Piecewise → Linear should clear segments state."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        # Switch to Piecewise first
        at.radio(key="rli_proj_basis").set_value("Piecewise linear (logit)").run()
        assert _has_widget(at, "radio", "rli_piecewise_n_seg")
        # Switch back to Linear
        at.radio(key="rli_proj_basis").set_value("Linear (logit)").run()
        _assert_no_error(at, "RLI Piecewise→Linear")
        assert not _has_widget(at, "radio", "rli_piecewise_n_seg"), \
            "RLI Segments radio should disappear after switching to Linear"
        assert "rli_piecewise_n_seg" not in at.session_state


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
        """Switching ECI from 1-segment to 2-segment should change CI
        (last-segment OLS vs full OLS)."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Epoch ECI")
        ppy_lo_1seg = at.number_input(key="eci_custom_ppy_lo").value
        # Switch to Piecewise first, then change segments
        at.radio(key="eci_proj_basis").set_value("Piecewise linear").run()
        at.radio(key="eci_piecewise_n_seg").set_value(2).run()
        _assert_no_error(at, "ECI 2-segment")
        ppy_lo_2seg = at.number_input(key="eci_custom_ppy_lo").value
        assert ppy_lo_2seg != ppy_lo_1seg, \
            f"ECI CI didn't change: {ppy_lo_1seg} → {ppy_lo_2seg}"

    def test_metr_custom_ci_renders_ok(self):
        """Changing CI values manually should render without error."""
        at = _fresh_app()
        at.run()
        at.number_input(key="custom_dt_lo").set_value(30).run()
        _assert_no_error(at, "custom dt_lo=30")
        at.number_input(key="custom_dt_hi").set_value(500).run()
        _assert_no_error(at, "custom dt_hi=500")

    def test_metr_toggles_render_ok(self):
        """Toggling sidebar controls should render without errors."""
        at = _fresh_app()
        at.run()
        at.toggle(key="milestones").set_value(False).run()
        _assert_no_error(at, "milestones off")
        at.toggle(key="log_scale").set_value(False).run()
        _assert_no_error(at, "log scale off")
        at.toggle(key="p80").set_value(True).run()
        _assert_no_error(at, "p80 on")

    def test_rli_custom_ci_renders_ok(self):
        """RLI: changing CI values renders without error."""
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Remote Labor Index")
        at.number_input(key="rli_custom_dt_lo").set_value(30.0).run()
        _assert_no_error(at, "RLI custom dt_lo=30")


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
        _assert_no_error(at, "after modifications")
        # Verify they actually changed
        assert at.number_input(key="custom_dt_lo").value == 10
        assert at.toggle(key="milestones").value is False
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


class TestEmploymentRenders:
    """Employment tab renders without error across projection bases."""

    def test_emp_default_linear_renders(self):
        """Employment tab renders with default Linear (logit) basis."""
        at = _emp_app()
        assert at.radio(key="emp_proj_basis").value == "Linear (logit)"
        assert _has_widget(at, "slider", "emp_rli_coverage")
        assert _has_widget(at, "slider", "emp_base_unemployment")

    def test_emp_piecewise_renders(self):
        at = _emp_app()
        at.radio(key="emp_proj_basis").set_value("Piecewise linear (logit)").run()
        _assert_no_error(at, "Employment / Piecewise")

    def test_emp_superexp_renders(self):
        at = _emp_app()
        at.radio(key="emp_proj_basis").set_value("Superexponential (logit)").run()
        _assert_no_error(at, "Employment / Superexponential")


class TestEmploymentDefaults:
    """Employment default values are reasonable and data-driven."""

    def test_emp_slider_defaults(self):
        at = _emp_app()
        assert at.slider(key="emp_rli_coverage").value == 70.0
        assert at.slider(key="emp_supervision_overhead").value == 10.0
        assert at.slider(key="emp_remote_digital_share").value == 38.0
        assert at.slider(key="emp_base_unemployment").value == 4.0
        assert at.slider(key="emp_jevons_recovery").value == 30.0
        assert at.slider(key="emp_adoption_lag").value == 365.0

    def test_emp_ci_defaults_are_data_driven(self):
        at = _emp_app()
        dt_lo = at.number_input(key="emp_custom_dt_lo").value
        dt_hi = at.number_input(key="emp_custom_dt_hi").value
        assert dt_lo > 0 and dt_hi > dt_lo, \
            f"DT CI should be positive with lo < hi: lo={dt_lo}, hi={dt_hi}"
        assert 3.0 <= dt_hi / dt_lo <= 5.0, \
            f"Unexpected CI spread: lo={dt_lo}, hi={dt_hi}"

    def test_emp_display_mode_default(self):
        at = _emp_app()
        assert at.radio(key="emp_display_mode").value == "Unemployment Rate (%)"

    def test_emp_end_year_default(self):
        at = _emp_app()
        assert at.radio(key="emp_end_year").value == 2028


class TestEmploymentSliderChanges:
    """Changing economic model sliders renders without error."""

    def test_emp_low_rli_coverage(self):
        """Low RLI coverage (10%) should not crash."""
        at = _emp_app()
        at.slider(key="emp_rli_coverage").set_value(10.0).run()
        _assert_no_error(at, "emp_rli_coverage=10")

    def test_emp_zero_rli_coverage(self):
        """Zero RLI coverage should not crash."""
        at = _emp_app()
        at.slider(key="emp_rli_coverage").set_value(0.0).run()
        _assert_no_error(at, "emp_rli_coverage=0")

    def test_emp_max_rli_coverage(self):
        """Max RLI coverage (100%) should not crash."""
        at = _emp_app()
        at.slider(key="emp_rli_coverage").set_value(100.0).run()
        _assert_no_error(at, "emp_rli_coverage=100")

    def test_emp_zero_jevons(self):
        """Zero Jevons recovery should not crash."""
        at = _emp_app()
        at.slider(key="emp_jevons_recovery").set_value(0.0).run()
        _assert_no_error(at, "emp_jevons=0")

    def test_emp_zero_lag(self):
        """Zero adoption lag should not crash."""
        at = _emp_app()
        at.slider(key="emp_adoption_lag").set_value(0.0).run()
        _assert_no_error(at, "emp_lag=0")

    def test_emp_max_lag(self):
        """Max adoption lag (1460 days) should not crash."""
        at = _emp_app()
        at.slider(key="emp_adoption_lag").set_value(1460.0).run()
        _assert_no_error(at, "emp_lag=1460")

    def test_emp_high_base_unemployment(self):
        """High base unemployment should not crash."""
        at = _emp_app()
        at.slider(key="emp_base_unemployment").set_value(12.0).run()
        _assert_no_error(at, "emp_base_unemp=12")


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

    def test_emp_jobs_lost_low_coverage(self):
        """Jobs Lost mode with low RLI coverage should not crash."""
        at = _emp_app()
        at.radio(key="emp_display_mode").set_value("Jobs Lost Above Baseline").run()
        at.slider(key="emp_rli_coverage").set_value(10.0).run()
        _assert_no_error(at, "Jobs Lost + low coverage")


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

    def test_metric_switch(self):
        at = self._dc_app()
        at.selectbox(key="dc_metric").set_value("Power (MW)").run()
        _assert_no_error(at, "Data Centers / Power (MW)")

    def test_train_log_op_metrics_render(self):
        for label in ("2mo train log OP", "6mo train log OP"):
            at = self._dc_app()
            at.selectbox(key="dc_metric").set_value(label).run()
            _assert_no_error(at, f"Data Centers / {label}")
            text = " ".join(str(c.value) for c in at.caption)
            assert f"Methodology: *{label}*" in text

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

    def test_train_log_op_timing_shift(self):
        at = self._dc_app()
        at.selectbox(key="dc_metric").set_value("6mo train log OP").run()
        at.selectbox(key="dc_timing").set_value("Model release").run()
        _assert_no_error(at, "Data Centers / 6mo train log OP + model release")

    def test_train_time_metrics_render_durations(self):
        for label in ("Capacity (time to GPT-5)", "Capacity (time to Mythos)"):
            at = self._dc_app()
            at.selectbox(key="dc_metric").set_value(label).run()
            _assert_no_error(at, f"Data Centers / {label}")
            text = " ".join(str(m.value) for m in at.markdown) + \
                " ".join(str(c.value) for c in at.caption)
            # Captions are in time units, never a bare run count.
            assert "Methodology: time to train one" in text
            assert any(u in text for u in ("hour", "day", "week", "month"))

    def test_include_future(self):
        at = self._dc_app()
        at.checkbox(key="dc_future").set_value(True).run()
        _assert_no_error(at, "Data Centers / include future")

    def test_projection_range_defaults(self):
        at = self._dc_app()
        assert at.radio(key="dc_start_year").value == 2025
        assert at.radio(key="dc_end_year").value == 2027

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

    def test_fit_to_last_n_still_reducible(self):
        """The slider remains user-adjustable to fit on fewer recent points."""
        at = self._rev_app()
        at.slider(key="oai_n_recent").set_value(3).run()
        _assert_no_error(at, "Revenue / reduced fit-N")
        assert at.slider(key="oai_n_recent").value == 3

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


# ===========================================================================
# Compute vs Capabilities tab
# ===========================================================================

class TestComputeVsCapabilities:
    """The Compute vs Capabilities tab renders both requested views."""

    def _cc_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Compute/capabilities/diffusion")
        return at

    def test_renders(self):
        at = self._cc_app()
        _assert_no_error(at, "Compute vs Capabilities default")
        text = (" ".join(str(m.value) for m in at.markdown) +
                " ".join(str(h.value) for h in at.subheader) +
                " ".join(str(w.value) for w in at.warning)).lower()
        # All three sections render: 1 (exchange rate + two-engine flow),
        # 2 (ECI forecasts), 3 (US vs. China).
        assert "exchange rate" in text
        assert "two engines" in text
        assert "eci forecasts" in text
        assert "us vs. china" in text

    def test_forecast_section_renders_milestones(self):
        at = self._cc_app()
        # Section 2 projects to end-2029 with year-end milestone cards and a
        # China distillation-scenario table carrying all four channel rows.
        labels = " ".join(str(m.label) for m in at.metric)
        assert "End 2029" in labels
        text = " ".join(str(m.value) for m in at.markdown).lower()
        assert "distillation scenario" in text
        assert "no external distillation" in text
        assert "indigenous only" in text

    def test_has_two_engine_metrics(self):
        at = self._cc_app()
        # The effective-compute decomposition exposes its two engines as metrics.
        labels = " ".join(str(m.label) for m in at.metric)
        assert "Physical compute" in labels
        assert "Algorithmic efficiency" in labels
        assert "Share of growth of compute" in labels

    def test_channel_decomposition_renders(self):
        """The three-channel section shows the derivation table, the stacked
        US/China bars, and the residual consistency check."""
        at = self._cc_app()
        text = " ".join(str(m.value) for m in at.markdown)
        assert "Where frontier growth comes from" in text
        assert "Diffusion" in text and "Distillation" in text
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

    def test_include_future_toggle(self):
        at = self._cc_app()
        at.checkbox(key="cc_future").set_value(False).run()
        _assert_no_error(at, "Compute vs Capabilities / no future")

    def test_dc_derivation_is_stated(self):
        """The tab says where its compute inputs come from: the
        Physical-compute provenance caption and the US-vs-China cross-check
        caption both name the Data Centers tab, and the China band is quoted
        against the catalogued paces rather than asserted in a vacuum."""
        at = self._cc_app()
        caps = " ".join(str(c.value) for c in at.caption).lower()
        assert "data centers tab" in caps
        assert "cross-check" in caps
        assert "china-accessible" in caps

    def test_run_length_toggle(self):
        """Switching to the 6-month window (the Pacing tab's options,
        verbatim) renders clean and relabels the capacity captions and the
        ceiling caveat with the chosen window."""
        at = self._cc_app()
        at.selectbox(key="cc_run").set_value("6-month run").run()
        _assert_no_error(at, "Compute vs Capabilities / 6-month run")
        caps = " ".join(str(c.value) for c in at.caption)
        assert "6-month run" in caps
        # The ceiling caveat is a hover on the caveats line (markdown), not
        # a warning box — st.warning takes no HTML.
        md = " ".join(str(m.value) for m in at.markdown)
        assert "6mo-capacity" in md

    def test_project_through_horizon(self):
        """The Projection range expander's 'Project through' year moves the
        forecast horizon: milestone cards, the end-year gap metric and the
        captions all follow it, at both ends of the offered range."""
        at = self._cc_app()
        at.radio(key="cc_end_year").set_value(2031).run()
        _assert_no_error(at, "Compute vs Capabilities / through 2031")
        labels = " ".join(str(m.label) for m in at.metric)
        assert "End 2031" in labels
        assert "ECI gap end-2031" in labels
        at.radio(key="cc_end_year").set_value(2027).run()
        _assert_no_error(at, "Compute vs Capabilities / through 2027")
        labels = " ".join(str(m.label) for m in at.metric)
        assert "End 2027" in labels and "End 2029" not in labels

    def test_reset(self):
        at = self._cc_app()
        at.checkbox(key="cc_future").set_value(False).run()
        at.button(key="cc_reset").click().run()
        _assert_no_error(at, "after cc reset")
        assert at.checkbox(key="cc_future").value is True


# ===========================================================================
# UK Cyber (AISI narrow cyber tasks)
# ===========================================================================

class TestCcWorldSharesTab:
    """The global compute distribution, last section of the CC tab."""

    def _app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Compute/capabilities/diffusion")
        return at

    _HEAD = "Global compute distribution"

    def test_renders_last_with_both_charts(self):
        at = self._app()
        _assert_no_error(at, "CC / world shares")
        assert [str(h.value) for h in at.subheader][-1] == self._HEAD
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
        assert us_tracked > us_guess + 15 and cn_guess > cn_tracked


class TestRsiTab:
    """CoBench score vs release date, plus the fitted trend."""

    def _rsi_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "RSI")
        return at

    def test_renders(self):
        at = self._rsi_app()
        _assert_no_error(at, "RSI / default")

    def test_deep_link_slug(self):
        at = _fresh_app()
        at.query_params["tab"] = "rsi"
        at.run()
        _assert_no_error(at, "RSI / deep link")
        assert at.session_state["_active_tab"] == "RSI"

    def test_metr_174h_panel_renders_at_the_top(self):
        """Both reliability levels at the month-scale bar, as their own cards."""
        at = self._rsi_app()
        labels = [str(m.label) for m in at.metric]
        assert "METR p50 horizon reaches 174h" in labels
        assert "METR p80 horizon reaches 174h" in labels
        assert not any("40h" in l for l in labels)
        assert "ECI reaches 207.5" in labels
        assert "ECI reaches 227" in labels
        assert not any("ECI reaches 170" in l for l in labels)
        assert "RLI reaches 90%" in labels
        caps = " ".join(str(c.value) for c in at.caption)
        assert caps.count("80% CI:") >= 4


    def test_research_direction_section_renders(self):
        at = self._rsi_app()
        assert "Research direction" in [str(h.value) for h in at.subheader]
        caps = " ".join(str(c.value) for c in at.caption)
        assert "When AI builds itself" in caps

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
        """All four RSI charts carry the dashed "Today" divider the other
        tabs' projection charts have — without it the fan's start reads as
        the last data point."""
        import json
        at = self._rsi_app()
        # The blend's CDF is excluded: it starts at today, so a divider there
        # would sit on its own left edge. It is the one chart with no x title.
        charts = [json.loads(el.proto.spec)["layout"]
                  for el in at.get("plotly_chart")]
        series = [L for L in charts if (L.get("xaxis") or {}).get("title")]
        assert len(series) == 4 and len(charts) == 5
        for L in series:
            notes = [a for a in L.get("annotations", [])
                     if a.get("text") == "Today"]
            assert len(notes) == 1, L["yaxis"]["title"]
            assert any(sh.get("line", {}).get("dash") == "dash"
                       and sh.get("x0") == sh.get("x1")
                       for sh in L.get("shapes", []))

    def test_revenue_milestone_card_and_blend_row(self):
        """The $1T card renders alongside the benchmark milestones and gets
        its own weighted row in the blend."""
        at = self._rsi_app()
        labels = {str(m.label) for m in at.metric}
        assert "Leading company revenue >$1T" in labels
        t = next(x.value for x in at.table if "Milestone" in x.value.columns)
        row = t[t["Milestone"] == "Leading company revenue >$1T"]
        assert len(row) == 1
        assert row.iloc[0]["Weight"].startswith("10%")
        # Every milestone gets a row, and the weights editor an input each.
        assert len(t) == 10

    def test_merged_code_section_renders(self):
        at = self._rsi_app()
        assert "Code merged per Anthropic engineer" in [str(h.value)
                                                        for h in at.subheader]
        assert "Code per person reaches 30x" in [str(m.label) for m in at.metric]

    def test_merged_code_row_in_the_blend(self):
        at = self._rsi_app()
        t = next(x.value for x in at.table if "Milestone" in x.value.columns)
        row = t[t["Milestone"] == "Code per person reaches 30x"]
        assert len(row) == 1
        assert row.iloc[0]["Weight"].startswith("7%")
        staff = t[t["Milestone"].str.contains("acceleration")]
        assert staff.iloc[0]["Weight"].startswith("8%")

    def test_every_milestone_card_carries_its_caveats_on_hover(self):
        """Caveats ride the card they belong to rather than piling into one
        caption nobody reads: every card has a hover naming its fit and its
        clock, and the caption under them stays short."""
        at = self._rsi_app()
        cards = [m for m in at.metric if "80% CI" not in str(m.label)
                 and str(m.label) not in ("Blended median", "Median")]
        helps = {str(m.label): (m.proto.help or "") for m in cards}
        milestones = {k: v for k, v in helps.items()
                      if "reaches" in k or "revenue" in k or "acceleration" in k}
        assert len(milestones) == 10
        for lab, h in milestones.items():
            assert "defaults" in h, lab
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
        import re
        tabs = ["METR Horizon", "Epoch ECI", "Remote Labor Index", "RSI",
                "UK Cyber", "Revenue", "Employment", "ECI Company Gap",
                "Data Centers", "Compute/capabilities/diffusion", "Pacing"]
        total = 0
        for tab in tabs:
            at = _fresh_app()
            at.run()
            _switch_tab(at, tab)
            for el in list(at.caption) + list(at.markdown):
                v = str(el.value)
                total += v.count('class="vp-fn-a')
                stray = re.findall(r'<span class="vp-fn(?: vp-fn-r)?">([^<]*)<',
                                   v)
                assert not stray, (tab, stray)
        assert total > 50            # the sweep is app-wide, not one tab

    def test_blend_renders(self):
        at = self._rsi_app()
        labels = {str(m.label) for m in at.metric}
        assert {"Blended median", "80% CI"} <= labels
        assert any("RSI projection (tentative)" in h.value for h in at.subheader)

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

    def test_horizon_selector(self):
        at = self._rsi_app()
        at.selectbox(key="rsi_end_year").set_value(2031).run()
        _assert_no_error(at, "RSI / project through 2031")


class TestUkCyberTab:
    """Frontier projection + open-weight lag."""

    def _ukc_app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "UK Cyber")
        return at

    def test_renders(self):
        at = self._ukc_app()
        _assert_no_error(at, "UK Cyber / default")

    def test_deep_link_slug(self):
        at = _fresh_app()
        at.query_params["tab"] = "ukcyber"
        at.run()
        _assert_no_error(at, "UK Cyber / deep link")
        assert at.session_state["_active_tab"] == "UK Cyber"

    def test_piecewise(self):
        at = self._ukc_app()
        at.radio(key="ukc_proj_basis").set_value("Piecewise linear (logit)").run()
        _assert_no_error(at, "UK Cyber / Piecewise linear (logit)")

    def test_superexp(self):
        at = self._ukc_app()
        at.radio(key="ukc_proj_basis").set_value("Superexponential (logit)").run()
        _assert_no_error(at, "UK Cyber / Superexponential (logit)")

    def test_backtest_vantage_point(self):
        at = self._ukc_app()
        at.selectbox(key="_ukc_proj_as_of").set_value("GPT-5").run()
        _assert_no_error(at, "UK Cyber / backtest from GPT-5")

    def test_toggles(self):
        at = self._ukc_app()
        at.toggle(key="ukc_show_open").set_value(False).run()
        _assert_no_error(at, "UK Cyber / open-weight hidden")
        at.toggle(key="ukc_show_lag").set_value(False).run()
        _assert_no_error(at, "UK Cyber / lag markers hidden")

    def test_target_eta_is_shown(self):
        """The headline output is when open-weight models reach the target."""
        at = self._ukc_app()
        labels = [m.label for m in at.metric]
        assert "China reaches 90%" in labels
        assert "Measured open-weight lag" in labels
        lag = [m for m in at.metric if m.label == "Measured open-weight lag"][0]
        assert lag.value == "4.7–7.4 mo"

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
        # The panel runs to the sidebar's "Project through" year.
        assert list(table["Year end"]) == ["2026", "2027"]
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
        assert at.sidebar.selectbox(key="dc_pool_n") is not None
        assert at.sidebar.radio(key="dc_cty_since") is not None

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
            ["2026", "2027", "2028", "2029", "2030"]
        at.radio(key="dc_start_year").set_value(2023).run()
        _assert_no_error(at, "by country / from 2023")

    def test_cones_toggle(self):
        at = self._dc_app()
        assert at.checkbox(key="dc_cty_cones").value is True
        at.checkbox(key="dc_cty_cones").set_value(False).run()
        _assert_no_error(at, "by country / cones off")
        # The numbers stay; only the shaded bands go.
        assert self._headline(at)

    def test_trend_only_when_planned_buildout_is_off(self):
        at = self._dc_app()
        at.checkbox(key="dc_future").set_value(False).run()
        _assert_no_error(at, "by country / no planned buildout")
        assert self._headline(at)

    def test_other_metrics(self):
        for label in ("6mo train log OP", "Power (MW)", "Capacity (time to Mythos)"):
            at = self._dc_app()
            at.selectbox(key="dc_metric").set_value(label).run()
            _assert_no_error(at, f"by country / {label}")
            assert self._headline(at), label

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
        assert heads[-1] == self._HEAD
        # The coverage caveat is body text above the chart, not fine print.
        warn = [str(m.value) for m in at.markdown
                if "Tracked data centers only" in str(m.value)]
        assert len(warn) == 1 and "Non-US shares are floors" in warn[0]
        assert "Tracked data centers only" not in \
            " ".join(str(c.value) for c in at.caption)

    def test_renders_under_every_metric(self):
        at = self._app()
        for label in at.selectbox(key="dc_metric").options:
            at.selectbox(key="dc_metric").set_value(label).run()
            _assert_no_error(at, f"region share / {label}")
            assert self._HEAD in [str(h.value) for h in at.subheader], label

    def test_renders_with_planned_buildout_off(self):
        at = self._app()
        at.checkbox(key="dc_future").set_value(False).run()
        _assert_no_error(at, "region share / no future")
        assert self._HEAD in [str(h.value) for h in at.subheader]


class TestDataCentersParty:
    """The DC tab's attribution radio: defaults to tenant (shared sites count
    under every listed user), flips to operator, and resets."""

    def _app(self):
        at = _fresh_app()
        at.run()
        _switch_tab(at, "Data Centers")
        return at

    def test_defaults_to_tenant_and_renders(self):
        at = self._app()
        _assert_no_error(at, "Data Centers / tenant")
        assert at.radio(key="dc_party").value == "Tenant (who trains there)"

    def test_operator_view_renders_and_resets(self):
        at = self._app()
        at.radio(key="dc_party").set_value(
            "Operator (who owns the building)").run()
        _assert_no_error(at, "Data Centers / operator")
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

    def test_us_pause_panel_renders(self):
        """The US-pause counterfactual renders with its crossing metric,
        race chart and assumption caption."""
        at = self._app()
        subs = " ".join(str(s.value) for s in at.subheader)
        assert "If the US paced" in subs
        labels = " ".join(str(m.label) for m in at.metric)
        assert "paced US frontier" in labels
        assert "Time for China to surpass" in labels
        caps = " ".join(str(c.value) for c in at.caption)
        assert "indigenous" in caps
        # The assumptions are hovers on the *Assumes* line now, so match
        # their bodies rather than the old run-on sentence.
        md = " ".join(str(m.value) for m in at.markdown)
        # Footnotes anchor to their phrases, so the line reads
        # "secure <span>weights</span>, ..." with the note in the bubble.
        assert "**Assumes**" in md
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
        base = _surpass_mo(at)
        # Serving the paused frontier publicly restores the full teacher:
        # China can only cross sooner (or the same, within MC jitter).
        # Each channel's setting is the body of its hover on the *Assumes*
        # line; the markers capitalise it, so match case-insensitively.
        def _md(at):
            return " ".join(str(m.value) for m in at.markdown).lower()

        at.checkbox(key="pc_withhold").uncheck().run()
        _assert_no_error(at, "Pacing / paused models served")
        assert _surpass_mo(at) <= base + 1
        assert "stays queryable" in _md(at)
        at.checkbox(key="pc_withhold").check().run()
        assert "release freeze" in _md(at)
        at.checkbox(key="pc_stop_dist").check().run()
        _assert_no_error(at, "Pacing / distillation stopped")
        dist_off = _surpass_mo(at)
        assert dist_off > base
        assert "cut today" in _md(at)
        at.checkbox(key="pc_stop_remote").check().run()
        _assert_no_error(at, "Pacing / remote compute cut")
        assert _surpass_mo(at) >= dist_off - 1  # MC jitter guard
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
        base = _surpass_mo(at)

        at.checkbox(key="pc_stop_dist").check().run()
        _assert_no_error(at, "Pacing / distillation cut today")
        now_d = _surpass_mo(at)
        late = at.select_slider(key="pc_dist_when").options[-1]
        at.select_slider(key="pc_dist_when").set_value(late).run()
        _assert_no_error(at, "Pacing / distillation cut late")
        assert base - 1 <= _surpass_mo(at) <= now_d + 1
        md = " ".join(str(m.value) for m in at.markdown).lower()
        assert f"cut {late}".lower() in md
        at.checkbox(key="pc_stop_dist").uncheck().run()

        at.checkbox(key="pc_stop_remote").check().run()
        _assert_no_error(at, "Pacing / remote cut today")
        now_r = _surpass_mo(at)
        at.select_slider(key="pc_remote_when").set_value(late).run()
        _assert_no_error(at, "Pacing / remote cut late")
        assert base - 1 <= _surpass_mo(at) <= now_r + 1
        md = " ".join(str(m.value) for m in at.markdown)
        assert late in md

    def test_chinese_run_length_trades_compute_against_wall_clock(self):
        """A longer Chinese run lifts the path (more compute in one model)
        and shifts it right (the run takes longer), so the crossing is a
        U-curve in run length: a few extra months help, a year hurts.
        Matching the bar's length is the default and changes nothing."""
        at = self._app()

        def _surpass_mo(at):
            m = next(m for m in at.metric
                     if "Time for China to surpass" in str(m.label))
            return float(str(m.value).strip("~ mo"))

        sl = at.slider(key="pc_cn_run")
        assert sl.value == 2 and sl.max == 12
        base = _surpass_mo(at)
        assert not [c for c in at.caption if "Chinese run" in str(c.value)]

        # 4 months is the live optimum. The curve is shallow near its floor and
        # the metric renders to 0.1 mo, so testing the arm at 5 reads a tie.
        at.slider(key="pc_cn_run").set_value(4).run()
        _assert_no_error(at, "Pacing / 4-month Chinese run")
        mid = _surpass_mo(at)
        cap = " ".join(str(c.value) for c in at.caption)
        assert "4-month Chinese run" in cap and "2 months of extra" in cap
        at.slider(key="pc_cn_run").set_value(12).run()
        _assert_no_error(at, "Pacing / 12-month Chinese run")
        long_run = _surpass_mo(at)
        # The left arm is shallow (~0.5 mo on a ~8 mo crossing) and both numbers
        # are 400-sample MC medians off an unseeded RNG, so it gets a tolerance;
        # the right arm is ~4 mo and stays strict.
        assert mid < base + 0.3      # a moderate stretch pays
        assert long_run > mid + 1    # a year of wall clock does not

    def test_domestic_slowdown_lever_delays_the_crossing(self):
        """The 0-90% domestic-growth lever comes off the domestic share of
        the compute band, so it delays China's crossing on its own *and*
        on top of the remote-access cut, where the whole band is domestic
        and the setback also takes longer to regrow."""
        at = self._app()

        def _surpass_mo(at):
            m = next(m for m in at.metric
                     if "Time for China to surpass" in str(m.label))
            return float(str(m.value).strip("~ mo"))

        sl = at.slider(key="pc_dom_slow")
        assert sl.value == 0 and sl.max == 90
        base = _surpass_mo(at)
        md = " ".join(str(m.value) for m in at.markdown)
        assert "slowed" not in md

        at.slider(key="pc_dom_slow").set_value(90).run()
        _assert_no_error(at, "Pacing / domestic growth slowed 90%")
        slowed = _surpass_mo(at)
        assert slowed > base
        md = " ".join(str(m.value) for m in at.markdown)
        assert "domestic buildout slowed 90%" in md

        # Stacks with the remote cut rather than being subsumed by it.
        at.slider(key="pc_dom_slow").set_value(0).run()
        at.checkbox(key="pc_stop_remote").check().run()
        _assert_no_error(at, "Pacing / remote cut, domestic pace intact")
        cut = _surpass_mo(at)
        at.slider(key="pc_dom_slow").set_value(90).run()
        _assert_no_error(at, "Pacing / remote cut + domestic growth slowed")
        assert _surpass_mo(at) > cut

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

        base = _rows(at)
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

        at.checkbox(key="pc_stop_dist").check().run()
        _assert_no_error(at, "Pacing / why with distillation cut")
        cut = _rows(at)
        assert cut["distillation"][0] < base["distillation"][0] / 2
        assert cut["total"][0] == pytest.approx(gap, abs=1.0)
        at.checkbox(key="pc_stop_dist").uncheck().run()

        # Cutting remote access collapses exactly the row that prices it:
        # the shadow the breakdown subtracts becomes the run itself.
        at.checkbox(key="pc_stop_remote").check().run()
        _assert_no_error(at, "Pacing / why with remote access cut")
        rem = _rows(at)
        assert rem["abroad"][0] == pytest.approx(0.0, abs=0.05)
        assert rem["abroad"][2] == "+0.0 mo"
        assert rem["domestic"][0] == pytest.approx(base["domestic"][0],
                                                   abs=1.0)
        # Dating the cut later leaves part of it standing.
        at.select_slider(key="pc_remote_when").set_value("Aug 2028").run()
        _assert_no_error(at, "Pacing / why with remote access cut later")
        later = _rows(at)
        # Between the two: more than a cut today leaves, no more than never
        # cutting. The upper bound carries a tolerance because the test
        # suite runs a small Monte Carlo (conftest's _VP_SAMPLES).
        assert later["abroad"][0] > rem["abroad"][0]
        assert later["abroad"][0] < base["abroad"][0] + 1.0
        at.select_slider(key="pc_remote_when").set_value("Now").run()
        at.checkbox(key="pc_stop_remote").uncheck().run()

        at.slider(key="pc_cn_run").set_value(6).run()
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

        at.selectbox(key="pc_timing").set_value(
            "Data center construction").run()
        _assert_no_error(at, "Pacing / pause on construction clock")
        d_con, s_con = _cross_date(at), _surpass_mo(at)
        at.selectbox(key="pc_timing").set_value("Model release").run()
        _assert_no_error(at, "Pacing / pause on release clock")
        d_rel, s_rel = _cross_date(at), _surpass_mo(at)
        # 2-mo run + 30d prep ≈ 3 months later, ±MC/rounding jitter.
        diff_mo = ((d_rel.year - d_con.year) * 12 + d_rel.month - d_con.month)
        assert 1 <= diff_mo <= 5
        assert abs(s_rel - s_con) <= 1
        caps = " ".join(str(c.value) for c in at.caption)
        assert "Dates = model releases" in caps

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
        for col in ("METR horizon (p50)", "METR horizon (p80)"):
            assert all(t[col])
        assert us["METR horizon (p80)"] != us["METR horizon (p50)"]
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

    def test_projection_range(self):
        """'Project through' moves the crossing-search horizon."""
        at = self._app()
        at.radio(key="pc_end_year").set_value(2027).run()
        _assert_no_error(at, "Pacing / through 2027")
        at.radio(key="pc_end_year").set_value(2031).run()
        _assert_no_error(at, "Pacing / through 2031")

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

    def test_run_length_toggle_renders(self):
        at = self._app()
        at.radio(key="pc_run").set_value("6-month run").run()
        _assert_no_error(at, "Pacing / 6-month run")

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
        for label in ["Single site (no networking)", "Nearby only",
                      "Nearby + plausible fabric", "Every site (implausible)"]:
            at.selectbox(key="pc_pool").set_value(label).run()
            _assert_no_error(at, f"Pacing / {label}")

    def test_operator_attribution_changes_the_roster(self):
        at = self._app()
        tenant_ents = set(self._entities(at)["Entity"])
        at.radio(key="pc_party").set_value(
            "Operator (who owns the building)").run()
        _assert_no_error(at, "Pacing / operator")
        op_ents = set(self._entities(at)["Entity"])
        # Anthropic's biggest clusters are leased (Colossus, Lake Mariner);
        # under operator attribution the landlords appear instead.
        assert op_ents != tenant_ents
        assert "Oracle" in op_ents

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
