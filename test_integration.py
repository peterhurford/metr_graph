"""
Integration tests for visualize_projection.py using Streamlit's AppTest.

These run the actual Streamlit app in a headless runtime, catching issues
that unit tests with a fake Streamlit module cannot (e.g., type mismatches
in number_input, missing session state keys, widget rendering errors).

Run: pytest test_integration.py -v
"""

import pytest
from datetime import datetime
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
        warns = " ".join(str(w.value) for w in at.warning)
        assert "6mo-capacity" in warns

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

    def test_headline_metrics(self):
        at = self._rsi_app()
        labels = {m.label: m.value for m in at.metric}
        assert labels["Best CoBench score"] == "62.8%"
        assert labels["Full-substitution bar"] == "85%"
        assert labels["Gap remaining"] == "22.2 pts"

    def test_projection_to_the_substitution_bar_is_shown(self):
        at = self._rsi_app()
        assert any("When does CoBench reach 85%?" in h.value for h in at.subheader)
        labels = {m.label: m.value for m in at.metric}
        assert "Median" in labels and "80% CI" in labels
        assert "–" in labels["80% CI"]

    def test_no_table(self):
        """The tab is a chart, not a listing."""
        at = self._rsi_app()
        assert len(at.get("table")) == 0

    def test_slower_rate_pushes_the_bar_out(self):
        at = self._rsi_app()
        before = [m for m in at.metric if m.label == "Median"][0].value
        at.number_input(key="rsi_custom_dt_lo").set_value(300.0).run()
        at.number_input(key="rsi_custom_dt_hi").set_value(900.0).run()
        _assert_no_error(at, "RSI / slow rate")
        after = [m for m in at.metric if m.label == "Median"][0].value
        assert before != after

    def test_pacing_quotes_the_same_milestone(self):
        """The Pacing tab's CoBench card is this tab's own fit, so the two
        cannot date the milestone differently. Compared with a tolerance, not
        by string: both are Monte Carlo medians off an unseeded RNG."""
        at = self._rsi_app()
        here = datetime.strptime(
            [m for m in at.metric if m.label == "Median"][0].value, "%b %Y")
        at2 = _fresh_app()
        at2.run()
        _switch_tab(at2, "Pacing")
        there = datetime.strptime(
            [m for m in at2.metric
             if m.label == "CoBench reaches 85%"][0].value, "%b %Y")
        assert abs((here - there).days) <= 62

    def test_toggles(self):
        at = self._rsi_app()
        at.checkbox(key="rsi_labels").set_value(False).run()
        _assert_no_error(at, "RSI / labels hidden")
        at.checkbox(key="rsi_show_bar").set_value(False).run()
        _assert_no_error(at, "RSI / substitution bar hidden")

    def test_horizon_selector(self):
        at = self._rsi_app()
        at.selectbox(key="rsi_end_year").set_value(2031).run()
        _assert_no_error(at, "RSI / project through 2031")

    def test_source_and_tilde_are_explained(self):
        at = self._rsi_app()
        caps = " ".join(c.value for c in at.caption)
        assert "Redacted Risk Report" in caps
        assert "~" in caps


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
    def _entities(at):
        """The race table, picked by its columns — the pause panel's own
        breakdown table renders after it, so position is not an address."""
        return next(t.value for t in at.table if "Entity" in t.value.columns)

    def test_metr_40h_panel_renders_at_the_top(self):
        """Both reliability levels get a dated metric with an 80% CI."""
        at = self._app()
        labels = [str(m.label) for m in at.metric]
        assert "METR p50 horizon reaches 40h" in labels
        assert "METR p80 horizon reaches 40h" in labels
        assert "US ECI reaches 170" in labels
        assert "US ECI reaches 195" in labels
        assert "RLI reaches 90%" in labels
        caps = " ".join(str(c.value) for c in at.caption)
        assert caps.count("80% CI:") >= 5

    def test_us_pause_panel_renders(self):
        """The US-pause counterfactual renders with its crossing metric,
        race chart and assumption caption."""
        at = self._app()
        subs = " ".join(str(s.value) for s in at.subheader)
        assert "If the US paused" in subs
        labels = " ".join(str(m.label) for m in at.metric)
        assert "paused US frontier" in labels
        assert "Time for China to surpass after US pause" in labels
        caps = " ".join(str(c.value) for c in at.caption)
        assert "indigenous" in caps
        md = " ".join(str(m.value) for m in at.markdown)
        assert "weights stay secure" in md
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
        at.checkbox(key="pc_withhold").uncheck().run()
        _assert_no_error(at, "Pacing / paused models served")
        assert _surpass_mo(at) <= base + 1
        md = " ".join(str(m.value) for m in at.markdown)
        assert "stays queryable" in md
        at.checkbox(key="pc_withhold").check().run()
        md = " ".join(str(m.value) for m in at.markdown)
        assert "release freeze" in md
        at.checkbox(key="pc_stop_dist").check().run()
        _assert_no_error(at, "Pacing / distillation stopped")
        dist_off = _surpass_mo(at)
        assert dist_off > base
        md = " ".join(str(m.value) for m in at.markdown)
        assert "cut today" in md
        at.checkbox(key="pc_stop_remote").check().run()
        _assert_no_error(at, "Pacing / remote compute cut")
        assert _surpass_mo(at) >= dist_off - 1  # MC jitter guard
        md = " ".join(str(m.value) for m in at.markdown)
        assert "largest domestic cluster" in md or "domestic pace only" in md

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
        md = " ".join(str(m.value) for m in at.markdown)
        assert f"cut {late}" in md
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

        at.slider(key="pc_cn_run").set_value(5).run()
        _assert_no_error(at, "Pacing / 5-month Chinese run")
        mid = _surpass_mo(at)
        cap = " ".join(str(c.value) for c in at.caption)
        assert "5-month Chinese run" in cap and "3 months of extra" in cap
        at.slider(key="pc_cn_run").set_value(12).run()
        _assert_no_error(at, "Pacing / 12-month Chinese run")
        long_run = _surpass_mo(at)
        assert mid < base            # a moderate stretch pays
        assert long_run > mid        # a year of wall clock does not

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
        assert parts == pytest.approx(gap, abs=0.15)
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
                     if "paused US frontier" in str(m.label))
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

    def test_pause_date_respects_run_length(self):
        """The US pauses at its first *completed* threshold run: a 2-month
        run needs 3x the cluster, so the pause comes later and the frozen
        bar — the US frontier by then — sits higher than under 6-month."""
        at = self._app()

        def _bar(at):
            lab = next(str(m.label) for m in at.metric
                       if "paused US frontier" in str(m.label))
            return float(lab.split("ECI")[-1].strip(" ~)"))

        bar_2mo = _bar(at)
        caps = " ".join(str(c.value) for c in at.caption)
        assert "2-month" in caps and "-op run" in caps
        at.radio(key="pc_run").set_value("6-month run").run()
        _assert_no_error(at, "Pacing / pause with 6-month run")
        assert _bar(at) < bar_2mo

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
        for yr in (2028, 2031):
            at.radio(key="pc_end_year").set_value(yr).run()
            _assert_no_error(at, f"Pacing / through {yr}")
            ends = _ends(at)
            assert len(ends) == 2, ends
            # The timeline pads a few months past the grid; the pause chart
            # stops on it exactly.
            assert ends[0] in (yr, yr + 1), ends
            assert ends[1] == yr, ends

    def test_us_pause_bar_syncs_with_sidebar_threshold(self):
        """The pause bar follows the sidebar's training-run threshold: a
        bigger run maps to a higher ECI bar via the exchange rate."""
        at = self._app()

        def _bar(at):
            lab = next(str(m.label) for m in at.metric
                       if "paused US frontier" in str(m.label))
            return float(lab.split("ECI")[-1].strip(" ~)"))

        base = _bar(at)
        at.selectbox(key="pc_threshold").set_value("1e29").run()
        _assert_no_error(at, "Pacing / pause bar at 1e29")
        assert _bar(at) > base

    def test_renders_with_headline_and_table(self):
        at = self._app()
        _assert_no_error(at, "Pacing")
        assert "Pacing" in [str(h.value) for h in at.header]
        # The US-vs-China headline is a country line; the default roster is
        # companies, so it appears only under the 'Country' attribution.
        assert not [m for m in at.markdown if "China-accessible" in str(m.value)]
        assert at.selectbox(key="pc_threshold").value == "1e28"
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

    def test_threshold_drives_the_race(self):
        """The headline no longer names the threshold, so the chart title
        is what carries it into the display."""
        at = self._app()
        at.selectbox(key="pc_threshold").set_value("1e29").run()
        _assert_no_error(at, "Pacing @ 1e29")
        import json
        titles = [json.loads(el.proto.spec)["layout"].get("title", {}).get("text", "")
                  for el in at.get("plotly_chart")]
        assert any("1e29" in str(t) for t in titles)

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
        base = dict(zip(self._entities(at)["Entity"],
                        self._entities(at)["Plan crosses"]))
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
        at.selectbox(key="pc_threshold").set_value("1e27").run()
        at.radio(key="pc_party").set_value(
            "Operator (who owns the building)").run()
        [b for b in at.button if b.key == "pc_reset"][0].click().run()
        _assert_no_error(at, "Pacing / reset")
        assert at.selectbox(key="pc_threshold").value == "1e28"
        assert at.radio(key="pc_party").value == "Tenant (who trains there)"
        assert at.selectbox(key="pc_pool").value == \
            "Nearby + announced fabric"
        assert at.selectbox(key="pc_timing").value == \
            "Training run finished"
