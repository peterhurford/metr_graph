"""
Integration tests for visualize_projection.py using Streamlit's AppTest.

These run the actual Streamlit app in a headless runtime, catching issues
that unit tests with a fake Streamlit module cannot (e.g., type mismatches
in number_input, missing session state keys, widget rendering errors).

Run: pytest test_integration.py -v
"""

import pytest
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
