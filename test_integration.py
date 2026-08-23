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
        _switch_tab(at, "Compute vs Capabilities")
        return at

    def test_renders(self):
        at = self._cc_app()
        _assert_no_error(at, "Compute vs Capabilities default")
        text = (" ".join(str(m.value) for m in at.markdown) +
                " ".join(str(h.value) for h in at.subheader) +
                " ".join(str(w.value) for w in at.warning)).lower()
        # All four sections render: 1 (compute slowing?), 2 (exchange rate +
        # two-engine flow), 3 (ECI forecasts), 4 (US vs. China).
        assert "is compute slowing" in text
        assert "exchange rate" in text
        assert "two engines" in text
        assert "eci forecasts" in text
        assert "us vs. china" in text

    def test_forecast_section_renders_milestones(self):
        at = self._cc_app()
        # Section 3 projects to end-2029 with year-end milestone cards and a
        # China distillation-scenario table.
        labels = " ".join(str(m.label) for m in at.metric)
        assert "End 2029" in labels
        text = " ".join(str(m.value) for m in at.markdown).lower()
        assert "distillation scenario" in text

    def test_has_two_engine_metrics(self):
        at = self._cc_app()
        # The effective-compute decomposition exposes its two engines as metrics.
        labels = " ".join(str(m.label) for m in at.metric)
        assert "Physical compute" in labels
        assert "Algorithmic efficiency" in labels
        assert "Share of growth of compute" in labels

    def test_include_future_toggle(self):
        at = self._cc_app()
        at.checkbox(key="cc_future").set_value(False).run()
        _assert_no_error(at, "Compute vs Capabilities / no future")

    def test_reset(self):
        at = self._cc_app()
        at.checkbox(key="cc_future").set_value(False).run()
        at.button(key="cc_reset").click().run()
        _assert_no_error(at, "after cc reset")
        assert at.checkbox(key="cc_future").value is True


# ===========================================================================
# UK Cyber (AISI narrow cyber tasks)
# ===========================================================================

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

    def test_renders_with_headline_and_table(self):
        at = self._app()
        _assert_no_error(at, "Pacing")
        assert "Pacing" in [str(h.value) for h in at.header]
        head = [str(m.value) for m in at.markdown
                if "First over" in str(m.value)]
        assert len(head) == 1 and "1e28" in head[0]
        assert at.selectbox(key="pc_threshold").value == "1e28"
        assert at.radio(key="pc_run").value == "6-month run"
        table = at.table[-1].value
        ents = list(table["Entity"])
        assert "United States" in ents
        assert "China-accessible" in ents
        assert "China (domestic only)" in ents
        assert "Plan crosses" in table.columns

    def test_threshold_drives_the_headline(self):
        at = self._app()
        at.selectbox(key="pc_threshold").set_value("1e29").run()
        _assert_no_error(at, "Pacing @ 1e29")
        head = [str(m.value) for m in at.markdown
                if "First over" in str(m.value)]
        assert len(head) == 1 and "1e29" in head[0]

    def test_run_length_toggle_renders(self):
        at = self._app()
        at.radio(key="pc_run").set_value("2-month run").run()
        _assert_no_error(at, "Pacing / 2-month run")

    def test_every_pooling_option_renders(self):
        at = self._app()
        for label in ["Single site (no networking)", "Nearby only",
                      "Nearby + plausible fabric", "Every site (implausible)"]:
            at.selectbox(key="pc_pool").set_value(label).run()
            _assert_no_error(at, f"Pacing / {label}")

    def test_operator_attribution_changes_the_roster(self):
        at = self._app()
        tenant_ents = set(at.table[-1].value["Entity"])
        at.radio(key="pc_party").set_value(
            "Operator (who owns the building)").run()
        _assert_no_error(at, "Pacing / operator")
        op_ents = set(at.table[-1].value["Entity"])
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
