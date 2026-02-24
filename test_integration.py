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


# ===========================================================================
# Each tab renders without error on default settings
# ===========================================================================

class TestDefaultRender:
    def test_metr_default(self):
        at = _fresh_app()
        at.run()
        _assert_no_error(at, "METR default")

    def test_eci_default(self):
        at = _fresh_app()
        at.run()
        [r for r in at.radio if r.label == "Tab"][0].set_value("Epoch ECI").run()
        _assert_no_error(at, "ECI default")

    def test_rli_default(self):
        at = _fresh_app()
        at.run()
        [r for r in at.radio if r.label == "Tab"][0].set_value("Remote Labor Index").run()
        _assert_no_error(at, "RLI default")


# ===========================================================================
# Every tab x projection basis combination renders without error
# ===========================================================================

class TestAllProjectionBases:
    """3 tabs x 3 projection bases = 9 combinations."""

    # -- METR (no explicit key on projection basis radio) -------------------

    def _metr_with_basis(self, basis):
        at = _fresh_app()
        at.run()
        proj = [r for r in at.radio if r.label == "Projection basis"][0]
        proj.set_value(basis).run()
        _assert_no_error(at, f"METR / {basis}")

    def test_metr_linear(self):
        self._metr_with_basis("Linear")

    def test_metr_piecewise(self):
        self._metr_with_basis("Piecewise linear")

    def test_metr_superexp(self):
        self._metr_with_basis("Superexponential")

    # -- ECI ----------------------------------------------------------------

    def _eci_with_basis(self, basis):
        at = _fresh_app()
        at.run()
        [r for r in at.radio if r.label == "Tab"][0].set_value("Epoch ECI").run()
        at.radio(key="eci_proj_basis").set_value(basis).run()
        _assert_no_error(at, f"ECI / {basis}")

    def test_eci_linear(self):
        self._eci_with_basis("Linear")

    def test_eci_piecewise(self):
        self._eci_with_basis("Piecewise linear")

    def test_eci_superexp(self):
        self._eci_with_basis("Superexponential")

    # -- RLI ----------------------------------------------------------------

    def _rli_with_basis(self, basis):
        at = _fresh_app()
        at.run()
        [r for r in at.radio if r.label == "Tab"][0].set_value("Remote Labor Index").run()
        at.radio(key="rli_proj_basis").set_value(basis).run()
        _assert_no_error(at, f"RLI / {basis}")

    def test_rli_linear(self):
        self._rli_with_basis("Linear (logit)")

    def test_rli_piecewise(self):
        self._rli_with_basis("Piecewise linear (logit)")

    def test_rli_superexp(self):
        self._rli_with_basis("Superexponential (logit)")
