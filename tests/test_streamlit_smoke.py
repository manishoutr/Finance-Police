"""Optional Streamlit smoke test.

Runs when Streamlit is installed. The backend is tested independently so the
suite remains runnable in lightweight CI environments without Streamlit.
"""
from pathlib import Path

import pytest


@pytest.mark.skipif(__import__('importlib').util.find_spec('streamlit') is None, reason='Streamlit is not installed in this test environment')
def test_streamlit_app_starts_without_exception():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "src" / "app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not at.exception
