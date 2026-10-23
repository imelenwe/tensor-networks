"""Shared setup for the hadron-aqc tests.

Run from the hadron-aqc folder:
    python -m pytest tests                       # fast checks (seconds)
    python -m pytest tests --run-slow            # also the 30-second training check
    HADRON_REFERENCE_QASM=<folder> python -m pytest tests   # also compare with the group's QASM files
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so the tests can import hadron_aqc_utils


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="also run the slow checks (training)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes ~30 s or more; runs only with --run-slow")
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")   # library notices, not our problem


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow: add --run-slow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def reference_qasm_folder():
    """The group's circuit files (T1/, T2/, ... each with SU2_T*_SCV_*.qasm and SU2_T*_MID_*.qasm).
    Not committed (they still change); point HADRON_REFERENCE_QASM at a local copy."""
    folder = os.environ.get("HADRON_REFERENCE_QASM")
    if not folder:
        pytest.skip("set HADRON_REFERENCE_QASM to the folder of reference QASM files")
    return Path(folder).expanduser()
