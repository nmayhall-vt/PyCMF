"""
Unit and regression test for the pycmf package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pycmf


def test_pycmf_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pycmf" in sys.modules
