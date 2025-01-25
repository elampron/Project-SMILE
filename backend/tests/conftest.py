"""
Pytest configuration file for the test suite.
"""

import pytest
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def pytest_configure(config):
    """Configure pytest-asyncio to use strict mode."""
    config.inicfg['asyncio_mode'] = 'strict' 