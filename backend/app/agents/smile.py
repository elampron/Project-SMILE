"""
SMILE agent module.

This module provides backward compatibility by re-exporting the Smile class
from the new modular structure.
"""

from .smile.agent import Smile

__all__ = ['Smile']


