"""Type hints for the SMILE agent."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Smile

__all__ = ["Smile"] 