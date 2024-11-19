"""Quasi-degenerate perturbation theory"""

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from . import series
from .block_diagonalization import (
    block_diagonalize, operator_to_BlockSeries
)

__all__ = [
    "block_diagonalize",
    "operator_to_BlockSeries",
    "series",
    "__version__",
    "__version_tuple__",
]
