"Quasi-degenerate perturbation theory"

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .block_diagonalization import (
    block_diagonalize,
)
from . import series

__all__ = [
    "block_diagonalize",
    "series",
    "__version__",
    "__version_tuple__",
]
