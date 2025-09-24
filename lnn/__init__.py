"""Lagrangian Neural Networks package."""

# Core functionality
from .core import lagrangian_eom, unconstrained_eom
from .models import MLP, MLPAutoencoder
from .utils import wrap_coords, rk4_step
from .plotting import get_args

__version__ = "0.1.0"

__all__ = [
    "lagrangian_eom",
    "unconstrained_eom",
    "MLP",
    "MLPAutoencoder",
    "wrap_coords",
    "rk4_step",
    "get_args",
]