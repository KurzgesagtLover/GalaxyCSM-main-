"""Galactic Chemical Evolution (GCE) Module.

Computes element abundance tables A_X(r, t) as a function of
galactocentric radius and cosmic time since the Big Bang.
"""
from .solver import GCESolver
from .config import ELEMENTS, DEFAULT_PARAMS
