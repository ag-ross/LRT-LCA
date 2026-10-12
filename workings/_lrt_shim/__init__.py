"""Numpy-only fallback for scipy.linalg.expm, used when scipy is absent."""

from . import linalg

__all__ = ["linalg"]
