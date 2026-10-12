"""Minimal numpy-only fallback for scipy.linalg.expm."""

from __future__ import annotations

import numpy as np

__all__ = ["expm"]


def expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential via eigendecomposition, with a power-series fallback."""
    A = np.asarray(A, dtype=complex)
    vals, vecs = np.linalg.eig(A)
    try:
        result = vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs)
    except np.linalg.LinAlgError:
        result = np.eye(A.shape[0], dtype=complex)
        term   = np.eye(A.shape[0], dtype=complex)
        for k in range(1, 21):
            term   = term @ A / k
            result = result + term

    real_scale = max(np.max(np.abs(result.real)), 1e-30)
    if np.max(np.abs(result.imag)) < 1e-10 * real_scale:
        return result.real
    return result
