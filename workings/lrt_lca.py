"""Closed-form LRT-LCA dynamics: M = A_io - I, Lambda = -M^-T, P(t1) = M^-T(exp(M^T t1) - I)."""

from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import expm as _expm
except ImportError:
    from _lrt_shim.linalg import expm as _expm  # type: ignore[no-redef]


STABILITY_TOL: float = 1e-10


def check_stable(M: np.ndarray, context: str = "") -> None:
    eigs = np.linalg.eigvals(M)
    if not np.all(eigs.real < -STABILITY_TOL):
        worst = float(np.max(eigs.real))
        tag = f" [{context}]" if context else ""
        raise RuntimeError(
            f"Drift matrix M is not stable (max Re(λ) = {worst:.6g}).{tag}"
        )


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def eigenvalues_M(M: np.ndarray) -> np.ndarray:
    """Eigenvalues sorted by Re ascending, fastest mode first."""
    eigs = np.linalg.eigvals(M)
    return eigs[np.argsort(eigs.real)]


def relaxation_timescales(M: np.ndarray) -> np.ndarray:
    """-1/Re per eigenvalue, slowest to fastest, in years."""
    eigs = np.linalg.eigvals(M)
    idx = np.argsort(np.abs(eigs.real))
    return -1.0 / eigs.real[idx]


def leontief_inverse(A_io: np.ndarray) -> np.ndarray:
    """L = (I - A_io)^-1 = -M^-1."""
    N = A_io.shape[0]
    M = A_io - np.eye(N)
    check_stable(M, "leontief_inverse")
    return np.linalg.solve(M, -np.eye(N))


def type1_multiplier(A_io: np.ndarray) -> np.ndarray:
    """Lambda = L^T = -M^-T, the Type I multiplier."""
    N = A_io.shape[0]
    M = A_io - np.eye(N)
    check_stable(M, "type1_multiplier")
    return np.linalg.solve(M.T, -np.eye(N))


def susceptibility_matrix(M: np.ndarray, t1: float) -> np.ndarray:
    """P(t1) = M^-T(exp(M^T t1) - I); t1*I as t1->0, Lambda as t1->inf."""
    check_stable(M, "susceptibility_matrix")
    N = M.shape[0]
    Mt = M.T
    return np.linalg.solve(Mt, _expm(Mt * t1) - np.eye(N))


def implied_perturbation(M: np.ndarray, delta_z: np.ndarray, t1: float) -> np.ndarray:
    """u~ = (exp(M^T t1) - I)^-1 M^T dz, the constant forcing giving dz over [0, t1]."""
    check_stable(M, "implied_perturbation")
    Mt = M.T
    return np.linalg.solve(_expm(Mt * t1) - np.eye(M.shape[0]), Mt @ delta_z)


def perturbation_budget(M: np.ndarray, delta_z: np.ndarray, t1: float) -> float:
    """||u~||_2, the network-weighted structural effort of dz."""
    return float(np.linalg.norm(implied_perturbation(M, delta_z, t1)))


def impulse_response(
    M: np.ndarray,
    epsilon: np.ndarray,
    tau_max: float,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """R[k] = exp(M^T taus[k]) epsilon, stepped by the semigroup so one expm covers the grid."""
    check_stable(M, "impulse_response")
    taus = np.arange(0.0, tau_max + dt / 2.0, dt)
    T, N = len(taus), M.shape[0]
    exp_dt = _expm(M.T * dt)
    R = np.zeros((T, N))
    R[0] = epsilon
    for k in range(1, T):
        R[k] = exp_dt @ R[k - 1]
    return taus, R


def unit_impulse_response(
    M: np.ndarray,
    sector_idx: int,
    tau_max: float,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    N = M.shape[0]
    ei = np.zeros(N)
    ei[sector_idx] = 1.0
    return impulse_response(M, ei, tau_max, dt)


def step_response(
    M: np.ndarray,
    delta_z: np.ndarray,
    tau_max: float,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """S[k] = (I - exp(M^T taus[k])) dz, rising from 0 to dz."""
    check_stable(M, "step_response")
    taus, E = impulse_response(M, delta_z, tau_max, dt)   # E[k] = exp(M^T tau_k) dz
    return taus, delta_z[np.newaxis, :] - E


def emission_irf(
    B: np.ndarray,
    M: np.ndarray,
    epsilon: np.ndarray,
    tau_max: float,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """g[k] = B . R(taus[k]); with epsilon = u~ for dz at t1, g integrates to B.dz over [0, t1]."""
    taus, R = impulse_response(M, epsilon, tau_max, dt)
    return taus, R @ B


def emission_step(
    B: np.ndarray,
    M: np.ndarray,
    delta_z: np.ndarray,
    tau_max: float,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """g_step[k] = B . S(taus[k]), rising from 0 to B.dz."""
    taus, S = step_response(M, delta_z, tau_max, dt)
    return taus, S @ B


def irf_norm_trajectory(
    M: np.ndarray,
    epsilon: np.ndarray,
    tau_max: float,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    taus, R = impulse_response(M, epsilon, tau_max, dt)
    return taus, np.linalg.norm(R, axis=1)


def minimum_effort_delta_z(B: np.ndarray, target_emission_reduction: float) -> np.ndarray:
    """Minimum-norm dz solving B.dz = -target: dz = B * (-target) / (B . B)."""
    denom = float(B @ B)
    if denom < 1e-14:
        raise ValueError("B is zero; pseudoinverse is undefined.")
    return B * (-target_emission_reduction / denom)


def pseudoinverse_delta_z(B: np.ndarray, delta_g_target: np.ndarray) -> np.ndarray:
    """dz = B^+ dg_target (Moore-Penrose) for an m x n intensity matrix B."""
    B = np.atleast_2d(np.asarray(B, dtype=float))
    delta_g_target = np.atleast_1d(np.asarray(delta_g_target, dtype=float))
    if B.shape[0] != delta_g_target.shape[0]:
        raise ValueError(
            f"Shape mismatch: B has {B.shape[0]} impact categories but "
            f"delta_g_target has length {delta_g_target.shape[0]}."
        )
    return np.linalg.pinv(B) @ delta_g_target
