"""Provenance log for every manuscript-quoted value, written to outputs/diagnostics.log."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from scipy.linalg import expm as _expm
except ImportError:
    from _lrt_shim.linalg import expm as _expm  # type: ignore[no-redef]

from data import (
    GROUPS,
    SECTOR_NAMES,
    IOSystem,
    T1_LONG,
    T1_SHORT,
    _code_to_macro,
    _load_matrices,
    DATA_PATH,
    build_io_system,
)
from lrt_lca import (
    implied_perturbation,
    perturbation_budget,
    spectral_radius,
    susceptibility_matrix,
    type1_multiplier,
)

TRANSITION_KEYS = ("T1", "T2", "T3", "T4")

# Fine IRF lag grid and budget-horizon grid.
IRF_TAU_MAX: float = 10.0
IRF_DT: float = 0.002
HORIZON_GRID: np.ndarray = np.linspace(0.1, 10.0, 199)
STEP_HORIZON: float = 10.0


def _irf_curve(M: np.ndarray, b: np.ndarray, dz: np.ndarray, t1: float):
    """Return (taus, g) for the scalar GHG IRF on the fine lag grid."""
    u = implied_perturbation(M, dz, t1)
    taus = np.arange(0.0, IRF_TAU_MAX + IRF_DT / 2.0, IRF_DT)
    E = _expm(M.T * IRF_DT)
    g = np.empty(len(taus))
    r = u.copy()
    g[0] = float(b @ r)
    for k in range(1, len(taus)):
        r = E @ r
        g[k] = float(b @ r)
    return taus, g


def _first_zero_crossing(taus: np.ndarray, y: np.ndarray) -> float | None:
    """Return the lag of the first sign change of y, or None."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    return float(taus[idx[0] + 1]) if len(idx) else None


def write_diagnostics(
    io: IOSystem | None = None,
    out_path: Path | None = None,
) -> Path:
    """Compute all manuscript-quoted diagnostics and write the log."""
    if io is None:
        io = build_io_system()
    if out_path is None:
        out_path = Path(__file__).parent / "outputs" / "diagnostics.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    A, b, M = io.A_io, io.B.ravel(), io.M
    N = io.N
    Lam = type1_multiplier(A)
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    # spectrum
    eigs = np.linalg.eigvals(M)
    lam0 = eigs[np.argmin(np.abs(eigs.real))]
    relax = -1.0 / lam0.real
    w("[spectrum]")
    w(f"rho_A_io = {spectral_radius(A):.4f}")
    w(f"slowest_relaxation_yr = {relax:.4f}")
    w(f"t1_4yr_over_relaxation = {T1_LONG / relax:.3f}")
    w(f"complex_eigenvalues = {int(np.sum(np.abs(eigs.imag) > 1e-10))} of {N}")
    w()

    w("[sales_ratio]  (intermediate use per unit of domestic output, imports included)")
    for i, s in enumerate(SECTOR_NAMES):
        w(f"{s:14s} = {float(A[i, :] @ io.X) / io.X[i]:.4f}")
    w()
    w("[intensities]  (B_i, Mt CO2-eq per unit of normalised sector output)")
    for i, s in enumerate(SECTOR_NAMES):
        w(f"{s:14s} = {b[i]:.2f}")
    w()

    # aggregation coverage
    A_df, _, _, _ = _load_matrices(DATA_PATH)
    codes = list(A_df.columns)
    dropped = [c for c in codes if _code_to_macro(c) not in SECTOR_NAMES]
    w("[aggregation]")
    w(f"useeio_sectors = {len(codes)}")
    w(f"kept_sectors = {len(codes) - len(dropped)}")
    w(f"excluded_sectors = {len(dropped)}")
    w(f"excluded_codes = {', '.join(sorted(dropped))}")
    w()

    # Monte Carlo draw survival (seeded, deterministic)
    from figures import N_MC, _mc_draws
    w("[monte_carlo]")
    w(f"stable_draws = {len(_mc_draws(io))} of {N_MC}")
    w()

    # accumulation
    w("[accumulation]  (acc_i(t1) = (B.P)_i / (B.Lambda)_i, per cent)")
    for t1 in (T1_SHORT, T1_LONG):
        acc = (b @ susceptibility_matrix(M, t1)) / (b @ Lam)
        w(
            f"t1={t1:g}: min = {100 * acc.min():.2f} "
            f"({SECTOR_NAMES[int(np.argmin(acc))]}), "
            f"max = {100 * acc.max():.2f} "
            f"({SECTOR_NAMES[int(np.argmax(acc))]}), "
            f"mean = {100 * acc.mean():.2f}"
        )
    w()

    # transitions
    w("[transitions]  (primary horizon t1 = 4 yr)")
    for key in TRANSITION_KEYS:
        dz = io.delta_z(key)
        bud = perturbation_budget(M, dz, T1_LONG)
        e = float(b @ dz)
        taus, g = _irf_curve(M, b, dz, T1_LONG)
        wnd = taus <= T1_LONG + IRF_DT / 2.0
        integral = float(np.trapezoid(g[wnd], taus[wnd])) if hasattr(np, "trapezoid") \
            else float(np.trapz(g[wnd], taus[wnd]))
        w(
            f"{key}: budget = {bud:.4f}, B.dz_Mt = {e:.2f}, "
            f"norm_dz = {np.linalg.norm(dz):.4f}, "
            f"budget_per_Mt = {bud / e:.5f}, "
            f"integral_identity_relerr = {abs(integral - e) / abs(e):.2e}"
        )
    for key in TRANSITION_KEYS:
        p = io.transitions[key].demand_shift
        dz = io.delta_z(key)
        idle = np.flatnonzero(p == 0.0)
        j = int(idle[int(np.argmax(np.abs(dz[idle])))])
        w(
            f"{key}_realised: "
            + ", ".join(f"{SECTOR_NAMES[i]}={dz[i]:+.3f}" for i in np.flatnonzero(p))
            + f"; largest_unshifted = {SECTOR_NAMES[j]}={dz[j]:+.3f}"
            + f", sum_dz = {float(dz.sum()):+.3f}"
        )
    for t1 in (T1_SHORT, T1_LONG):
        buds = {k: perturbation_budget(M, io.delta_z(k), t1)
                for k in TRANSITION_KEYS}
        w(f"budgets t1={t1:g}: "
          + ", ".join(f"{k}={v:.4f}" for k, v in buds.items())
          + "  rank: " + " < ".join(sorted(buds, key=buds.__getitem__)))
    w("budget_long_run: " + ", ".join(
        f"{k}={float(np.linalg.norm(M.T @ io.delta_z(k))):.4f}"
        for k in TRANSITION_KEYS))
    w("full_area_over_B_dz t1=4: " + ", ".join(
        f"{k}={float(b @ Lam @ implied_perturbation(M, io.delta_z(k), T1_LONG)) / float(b @ io.delta_z(k)):.3f}"
        for k in TRANSITION_KEYS))
    w()

    # IRF features
    w("[irf_features]  (scalar GHG IRF, Mt CO2-eq/yr; crossings on [0, 10])")
    for t1 in (T1_SHORT, T1_LONG):
        for key in TRANSITION_KEYS:
            dz = io.delta_z(key)
            taus, g = _irf_curve(M, b, dz, t1)
            wnd = taus <= t1 + IRF_DT / 2.0
            cross = _first_zero_crossing(taus, g)
            w(
                f"t1={t1:g} {key}: g(0) = {g[0]:.1f}, "
                f"min_on_window = {g[wnd].min():.3f}, "
                f"first_zero_crossing_yr = "
                f"{'none' if cross is None else f'{cross:.2f}'}"
            )
    sweep = []
    for t1 in (0.1, 1.0, 4.0, 10.0):
        taus, g = _irf_curve(M, b, io.delta_z("T1"), t1)
        cross = _first_zero_crossing(taus, g)
        sweep.append(f"t1={t1:g}: {'none' if cross is None else f'{cross:.2f}'}")
    w("T1_crossing_by_horizon_yr: " + ", ".join(sweep))
    w()

    # grid scan to bracket features, then refine so logged values are grid-independent
    w("[budget_horizon]  (scanned 0.1..10 yr; rises below 1e-4 relative ignored; minima and crossings refined)")
    dzs = {k: io.delta_z(k) for k in TRANSITION_KEYS}
    curves = {
        k: np.array([perturbation_budget(M, dzs[k], float(t))
                     for t in HORIZON_GRID])
        for k in TRANSITION_KEYS
    }

    def _bud(key: str, t: float) -> float:
        return perturbation_budget(M, dzs[key], float(t))

    try:
        from scipy.optimize import brentq, minimize_scalar
    except ImportError:
        brentq = minimize_scalar = None  # type: ignore[assignment]

    for key, c in curves.items():
        d = np.diff(c)
        if np.any(d > 1e-4 * c[:-1]):
            i0 = int(np.argmax(d > 1e-4 * c[:-1]))
            lo = HORIZON_GRID[max(i0 - 1, 0)]
            hi = HORIZON_GRID[min(i0 + 1, len(HORIZON_GRID) - 1)]
            if minimize_scalar is not None:
                res = minimize_scalar(lambda t: _bud(key, t),
                                      bounds=(lo, hi), method="bounded")
                t_min, b_min = float(res.x), float(res.fun)
            else:
                t_min, b_min = float(HORIZON_GRID[i0]), float(c[i0])
            rise = 100.0 * (_bud(key, 10.0) - b_min) / b_min
            w(f"{key}: monotone = no, min_at_t1 = {t_min:.2f}, "
              f"budget_at_min = {b_min:.4f}, rise_to_10yr_pct = {rise:.1f}")
        else:
            w(f"{key}: monotone = yes")
    for a, bb in (("T1", "T2"), ("T1", "T4"), ("T2", "T4")):
        diff = curves[a] - curves[bb]
        sw = np.where(np.diff(np.sign(diff)))[0]
        if len(sw) == 0:
            w(f"ordering_crossing {a} vs {bb}: t1 = none")
            continue
        vals = []
        for i in sw:
            lo, hi = float(HORIZON_GRID[i]), float(HORIZON_GRID[i + 1])
            if brentq is not None:
                t_c = float(brentq(lambda t: _bud(a, t) - _bud(bb, t), lo, hi))
            else:
                t_c = 0.5 * (lo + hi)
            vals.append(f"{t_c:.2f}")
        w(f"ordering_crossing {a} vs {bb}: t1 = {', '.join(vals)}")
    w()

    w("[forward_vs_backward]  (t1 = 4 yr, fractional units)")
    P4 = susceptibility_matrix(M, T1_LONG)
    P4_bwd = P4.T * io.X[np.newaxis, :] / io.X[:, np.newaxis]
    bwd = {}
    for key in TRANSITION_KEYS:
        dz = io.delta_z(key)
        u_fwd = np.linalg.solve(P4, dz)
        u_bwd = np.linalg.solve(P4_bwd, dz)
        n_fwd = float(np.linalg.norm(u_fwd))
        bwd[key] = float(np.linalg.norm(u_bwd))
        w(f"{key}: forward = {n_fwd:.4f}, backward = {bwd[key]:.4f}, "
          f"rel_diff_pct = {100 * abs(n_fwd - bwd[key]) / n_fwd:.1f}, "
          f"onset_forward = {float(b @ u_fwd):.1f}, onset_backward = {float(b @ u_bwd):.1f}")
    w("backward_rank: " + " < ".join(sorted(bwd, key=bwd.__getitem__)))
    w()

    # asymmetry
    from asymmetry_check import asymmetry_budgets
    outbound, ret_m, ret_mp = asymmetry_budgets(io, T1_LONG)
    w("[asymmetry]  (transition T1, t1 = 4 yr)")
    w(f"outbound = {outbound:.4f}")
    w(f"return_at_M = {ret_m:.4f}")
    w(f"return_at_M_displaced = {ret_mp:.4f}")
    w(f"asymmetry_pct = {100.0 * (ret_mp - ret_m) / ret_m:+.2f}")
    w("asymmetry_pct_by_coefficient_change: " + ", ".join(
        f"d={d:g}: {100.0 * (asymmetry_budgets(io, T1_LONG, d=d)[2] - ret_m) / ret_m:+.2f}"
        for d in (0.05, 0.10, 0.30, 0.40, 0.50)))
    w()

    # step-response horizon accounting
    w(f"[step_accounting]  (T = {STEP_HORIZON:g} yr)")
    trap = getattr(np, "trapezoid", None) or np.trapz
    for key in TRANSITION_KEYS:
        dz = io.delta_z(key)
        taus = np.arange(0.0, STEP_HORIZON + IRF_DT / 2.0, IRF_DT)
        E = _expm(M.T * IRF_DT)
        gs = np.empty(len(taus))
        r = dz.copy()
        gs[0] = float(b @ (dz - r))
        for k in range(1, len(taus)):
            r = E @ r
            gs[k] = float(b @ (dz - r))
        integral = float(trap(gs, taus))
        P_T = susceptibility_matrix(M, STEP_HORIZON)
        expected = float(b @ dz) * STEP_HORIZON - float(b @ (P_T @ dz))
        w(f"{key}: integral = {integral:.4f}, closed_form = {expected:.4f}, "
          f"relerr = {abs(integral - expected) / abs(expected):.2e}, "
          f"transient_limit_B_Lambda_dz = {float(b @ (Lam @ dz)):.4f}")
    w()

    # FIG_03b routing onsets
    w("[fig03b_routing]  (T3, t1 = 4 yr; onsets sum_{i in group} B_i u_i)")
    dz3 = io.delta_z("T3")
    u3 = implied_perturbation(M, dz3, T1_LONG)
    w(f"total_onset_B_u = {float(b @ u3):.3f}")
    taus3, g3 = _irf_curve(M, b, dz3, T1_LONG)
    cross3 = _first_zero_crossing(taus3, g3)
    w(f"total_zero_crossing_yr = {'none' if cross3 is None else f'{cross3:.2f}'}, "
      f"total_peak = {g3.max():.2f} at tau = {taus3[int(np.argmax(g3))]:.2f}")
    for gkey, sl in GROUPS.items():
        idx = list(range(*sl.indices(N)))
        w(f"group_{gkey}_onset = {float(b[idx] @ u3[idx]):+.3f}")
    # displaced sectors' forcings, then the Utilities row of P.u = dz decomposed by source
    for name, i in (("Machinery", 10), ("ProfServices", 16),
                    ("OtherServices", 19), ("Utilities", 5)):
        w(f"u_{name} = {u3[i]:+.6f}")
    w(f"utilities_onset_B_u = {b[5] * u3[5]:+.3f}")
    P4b = susceptibility_matrix(M, T1_LONG)
    contrib = P4b[5, :] * u3
    w("utilities_row_of_P.u_by_source: "
      f"from_ProfServices = {contrib[16]:+.6f}, "
      f"from_OtherServices = {contrib[19]:+.6f}, "
      f"from_Machinery = {contrib[10]:+.6f}, "
      f"own = {contrib[5]:+.6f}, "
      f"row_total = {float(contrib.sum()):+.6f}")
    taus = np.arange(0.0, IRF_TAU_MAX + IRF_DT / 2.0, IRF_DT)
    E = _expm(M.T * IRF_DT)
    e_idx = list(range(*GROUPS["E"].indices(N)))
    gE = np.empty(len(taus))
    r = u3.copy()
    gE[0] = float(b[e_idx] @ r[e_idx])
    for k in range(1, len(taus)):
        r = E @ r
        gE[k] = float(b[e_idx] @ r[e_idx])
    cross = _first_zero_crossing(taus, gE)
    w(f"energy_group_zero_crossing_yr = "
      f"{'none' if cross is None else f'{cross:.2f}'}, "
      f"energy_group_peak = {gE.max():.2f} at tau = {taus[int(np.argmax(gE))]:.2f}")
    w()

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"diagnostics written to {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    write_diagnostics()
