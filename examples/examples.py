#!/usr/bin/env python3
# examples.py: the LRT-LCA analytical objects on a fabricated toy system of any size.
#
# All numbers are fabricated for illustration, not data. The paper's USEEIO
# results are produced by ../replicate.py, NOT by this file!
#
# To apply the framework to another system, edit the toy-system block and the
# transitions beneath it. The blocks run for any number of sectors, provided the
# system is productive and stable.
#
# Requires numpy, scipy, and matplotlib:
#   python3 examples/examples.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import matplotlib
matplotlib.use("Agg")   # file output only, no interactive display
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


# --- Console output copied to a text file. -----------------------------------
class Tee:
    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file_handle = file_handle

    def write(self, text):
        self.stream.write(text)
        self.file_handle.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file_handle.flush()


_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def show(name, obj):
    if np.isscalar(obj) or (hasattr(obj, "ndim") and obj.ndim == 0):
        print(f"{name} = {float(obj):.4f}")
    else:
        print(f"{name} =")
        print(np.asarray(obj))


# --- Toy system (replace to apply the framework to another case) -------------
# A_io[i, j] is the monetary input from sector i per unit output of sector j,
# X_REF the reference outputs, demand-shift entries fractions of own-sector
# reference output, and B_i the reference annual emission of sector i. B is a single
# pre-aggregated GHG row, so Q of Block 7 is 1x1. In general B is m x N and Q is p x m.
# FORM is "forward" (K = M^T, the paper) or "backward" (K = D^-1 M D). See README.
#
# The embedding is agnostic to the provenance of A_io. That is, any square
# technical-coefficients matrix admits the same construction, whether drawn from
# a single-region national table or a multi-regional system, and whether sourced
# from databases such as WIOD, FIGARO, ADB, EXIOBASE, GLORIA, EORA, or GTAP.

SECTORS = ["Energy", "Manufacturing", "Services"]

A_io = np.array([
    [0.53, 0.48, 0.05],
    [0.07, 0.36, 0.67],
    [0.04, 0.09, 0.25],
])

B = np.array([0.25, 1.00, 0.10])   # reference annual GHG emissions by sector, t CO2-eq
X_REF = np.array([2.4, 1.7, 1.0])   # reference outputs, monetary units
FORM = "forward"

N = len(SECTORS)
I = np.eye(N)
assert A_io.shape == (N, N) and B.shape == (N,) and X_REF.shape == (N,), \
    "A_io, B and X_REF must match the number of SECTORS"
assert FORM in ("forward", "backward"), 'FORM must be "forward" or "backward"'
assert np.all(X_REF > 0), "X_REF must be positive"
assert np.max(np.abs(np.linalg.eigvals(A_io))) < 1.0, "A_io is not productive"


def vec(entries):
    """Length-N vector from a {sector name: value} mapping, zero elsewhere."""
    v = np.zeros(N)
    for name, value in entries.items():
        assert name in SECTORS, f"{name!r} is not in SECTORS: restate the transitions by sector name"
        v[SECTORS.index(name)] = value
    return v


def displacement(A, p):
    """Fractional output displacement of a final-demand shift p, dz = L (p X_REF) / X_REF."""
    return np.linalg.solve(I - A, p * X_REF) / X_REF


def unit(v):
    """v rescaled to unit Euclidean norm."""
    return v / np.linalg.norm(v)


T1_PRIMARY = 4.0    # calibration horizon t1, years
T1_SHORT = 1.0      # short comparison horizon, years
T_ACC = 6.0         # horizon-accounting window for the step response, years
T_FAR = 40.0        # far horizon where B P(T) dz -> B Lambda dz, years
PLANNING_HORIZON = 8.0   # upper bound on t1 for the block 12 horizon sweep, years

# Transitions shared by more than one block, each stated as a shift in final demand.
DEMAND_SHIFTS = {
    "contract Energy":  vec({"Energy": -0.20, "Manufacturing": 0.20}),
    "repair mix":       vec({"Manufacturing": -0.15, "Services": 0.15}),
    "expand Services":  vec({"Energy": 0.10, "Manufacturing": -0.20, "Services": 0.10}),
}
SCENARIO = "contract Energy"   # scenario of blocks 5 to 8, and the transition of the sweeps
REPAIR = "repair mix"          # scenario of blocks 10 and 11
DELTA_Z = displacement(A_io, DEMAND_SHIFTS[SCENARIO])
DZ_REPAIR = displacement(A_io, DEMAND_SHIFTS[REPAIR])   # output-mix repair
TARGET_REDUCTION = 0.10                     # emission-space target
DELTA_B = vec({"Manufacturing": -0.30})   # change in reference annual emissions
RECYCLE = {("Energy", "Manufacturing"): -0.20, ("Services", "Manufacturing"): 0.10}

# Candidate transitions for the block 12 sweeps, as unit-norm directions.
TRANSITIONS = {name: unit(displacement(A_io, p)) for name, p in DEMAND_SHIFTS.items()}


# --- Monte Carlo draws (from workings/figures.py) ----------------------------
# Off-diagonal A_io entries carry multiplicative noise, the diagonal is fixed,
# and b carries Gaussian noise. Draws with spectral radius >= 1 or unstable M
# are discarded. One shared set of accepted draws feeds every block.

N_MC = 400
MC_SEED = 12345
A_NOISE = 0.10      # off-diagonal A_io noise, uniform on [-0.10, 0.10]
B_NOISE = 0.15      # sector-emission noise, Gaussian, one standard deviation
PCT_LO, PCT_HI = 5, 95   # percentile band edges
MC_MIN_ACCEPT = 100      # below this the bands are unreliable and a warning is printed


def drift(A, form=None):
    """Propagation matrix on fractional deviations for the chosen form."""
    if (form or FORM) == "forward":
        return A.T - I
    return A * X_REF[np.newaxis, :] / X_REF[:, np.newaxis] - I


def _mc_draws(A, b, n_mc=N_MC, seed=MC_SEED, a_noise=A_NOISE, b_noise=B_NOISE):
    """Accepted draws as dicts A_mc, K_mc, b_mc. A draw is kept only if A_mc has
    spectral radius < 1 and M_mc is strictly stable."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_mc):
        noise = 1.0 + a_noise * rng.uniform(-1.0, 1.0, size=(N, N))
        np.fill_diagonal(noise, 1.0)
        A_mc = A * noise
        if np.max(np.abs(np.linalg.eigvals(A_mc))) >= 1.0:
            continue
        M_mc = A_mc - I
        if not np.all(np.linalg.eigvals(M_mc).real < -1e-10):
            continue
        b_mc = np.clip(b * (1.0 + b_noise * rng.standard_normal(N)), 0.0, None)
        draws.append({"A": A_mc, "K": drift(A_mc), "b": b_mc})
    return draws


def _implied(K, dz, t1):
    """u = (exp(K t1) - I)^-1 K dz, equal to P(t1)^-1 dz. SM forcing."""
    return np.linalg.solve(expm(K * t1) - I, K @ dz)


def _irf(K, u0, taus):
    """R[k] = exp(K tau_k) u0 on a uniform tau grid, by the semigroup step.

    The errstate guard suppresses spurious floating-point flag warnings.
    """
    step = expm(K * (taus[1] - taus[0]))
    R = np.zeros((len(taus), N))
    R[0] = u0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for k in range(1, len(taus)):
            R[k] = step @ R[k - 1]
    return R


def _band(runs):
    """(lo, hi) 5th and 95th percentiles across a list of MC arrays."""
    arr = np.array(runs)
    return np.nanpercentile(arr, PCT_LO, axis=0), np.nanpercentile(arr, PCT_HI, axis=0)


def _grid(t_max, dt):
    """Inclusive time grid on [0, t_max] at step dt, keeping the endpoint."""
    return np.arange(0.0, t_max + dt / 2.0, dt)


# --- Local plotting style ----------
# Okabe and Ito colourblind-safe palette. Source: https://jfly.uni-koeln.de/color/
_OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


# Per-sector emission measures of manuscript Table 2, matching workings/tables.py.
MULT_KEYS = ("direct", "standard", "type1")
MULT_LABELS = {
    "direct":   "direct (B_i)",
    "standard": "IO-LCA (B.L)",
    "type1":    "LRT-LCA (B.Lam)",
}
MULT_COLORS = {"direct": _OKABE_ITO[0], "standard": _OKABE_ITO[1],
               "type1": _OKABE_ITO[2]}
FIGSIZE = (5.2, 3.3)
TAU_MAX = 14.0         # lag grid for the IRF figures (years)
TAU_MAX_DECOMP = 8.0   # shorter grid for the block 10 decomposition (years)
DT_FINE = 0.01         # fine step for the deterministic curves (years)
DT_MC = 0.05           # coarser step for the MC ribbons (years)

FS_TITLE  = 10   # axes title
FS_LABEL  = 9    # axis labels
FS_LEGEND = 8    # legend entries
FS_TICK   = 8    # categorical tick labels
FS_ANNOT  = 7    # inline annotations

_FIGURES = []


def _new_ax(figsize=FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _style(ax, title, xlabel, ylabel):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.15)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)


def _zero_line(ax):
    ax.axhline(0.0, color="0.6", lw=0.7, ls="--", zorder=1)


def _ribbon(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.20, lw=0, zorder=2)


def _save(fig, stem):
    fig.savefig(_OUTPUT_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    _FIGURES.append(f"{stem}.png")
    plt.close(fig)


def susceptibility(K, t1):
    """P(t1) = K^-1 (exp(K t1) - I), with K = M^T in the forward form. Eq. (6)."""
    return np.linalg.solve(K, expm(K * t1) - I)


def trajectory(K, u0, tau_max, dt=DT_FINE):
    """(taus, R) with R[k] = exp(K tau_k) u0, stepped by R[k] = exp(K dt) R[k-1]."""
    taus = _grid(tau_max, dt)
    step = expm(K * dt)
    R = np.zeros((len(taus), N))
    R[0] = u0
    for k in range(1, len(taus)):
        R[k] = step @ R[k - 1]
    return taus, R


def project(R, vec):
    """R @ vec for a (T x N) trajectory. The errstate guard suppresses spurious
    floating-point flag warnings."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return R @ vec


def main():
    """Run every block, teeing console output to outputs/examples_output.txt."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = open(_OUTPUT_DIR / "examples_output.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, output_file)
    try:
        print("LRT-LCA toy example (all numbers fabricated for illustration)")

        MC_DRAWS = _mc_draws(A_io, B)
        print(f"Monte Carlo: {len(MC_DRAWS)} of {N_MC} draws accepted "
              f"(seed {MC_SEED}, off-diagonal A noise {A_NOISE}, B noise {B_NOISE}).")
        if len(MC_DRAWS) < MC_MIN_ACCEPT:
            print(f"  warning: fewer than {MC_MIN_ACCEPT} draws accepted, so the "
                  "uncertainty bands are unreliable. Reduce A_NOISE or check stability.")

        print(f"\nScenario '{SCENARIO}': a shift p in final demand, with the displacement")
        print("Delta z = L (p X_REF) / X_REF it implies. Every sector carries a displacement.")
        show("p", DEMAND_SHIFTS[SCENARIO])
        show("Delta z", DELTA_Z)


        # --- [1] Drift matrix and stability. Eq. (5). --------------------------------
        print("\n[1] Drift matrix M = A_io - I. Eq. (5).")
        M = A_io - I
        show("A_io", A_io)
        show("M", M)
        K = drift(A_io)
        print(f"form = {FORM}: objects are built on K = "
              f"{'M^T' if FORM == 'forward' else 'D^-1 M D'}.")

        rho = np.max(np.abs(np.linalg.eigvals(A_io)))
        eigM = np.linalg.eigvals(M)
        timescales = -1.0 / eigM.real
        print(f"spectral radius rho(A_io) = {rho:.4f} (stable iff < 1)")
        print(f"eigenvalues Re(lambda) of M = {np.sort(eigM.real)}")
        print(f"relaxation timescales 1/|Re(lambda)| = {np.sort(timescales)[::-1]} yr")

        assert rho < 1.0, "A_io is not productive: spectral radius of A_io >= 1"
        assert np.all(eigM.real < 0), "M is not stable: an eigenvalue has non-negative real part"


        # --- [2] Susceptibility matrix. Eq. (6). Analytical object 1. ----------------
        print("\n[2] Susceptibility P(t1) = K^-1 (exp(K t1) - I). Eq. (6).")
        print("Short-horizon limit P(t1) -> t1 I:")
        show("P(0.1)", susceptibility(K, 0.1))
        show(f"P({T1_PRIMARY})", susceptibility(K, T1_PRIMARY))


        # --- [3] Type I LRT-LCA multiplier and hotspot rankings. Eq. (7). Object 2. --
        print("\n[3] Type I LRT-LCA multiplier Lambda = -K^-1 (L^T in the forward form). Eq. (7).")
        L = np.linalg.inv(I - A_io)          # Leontief inverse, Eq. (2)
        Lam = -np.linalg.inv(K)              # L^T in the forward form
        show("L", L)
        show("Lambda", Lam)
        print(f"asymmetry ||Lambda - L||_F = {np.linalg.norm(Lam - L):.4f}"
              + (" (0 iff A_io symmetric)" if FORM == "forward" else ""))

        # Manuscript Table 2 measures.
        direct   = B          # B_i: direct annual GHG emission
        std_mult = ((B / X_REF) @ L) * X_REF   # (B.L)_i in the normalised units of B
        lrt_mult = B @ Lam    # (B.Lambda)_{.i}: Type I LRT-LCA multiplier
        # (B.Lam)/B_i amplification, undefined where a sector has zero direct emission.
        with np.errstate(divide="ignore", invalid="ignore"):
            amplification = np.where(direct > 0.0, lrt_mult / direct, np.nan)
        print("  columns: B_i direct emission, (B.L) standard IO-LCA multiplier,")
        print("  (B.Lam) Type I LRT-LCA multiplier, ratio = (B.Lam)/B_i amplification")
        print(f"\n  {'sector':14s}{'B_i':>8s}{'(B.L)':>9s}{'(B.Lam)':>10s}{'ratio':>11s}")
        for i in range(N):
            ratio = f"{amplification[i]:11.2f}" if np.isfinite(amplification[i]) else f"{'n/a':>11s}"
            print(f"  {SECTORS[i]:14s}{direct[i]:8.3f}{std_mult[i]:9.3f}{lrt_mult[i]:10.3f}{ratio}")
        def order(v):
            return [SECTORS[i] for i in np.argsort(-v)]
        print(f"  ranking by direct emission B_i    = {order(direct)}")
        print(f"  ranking by standard IO-LCA (B.L)  = {order(std_mult)}")
        print(f"  ranking by Type I LRT-LCA (B.Lam) = {order(lrt_mult)}")

        det_foot = {"direct": direct, "standard": std_mult, "type1": lrt_mult}
        mc_foot = {k: [] for k in MULT_KEYS}
        for d in MC_DRAWS:
            L_mc = np.linalg.inv(I - d["A"])
            b_mc = d["b"]
            mc_foot["direct"].append(b_mc)
            mc_foot["standard"].append(((b_mc / X_REF) @ L_mc) * X_REF)
            mc_foot["type1"].append(b_mc @ -np.linalg.inv(d["K"]))
        fig, ax = _new_ax()
        x = np.arange(N)
        width = 0.26
        for j, k in enumerate(MULT_KEYS):
            lo_f, hi_f = _band(mc_foot[k])
            xr = x + (j - 1) * width
            ax.bar(xr, det_foot[k], width, color=MULT_COLORS[k], alpha=0.85, zorder=3,
                   label=MULT_LABELS[k])
            ax.errorbar(xr, det_foot[k], yerr=[det_foot[k] - lo_f, hi_f - det_foot[k]],
                        fmt="none", ecolor="0.3", capsize=2, lw=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(SECTORS, fontsize=FS_TICK, rotation=0 if N <= 6 else 90)
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12 if N <= 6 else -0.32), ncol=3, columnspacing=1.2, handletextpad=0.5)
        _style(ax, "Direct emission and multiplier hotspots", "",
               "Emission (t CO2-eq per year)")
        _save(fig, "fig_multiplier")


        # --- [4] Accumulation fractions. Fig 1b. -------------------------------------
        print("\n[4] Accumulation acc_i(t1) = (B P(t1))_i / (B Lambda)_i. Fig 1b.")
        BLam = B @ Lam
        for t1 in (T1_SHORT, T1_PRIMARY):
            acc = (B @ susceptibility(K, t1)) / BLam
            print(f"  t1 = {t1:.0f} yr: " +
                  "  ".join(f"{SECTORS[i]}={acc[i]:.2f}" for i in range(N)))


        # --- [5] Perturbation budget. Eq. (8). Object 3. -----------------------------
        print("\n[5] Implied forcing u = P(t1)^-1 Delta z, budget ||u||. Eq. (8).")
        delta_z = DELTA_Z
        P1 = susceptibility(K, T1_PRIMARY)
        u_tilde = np.linalg.solve(P1, delta_z)
        budget = np.linalg.norm(u_tilde)
        show("u", u_tilde)
        print(f"budget ||u||_2 = {budget:.4f}")
        print(f"static change B Delta z = {B @ delta_z:+.4f}")

        budget_rev = np.linalg.norm(np.linalg.solve(P1, -delta_z))
        print(f"return budget at M = {budget_rev:.4f}")
        assert np.isclose(budget, budget_rev), "budget not symmetric under reversal of Delta z"


        # --- [6] Environmental IRF: the temporary shock. Eqs. (9) to (11). Object 4. -
        print("\n[6] Environmental IRF g(tau) = B exp(K tau) u. Eqs. (9), (10).")
        taus, R = trajectory(K, u_tilde, tau_max=TAU_MAX)
        g = project(R, B)
        crossings = taus[np.where(g[:-1] * g[1:] < 0)[0]]
        print(f"g(0) = {g[0]:+.4f}")
        if crossings.size:
            print(f"g changes sign at tau = {np.round(crossings, 2)} yr.")
        else:
            print("g does not change sign over the window.")

        within = taus <= T1_PRIMARY
        integral = np.trapezoid(g[within], taus[within])
        static = float(B @ delta_z)
        print(f"integral identity, Eq. (11): int_0^t1 g dtau = {integral:+.4f}, "
              f"B Delta z = {static:+.4f}")
        assert np.isclose(integral, static, atol=1e-3), "integral identity Eq. (11) violated"

        taus_mc = _grid(TAU_MAX, DT_MC)
        mc_g = []
        for d in MC_DRAWS:
            dz_mc = displacement(d["A"], DEMAND_SHIFTS[SCENARIO])
            u_mc = _implied(d["K"], dz_mc, T1_PRIMARY)
            mc_g.append(project(_irf(d["K"], u_mc, taus_mc), d["b"]))
        lo_g, hi_g = _band(mc_g)
        fig, ax = _new_ax()
        _ribbon(ax, taus_mc, lo_g, hi_g, _OKABE_ITO[0])
        ax.plot(taus, g, color=_OKABE_ITO[0], lw=1.5, zorder=3, label="g(tau)")
        _zero_line(ax)
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right")
        _style(ax, "Temporary shock: environmental IRF g(tau)", "Lag tau (years)",
               "Emission rate g(tau) (t CO2-eq per year)")
        _save(fig, "fig_temporary_shock")


        # --- [7] Impact IRF. Manuscript Sec. 3.3. ------------------------------------
        print("\n[7] Impact IRF h(tau) = Q g(tau). Sec. 3.3.")
        Q = np.array([[1.0]])   # single stressor, so Q is 1x1 and h = g
        h = (Q @ g[np.newaxis, :]).ravel()
        print(f"Q = [[1.0]], so h(tau) = g(tau); h(0) = {h[0]:+.4f}")


        # --- [8] Step response: the permanent shock. SM Eqs. (S17) to (S20). Extension. -
        print("\n[8] Step response, sustained forcing. SM Eqs. (S17) to (S20).")
        u_inf = -K @ delta_z                       # -M^T dz in the forward form, SM Eq. (S17)
        show("u_inf = -K Delta z", u_inf)
        # Horizon accounting int_0^T g_step dtau = B dz T - B P(T) dz, SM Eq. (S20).
        linear = static * T_ACC
        correction = float(B @ susceptibility(K, T_ACC) @ delta_z)
        print(f"over T = {T_ACC:.0f} yr, linear B dz T = {linear:+.4f}, "
              f"correction B P(T) dz = {correction:+.4f}, total = {linear - correction:+.4f}")
        correction_limit = float(B @ Lam @ delta_z)
        correction_far = float(B @ susceptibility(K, T_FAR) @ delta_z)
        print(f"correction B P(T) dz -> B Lambda dz: {correction_far:.4f} -> {correction_limit:.4f}")
        assert np.isclose(correction_far, correction_limit, atol=1e-2), \
            "correction has not reached B Lambda dz by T_FAR; increase T_FAR for a slower system"

        taus_step = _grid(TAU_MAX, DT_FINE)
        g_step = project(delta_z[np.newaxis, :] - _irf(K, delta_z, taus_step), B)
        mc_step = []
        for d in MC_DRAWS:
            dz_mc = displacement(d["A"], DEMAND_SHIFTS[SCENARIO])
            mc_step.append(project(dz_mc[np.newaxis, :] - _irf(d["K"], dz_mc, taus_mc), d["b"]))
        lo_step, hi_step = _band(mc_step)
        fig, ax = _new_ax()
        _ribbon(ax, taus_mc, lo_step, hi_step, _OKABE_ITO[1])
        ax.plot(taus_step, g_step, color=_OKABE_ITO[1], lw=1.5, zorder=3,
                label="g_step(tau)")
        ax.axhline(static, color="0.4", lw=0.7, ls=":", zorder=1)
        ax.text(0.3, static, "B.dz", va="bottom", ha="left", fontsize=FS_ANNOT, color="0.4")
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower right")
        _style(ax, "Permanent shock: step response g_step(tau)", "Lag tau (years)",
               "Emission-rate change (t CO2-eq per year)")
        _save(fig, "fig_permanent_shock")


        # --- [9] Emission-space target. SM Eq. (S21). Extension. ---------------------
        print("\n[9] Emission-space target Delta z = B^+ Delta g_target. SM Eq. (S21).")
        dz_target = B * (-TARGET_REDUCTION / (B @ B))   # minimum-norm B^+ solution
        show("Delta z", dz_target)
        print(f"B Delta z = {B @ dz_target:+.4f} (target {-TARGET_REDUCTION:+.2f})")
        u_target = np.linalg.solve(P1, dz_target)
        budget_target = np.linalg.norm(u_target)
        print(f"budget ||u|| = {budget_target:.4f}")


        # --- [10] Intensity change and joint decomposition. SM Eqs. (S22) to (S24). Extension. -
        print("\n[10] Joint change g_joint(tau) = B r(tau;dz) + dB r(tau;z0). SM Eq. (S24).")
        # Reference-activity IRF r(tau;v) = exp(K tau) P(t1)^-1 v, SM Eq. (S22).
        z0 = np.ones(N)                          # reference activity in fractional units
        delta_B = DELTA_B
        delta_z_mix = DZ_REPAIR

        # Integral identity int_0^t1 r(tau;z0) dtau = z0, SM Eq. (S22).
        _, R_z0 = trajectory(K, np.linalg.solve(P1, z0), tau_max=T1_PRIMARY)
        integ_r = np.trapezoid(R_z0, _grid(T1_PRIMARY, DT_FINE), axis=0)
        show("int_0^t1 r(tau;z0) dtau", integ_r)
        assert np.allclose(integ_r, z0, atol=1e-2), "integral identity SM Eq. (S22) violated"

        taus_j, R_dz = trajectory(K, np.linalg.solve(P1, delta_z_mix), tau_max=TAU_MAX_DECOMP)
        _, R_z0b = trajectory(K, np.linalg.solve(P1, z0), tau_max=TAU_MAX_DECOMP)
        routing = project(R_dz, B)          # B r(tau;dz)
        intensity = project(R_z0b, delta_B)  # dB r(tau;z0)
        print("  tau      routing   intensity     joint")
        for tt in (0.0, T1_SHORT, T1_PRIMARY):
            k = int(round(tt / DT_FINE))
            print(f"  {tt:.0f} yr   {routing[k]:+.4f}    {intensity[k]:+.4f}   {routing[k]+intensity[k]:+.4f}")
        print(f"ongoing rates: B dz = {B @ delta_z_mix:+.4f}, dB z0 = {delta_B @ z0:+.4f}")
        cross_ratio = (np.linalg.norm(delta_z_mix) / np.linalg.norm(z0)
                       * np.linalg.norm(delta_B) / np.linalg.norm(B))
        print(f"cross-term relative size = {cross_ratio:.3f}")

        tau8_mc = _grid(TAU_MAX_DECOMP, DT_MC)
        joint = routing + intensity
        mc_route, mc_int = [], []
        for d in MC_DRAWS:
            dz_mc = displacement(d["A"], DEMAND_SHIFTS[REPAIR])
            R_dz = _irf(d["K"], _implied(d["K"], dz_mc, T1_PRIMARY), tau8_mc)
            R_z0 = _irf(d["K"], _implied(d["K"], z0, T1_PRIMARY), tau8_mc)
            mc_route.append(project(R_dz, d["b"]))
            mc_int.append(project(R_z0, delta_B))
        lo_r, hi_r = _band(mc_route)
        lo_i, hi_i = _band(mc_int)
        lo_j, hi_j = _band([r + i for r, i in zip(mc_route, mc_int)])
        fig, ax = _new_ax()
        for (curve, lo_c, hi_c, col, lbl) in (
            (routing, lo_r, hi_r, _OKABE_ITO[0], "routing  B.r(tau;dz)"),
            (intensity, lo_i, hi_i, _OKABE_ITO[1], "intensity  dB.r(tau;z0)"),
            (joint, lo_j, hi_j, _OKABE_ITO[2], "joint  (routing + intensity)"),
        ):
            _ribbon(ax, tau8_mc, lo_c, hi_c, col)
            ax.plot(taus_j, curve, color=col, lw=1.4, zorder=3, label=lbl)
        _zero_line(ax)
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower right")
        _style(ax, "Joint routing and intensity decomposition", "Lag tau (years)",
               "Emission rate (t CO2-eq per year)")
        _save(fig, "fig_intensity_change")


        # --- [11] Circular-economy interventions. Sec. 3.3 footnote. -----------------
        print("\n[11] Circular economy: two entry routes. Sec. 3.3 footnote.")

        # (a) Repair: output-mix displacement.
        dz_repair = DZ_REPAIR
        budget_repair = np.linalg.norm(np.linalg.solve(P1, dz_repair))
        print(f"(a) repair (output mix): budget = {budget_repair:.4f}, "
              f"B dz = {B @ dz_repair:+.4f}")

        # (b) Recycling: coefficient revision, giving a displaced drift M'.
        A_recycle = A_io.copy()
        for (supplier, user), change in RECYCLE.items():
            A_recycle[SECTORS.index(supplier), SECTORS.index(user)] += change
        K_prime = drift(A_recycle)
        L_prime = np.linalg.inv(I - A_recycle)
        rho_prime = np.max(np.abs(np.linalg.eigvals(A_recycle)))
        y_ref = (I - A_io) @ X_REF                # final demand consistent with X_REF
        dz_recycle = (L_prime - L) @ y_ref / X_REF   # fractional output change, final demand fixed
        budget_recycle = np.linalg.norm(np.linalg.solve(P1, dz_recycle))
        print(f"(b) recycling (coefficients, M'): rho' = {rho_prime:.3f}, "
              f"budget = {budget_recycle:.4f}, B dz = {B @ dz_recycle:+.4f}")
        assert rho_prime < 1.0, "recycled system A_recycle is not productive: spectral radius >= 1"

        comp = "<" if budget_repair < budget_recycle else ">"
        print(f"budget ranking: repair {budget_repair:.4f} {comp} recycling {budget_recycle:.4f}")

        # Outbound budget at M vs return budget at M'.
        out_budget = np.linalg.norm(np.linalg.solve(susceptibility(K, T1_PRIMARY), delta_z))
        ret_P = susceptibility(K_prime, T1_PRIMARY)
        ret_budget = np.linalg.norm(np.linalg.solve(ret_P, -delta_z))
        print(f"outbound budget at M = {out_budget:.4f}, return budget at M' = {ret_budget:.4f} "
              f"({100 * (ret_budget / out_budget - 1):+.1f}%)")


        # --- [12] Sensitivity to free model choices. Figs 2 and 3; SM. ---------------
        # On fabricated coefficients the absolute values carry no meaning.
        print("\n[12] Sensitivity to free model choices.")

        def rank(scores):
            return [name for name, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

        # (a) Calibration horizon t1. Fig 2. Floor = slowest relaxation time, ceiling
        # = planning horizon.
        lam0 = np.max(eigM.real)                 # eigenvalue of M closest to zero
        t_lo = -1.0 / lam0
        horizons = [T1_SHORT, t_lo, PLANNING_HORIZON]   # short, floor, planning
        print(f"(a) horizon t1: relaxation floor 1/|Re(lambda0)| = {t_lo:.2f} yr, "
              f"planning horizon {PLANNING_HORIZON:.0f} yr.")
        orders_h = []
        for t1 in horizons:
            Pt = susceptibility(K, t1)
            budgets = {n: np.linalg.norm(np.linalg.solve(Pt, dz))
                       for n, dz in TRANSITIONS.items()}
            orders_h.append(rank(budgets))
            below = " (below floor)" if t1 < t_lo else ""
            print(f"  t1 = {t1:5.2f} yr{below}: {SCENARIO} budget = "
                  f"{budgets[SCENARIO]:.4f}, ranking = {rank(budgets)}")
        print(f"  ranking {'stable' if all(o == orders_h[0] for o in orders_h) else 'changes'} "
              "across the horizon range.")

        # (b) Forward form (M^T, the paper) against backward form (D^-1 M D). Sec. 4.4.
        print(f"(b) forward (M^T) vs backward (D^-1 M D) form, t1 = {T1_PRIMARY:.0f} yr:")
        fw, bw = {}, {}
        P_f = susceptibility(drift(A_io, "forward"), T1_PRIMARY)
        P_b = susceptibility(drift(A_io, "backward"), T1_PRIMARY)
        for n, dz in TRANSITIONS.items():
            fw[n] = np.linalg.norm(np.linalg.solve(P_f, dz))
            bw[n] = np.linalg.norm(np.linalg.solve(P_b, dz))
            print(f"  {n:16s} forward = {fw[n]:.4f}  backward = {bw[n]:.4f}  "
                  f"diff = {100 * (bw[n] / fw[n] - 1):+.1f}%")
        print(f"  ranking, forward  = {rank(fw)}")
        print(f"  ranking, backward = {rank(bw)}")
        for form, P_x in (("forward", P_f), ("backward", P_b)):
            u_x = np.linalg.solve(P_x, DELTA_Z)
            mult = B @ -np.linalg.inv(drift(A_io, form))
            print(f"  {form:8s}: onset B.u = {B @ u_x:+.4f}, long-run multipliers = {mult}")
        print("  The forward form follows a displacement to the users of a sector's output,")
        print("  the backward form to its suppliers. The paper reports the forward form.")

        # (c) Monte Carlo noise scale. Fig 1 practitioner footnote.
        print(f"(c) structural noise-scale sweep on the budget of {SCENARIO}:")
        for scale in (0.05, 0.10, 0.20):
            draws = _mc_draws(A_io, B, a_noise=scale)
            buds = [np.linalg.norm(np.linalg.solve(susceptibility(d["K"], T1_PRIMARY),
                                                   unit(displacement(d["A"], DEMAND_SHIFTS[SCENARIO]))))
                    for d in draws]
            lo, hi = np.nanpercentile(buds, [PCT_LO, PCT_HI])
            print(f"  noise {scale:.2f}: {len(draws):3d} draws, "
                  f"budget {PCT_LO}-{PCT_HI} band [{lo:.4f}, {hi:.4f}], width {hi - lo:.4f}")

        # (d) Norm choice, Euclidean vs sector-output-weighted. SM.
        print(f"(d) Euclidean vs output-weighted budget norm, t1 = {T1_PRIMARY:.0f} yr:")
        euc, wgt = {}, {}
        for n, dz in TRANSITIONS.items():
            u = np.linalg.solve(P1, dz)
            euc[n] = np.linalg.norm(u)
            wgt[n] = float(np.sqrt(X_REF @ (u * u)))
            print(f"  {n:16s} Euclidean = {euc[n]:.4f}  weighted = {wgt[n]:.4f}")
        print(f"  ranking under Euclidean norm = {rank(euc)}")
        print(f"  ranking under weighted norm  = {rank(wgt)}")
        print(f"  ranking {'stable' if rank(euc) == rank(wgt) else 'changes'} "
              "between the two norms.")


        # --- [13] Summary. -----------------------------------------------------------
        print("\nSummary of objects and manuscript references:")
        rows = [
            ("1  drift M",                     "Eq. (5)"),
            ("2  susceptibility P(t1)",        "Eq. (6)"),
            ("3  Type I LRT-LCA multiplier -K^-1", "Eq. (7)"),
            ("4  accumulation fractions",      "Fig 1b"),
            ("5  perturbation budget",         "Eq. (8)"),
            ("6  environmental IRF g(tau)",    "Eqs. (9) to (11)"),
            ("7  impact IRF h = Q g",          "Sec. 3.3"),
            ("8  step response",               "SM Eqs. (S17) to (S20)"),
            ("9  emission-space target B^+",   "SM Eq. (S21)"),
            ("10 joint routing + intensity",   "SM Eq. (S24)"),
            ("11 circular economy",            "Sec. 3.3 footnote"),
            ("12 sensitivity to free choices", "Figs 2, 3; SM"),
        ]
        for name, eq in rows:
            print(f"  {name:34s} {eq}")
        print("All checks passed.")

        print("\nFigures written to outputs/:")
        for fname in _FIGURES:
            print(f"  {fname}")
    finally:
        # Restore stdout before closing so a late flush cannot reach a closed file.
        sys.stdout = sys.__stdout__
        output_file.flush()
        output_file.close()


if __name__ == "__main__":
    # Spurious floating-point flags from BLAS matmul are silenced.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        main()
