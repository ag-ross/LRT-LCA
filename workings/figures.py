"""Main-paper figures FIG_01, FIG_02 and FIG_03, written to outputs/ as pdf and png."""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from data import (
    GROUP_COLORS,
    GROUP_LABELS,
    IOSystem,
    T1_LONG,
    T1_SHORT,
    build_io_system,
    displacement,
)
from lrt_lca import (
    _expm,
    check_stable,
    implied_perturbation,
    impulse_response,
    perturbation_budget,
    spectral_radius,
    susceptibility_matrix,
    type1_multiplier,
)
from figure_style import (
    DPI_PNG,
    FIG_WIDTH_FULL_IN,
    FONT_SIZE_BODY,
    LINE_ALPHA,
    LINE_WIDTH_DATA,
    MARKER_EDGEWIDTH,
    MARKER_SIZE,
    OUTPUT_DIR,
    apply_rcparams,
    panel_label,
    set_layout,
    style_ax,
    t1_vline,
)


TAU_MAX:       float = 10.0   # upper limit of IRF time grids (years)
DT:            float = 0.05   # time step (years)
N_MC:          int   = 200    # Monte Carlo realisations
A_NOISE_LEVEL: float = 0.10   # off-diagonal A_io structural noise
B_NOISE_LEVEL: float = 0.15   # emission-intensity noise
MC_SEED:       int   = 0

# Accumulation curve grid (FIG_01b).
T_MAX_ACC:     float = 15.0
N_T_ACC:       int   = 150

# Budget-vs-horizon grid (FIG_02b).
T1_GRID: np.ndarray = np.linspace(0.1, TAU_MAX, 80)

GROUP_NAMES: dict[str, str] = {
    "A": "Agriculture", "E": "Energy",
    "M": "Manufacturing", "S": "Services",
}

TRANSITION_KEYS: list[str] = ["T1", "T2", "T3", "T4"]
TRANSITION_COLORS: dict[str, str] = {
    "T1": GROUP_COLORS["E"],
    "T2": GROUP_COLORS["M"],
    "T3": GROUP_COLORS["S"],
    "T4": GROUP_COLORS["A"],
}
TRANSITION_MARKERS: dict[str, str] = {
    "T1": "o", "T2": "s", "T3": "^", "T4": "D",
}

# FIG_02a: two planning horizons compared.
T1_VALS: tuple[float, float] = (T1_SHORT, T1_LONG)


def _mc_draws(
    io: IOSystem,
    n_mc: int = N_MC,
    noise_level: float = A_NOISE_LEVEL,
    seed: int = MC_SEED,
) -> list[np.ndarray]:
    """Noisy drift draws: multiplicative noise off-diagonal, spectral radius >= 1 discarded."""
    rng   = np.random.default_rng(seed)
    N     = io.N
    draws: list[np.ndarray] = []
    for _ in range(n_mc):
        noise = 1.0 + noise_level * rng.uniform(-1.0, 1.0, size=(N, N))
        np.fill_diagonal(noise, 1.0)
        A_mc = io.A_io * noise
        if spectral_radius(A_mc) >= 1.0:
            continue
        M_mc = A_mc - np.eye(N)
        if np.all(np.linalg.eigvals(M_mc).real < -1e-10):
            draws.append(M_mc)
    return draws


def _accumulation(
    A_io: np.ndarray,
    b: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Return acc (T, N) with acc_i(t1) = (B.P(t1))_i / (B.Lambda)_i, accumulated at each t."""
    N      = A_io.shape[0]
    M      = A_io - np.eye(N)
    check_stable(M, "_accumulation")
    Lambda = type1_multiplier(A_io)        # (N, N)
    denom  = b @ Lambda                    # (N,)
    T      = len(t_grid)
    acc    = np.zeros((T, N))
    for k, t1 in enumerate(t_grid):
        if t1 < 1e-12:
            acc[k] = 0.0
            continue
        rho    = susceptibility_matrix(M, t1)
        numer  = b @ rho
        acc[k] = np.where(np.abs(denom) > 1e-14, numer / denom, np.nan)
    return acc


def _mc_sector_bands(
    io: IOSystem,
    t_grid: np.ndarray,
    n_mc: int = N_MC,
    noise_level: float = A_NOISE_LEVEL,
    seed: int = MC_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sector MC 5th/95th bands (lo, hi), each (T, N), for the accumulation curves."""
    rng  = np.random.default_rng(seed)
    N    = io.N
    b    = io.B.ravel()
    T    = len(t_grid)
    runs: list[np.ndarray] = []
    for _ in range(n_mc):
        noise = 1.0 + noise_level * rng.uniform(-1.0, 1.0, size=(N, N))
        np.fill_diagonal(noise, 1.0)
        A_mc = io.A_io * noise
        if spectral_radius(A_mc) >= 1.0:
            continue
        M_mc = A_mc - np.eye(N)
        if not np.all(np.linalg.eigvals(M_mc).real < -1e-10):
            continue
        try:
            runs.append(_accumulation(A_mc, b, t_grid))
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    if len(runs) < 2:
        empty = np.full((T, N), np.nan)
        return empty, empty
    arr = np.stack(runs, axis=0)
    return np.nanpercentile(arr, 5, axis=0), np.nanpercentile(arr, 95, axis=0)


def _emission_irf_bands(
    draws: list[np.ndarray],
    io: IOSystem,
    key: str,
    t1: float,
    tau_max: float,
    dt: float,
    b_noise: float = B_NOISE_LEVEL,
    seed: int = MC_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (taus, lo, hi) for B.R(tau), combining structural and intensity uncertainty."""
    rng  = np.random.default_rng(seed + 2)
    b    = io.B.ravel()
    p    = io.transitions[key].demand_shift
    N    = io.N
    taus = np.arange(0.0, tau_max + dt / 2.0, dt)
    runs: list[np.ndarray] = []
    for M_mc in draws:
        b_mc = np.clip(b * (1.0 + b_noise * rng.standard_normal(N)), 0.0, None)
        try:
            dz_mc   = displacement(M_mc + np.eye(N), io.X, p)
            eps_mc  = implied_perturbation(M_mc, dz_mc, t1)
            _, R_mc = impulse_response(M_mc, eps_mc, tau_max, dt)
            runs.append(R_mc @ b_mc)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    if len(runs) < 2:
        empty = np.full(len(taus), np.nan)
        return taus, empty, empty
    arr = np.stack(runs, axis=0)
    return taus, np.nanpercentile(arr, 5, axis=0), np.nanpercentile(arr, 95, axis=0)


def _budget_fast(M_mc: np.ndarray, delta_z: np.ndarray, t1: float) -> float:
    """Perturbation budget without the stability check, for inner-loop use."""
    Mt  = M_mc.T
    N   = M_mc.shape[0]
    eps = np.linalg.solve(_expm(Mt * t1) - np.eye(N), Mt @ delta_z)
    return float(np.linalg.norm(eps))


def _budget_horizon_bands(
    draws: list[np.ndarray],
    io: IOSystem,
    key: str,
    t1_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) of shape (len(t1_grid),) for the budget across MC draws."""
    p    = io.transitions[key].demand_shift
    Id   = np.eye(io.N)
    runs = []
    for M_mc in draws:
        dz_mc = displacement(M_mc + Id, io.X, p)
        runs.append([_budget_fast(M_mc, dz_mc, float(t1)) for t1 in t1_grid])
    return np.nanpercentile(runs, 5, axis=0), np.nanpercentile(runs, 95, axis=0)


def _draw_fig01a(
    ax: plt.Axes,
    io: IOSystem,
    draws: list[np.ndarray],
    t1: float = T1_LONG,
    tau_max: float = TAU_MAX,
    dt: float = DT,
) -> None:
    """IRF g(tau) = B.R(tau) at t1; each curve integrates to B.dz, the T1 area hatched."""
    b = io.B.ravel()
    M = io.M

    for key in TRANSITION_KEYS:
        tr  = io.transitions[key]
        col = TRANSITION_COLORS[key]

        eps     = implied_perturbation(M, io.delta_z(key), t1)
        taus, R = impulse_response(M, eps, tau_max, dt)
        g_nom   = R @ b

        _, lo, hi = _emission_irf_bands(draws, io, key, t1, tau_max, dt)

        step  = max(1, len(taus) // 4)
        mk_kw = dict(marker=TRANSITION_MARKERS[key], markersize=MARKER_SIZE,
                     markerfacecolor="none", markeredgecolor=col,
                     markeredgewidth=MARKER_EDGEWIDTH,
                     markevery=(0, step)) if key == "T4" else {}

        ax.fill_between(taus, lo, hi, color=col, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(taus, g_nom, color=col, linewidth=LINE_WIDTH_DATA,
                alpha=LINE_ALPHA, zorder=3, label=tr.label, **mk_kw)

        # Hatch the T1 integral area only; no face colour keeps the ribbon visible.
        if key == "T1":
            g_t1 = g_nom
            mask = taus <= t1 + dt / 2.0
            ax.fill_between(
                taus[mask], g_nom[mask], 0.0,
                facecolor="none", edgecolor="0.45",
                hatch="////", linewidth=0, zorder=3,
            )

    ax.axhline(0.0, color="0.6", linewidth=0.3, zorder=1)
    t1_vline(ax, t1)

    ax.set_xlim(-0.25, tau_max + 0.25)
    ax.set_xlabel(r"Lag $\tau$ (yr)", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel(
        r"Emission rate $g(\tau)$ (Mt CO$_2$-eq yr$^{-1}$)",
        fontsize=FONT_SIZE_BODY,
    )
    # Arrow annotation into the hatched area.
    ax.text(
        1.3, 282.0,
        r"$\int_0^{t_1}\!g\,\mathrm{d}\tau = B\!\cdot\!\Delta z$",
        fontsize=FONT_SIZE_BODY, color="0.45",
        ha="left", va="center",
    )
    ax.annotate(
        "",
        xy=(0.25, 220.0),
        xytext=(1.25, 280.0),
        arrowprops=dict(
            arrowstyle="->",
            color="0.45",
            lw=0.6,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels,
              frameon=False, loc="upper right",
              handlelength=1.0, handletextpad=0.4, fontsize=FONT_SIZE_BODY)
    style_ax(ax)
    panel_label(ax, "a")


def _draw_fig01b(
    ax: plt.Axes,
    io: IOSystem,
    t_grid: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    tau_dashed: tuple[float, float] = (T1_SHORT, T1_LONG),
) -> None:
    """Susceptibility accumulation curves, 20 sectors coloured by group, with MC bands."""
    b   = io.B.ravel()
    acc = _accumulation(io.A_io, b, t_grid)    # (T, N)

    # MC ribbons behind nominal curves.
    for i in range(io.N):
        g = GROUP_LABELS[i]
        ax.fill_between(t_grid, lo[:, i], hi[:, i],
                        color=GROUP_COLORS[g], alpha=0.20, linewidth=0, zorder=2)

    # Nominal curves.
    for i in range(io.N):
        g = GROUP_LABELS[i]
        ax.plot(t_grid, acc[:, i],
                color=GROUP_COLORS[g], linewidth=LINE_WIDTH_DATA,
                alpha=LINE_ALPHA, zorder=3)

    for tau in tau_dashed:
        t1_vline(ax, tau)

    ax.set_xlim(-0.25, T_MAX_ACC + 0.25)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel(r"Planning horizon $t_1$ (yr)", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel(
        r"Fraction of long-run multiplier"
        "\n"
        r"$(B{\cdot}P(t_1))_i \;/\; (B{\cdot}\Lambda)_i$",
        fontsize=FONT_SIZE_BODY,
    )
    legend_handles = [
        mlines.Line2D([], [], color=GROUP_COLORS[g], label=GROUP_NAMES[g],
                      linewidth=LINE_WIDTH_DATA)
        for g in GROUP_COLORS
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right",
              handlelength=0.8, handletextpad=0.4, fontsize=FONT_SIZE_BODY)
    style_ax(ax)
    panel_label(ax, "b")


def _draw_fig02a(
    ax: plt.Axes,
    io: IOSystem,
    draws: list[np.ndarray],
    tau_max: float = TAU_MAX,
    dt: float = DT,
) -> None:
    """g(tau) at t1 = 1 and 4 yr; colour and marker encode transition, linestyle the horizon."""
    b = io.B.ravel()
    M = io.M

    horizon_ls = {T1_SHORT: "-", T1_LONG: "--"}
    for t1_idx, t1 in enumerate(T1_VALS):
        for key in TRANSITION_KEYS:
            tr  = io.transitions[key]
            col = TRANSITION_COLORS[key]
            ls  = horizon_ls[t1]

            eps     = implied_perturbation(M, io.delta_z(key), t1)
            taus, R = impulse_response(M, eps, tau_max, dt)
            g_nom   = R @ b

            _, lo, hi = _emission_irf_bands(draws, io, key, t1, tau_max, dt)
            ax.fill_between(taus, lo, hi, color=col, alpha=0.10, linewidth=0, zorder=2)

            step  = max(1, len(taus) // 4)
            mk_kw = dict(marker=TRANSITION_MARKERS[key], markersize=MARKER_SIZE,
                         markerfacecolor="none", markeredgecolor=col,
                         markeredgewidth=MARKER_EDGEWIDTH,
                         markevery=(t1_idx * step // 2, step)) if key == "T4" else {}

            lbl = tr.label if t1_idx == 0 else None
            ax.plot(taus, g_nom, color=col, linewidth=LINE_WIDTH_DATA,
                    linestyle=ls, alpha=LINE_ALPHA, zorder=3, label=lbl, **mk_kw)

    # Planning-horizon reference lines and zero baseline.
    t1_vline(ax, T1_SHORT)
    t1_vline(ax, T1_LONG)
    ax.axhline(0.0, color="0.6", linewidth=0.3, zorder=1)

    handles, _ = ax.get_legend_handles_labels()
    handles += [
        mlines.Line2D([], [], color="0.4", linestyle="-",
                      linewidth=LINE_WIDTH_DATA, label=r"$t_1 = 1$ yr"),
        mlines.Line2D([], [], color="0.4", linestyle="--",
                      linewidth=LINE_WIDTH_DATA, label=r"$t_1 = 4$ yr"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right",
              handlelength=1.2, handletextpad=0.4, fontsize=FONT_SIZE_BODY)

    ax.set_xlim(-0.25, tau_max + 0.25)
    ax.set_xlabel(r"Lag $\tau$ (yr)", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel(
        r"Emission rate $g(\tau)$ (Mt CO$_2$-eq yr$^{-1}$)",
        fontsize=FONT_SIZE_BODY,
    )
    style_ax(ax)
    panel_label(ax, "a")


def _draw_fig02b(
    ax: plt.Axes,
    io: IOSystem,
    draws: list[np.ndarray],
    t1_grid: np.ndarray = T1_GRID,
) -> None:
    """Budget ||u~(t1)||_2 vs horizon t1; curves need not decrease and the ordering can change."""
    M = io.M
    for key in TRANSITION_KEYS:
        tr  = io.transitions[key]
        col = TRANSITION_COLORS[key]

        dz      = io.delta_z(key)
        budgets = np.array([
            perturbation_budget(M, dz, float(t1))
            for t1 in t1_grid
        ])
        lo, hi = _budget_horizon_bands(draws, io, key, t1_grid)

        step = max(1, len(t1_grid) // 4)
        mk_kw = dict(marker=TRANSITION_MARKERS[key], markersize=MARKER_SIZE,
                     markerfacecolor="none", markeredgecolor=col,
                     markeredgewidth=MARKER_EDGEWIDTH,
                     markevery=(0, step)) if key == "T4" else {}

        ax.fill_between(t1_grid, lo, hi, color=col, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(t1_grid, budgets, color=col, linewidth=LINE_WIDTH_DATA,
                alpha=LINE_ALPHA, zorder=3, label=tr.label, **mk_kw)

    ax.set_xlim(-0.25, TAU_MAX + 0.25)
    ax.set_xlabel(r"Planning horizon $t_1$ (yr)", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel(
        r"Perturbation budget $\|\tilde{u}(t_1)\|_2$ (yr$^{-1}$)",
        fontsize=FONT_SIZE_BODY,
    )
    ax.legend(frameon=False, loc="upper right",
              handlelength=1.2, handletextpad=0.4, fontsize=FONT_SIZE_BODY)
    style_ax(ax)
    panel_label(ax, "b")


def _draw_fig03a(
    ax: plt.Axes,
    io: IOSystem,
    t1: float = T1_LONG,
) -> None:
    """Budget scatter B.dz against ||u~||_2; a ranking reversal shows as further right but lower."""
    b = io.B.ravel()
    M = io.M

    xs: list[float] = []
    ys: list[float] = []
    for key in TRANSITION_KEYS:
        tr  = io.transitions[key]
        col = TRANSITION_COLORS[key]
        mk  = TRANSITION_MARKERS[key]

        dz = io.delta_z(key)
        x  = float(b @ dz)
        y  = perturbation_budget(M, dz, t1)
        xs.append(x)
        ys.append(y)

        ax.plot(x, y, mk, color=col, markersize=MARKER_SIZE * 1.4,
                zorder=4, label=tr.label)

    # T1 reverse (open marker, same colour and shape as T1).
    dz_rev = -io.delta_z("T1")
    x_rev  = float(b @ dz_rev)
    y_rev  = perturbation_budget(M, dz_rev, t1)
    xs.append(x_rev)
    ys.append(y_rev)

    ax.plot(x_rev, y_rev, TRANSITION_MARKERS["T1"], color=TRANSITION_COLORS["T1"],
            markerfacecolor="white", markeredgewidth=0.8,
            markersize=MARKER_SIZE * 1.4,
            zorder=4, label="T1 reverse")

    ax.axhline(0.0, color="0.6", linewidth=0.3, zorder=1)
    ax.axvline(0.0, color="0.6", linewidth=0.3, zorder=1)

    ax.set_xlabel(
        r"Emission change $B{\cdot}\Delta z$ (Mt CO$_2$-eq)",
        fontsize=FONT_SIZE_BODY,
    )
    ax.set_ylabel(
        r"Perturbation budget $\|\tilde{u}(t_1)\|_2$ (yr$^{-1}$)",
        fontsize=FONT_SIZE_BODY,
    )
    # Symmetric x-limits (T1 and T1 reverse mirror about zero).
    x_max = 1.15 * max(abs(v) for v in xs)
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(0.0, 1.15 * max(ys))
    ax.legend(frameon=False, loc="lower left",
              handlelength=0.8, handletextpad=0.4, fontsize=FONT_SIZE_BODY)
    style_ax(ax, grid=False)
    panel_label(ax, "a")


def _draw_fig03b(
    ax: plt.Axes,
    io: IOSystem,
    draws: list[np.ndarray],
    t1: float = T1_LONG,
    tau_max: float = TAU_MAX,
    dt: float = DT,
    transition_key: str = "T3",
) -> None:
    """Sector-group routing of the T3 IRF: g_group(tau) = sum_{i in group} B_i R_i(tau)."""
    b    = io.B.ravel()
    M    = io.M
    p    = io.transitions[transition_key].demand_shift
    Id   = np.eye(io.N)
    rng  = np.random.default_rng(MC_SEED + 5)

    # Nominal IRF: R shape (T, N).
    eps     = implied_perturbation(M, io.delta_z(transition_key), t1)
    taus, R = impulse_response(M, eps, tau_max, dt)
    g_sector = R * b[np.newaxis, :]   # (T, N) per-sector contributions

    # MC group-level bands.
    group_runs: dict[str, list[np.ndarray]] = {g: [] for g in GROUP_COLORS}
    for M_mc in draws:
        b_mc = np.clip(b * (1.0 + B_NOISE_LEVEL * rng.standard_normal(io.N)), 0.0, None)
        try:
            dz_mc   = displacement(M_mc + Id, io.X, p)
            eps_mc  = implied_perturbation(M_mc, dz_mc, t1)
            _, R_mc = impulse_response(M_mc, eps_mc, tau_max, dt)
            gs_mc   = R_mc * b_mc[np.newaxis, :]
            for g in GROUP_COLORS:
                idx = [i for i, gl in enumerate(GROUP_LABELS) if gl == g]
                group_runs[g].append(gs_mc[:, idx].sum(axis=1))
        except (RuntimeError, np.linalg.LinAlgError):
            continue

    # Group curves and ribbons.
    for g, col in GROUP_COLORS.items():
        idx   = [i for i, gl in enumerate(GROUP_LABELS) if gl == g]
        g_nom = g_sector[:, idx].sum(axis=1)

        if len(group_runs[g]) >= 2:
            arr = np.stack(group_runs[g], axis=0)
            lo  = np.nanpercentile(arr,  5, axis=0)
            hi  = np.nanpercentile(arr, 95, axis=0)
            ax.fill_between(taus, lo, hi, color=col, alpha=0.20, linewidth=0, zorder=2)

        ax.plot(taus, g_nom, color=col, linewidth=LINE_WIDTH_DATA,
                linestyle="-", alpha=LINE_ALPHA, zorder=3,
                label=GROUP_NAMES[g])

    # Total g(tau) as a dotted reference line.
    ax.plot(taus, R @ b, color="0.25", linewidth=LINE_WIDTH_DATA * 1.3,
            linestyle=":", alpha=0.85, zorder=4, label="Total")

    ax.axhline(0.0, color="0.6", linewidth=0.3, zorder=1)
    t1_vline(ax, t1)

    ax.set_xlim(-0.25, tau_max + 0.25)
    ax.set_xlabel(r"Lag $\tau$ (yr)", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel(
        r"Emission rate contribution (Mt CO$_2$-eq yr$^{-1}$)",
        fontsize=FONT_SIZE_BODY,
    )
    ax.legend(frameon=False, loc="upper right",
              handlelength=1.0, handletextpad=0.4, fontsize=FONT_SIZE_BODY)
    style_ax(ax)
    panel_label(ax, "b")


def make_fig01(
    io: IOSystem | None = None,
    out_dir: Path = OUTPUT_DIR,
    out_stem: str = "FIG_01",
    n_mc: int = N_MC,
) -> None:
    """FIG_01: environmental IRF (a) and susceptibility accumulation curves (b)."""
    if io is None:
        io = build_io_system()
    apply_rcparams()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("FIG_01: computing IRF MC draws ...")
    draws = _mc_draws(io, n_mc=n_mc)
    print(f"  {len(draws)} stable realisations.")

    t_grid = np.linspace(0.0, T_MAX_ACC, N_T_ACC)
    print("FIG_01: computing accumulation MC bands ...")
    lo_acc, hi_acc = _mc_sector_bands(io, t_grid, n_mc=n_mc)

    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    set_layout(fig, nrows=1, ncols=2, w_in=FIG_WIDTH_FULL_IN)

    _draw_fig01a(ax_a, io, draws)
    _draw_fig01b(ax_b, io, t_grid, lo_acc, hi_acc)

    for ext in (".pdf", ".png"):
        path = out_dir / f"{out_stem}{ext}"
        kw   = {"dpi": DPI_PNG} if ext == ".png" else {}
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  {out_stem} written to {out_dir.resolve()}")


def make_fig02(
    io: IOSystem | None = None,
    out_dir: Path = OUTPUT_DIR,
    out_stem: str = "FIG_02",
    n_mc: int = N_MC,
) -> None:
    """FIG_02: g(tau) at two horizons (a) and budget vs planning horizon (b)."""
    if io is None:
        io = build_io_system()
    apply_rcparams()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("FIG_02: computing MC draws ...")
    draws = _mc_draws(io, n_mc=n_mc)
    print(f"  {len(draws)} stable realisations.")

    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    set_layout(fig, nrows=1, ncols=2, w_in=FIG_WIDTH_FULL_IN)

    _draw_fig02a(ax_a, io, draws)
    _draw_fig02b(ax_b, io, draws)

    for ext in (".pdf", ".png"):
        path = out_dir / f"{out_stem}{ext}"
        kw   = {"dpi": DPI_PNG} if ext == ".png" else {}
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  {out_stem} written to {out_dir.resolve()}")


def make_fig03(
    io: IOSystem | None = None,
    out_dir: Path = OUTPUT_DIR,
    out_stem: str = "FIG_03",
    n_mc: int = N_MC,
) -> None:
    """FIG_03: budget scatter (a) and sector-group routing of the T3 IRF (b)."""
    if io is None:
        io = build_io_system()
    apply_rcparams()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("FIG_03: computing MC draws ...")
    draws = _mc_draws(io, n_mc=n_mc)
    print(f"  {len(draws)} stable realisations.")

    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    set_layout(fig, nrows=1, ncols=2, w_in=FIG_WIDTH_FULL_IN)

    _draw_fig03a(ax_a, io)
    _draw_fig03b(ax_b, io, draws)

    for ext in (".pdf", ".png"):
        path = out_dir / f"{out_stem}{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == ".png":
            kw["dpi"] = DPI_PNG
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  {out_stem} written to {out_dir.resolve()}")


if __name__ == "__main__":
    io = build_io_system()
    make_fig01(io)
    make_fig02(io)
    make_fig03(io)

    # log key numerical values
    eigs = np.linalg.eigvals(io.M)
    lambda0 = eigs[np.argmin(np.abs(eigs.real))]
    budgets = {k: perturbation_budget(io.M, io.delta_z(k), T1_LONG) for k in io.transitions}
    budget_rank = tuple(sorted(budgets, key=budgets.__getitem__))
    log = (
        f"relaxation_time_yr = {-1.0 / lambda0.real:.4f}\n"
        f"complex_eigenvalues = {int(np.sum(np.abs(eigs.imag) > 1e-10))} of {io.N}\n"
        f"spectral_radius = {spectral_radius(io.A_io):.4f}\n"
        f"budget_rank = {' < '.join(budget_rank)}\n"
    )
    (OUTPUT_DIR / "figures.log").write_text(log)
    print(log, end="")
