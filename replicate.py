"""
replicate.py: run the full LRT-LCA replication pipeline from the project root.

Generates all paper outputs (figures and tables) in workings/outputs/.

Usage
-----
    python replicate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add workings/ to sys.path so that the pipeline modules resolve their
# local imports (data, lrt_lca, figure_style).
_WORKINGS = Path(__file__).resolve().parent / "workings"
if str(_WORKINGS) not in sys.path:
    sys.path.insert(0, str(_WORKINGS))

from figure_style import OUTPUT_DIR  # noqa: E402


def _write_log(io) -> None:  # type: ignore[no-untyped-def]
    """Write figures.log with key diagnostic values."""
    from lrt_lca import perturbation_budget, spectral_radius
    from figures import T1_LONG

    eigs    = np.linalg.eigvals(io.M)
    lambda0 = eigs[np.argmin(np.abs(eigs.real))]
    budgets = {
        k: perturbation_budget(io.M, io.delta_z(k), T1_LONG)
        for k in io.transitions
    }
    budget_rank = tuple(sorted(budgets, key=budgets.__getitem__))

    log = (
        f"relaxation_time_yr = {-1.0 / lambda0.real:.4f}\n"
        f"complex_eigenvalues = {int(np.sum(np.abs(eigs.imag) > 1e-10))} of {io.N}\n"
        f"spectral_radius = {spectral_radius(io.A_io):.4f}\n"
        f"budget_rank = {' < '.join(budget_rank)}\n"
    )
    (OUTPUT_DIR / "figures.log").write_text(log)
    print(log, end="")


def main() -> None:
    from data import build_io_system
    from diagnostics import write_diagnostics
    from figures import make_fig01, make_fig02, make_fig03
    from tables import multiplier_table

    print("Building IO system ...")
    io = build_io_system()

    make_fig01(io)
    make_fig02(io)
    make_fig03(io)
    multiplier_table(io)
    write_diagnostics(io)
    _write_log(io)


if __name__ == "__main__":
    main()
