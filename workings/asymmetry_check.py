"""Reference-state asymmetry: the T1 return leg evaluated at the displaced state M' = A' - I."""

from __future__ import annotations

import numpy as np

from data import build_io_system, IOSystem
from lrt_lca import perturbation_budget

T1_HORIZON: float = 4.0

# Manuscript reference values (tolerance 1e-3).
EXPECTED_OUTBOUND: float = 0.2785
EXPECTED_RETURN_M: float = 0.2785
EXPECTED_RETURN_MP: float = 0.2759


def asymmetry_budgets(
    io: IOSystem | None = None,
    t1: float = T1_HORIZON,
    d: float = 0.2,
    src: int = 3,
    dst: int = 5,
) -> tuple[float, float, float]:
    """Return (outbound, return_at_M, return_at_Mp) budgets for T1, moving src row weight to dst."""
    if io is None:
        io = build_io_system()

    dz = io.delta_z("T1")
    A = io.A_io

    Ap = A.copy()
    Ap[dst, :] += d * A[src, :]
    Ap[src, :] *= 1.0 - d

    M = A - np.eye(A.shape[0])
    Mp = Ap - np.eye(A.shape[0])

    outbound = perturbation_budget(M, dz, t1)
    return_at_M = perturbation_budget(M, -dz, t1)
    return_at_Mp = perturbation_budget(Mp, -dz, t1)

    return outbound, return_at_M, return_at_Mp


def main() -> None:
    io = build_io_system()
    outbound, ret_M, ret_Mp = asymmetry_budgets(io)
    asymmetry_pct = 100.0 * (ret_Mp - ret_M) / ret_M

    print(f"t1 = {T1_HORIZON}, transition T1 (OilGas to Utilities)")
    print(f"outbound budget ||P(t1)^-1 dz||   at M  = A  - I : {outbound:.4f}")
    print(f"return budget   ||P(t1)^-1 (-dz)|| at M  = A  - I : {ret_M:.4f}")
    print(f"return budget   ||P'(t1)^-1 (-dz)|| at M' = A' - I : {ret_Mp:.4f}")
    print(f"reference-state asymmetry: {asymmetry_pct:+.2f}%")

    assert abs(outbound - EXPECTED_OUTBOUND) < 1e-3, outbound
    assert abs(ret_M - EXPECTED_RETURN_M) < 1e-3, ret_M
    assert abs(ret_Mp - EXPECTED_RETURN_MP) < 1e-3, ret_Mp
    print("all reference values reproduced (tol 1e-3)")

    print("asymmetry vs coefficient-change size d:")
    for d in (0.05, 0.10, 0.30, 0.40, 0.50):
        _, _, displaced = asymmetry_budgets(io, d=d)
        print(f"  d = {d:.2f}: {100.0 * (displaced - ret_M) / ret_M:+.2f}%")

    print("unaligned control pairs, d = 0.20:")
    for src, dst in ((1, 2), (7, 8), (10, 12)):
        _, _, displaced = asymmetry_budgets(io, src=src, dst=dst)
        print(f"  {src} -> {dst}: {100.0 * (displaced - ret_M) / ret_M:+.1f}%")


if __name__ == "__main__":
    main()
