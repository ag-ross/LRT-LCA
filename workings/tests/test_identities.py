"""Analytical identity tests for the LRT-LCA objects against the USEEIO system."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_WORKINGS = Path(__file__).resolve().parent.parent
if str(_WORKINGS) not in sys.path:
    sys.path.insert(0, str(_WORKINGS))

from data import IOSystem, build_io_system
from lrt_lca import (
    emission_irf,
    emission_step,
    implied_perturbation,
    leontief_inverse,
    perturbation_budget,
    susceptibility_matrix,
    type1_multiplier,
)
from asymmetry_check import (
    EXPECTED_OUTBOUND,
    EXPECTED_RETURN_M,
    EXPECTED_RETURN_MP,
    asymmetry_budgets,
)

# np.trapezoid replaces the deprecated np.trapz (numpy >= 2.0).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

T1: float = 4.0
TRANSITION_KEYS = ("T1", "T2", "T3", "T4")


class TestIdentities(unittest.TestCase):

    io: IOSystem

    @classmethod
    def setUpClass(cls) -> None:
        cls.io = build_io_system()

    def test_integral_identity(self) -> None:
        B = self.io.B.ravel()
        M = self.io.M
        for key in TRANSITION_KEYS:
            dz = self.io.delta_z(key)
            u = implied_perturbation(M, dz, T1)
            taus, g = emission_irf(B, M, u, tau_max=T1, dt=0.001)
            integral = float(_trapezoid(g, taus))
            expected = float(B @ dz)
            np.testing.assert_allclose(
                integral, expected, rtol=1e-3,
                err_msg=f"integral identity violated for {key}",
            )

    def test_type1_multiplier_is_leontief_transpose(self) -> None:
        L = leontief_inverse(self.io.A_io)
        Lambda = type1_multiplier(self.io.A_io)
        np.testing.assert_allclose(Lambda, L.T, atol=1e-12)

    def test_outbound_equals_return_budget_at_fixed_M(self) -> None:
        M = self.io.M
        for key in TRANSITION_KEYS:
            dz = self.io.delta_z(key)
            fwd = perturbation_budget(M, dz, T1)
            rev = perturbation_budget(M, -dz, T1)
            self.assertAlmostEqual(
                fwd, rev, delta=1e-12,
                msg=f"outbound/return budget asymmetry at fixed M for {key}",
            )

    def test_susceptibility_long_run_limit(self) -> None:
        P_inf = susceptibility_matrix(self.io.M, 200.0)
        Lambda = type1_multiplier(self.io.A_io)
        np.testing.assert_allclose(P_inf, Lambda, atol=1e-8)

    def test_forward_vs_backward_budget(self) -> None:
        expected_rel = {"T1": 0.029, "T2": 0.027, "T3": 0.017, "T4": 0.024}
        X = self.io.X
        P = susceptibility_matrix(self.io.M, T1)
        P_bwd = P.T * X[np.newaxis, :] / X[:, np.newaxis]
        bwd_budgets: dict[str, float] = {}
        for key in TRANSITION_KEYS:
            dz = self.io.delta_z(key)
            n_fwd = float(np.linalg.norm(np.linalg.solve(P, dz)))
            n_bwd = float(np.linalg.norm(np.linalg.solve(P_bwd, dz)))
            bwd_budgets[key] = n_bwd
            rel = abs(n_fwd - n_bwd) / n_fwd
            self.assertAlmostEqual(
                rel, expected_rel[key], delta=0.005,
                msg=f"forward vs backward budget difference for {key} is {100 * rel:.1f}%, "
                    f"expected about {100 * expected_rel[key]:.1f}%",
            )
        bwd_rank = sorted(bwd_budgets, key=bwd_budgets.__getitem__)
        self.assertEqual(
            bwd_rank, ["T3", "T1", "T2", "T4"],
            msg=f"backward budget ordering changed: {bwd_rank}",
        )

    def test_step_response_horizon_accounting(self) -> None:
        # sustained case: extra emissions over [0, T] equal B.dz*T - B.P(T).dz
        B = self.io.B.ravel()
        M = self.io.M
        T_hor = 10.0
        for key in TRANSITION_KEYS:
            dz = self.io.delta_z(key)
            taus, g_step = emission_step(B, M, dz, tau_max=T_hor, dt=0.001)
            integral = float(_trapezoid(g_step, taus))
            P_T = susceptibility_matrix(M, T_hor)
            expected = float(B @ dz) * T_hor - float(B @ (P_T @ dz))
            np.testing.assert_allclose(
                integral, expected, rtol=1e-3,
                err_msg=f"step-response horizon accounting violated for {key}",
            )

    def test_asymmetry_check_numbers(self) -> None:
        outbound, ret_M, ret_Mp = asymmetry_budgets(self.io, t1=T1)
        self.assertAlmostEqual(outbound, EXPECTED_OUTBOUND, delta=1e-3)
        self.assertAlmostEqual(ret_M, EXPECTED_RETURN_M, delta=1e-3)
        self.assertAlmostEqual(ret_Mp, EXPECTED_RETURN_MP, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
