"""Accounting tests for data.py and lrt_lca.py; sectors A 0:3, E 3:6, M 6:12, S 12:20."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_WORKINGS = Path(__file__).resolve().parent.parent
if str(_WORKINGS) not in sys.path:
    sys.path.insert(0, str(_WORKINGS))

import data as data_module
from data import (
    N,
    GROUPS,
    GROUP_LABELS,
    SECTOR_NAMES,
    IOSystem,
    build_io_system,
    displacement,
)
from lrt_lca import (
    check_stable,
    implied_perturbation,
    impulse_response,
    leontief_inverse,
    perturbation_budget,
    type1_multiplier,
)


class _SharedIO:
    """Build the IOSystem once per concrete class; not a TestCase, so the runner ignores it."""

    io: IOSystem

    @classmethod
    def setUpClass(cls) -> None:
        cls.io = build_io_system()  # type: ignore[attr-defined]


class TestModuleConstants(unittest.TestCase):

    def test_N_is_20(self) -> None:
        self.assertEqual(N, 20)

    def test_sector_names_length(self) -> None:
        self.assertEqual(len(SECTOR_NAMES), N)

    def test_group_labels_length(self) -> None:
        self.assertEqual(len(GROUP_LABELS), N)

    def test_groups_partition_all_sectors(self) -> None:
        covered: set[int] = set()
        for sl in GROUPS.values():
            covered.update(range(*sl.indices(N)))
        self.assertEqual(covered, set(range(N)))

    def test_groups_are_non_overlapping(self) -> None:
        all_indices: list[int] = []
        for sl in GROUPS.values():
            all_indices.extend(range(*sl.indices(N)))
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_group_A_size(self) -> None:
        self.assertEqual(len(range(*GROUPS["A"].indices(N))), 3)

    def test_group_E_size(self) -> None:
        self.assertEqual(len(range(*GROUPS["E"].indices(N))), 3)

    def test_group_M_size(self) -> None:
        self.assertEqual(len(range(*GROUPS["M"].indices(N))), 6)

    def test_group_S_size(self) -> None:
        self.assertEqual(len(range(*GROUPS["S"].indices(N))), 8)


class TestSectorNames(unittest.TestCase):

    def test_oilgas_at_index_3(self) -> None:
        self.assertEqual(SECTOR_NAMES[3], "OilGas")

    def test_mining_at_index_4(self) -> None:
        self.assertEqual(SECTOR_NAMES[4], "Mining")

    def test_utilities_at_index_5(self) -> None:
        self.assertEqual(SECTOR_NAMES[5], "Utilities")

    def test_chemicals_at_index_8(self) -> None:
        self.assertEqual(SECTOR_NAMES[8], "Chemicals")

    def test_machinery_at_index_10(self) -> None:
        self.assertEqual(SECTOR_NAMES[10], "Machinery")

    def test_profservices_at_index_16(self) -> None:
        self.assertEqual(SECTOR_NAMES[16], "ProfServices")

    def test_otherservices_at_index_19(self) -> None:
        self.assertEqual(SECTOR_NAMES[19], "OtherServices")

    def test_group_labels_match_groups(self) -> None:
        for g, sl in GROUPS.items():
            for i in range(*sl.indices(N)):
                self.assertEqual(GROUP_LABELS[i], g,
                                 msg=f"index {i} ({SECTOR_NAMES[i]}): "
                                     f"expected {g}, got {GROUP_LABELS[i]}")


class TestIOSystemStructure(_SharedIO, unittest.TestCase):

    def test_A_io_shape(self) -> None:
        self.assertEqual(self.io.A_io.shape, (N, N))

    def test_B_shape(self) -> None:
        self.assertEqual(self.io.B.ravel().shape, (N,))

    def test_M_shape(self) -> None:
        self.assertEqual(self.io.M.shape, (N, N))

    def test_N_property(self) -> None:
        self.assertEqual(self.io.N, N)

    def test_M_equals_A_minus_I(self) -> None:
        diff = self.io.M - (self.io.A_io - np.eye(N))
        self.assertLess(np.max(np.abs(diff)), 1e-12)

    def test_transitions_keys_present(self) -> None:
        for key in ("T1", "T2", "T3", "T4"):
            self.assertIn(key, self.io.transitions)

    def test_transitions_delta_z_shape(self) -> None:
        for key in self.io.transitions:
            self.assertEqual(self.io.delta_z(key).shape, (N,),
                             msg=f"delta_z shape wrong for {key}")


class TestIOMatrixProperties(_SharedIO, unittest.TestCase):

    def test_A_io_non_negative(self) -> None:
        self.assertTrue(np.all(self.io.A_io >= 0.0),
                        "A_io contains negative entries")

    def test_A_io_column_sums_below_one(self) -> None:
        col_max = float(self.io.A_io.sum(axis=0).max())
        self.assertLess(col_max, 1.0, msg=f"A_io column sum {col_max:.4f} >= 1.0")

    def test_spectral_radius_less_than_one(self) -> None:
        rho = float(np.max(np.abs(np.linalg.eigvals(self.io.A_io))))
        self.assertLess(rho, 1.0,
                        msg=f"Spectral radius {rho:.4f} >= 1.0 (system not stable)")
        self.assertGreaterEqual(rho, 0.40,
                                msg=f"Spectral radius {rho:.4f} unexpectedly low (< 0.40)")
        self.assertLessEqual(rho, 0.60,
                             msg=f"Spectral radius {rho:.4f} outside expected range [0.40, 0.60]")

    def test_M_is_stable(self) -> None:
        try:
            check_stable(self.io.M, "test_M_is_stable")
        except RuntimeError as exc:
            self.fail(f"M is not stable: {exc}")

    def test_productive_spectral_condition(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.io.A_io)))
        L = leontief_inverse(self.io.A_io)
        self.assertTrue(np.all(np.isfinite(L)))

    def test_budget_rank_ordering(self) -> None:
        # Reference ordering at both horizons: T3 < T1 < T2 < T4.
        for t1 in (1.0, 4.0):
            budgets = {
                k: perturbation_budget(self.io.M, self.io.delta_z(k), t1)
                for k in self.io.transitions
            }
            rank = sorted(budgets, key=budgets.__getitem__)
            self.assertEqual(
                rank, ["T3", "T1", "T2", "T4"],
                msg=f"budget ordering at t1={t1:g} is {rank}, "
                    + ", ".join(f"{k}={v:.6f}" for k, v in budgets.items()),
            )

    def test_irf_signs_on_window(self) -> None:
        t1 = 4.0
        b = self.io.B.ravel()
        minima = {}
        for k in self.io.transitions:
            u = implied_perturbation(self.io.M, self.io.delta_z(k), t1)
            _, R = impulse_response(self.io.M, u, t1)
            g = R @ b
            self.assertGreater(g[0], 0.0, msg=f"{k} onset is not positive")
            minima[k] = float(g.min())
        self.assertGreater(minima["T1"], 0.0)
        self.assertGreater(minima["T3"], 0.0)
        self.assertLess(minima["T2"], 0.0)
        self.assertLess(minima["T4"], 0.0)


class TestEmissionIntensity(_SharedIO, unittest.TestCase):

    def test_B_non_negative(self) -> None:
        self.assertTrue(np.all(self.io.B.ravel() >= 0.0))

    def test_B_not_all_zero(self) -> None:
        self.assertGreater(float(np.max(self.io.B)), 0.0)

    def test_energy_group_mean_intensity_above_overall_mean(self) -> None:
        b = self.io.B.ravel()
        E_mean      = b[GROUPS["E"]].mean()
        overall_mean = b.mean()
        self.assertGreater(E_mean, overall_mean,
                           msg="Energy group mean emission intensity should exceed overall mean")

    def test_ghg_row_name_recorded(self) -> None:
        # build_io_system() ran in setUpClass, so GHG_ROW_NAME must be set.
        name = data_module.GHG_ROW_NAME
        self.assertIsNotNone(name,
                             msg="GHG_ROW_NAME not set after build_io_system()")
        if name != "GWP100-from-B-flows (AR5)":
            # C-matrix path: the row label must match the search keywords.
            self.assertRegex(name.lower(), r"global warm|greenhouse|ghg",
                             msg=f"Unexpected GHG row label: {name!r}")


class TestDemandShifts(_SharedIO, unittest.TestCase):

    EXPECTED = {
        "T1": {3: -0.2, 5: +0.2},
        "T2": {8: -0.3, 3: +0.15, 5: +0.15},
        "T3": {10: -0.1, 16: +0.05, 19: +0.05},
        "T4": {8: -0.3, 3: +0.15, 5: +0.15, 10: -0.1, 16: +0.05, 19: +0.05},
    }

    def _p(self, key: str) -> np.ndarray:
        return self.io.transitions[key].demand_shift

    def test_shapes(self) -> None:
        for key in self.EXPECTED:
            self.assertEqual(self._p(key).shape, (N,),
                             msg=f"demand_shift for {key} has wrong shape")

    def test_entries_exactly_as_stated(self) -> None:
        for key, entries in self.EXPECTED.items():
            expected = np.zeros(N)
            for i, v in entries.items():
                expected[i] = v
            np.testing.assert_allclose(
                self._p(key), expected, atol=1e-12,
                err_msg=f"demand_shift for {key} does not match the stated entries",
            )

    def test_sum_zero(self) -> None:
        for key in self.EXPECTED:
            self.assertAlmostEqual(float(self._p(key).sum()), 0.0, places=10,
                                   msg=f"demand_shift for {key} does not sum to zero")

    def test_T2_oilgas_and_utilities_equal_split(self) -> None:
        p = self._p("T2")
        self.assertAlmostEqual(float(p[3]), float(p[5]), places=10)

    def test_T3_services_equal_split(self) -> None:
        p = self._p("T3")
        self.assertAlmostEqual(float(p[16]), float(p[19]), places=10)

    def test_T4_is_T2_plus_T3(self) -> None:
        np.testing.assert_allclose(self._p("T4"), self._p("T2") + self._p("T3"),
                                   atol=1e-12)


class TestDisplacement(_SharedIO, unittest.TestCase):

    def _dz(self, key: str) -> np.ndarray:
        return self.io.delta_z(key)

    def test_delta_z_matches_displacement(self) -> None:
        for key, tr in self.io.transitions.items():
            np.testing.assert_allclose(
                self._dz(key),
                displacement(self.io.A_io, self.io.X, tr.demand_shift),
                atol=1e-12, err_msg=f"delta_z for {key} is not the Leontief displacement",
            )

    def test_displacement_from_drift_matches(self) -> None:
        for key, tr in self.io.transitions.items():
            np.testing.assert_allclose(
                displacement(self.io.M + np.eye(self.io.N), self.io.X, tr.demand_shift),
                self._dz(key),
                atol=1e-12, err_msg=f"displacement from M + I differs for {key}",
            )

    def test_delta_z_is_dense(self) -> None:
        for key in self.io.transitions:
            self.assertEqual(int(np.count_nonzero(self._dz(key))), N,
                             msg=f"delta_z for {key} has zero entries")

    def test_leontief_relation_reproduces_demand_shift(self) -> None:
        # (I - A)(dz * X) = p * X, the defining relation of the displacement.
        for key, tr in self.io.transitions.items():
            lhs = (np.eye(N) - self.io.A_io) @ (self._dz(key) * self.io.X)
            np.testing.assert_allclose(
                lhs, tr.demand_shift * self.io.X, rtol=1e-8,
                atol=1e-8 * float(np.max(self.io.X)),
                err_msg=f"(I - A)(dz X) != p X for {key}",
            )

    def test_T4_displacement_is_T2_plus_T3(self) -> None:
        np.testing.assert_allclose(self._dz("T4"), self._dz("T2") + self._dz("T3"),
                                   atol=1e-12)

    def test_shifted_sectors_keep_their_sign(self) -> None:
        for key, tr in self.io.transitions.items():
            dz = self._dz(key)
            for i in np.flatnonzero(tr.demand_shift):
                self.assertEqual(np.sign(dz[i]), np.sign(tr.demand_shift[i]),
                                 msg=f"{key}: {SECTOR_NAMES[i]} displacement "
                                     f"{dz[i]:+.4f} opposes its demand shift")


class TestLRTLCAConsistency(_SharedIO, unittest.TestCase):

    def test_leontief_inverse_shape(self) -> None:
        L = leontief_inverse(self.io.A_io)
        self.assertEqual(L.shape, (N, N))

    def test_leontief_identity(self) -> None:
        L = leontief_inverse(self.io.A_io)
        residual = (np.eye(N) - self.io.A_io) @ L - np.eye(N)
        self.assertLess(float(np.max(np.abs(residual))), 1e-8,
                        msg="(I - A) L != I: Leontief identity violated")

    def test_leontief_inverse_positive(self) -> None:
        L = leontief_inverse(self.io.A_io)
        self.assertTrue(np.all(L > 0.0),
                        msg="Leontief inverse has non-positive entries, "
                            "which suggests the IO system is not productive")

    def test_type1_multiplier_equals_leontief_transpose(self) -> None:
        L      = leontief_inverse(self.io.A_io)
        Lambda = type1_multiplier(self.io.A_io)
        np.testing.assert_allclose(Lambda, L.T, atol=1e-8,
                                   err_msg="Lambda != L^T")

    def test_type1_multiplier_positive(self) -> None:
        Lambda = type1_multiplier(self.io.A_io)
        self.assertTrue(np.all(Lambda > 0.0))

    def test_leontief_diagonal_geq_1(self) -> None:
        # Diagonal of L >= 1 (direct + own-sector multiplier)
        L = leontief_inverse(self.io.A_io)
        self.assertTrue(np.all(np.diag(L) >= 1.0))


if __name__ == "__main__":
    unittest.main()
