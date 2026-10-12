"""20-sector US IO system aggregated from USEEIO v2.0.1-411, with A_agg[I, J] = F_IJ / X_J."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


DATA_PATH: Path = Path(__file__).parent / "data" / "USEEIOv2.0.1-411.xlsx"

# Lookup tries three BEA characters, then two; unmatched codes map to "Other" and are dropped.

_BEA3: dict[str, str] = {
    # Agriculture
    "111": "Crops",
    "112": "Livestock",
    "113": "ForestFish",
    "114": "ForestFish",
    "115": "ForestFish",
    # Energy
    "211": "OilGas",
    "212": "Mining",
    "213": "Mining",
    # Manufacturing
    "311": "FoodBeverage",
    "312": "FoodBeverage",
    "325": "Chemicals",
    "331": "Metals",
    "332": "Metals",
    "333": "Machinery",
    "334": "Machinery",
    "335": "Machinery",
    "336": "Machinery",
}

_BEA2: dict[str, str] = {
    # Energy
    "22": "Utilities",
    # Manufacturing
    "23": "Construction",
    "31": "OtherMfg",
    "32": "OtherMfg",
    "33": "OtherMfg",
    # Services
    "42": "Trade",
    "44": "Trade",
    "45": "Trade",
    "48": "Transport",
    "49": "Transport",
    "51": "Information",
    "52": "Finance",
    "53": "Finance",
    "54": "ProfServices",
    "55": "ProfServices",
    "56": "ProfServices",
    "61": "Education",
    "62": "Healthcare",
    "71": "OtherServices",
    "72": "OtherServices",
    "81": "OtherServices",
}

# Groups are contiguous so GROUPS can use slices.
SECTOR_NAMES: list[str] = [
    "Crops", "Livestock", "ForestFish",                                              # A: 0-2
    "OilGas", "Mining", "Utilities",                                                 # E: 3-5
    "Construction", "FoodBeverage", "Chemicals", "Metals", "Machinery", "OtherMfg",  # M: 6-11
    "Trade", "Transport", "Information", "Finance",
    "ProfServices", "Education", "Healthcare", "OtherServices",                      # S: 12-19
]

N: int = len(SECTOR_NAMES)

GROUP_LABELS: list[str] = ["A"] * 3 + ["E"] * 3 + ["M"] * 6 + ["S"] * 8

GROUPS: dict[str, slice] = {
    "A": slice(0, 3),
    "E": slice(3, 6),
    "M": slice(6, 12),
    "S": slice(12, 20),
}

GROUP_COLORS: dict[str, str] = {
    "E": "#B76E52",   # Energy
    "M": "#5A7FA8",   # Manufacturing
    "S": "#498D5D",   # Services
    "A": "#8A558A",   # Agriculture
}

T1_SHORT: float = 1.0
T1_LONG:  float = 4.0

# Characterisation row used by _ghg_intensity; set on first build.
GHG_ROW_NAME: str | None = None

# GWP-100 factors (AR5, IPCC 2013), used when the C matrix is unavailable.
_GWP100: dict[str, float] = {
    "carbon dioxide": 1.0,
    "methane":        28.0,
    "nitrous oxide":  265.0,
}


def displacement(A_io: np.ndarray, X: np.ndarray, demand_shift: np.ndarray) -> np.ndarray:
    """dz = L (p X) / X with L = (I - A_io)^-1: output displacement of a final-demand shift."""
    N_io = A_io.shape[0]
    return np.linalg.solve(np.eye(N_io) - A_io, demand_shift * X) / X


@dataclass
class Transition:
    """Final-demand scenario for one policy transition."""
    key:          str
    label:        str
    demand_shift: np.ndarray
    description:  str


@dataclass
class IOSystem:
    """IO system ready for LRT-LCA analysis."""
    A_io:         np.ndarray
    B:            np.ndarray
    M:            np.ndarray
    X:            np.ndarray
    transitions:  dict[str, Transition]
    sector_names: list[str] = field(default_factory=lambda: SECTOR_NAMES)
    group_labels: list[str] = field(default_factory=lambda: GROUP_LABELS)

    @property
    def N(self) -> int:
        return self.A_io.shape[0]

    def group_slice(self, group: str) -> slice:
        return GROUPS[group]

    def delta_z(self, key: str) -> np.ndarray:
        return displacement(self.A_io, self.X, self.transitions[key].demand_shift)

    def sector_colors(self) -> list[str]:
        return [GROUP_COLORS[g] for g in self.group_labels]


def _load_matrices(path: Path):
    """Return (A_df, B_df, C_df, q_ser) from the USEEIO workbook; C_df is None if absent."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas and openpyxl are required: pip install pandas openpyxl"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(
            f"USEEIO data file not found at {path}.\n"
            f"Download from:\n"
            f"  https://pasteur.epa.gov/uploads/10.23719/1524311/USEEIOv2.0.1-411.xlsx\n"
            f"and place at:\n"
            f"  {path}"
        )

    kw = {"index_col": 0, "header": 0}
    A_df = pd.read_excel(path, sheet_name="A", **kw)
    B_df = pd.read_excel(path, sheet_name="B", **kw)
    try:
        C_df = pd.read_excel(path, sheet_name="C", **kw)
    except Exception:
        C_df = None

    if not A_df.columns.equals(B_df.columns):
        raise ValueError(
            "A and B matrix columns do not match; the file may be corrupt "
            "or from an unsupported USEEIO version."
        )

    q_ser = pd.read_excel(path, sheet_name="q", index_col=0).iloc[:, 0]
    q_ser = q_ser.reindex(A_df.columns)
    if q_ser.isna().any():
        missing = list(q_ser.index[q_ser.isna()])[:5]
        raise ValueError(
            f"Output vector q does not cover all A columns "
            f"(first missing: {missing}); the file may be corrupt."
        )

    return A_df, B_df, C_df, q_ser


def _ghg_intensity(B_df, C_df) -> np.ndarray:
    """Direct GHG intensity (kg CO2-eq / 2012 USD) from the GWP-100 row of C, else B with AR5."""
    global GHG_ROW_NAME

    if C_df is not None:
        mask = C_df.index.str.lower().str.contains(
            "global warm|greenhouse|ghg", na=False
        )
        if mask.any():
            GHG_ROW_NAME = str(C_df.index[mask][0])
            c_ghg = C_df.loc[mask].values[0]
            return (c_ghg @ B_df.values).ravel().astype(float)

    GHG_ROW_NAME = "GWP100-from-B-flows (AR5)"
    b_ghg = np.zeros(B_df.shape[1])
    for keyword, gwp in _GWP100.items():
        mask = B_df.index.str.lower().str.contains(keyword, na=False)
        if mask.any():
            b_ghg += B_df.loc[mask].values.sum(axis=0) * gwp

    if not np.any(b_ghg != 0):
        warnings.warn(
            "No GHG flows matched in B matrix; emission intensities are zero. "
            "Check flow names against _GWP100 keywords.",
            RuntimeWarning,
            stacklevel=3,
        )
    return b_ghg.astype(float)


def _code_to_macro(code: str) -> str:
    """Map a BEA sector code to a macro-sector label, or 'Other'."""
    c = str(code).strip()
    return _BEA3.get(c[:3]) or _BEA2.get(c[:2], "Other")


def _aggregate(A_df, b_raw: np.ndarray, x_raw: np.ndarray):
    """Aggregate A and GHG intensity to macro-sectors, weighted by x_raw; "Other" is dropped."""
    sector_codes = list(A_df.columns)
    macro_labels = [_code_to_macro(c) for c in sector_codes]

    keep_codes  = [c for c, m in zip(sector_codes, macro_labels) if m in SECTOR_NAMES]
    keep_macros = [m for c, m in zip(sector_codes, macro_labels) if m in SECTOR_NAMES]

    if not keep_codes:
        raise ValueError(
            "No USEEIO sectors matched the aggregation map. "
            "Verify the file is USEEIOv2.0.1-411.xlsx."
        )

    n_dropped = len(sector_codes) - len(keep_codes)
    if n_dropped > 50:
        warnings.warn(
            f"{n_dropped} of {len(sector_codes)} sectors were not mapped "
            f"(expected 12 scrap/government/special rows); check _BEA3 and _BEA2.",
            RuntimeWarning,
            stacklevel=3,
        )

    keep_idx = [sector_codes.index(c) for c in keep_codes]
    A_sub = A_df.loc[keep_codes, keep_codes].values.astype(float)
    b_sub = b_raw[keep_idx]
    x_sub = x_raw[keep_idx]

    group_idx = {g: i for i, g in enumerate(SECTOR_NAMES)}
    S = np.zeros((N, len(keep_codes)))
    for j, m in enumerate(keep_macros):
        S[group_idx[m], j] = 1.0

    F = A_sub * x_sub[np.newaxis, :]   # fine-sector flows (2012 USD)
    X = S @ x_sub                      # macro reference outputs (2012 USD)
    if np.any(X <= 0.0):
        empty = [SECTOR_NAMES[i] for i in np.where(X <= 0.0)[0]]
        raise ValueError(f"Macro-sectors with zero reference output: {empty}.")

    A_agg = (S @ F @ S.T) / X[np.newaxis, :]
    b_agg = (S @ (b_sub * x_sub)) / 1e9    # kg -> Mt CO2-eq per sector-unit

    return A_agg, b_agg, X


def _build_transitions(scale: float = 1.0) -> dict[str, Transition]:
    """Four illustrative final-demand scenarios (T1-T4), in fractions of reference output."""
    p = {k: np.zeros(N) for k in ("T1", "T2", "T3", "T4")}

    p["T1"][3]  = -0.2 * scale
    p["T1"][5]  = +0.2 * scale

    p["T2"][8]  = -0.3 * scale
    p["T2"][3]  = +0.15 * scale
    p["T2"][5]  = +0.15 * scale

    p["T3"][10] = -0.1 * scale
    p["T3"][16] = +0.05 * scale
    p["T3"][19] = +0.05 * scale

    p["T4"] = p["T2"] + p["T3"]

    return {
        "T1": Transition(
            "T1", "OilGas to Utilities", p["T1"],
            "Final demand falls by 20% of reference output for OilGas (3) and rises by 20% for Utilities (5)",
        ),
        "T2": Transition(
            "T2", "Chemicals contraction", p["T2"],
            "Final demand for Chemicals (8) falls by 30% of reference output, "
            "with rises of 15% of reference output for OilGas and Utilities",
        ),
        "T3": Transition(
            "T3", "Manufacturing to Services", p["T3"],
            "Final demand falls by 10% of reference output for Machinery (10), "
            "with rises of 5% for ProfServices (16) and OtherServices (19)",
        ),
        "T4": Transition(
            "T4", "T2+T3 (joint)", p["T4"],
            "Combined final-demand shifts of T2 and T3",
        ),
    }


def build_io_system(
    path: Path | None = None,
) -> IOSystem:
    """Load USEEIO and aggregate to 20 macro-sectors; B in Mt CO2-eq per unit, M = A_io - I."""
    if path is None:
        path = DATA_PATH

    A_df, B_df, C_df, q_ser = _load_matrices(path)
    b_raw            = _ghg_intensity(B_df, C_df)
    A_io, b_agg, X   = _aggregate(A_df, b_raw, q_ser.values.astype(float))
    M                = A_io - np.eye(N)

    return IOSystem(
        A_io=A_io,
        B=b_agg,
        M=M,
        X=X,
        transitions=_build_transitions(),
        sector_names=SECTOR_NAMES,
        group_labels=GROUP_LABELS,
    )


if __name__ == "__main__":
    from lrt_lca import relaxation_timescales, spectral_radius

    io = build_io_system()
    ts  = relaxation_timescales(io.M)

    print(f"N sectors : {io.N}")
    print(f"rho(A_io) : {spectral_radius(io.A_io):.4f}")
    print("Timescales (yr), slowest to fastest:")
    for i, (name, t) in enumerate(
        sorted(zip(io.sector_names, ts), key=lambda x: -x[1])
    ):
        print(f"  {i+1:2d}  {name:<15s}  {t:.3f}")
