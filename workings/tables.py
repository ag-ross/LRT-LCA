"""Long-run emission hotspot ranking for the 20 macro-sectors, written to outputs/TAB_A1.tex."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data import (
    GROUP_LABELS,
    SECTOR_NAMES,
    IOSystem,
    build_io_system,
)
from lrt_lca import leontief_inverse, type1_multiplier
from figure_style import OUTPUT_DIR

import math


def _fmt_sigfig(x: float, n: int = 3) -> str:
    """Format x to n significant figures, keeping trailing zeros."""
    if not np.isfinite(x):
        return r"\textemdash"
    if x == 0:
        return f"{0:.{n - 1}f}"
    magnitude = math.floor(math.log10(abs(x)))
    dp = max(0, n - 1 - magnitude)
    return f"{x:.{dp}f}"


def _fmt_ratio(x: float) -> str:
    """Format the amplification ratio to one decimal place."""
    if not np.isfinite(x):
        return r"\textemdash"
    return f"{x:.1f}"


def multiplier_table(
    io: IOSystem | None = None,
    out_dir: Path = OUTPUT_DIR,
    out_stem: str = "TAB_A1",
) -> None:
    """Compute the multipliers and write the LaTeX table to out_dir/out_stem.tex."""
    if io is None:
        io = build_io_system()

    out_dir.mkdir(parents=True, exist_ok=True)

    b      = io.B.ravel()
    L      = leontief_inverse(io.A_io)
    Lambda = type1_multiplier(io.A_io)

    std_mult = ((b / io.X) @ L) * io.X
    lrt_mult = b @ Lambda  # B.Lambda_{.i}: Type I LRT-LCA multiplier
    ratio    = np.where(b > 1e-14, lrt_mult / b, np.nan)

    order = np.argsort(lrt_mult)[::-1]

    body_lines: list[str] = []
    for k in order:
        cells = [
            f"    {SECTOR_NAMES[k]}",
            f"{GROUP_LABELS[k]}",
            _fmt_sigfig(b[k]),
            _fmt_sigfig(std_mult[k]),
            _fmt_sigfig(lrt_mult[k]),
            _fmt_ratio(ratio[k]),
        ]
        body_lines.append(" & ".join(cells) + r" \\")

    col_spec = r"@{}l l r r r r@{}"

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\footnotesize",
        r"\centering",
        (
            r"\caption{Long-run emission multipliers for the 20-sector USEEIO"
            r" macro-sector system, sorted by Type~I LRT-LCA multiplier (descending).}"
        ),
        r"\label{tab:lrt_multipliers}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        (
            r"Sector & Group"
            r" & $B_i$"
            r" & $(B \cdot L)_i$"
            r" & $B \cdot \Lambda_{\cdot i}$"
            r" & Ratio \\"
        ),
        r"\midrule",
    ]

    lines.extend(body_lines)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\smallskip",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        (
            r"\textit{Notes.}"
            r" Units are Mt\,CO\textsubscript{2}-eq per unit of normalised sector output throughout"
            r" (one unit equals the sector's reference annual output). $B_i$: direct GHG emission intensity."
            r" $(B \cdot L)_i$: standard IO-LCA emission multiplier,"
            r" total cradle-to-gate emissions per unit of final demand for sector~$i$"
            r" (Leontief inverse in the same normalised units, backward linkage)."
            r" $B \cdot \Lambda_{\cdot i}$: Type~I LRT-LCA multiplier,"
            r" long-run cumulative emission impact per unit of sustained forcing"
            r" on sector~$i$ ($\Lambda = L^{\top} = {-M^{-\top}}$, forward linkage)."
            r" Ratio: $B \cdot \Lambda_{\cdot i} / B_i$, the indirect amplification factor."
            r" Groups: A\,=\,Agriculture, E\,=\,Energy, M\,=\,Manufacturing, S\,=\,Services."
            r" Data source: USEEIO \citep{Wesley2022},"
            r" aggregated to 20 macro-sectors."
        ),
        r"\end{minipage}",
        r"\end{table}",
    ]

    out_path = out_dir / f"{out_stem}.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out_stem} written to {out_path.resolve()}")


if __name__ == "__main__":
    multiplier_table()
