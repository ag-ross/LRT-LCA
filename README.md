# LRT-LCA: Closed-form dynamics in life cycle assessment: structural change and transition diagnostics

Replication code plus worked simple examples.

## Repository layout

**`replicate.py`** (single entry point that runs the full replication pipeline in one command)

**`workings/`** (full replication of the paper pipeline for the 20-sector USEEIO macro-sector system: all LRT objects, figures, and the multiplier table; see `workings/figures.py` and `workings/tables.py`)

**`examples/`** (a self-contained worked example that computes every LRT-LCA object on a fabricated 3-sector system, with no external data needed and no assumption on the number of sectors, and shows how to adapt the method to an input-output system of your own. See `examples/examples.py` and `examples/README.md`)

## Quick start

To replicate all paper results (figures and tables) in one command (data need to be downloaded first, see the Data section below):

```
python3 replicate.py   # run full pipeline; outputs -> workings/outputs/
```

The pipeline writes the figures (pdf and png), the multiplier table (`TAB_A1.tex`), and two provenance logs: `figures.log` (headline diagnostics) and `diagnostics.log` (every numerical value quoted in the manuscript).

Or run the pipeline stages individually from the `workings/` directory:

```
cd workings
python3 figures.py       # FIG_01, FIG_02, FIG_03 (pdf and png)
python3 tables.py        # TAB_A1.tex
python3 diagnostics.py   # diagnostics.log
```

To run the worked example instead:

```
cd examples
python3 examples.py   # prints every LRT-LCA object, writes figures and log to outputs/
```

## Data

Download `USEEIOv2.0.1-411.xlsx` from:

```
https://pasteur.epa.gov/uploads/10.23719/1524311/USEEIOv2.0.1-411.xlsx
```

and place it at `workings/data/USEEIOv2.0.1-411.xlsx`.

## Dependencies

Python >= 3.9 is required.

```
pip install -r requirements.txt
```

A minimal numpy-only fallback shim is provided in `workings/_lrt_shim/` if scipy is not installed.

Results were produced with Python 3.9.6, numpy 2.0.2, scipy 1.13.1, matplotlib 3.9.4, pandas 2.3.2 and openpyxl 3.1.0 on macOS.

## Licence

Details are provided in the LICENSE file.

### Cite as:

Ross, A. G. [et al.] (2026). Closed-form dynamics in life cycle assessment: structural change and transition diagnostics.
