# LRT-LCA worked examples

A worked example for applying LRT-LCA, as introduced in the paper, to an input-output system of your own. The objects are methods-level (the drift matrix, susceptibility, impulse responses, and the perturbation budget) and assume familiarity with the paper. No external data files are needed. The technology matrix is hardcoded, so the script runs on its own.

## Contents

`examples.py` computes every LRT-LCA analytical object on a fabricated 3-sector input-output system (Energy, Manufacturing, Services) and prints each object next to its manuscript equation reference. The console output is also written to `outputs/examples_output.txt`. All numbers are fabricated for illustration and are not data.

A scenario is stated as a shift `p` in final demand, each entry a fraction of that sector's reference output, and the displacement follows from the Leontief relation `dz = L (p X_REF) / X_REF`. Every sector then carries a displacement, including those whose final demand is untouched, since they supply the sectors whose demand moves. Setting an entry of `dz` by hand to zero instead imposes a target, holding that sector's output fixed while its customers move.

The script prints every object and writes four figures as PNG to `outputs/`, described in the table below. The figures do not reproduce the manuscript figures.

| Figure | Question it answers |
|---|---|
| `fig_temporary_shock` | how a one-off change propagates and then fades, `g(tau)` decaying to zero |
| `fig_permanent_shock` | how a sustained change's emission rate rises to the ongoing rate `B.dz` |
| `fig_multiplier` | which sectors carry the largest emission multipliers, and how the Type I LRT-LCA and standard IO-LCA rankings differ |
| `fig_intensity_change` | how much of an emission change is output routing and how much is intensity improvement |

The remaining objects (the susceptibility matrix, the perturbation budget, the emission-space target, and the circular-economy routes) are printed to the console rather than plotted. On fabricated coefficients their absolute values carry no meaning.

The lag curves carry shaded 5th to 95th percentile Monte Carlo ribbons and the multiplier bars carry Monte Carlo whiskers. Each draw perturbs the off-diagonal technology coefficients and the sector emissions, discarding any draw that is not stable, and one shared set of accepted draws feeds every figure. The displacement is recomputed from each draw's coefficients, so a draw carries the demand shift through its own Leontief inverse. The seed is fixed, so the figures reproduce exactly.

Block 12 exercises the four free model choices the manuscript reports with sensitivity, the calibration horizon `t1`, the forward form against the backward form, the Monte Carlo noise scale, and the norm for the budget. Each is applied to candidate transitions, as unit-norm directions.

The paper's own figures, on the real USEEIO system and with their Monte Carlo uncertainty bands, are produced by `../replicate.py` with the code in `../workings/`.

## Forward and backward forms

The objects can be built in two directions, set by `FORM` near the top of the script. `"forward"` is the default and the form the paper uses. It builds the objects on `M^T` and follows a displacement from a sector to the users of its output, so it answers questions about downstream reach. `"backward"` builds them on `D^-1 M D` with `D = diag(X_REF)`, the demand-driven response of the embedded process expressed in the same fractional units. It follows a displacement to a sector's suppliers, and its long-run emission multiplier equals the standard IO-LCA multiplier. Block 12 (b) prints both forms side by side whichever is selected. Leave `FORM` at its default to reproduce the paper's objects.

## Dependencies

`numpy` (version 2.0 or later), `scipy`, and `matplotlib` are required. `matplotlib` runs under the non-interactive Agg backend, so no display is needed.

```
pip install "numpy>=2.0" scipy matplotlib
```

## Usage

From the `examples` folder:

```
python3 examples.py
```

The console record is written to `examples/outputs/examples_output.txt`.

## Adapting to your own system

The embedding is agnostic to the provenance of `A_io`. That is, any square technical-coefficients matrix admits the same construction, whether drawn from a single-region national table or a multi-regional system, and whether sourced from databases such as WIOD, FIGARO, ADB, EXIOBASE, GLORIA, EORA, or GTAP. Edit the toy-system block near the top of the script and the transitions beneath it: replace `A_io`, `B`, `X_REF`, and `SECTORS`, and restate the demand shifts (`DEMAND_SHIFTS`, which `DELTA_Z`, `DZ_REPAIR` and `TRANSITIONS` are built from, with `SCENARIO` and `REPAIR` naming the two that the blocks use), the emission change `DELTA_B` and the coefficient revision `RECYCLE` by sector name. The script makes no assumption on the number of sectors. `A_io` holds technical coefficients in monetary units, `X_REF` the reference outputs in the same units, `B` the reference annual emissions of each sector (intensity times reference output), and each demand-shift entry a fraction of that sector's reference output. The remaining blocks then run on the adapted system, provided it is productive and stable, which the script checks. Properties that only illustrate the toy, such as the standard-versus-Type-I ranking difference, need not hold on another system. The manuscript requires the free model choices to be reported with sensitivity, so read block 12 on the adapted system, since the horizon, form, noise-scale, and norm effects differ in size from one system to another.

## Licence

Details are provided in the LICENSE file in the main project folder.

### Cite as:

See `README.md` in the main project folder.
