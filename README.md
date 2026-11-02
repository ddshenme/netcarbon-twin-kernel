# Net-carbon intelligent control for water-energy-carbon synergy in wastewater treatment plants - an accelerated digital-twin kernel

Reference implementation for the paper:

> Jiashu Ren, Ying Fang. *Net-carbon intelligent control for water-energy-carbon synergy in wastewater treatment plants - an accelerated digital-twin kernel*. Environmental Modelling & Software, 2026.

## What this repository contains

The numba-compiled ASM1 + Takács digital-twin kernel (4 modules, 772 lines) together with the closed-loop PI layer, the scan driver, the cross-validation script and the figure/table scripts (1989 lines in total).

## Quick start

```bash
pip install -r requirements.txt
python verify.py xval 13.9   # bitwise cross-check against the official bsm2-python
python sweep.py all           # 261-point steady-state scan (~12 min on 16 cores)
```

## Reproducing the figures of the paper

```bash
python make_figures.py
```

`figures/Fig1_emission_factor.*` and `figures/Fig2_operating_map.*` correspond to Figures 1 and 2 of the paper.

Every number in the paper can be regenerated from the scripts above; the archived
JSON/CSV outputs of the original runs are kept in `data/` so that the figures can be
redrawn without re-running the simulations (a full re-run takes minutes on 16 cores).

## Repository layout

```
    bsm1_fast.py      numba kernel of ASM1 + Takács clarifier
    n2o.py            two-step nitrification N2O overlay
    twin.py           steady-state twin surrogate (bilinear interpolation)
    sweep.py          steady-state setpoint scan (3 grids)
    control_arms.py   plant-level PI controllers
    loop.py           closed-loop integration driver
    env.py            simulation environment / influent handling
    verify.py         bitwise cross-check against bsm2-python
    bsm1_plant.py     reference plant used by verify.py
    make_figures.py   redraws Figures 1-2 from data/
    data/             archived scan results + official BSM1 influent
    figures/          Figure 1 and Figure 2 (PNG + vector PDF)
```

## Data

`data/` contains the archived results of the runs reported in the paper plus the
official IWA BSM1 influent files (`dryinfluent.csv`) and the pre-computed
steady-state state vector (`_ss_A_n2o.npy`, 2 KB) used as the initial condition.
No synthetic data are used anywhere.

## License and citation

MIT, see `LICENSE`. If you use this code please cite the paper above; a `CITATION.cff`
is provided so that GitHub/Zenodo can generate the BibTeX entry for you.
