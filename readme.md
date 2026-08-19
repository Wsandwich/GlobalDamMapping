# Uneven global dam change shapes divergent progress of free-flowing rivers

This repository contains analysis and visualization code supporting the manuscript **“Uneven global dam change shapes divergent progress of free-flowing rivers.”** The study maps global dams in 2010, 2015, and 2020 and reports a 27.0% increase in global dam numbers over the decade, together with geographically divergent changes in river fragmentation and free-flowing rivers.

## Repository status

This repository is a submission-stage release of the supporting analysis code. Additional workflow components, documentation, configuration examples, and reproducibility materials are being consolidated and will be added in subsequent versioned releases.

![dam mapping](https://github.com/user-attachments/assets/aff34b6b-d1e6-4098-9c51-fa5973992cc2)

## Repository structure

```text
├── result1/    # Dam-count aggregation, spatial summaries, and figure/table generation
├── result2/    # Policy and socio-environmental analyses, Monte Carlo propagation, and spatial regression
├── result3/    # River-fragmentation and free-flowing-river analyses
├── tools/      # Multi-temporal dam-annotation tools
└── doc/        # Overview figures and supporting documentation
```

## Code-to-analysis mapping

| Manuscript analysis | Representative code |
| --- | --- |
| Global dam counts, dam types, continental and latitudinal summaries | `result1/count_*.py`, `result1/compare_*.py`, and `result1/draw_*.py` |
| Monte Carlo uncertainty propagation and spatial-lag regression | `result2/common_mc.py` and `result2/exp8_spatial_mc_fast_nogdpc_repl2010.py` |
| Dam-to-river assignment and degree-of-fragmentation calculations | `result3/dam_river_proximity_*.py` and `result3/dof_v3.py` |
| Monte Carlo propagation for river-fragmentation metrics | `result3/mc_dof/02_mc_dof_simulation_v4.py` |
| Monte Carlo convergence diagnostics | `result3/mc_dof/04_mc_convergence_v2.py` |
| Figure and table generation | plotting and export scripts under `result1/`, `result2/`, and `result3/` |
| Multi-temporal annotation and quality control | `tools/roLabelImg4.py` and associated utilities |

## Environment and configuration

The analysis scripts use Python 3.8 or later. Principal dependencies include:

```text
NumPy
Pandas
GeoPandas
GDAL
Rasterio
Matplotlib
SciPy
statsmodels
libpysal
esda
spreg
joblib
scikit-learn
numba
```

The spatial-regression workflow accepts environment-specific input, output, and run settings through variables including:

```text
RESULT1_OUTPUT_ROOT
RESULT2_OUTPUT_ROOT
RESULT1_RUN_TAG
RESULT2_RUN_TAG
HLZ_DATA_DIR
HLZ_SHP_TEMPLATE
MC_CACHE_DIR
MC_CACHE_TEMPLATE
ATTR_EXP
ATTR_LEV
N_SIM
N_JOBS
OUTPUT_BASE_DIR
OUTPUT_DIR
```

For example, after setting the required paths, run the spatial-regression and Monte Carlo workflow from the repository root:

```text
python result2/exp8_spatial_mc_fast_nogdpc_repl2010.py
```

The river-fragmentation Monte Carlo scripts expose command-line arguments so that local paths do not need to match the default computing environment:

```text
python result3/mc_dof/02_mc_dof_simulation_v4.py --cache-path PATH/TO/basin_cache.pkl --results-path PATH/TO/mc_results_v4.pkl --n-sim 1000 --seed 42
python result3/mc_dof/04_mc_convergence_v2.py --cache-path PATH/TO/basin_cache.pkl --out-path PATH/TO/mc_dof_convergence.png --seed 42
```

Large input datasets and intermediate caches are not included in this repository. Input sources and processing definitions are documented in the manuscript and its Supplementary Information.

## Data availability

The complete machine-readable, high-resolution geospatial dataset describing individual dam locations, orientations, and geometries is not publicly released owing to sensitivities associated with critical water infrastructure. Aggregated dam counts and changes, non-georeferenced river-fragmentation and free-flowing-river metrics, and source data underlying the reported figures and tables will be made publicly available upon publication at the reserved Figshare DOI: https://doi.org/10.6084/m9.figshare.33245955.

## Licensing and provenance

This repository is distributed under the GNU General Public License v3.0; see `LICENSE`. The Free-Flowing-Rivers-derived components under `result3/` retain the same license in `result3/LICENSE` and are associated with Grill *et al.*, “Mapping the world's free-flowing rivers” (2019), https://doi.org/10.1038/s41586-019-1111-9.

## Contact

Questions about the manuscript and supporting code should be directed to the corresponding author through the contact information provided in the manuscript.
