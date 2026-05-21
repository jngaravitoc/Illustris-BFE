# Illustris-BFE

Illustris-BFE is a halo analysis pipeline for building Basis Function Expansion (BFE)
models of Illustris/TNG halos, reconstructing density fields, and evaluating fidelity
metrics against KDE reference fields.

## What the Pipeline Does

For each halo, the pipeline runs up to 5 stages in dependency order:

1. `profiles`
   Loads or computes halo density profiles.
2. `basis`
   Builds basis functions given the profile fit.
3. `coefficients`
   Computes BFE coefficients from particle snapshots.
4. `fields`
   Generates BFE and KDE density fields per snapshot.
5. `metrics`
   Computes fidelity metrics (MISE, MIRSE), writes metrics HDF5, and generates
   summary figures. Optional: generate per-snapshot 2-D metric maps.

## Configuration

Use a YAML config file (see `src/analysis/example_config.yaml`) with:

- simulation + halo information (`simulation`, `halo_id`)
- expansion orders (`nmax`, `lmax`; scalar or paired lists)
- snapshots (`snapshots` list or `all`)
- grid settings (`grid_range`, `grid_bins`)
- projection axis for metric-map reductions (`spatial_axis`, default `2`)
- stage toggles (`stages.*`)
- metrics map toggle (`metrics_generate_all_snapshot_maps`)
- data paths/patterns (`data_root`, `halo_params_file`, `profile_fit_file`, `particles_file_pattern`)

## How to Run

Run from repository root.

### 1) Run all enabled stages for one config

```bash
python src/run_pipeline.py src/analysis/example_config.yaml
```

### 2) Run only selected stages

```bash
python src/run_pipeline.py src/analysis/example_config.yaml --stages fields metrics
```

### 3) Validate config and resolved paths without executing

```bash
python src/run_pipeline.py src/analysis/example_config.yaml --dry-run
```

### 4) Override halo id from CLI

```bash
python src/run_pipeline.py src/analysis/example_config.yaml --halo-id 21537
```

### 5) Batch mode over halos in a CSV

```bash
python src/run_pipeline.py src/analysis/example_config.yaml --batch data/TNG_parameters.csv
```

Parallel batch mode:

```bash
python src/run_pipeline.py src/analysis/example_config.yaml --batch data/TNG_parameters.csv --parallel --workers 8
```

## Metrics Outputs

For each `(halo_id, nmax, lmax)` run:

- Metrics file:
  `data/<sim>/halo_<id>/metrics/halo_<id>_metrics_<nmax>_<lmax>.h5`
- Summary figure:
  `data/<sim>/halo_<id>/figures/halo_<id>_metrics_<nmax>_<lmax>.pdf`

If `metrics_generate_all_snapshot_maps: true`, the metrics stage also writes one
2-D map figure per snapshot under the halo `figures` directory.

## Notes

- If metrics files already exist, stages may skip recomputation.
- The metrics stage includes compatibility handling for legacy metrics files;
  when map generation is requested, it can upgrade legacy scalar-format files
  by recomputing matrix-valued metrics from available fields.
