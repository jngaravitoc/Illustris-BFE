# Tests Overview

This folder contains integration and helper tests for the halo 21537 BFE workflow.

## Prerequisites

- Python environment with `pyEXP` available.
- Recommended interpreter:
  - `/home/ngc/Work/research/codes/environments/pyexp310/bin/python`
- Run commands from repository root unless noted otherwise.

## Files in This Folder

- `test_basis.py`
  - Rebuilds two basis objects from the same fitted profile inputs.
  - `build_halo_21537_basis` defaults to `compute_covariance=True`, which sets
    `pcavar`, `samplesz`, `totalCovar`, and `fullCovar` in the basis parameters
    so pyEXP initialises covariance storage for downstream `writeCoefCovariance` calls.
  - Compares density arrays from:
    - `getBasis(1e-2, 1.2, numr=400)`
  - Uses `np.testing.assert_allclose` for the first 8 entries.

- `test_coefficients_computation.py`
  - Loads basis config from `src/tests/data/halo_21537_basis_config_08_02.yaml`
    (includes covariance keys: `pcavar`, `samplesz`, `totalCovar`, `fullCovar`).
  - Runs `compute_coefficients_for_snapshots` to generate coefficients.
  - Supports two modes via `ILLUSTRIS_BFE_COEFS_TEST_MODE`:
    - `lite`: every 10th snapshot only, quick sanity check (default for CI).
    - `full`: all snapshots in halo params and stanza comparison against reference.
  - Contains three test functions:
    - `test_compute_coefficients`: smoke test; asserts output HDF5 is created and non-empty.
    - `test_coefficients_computation_matches_reference`: compares against reference coefficients
      in `src/tests/data/halo_21537_coefficients_08_02.h5`.
    - `test_covariance_computation`: runs with `covariance=True`, asserts that
      `coefcovar.halo.cov.h5` is created in the output directory and is readable
      via `pyEXP.basis.CovarianceReader`.

- `test_bfe_profiles.py`
  - Helper loader utilities for exported profile tables and metadata parsing.
  - Currently no pytest test function in this file.

## Data/Asset Dependencies

Expected test assets include:

- `src/tests/data/halo_21537_basis_config_08_02.yaml` (includes covariance keys)
- `src/tests/data/halo_21537_basis_cache_08_02.txt`
- `src/tests/data/halo_21537_coefficients_08_02.h5` (reference coefficients)
- `src/tests/data/halo_21537_normalized_density_profile_fit.txt`
- `data/tng35-3-dark/halo_21537/halo_21537_params.hdf5`
- `data/tng35-3-dark/halo_21537/particle_data/` snapshot files

## How to Run

### 0) One-command test runner

From `src/tests`:

```bash
./run_tests.sh
```

Modes:

```bash
./run_tests.sh lite
./run_tests.sh full
./run_tests.sh basis
./run_tests.sh coeff-lite
./run_tests.sh coeff-full
```

Optional interpreter override:

```bash
PYTHON_BIN=/home/ngc/Work/research/codes/environments/pyexp310/bin/python ./run_tests.sh smoke
```

### 1) Basis comparison test

From repo root:

```bash
/home/ngc/Work/research/codes/environments/pyexp310/bin/python src/tests/test_basis.py
```

### 2) Coefficients test (lite mode, faster)

From repo root:

```bash
ILLUSTRIS_BFE_COEFS_TEST_MODE=lite /home/ngc/Work/research/codes/environments/pyexp310/bin/python src/tests/test_coefficients_computation.py
```

### 3) Coefficients test (full comparison)

From repo root:

```bash
/home/ngc/Work/research/codes/environments/pyexp310/bin/python src/tests/test_coefficients_computation.py
```

## Notes

- The reference basis config (`src/tests/data/halo_21537_basis_config_08_02.yaml`) includes
  `pcavar: true`, `samplesz: 1`, `totalCovar: true`, `fullCovar: false` so the basis loads
  with covariance storage initialised. If you regenerate this file, make sure those keys are present.
- Covariance files are written as `coefcovar.{component}.{runtag}.h5` (e.g. `coefcovar.halo.cov.h5`)
  in the same directory as the coefficients output file. The runtag used by `compute_exp_coefs` is `'cov'`.
- `covariance` defaults to `False` in `compute_coefficients_for_snapshots` and `PipelineConfig`
  (opt-in at the pipeline level). Set `covariance: true` in your pipeline YAML to enable it.
- If a coefficients run fails with YAML parsing errors, verify that `src/compute_coefficients.py`
  loads YAML file content and passes that content to `pyEXP.basis.Basis.factory(...)`.
