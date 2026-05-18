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
  - Compares density arrays from:
    - `getBasis(1e-2, 1.2, numr=400)`
  - Uses `np.testing.assert_allclose` for the first 8 entries.

- `test_coefficients_computation.py`
  - Loads basis config from `src/tests/_basis_tmp/halo_21537_basis_config_16_08.yaml`.
  - Runs `src/compute_coefficients.py` to generate:
    - `_coefs_tmp/test_halo_21537_coefficients_16_08.h5`
  - Compares generated coefficients with reference coefficients in `src/tests/`.
  - Supports two modes:
    - `smoke`: first 2 snapshots only, quick sanity check.
    - `full`: all snapshots in halo params and stanza comparison against reference.

- `test_bfe_profiles.py`
  - Helper loader utilities for exported profile tables and metadata parsing.
  - Currently no pytest test function in this file.

## Data/Asset Dependencies

Expected test assets include:

- `src/tests/_basis_tmp/halo_21537_basis_config_16_08.yaml`
- `src/tests/halo_21537_coefficients_16_08.h5` (or legacy spelling variant)
- `data/tng35-3-dark/halo_21537_params.hdf5`
- `data/tng35-3-dark/` snapshot files used by coefficient computation
- `data/dimer_density_profile_fit.txt` (used by basis-related workflows)

## How to Run

### 0) One-command test runner

From `src/tests`:

```bash
./run_tests.sh
```

Modes:

```bash
./run_tests.sh smoke
./run_tests.sh full
./run_tests.sh basis
./run_tests.sh coeff-smoke
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

### 2) Coefficients test (smoke mode, faster)

From `src/tests`:

```bash
cd src/tests
ILLUSTRIS_BFE_COEFS_TEST_MODE=smoke /home/ngc/Work/research/codes/environments/pyexp310/bin/python test_coefficients_computation.py
```

### 3) Coefficients test (full comparison)

From `src/tests`:

```bash
cd src/tests
/home/ngc/Work/research/codes/environments/pyexp310/bin/python test_coefficients_computation.py
```

## Notes

- If a coefficients run fails with YAML parsing errors, verify that `src/compute_coefficients.py` loads YAML file content and passes that content to `pyEXP.basis.Basis.factory(...)`.
- `test_coefficients_computation.py` prints a short debug preview of the basis YAML file path and first lines to help diagnose basis-load issues quickly.
