"""
Basis tests 
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pyEXP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))

from basis_helpers import compute_basis


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src/tests/data"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
TMP_BASIS_DIR = Path(__file__).resolve().parent / "_basis_tmp"
TEST_NMAX = 8
TEST_LMAX = 2


def load_basis_fit() -> tuple[np.ndarray, np.ndarray]:
    """Load the normalized density profile fit used by the notebook basis build."""

    fit_path = DATA_DIR / "halo_21537_normalized_density_profile_fit.txt"
    return np.loadtxt(fit_path)


def build_halo_21537_basis(nmax: int = 8, lmax: int = 2):
    """Compute the halo 21537 basis using the same inputs as notebook cell 9.

    The notebook cell uses:

    - ``nmax = 8``
    - ``lmax = 2``
    - ``basis = build_basis(r_norm_fit, rho_norm_fit, nmax=nmax, lmax=lmax)``

    This function reproduces that construction through ``basis_helpers.compute_basis``.
    """

    r_norm_fit, rho_norm_fit = load_basis_fit()
    TMP_BASIS_DIR.mkdir(parents=True, exist_ok=True)

    basis_params = {
        "rmin": float(r_norm_fit[0]),
        "rmax": float(r_norm_fit[-1]),
        "rmapping": 1.0,
        "Mtotal": 1.0,
        "nbins": 2000,
        "lmax": int(lmax),
        "nmax": int(nmax),
        "cachename": str(
            TMP_BASIS_DIR / f"test_halo_21537_basis_cache_{nmax:02d}_{lmax:02d}.txt"
        ),
        "modelname": str(TMP_BASIS_DIR / "halo_21537_model.txt"),
        "basis_id": "sphereSL",
    }

    basis = compute_basis(
        basis_params,
        r_norm_fit,
        rho_norm_fit,
        str(TMP_BASIS_DIR),
        f"test_halo_21537_basis_config_{nmax:02d}_{lmax:02d}.yaml",
    )
    return basis



def load_test_halo_21537_basis(nmax: int = 8, lmax: int = 2):
    """Compute the halo 21537 basis using the same inputs as notebook cell 9.

    The notebook cell uses:

    - ``nmax = 8``
    - ``lmax = 2``
    - ``basis = build_basis(r_norm_fit, rho_norm_fit, nmax=nmax, lmax=lmax)``

    This function reproduces that construction through ``basis_helpers.compute_basis``.
    """

    basis_configname = f"halo_21537_basis_config_{nmax:02d}_{lmax:02d}.yaml"
    basis_config_path = DATA_DIR / basis_configname
    with open(basis_config_path, "r", encoding="utf-8") as f:
        basis_yaml = f.read()

    # Config files reference model/cache with relative paths; resolve from DATA_DIR.
    cwd_path = os.getcwd()
    try:
        os.chdir(DATA_DIR)
        basis = pyEXP.basis.Basis.factory(basis_yaml)
    finally:
        os.chdir(cwd_path)
  
    return basis


def test_basis_density_matches_reference() -> None:
    basis_obj = load_test_halo_21537_basis(nmax=TEST_NMAX, lmax=TEST_LMAX)
    basis_tests = build_halo_21537_basis(nmax=TEST_NMAX, lmax=TEST_LMAX)

    basis_obj_values = basis_obj.getBasis(1e-2, 1.2, numr=400)
    basis_tests_values = basis_tests.getBasis(1e-2, 1.2, numr=400)

    for i in range(TEST_LMAX):
        np.testing.assert_allclose(
            basis_obj_values[i][0]["density"],
            basis_tests_values[i][0]["density"],
        )


if __name__ == "__main__":
    test_basis_density_matches_reference()
