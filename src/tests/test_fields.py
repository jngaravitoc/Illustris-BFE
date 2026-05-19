"""Regression tests for KDE and BFE 3-D density fields."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pyEXP

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src/tests/data"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compute_coefficients import load_basis_from_config_file
from exp.compute_fields import compute_bfe_fields, compute_kde_density
from exp.data_ios import read_halo_params, read_tng_halo_particles


def _build_test_grid() -> np.ndarray:
    """Build the same normalized Cartesian grid used to generate references."""
    dbins = np.linspace(-1.1, 1.1, 50)
    grid_arrays = np.meshgrid(dbins, dbins, dbins, indexing="ij")
    return np.stack(grid_arrays)


def _load_reference_bfe_density(path: Path) -> tuple[str, np.ndarray]:
    """Return time key and dens array from a reference BFE field file."""
    with h5py.File(path, "r") as f:
        time_keys = list(f.keys())
        assert len(time_keys) == 1, f"Expected one time key in {path}, got {time_keys}"
        time_key = time_keys[0]
        dens = np.asarray(f[time_key]["dens"])
    return time_key, dens


def _load_reference_kde_density(path: Path) -> np.ndarray:
    """Return kde_density array from a reference KDE field file."""
    with h5py.File(path, "r") as f:
        return np.asarray(f["kde_density"])


def test_kde_and_bfe_fields_match_reference(nmax: int = 8, lmax: int = 2) -> None:
    """Validate halo 21537 KDE/BFE fields at snapshots 99 and 50."""

    basis_config = DATA_DIR / f"halo_21537_basis_config_{nmax:02d}_{lmax:02d}.yaml"
    coefs_file = DATA_DIR / f"halo_21537_coefficients_{nmax:02d}_{lmax:02d}.h5"
    halo_params_file = REPO_ROOT / "data" / "tng35-3-dark" / "halo_21537_params.hdf5"

    assert basis_config.exists(), f"Missing basis config: {basis_config}"
    assert coefs_file.exists(), f"Missing coefficients file: {coefs_file}"
    assert halo_params_file.exists(), f"Missing halo params file: {halo_params_file}"

    basis = load_basis_from_config_file(str(basis_config))
    coefs = pyEXP.coefs.Coefs.factory(str(coefs_file))

    halo_params = read_halo_params(str(halo_params_file))
    snap_arr = np.asarray(halo_params["snap"], dtype=int)
    r200c_arr = np.asarray(halo_params["R200c"], dtype=float)
    rho200c_arr = np.asarray(halo_params["M200c"], dtype=float) / (
        4.0 / 3.0 * np.pi * r200c_arr**3
    )

    r200c_halo = {int(s): float(v) for s, v in zip(snap_arr, r200c_arr)}
    rho200c_halo = {int(s): float(v) for s, v in zip(snap_arr, rho200c_arr)}

    grid = _build_test_grid()
    sim = "tng35-3-dark"
    halo_subfind_id = 21537
    snap_basename = "galaxies_halo_{subfind_id}_tng50-3-dark_{snap:03d}.hdf5"
    snapshots = [99, 50]

    for snap in snapshots:
        kde_ref_file = DATA_DIR / f"halo_21537_kde_density_field_snap_{snap:03d}.h5"
        bfe_ref_file = DATA_DIR / f"halo_21537_bfe_density_{nmax:02d}_{lmax:02d}_snap_{snap:03d}.h5"
        particle_file = (
            REPO_ROOT
            / "data"
            / sim
            / snap_basename.format(subfind_id=halo_subfind_id, snap=snap)
        )

        assert kde_ref_file.exists(), f"Missing KDE reference: {kde_ref_file}"
        assert bfe_ref_file.exists(), f"Missing BFE reference: {bfe_ref_file}"
        assert particle_file.exists(), f"Missing particle file: {particle_file}"

        coords, masses = read_tng_halo_particles(str(particle_file))
        pos_norm = coords / r200c_halo[snap]
        mass_norm = masses / (rho200c_halo[snap] * r200c_halo[snap] ** 3)

        kde_calc = compute_kde_density(pos_norm, mass_norm, grid)
        kde_ref = _load_reference_kde_density(kde_ref_file)
        np.testing.assert_allclose(kde_calc, kde_ref)

        ref_time_key, bfe_ref_dens = _load_reference_bfe_density(bfe_ref_file)
        eval_time = float(ref_time_key)
        dens_bfe_list, _fp, points = compute_bfe_fields(grid, basis, coefs, [eval_time])

        # Compare both direct list output and raw points output with file reference.
        np.testing.assert_allclose(dens_bfe_list[0].reshape(-1), bfe_ref_dens)
        bfe_calc_dens = np.asarray(points[eval_time]["dens"])
        np.testing.assert_allclose(bfe_calc_dens, bfe_ref_dens)


if __name__ == "__main__":
    test_kde_and_bfe_fields_match_reference()
    print("Field regression test passed.")