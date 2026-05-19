
"""Test density profile computation from basis and coefficients."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pyEXP

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src/tests/data"
SRC_DIR = REPO_ROOT / "src"
TESTS_OUTPUT_DIR = Path(__file__).resolve().parent / "_temp_tests_outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exp.data_ios import read_density_profile, read_halo_params
from exp.fields import bfe_density_profiles


def test_density_profiles_two_snapshots(nmax: int = 8, lmax: int = 2) -> None:
    """
    Test density profile computation for halo 21537 at snapshots 99 and 50.
    
    Compares BFE density profiles (computed from basis + coefficients)
    against particle density profiles and generates comparison plots.
    """
    
    # Setup paths (matching test_coefficients_computation.py)
    tests_dir = Path(__file__).resolve().parent
    basis_config = DATA_DIR / f"halo_21537_basis_config_{nmax:02d}_{lmax:02d}.yaml"
    coefs_file = tests_dir / f"data/halo_21537_coefficients_{nmax:02d}_{lmax:02d}.h5"
    halo_params = REPO_ROOT / "data" / "tng35-3-dark" / "halo_21537" / "halo_21537_params.hdf5"
    density_filename = REPO_ROOT / "data" / "tng35-3-dark" / "halo_21537" / "profiles" / "halo_21537_density_profiles.hdf5"
    
    assert basis_config.exists(), f"Missing basis config: {basis_config}"
    assert coefs_file.exists(), f"Missing coefficients file: {coefs_file}"
    assert halo_params.exists(), f"Missing halo params: {halo_params}"
    assert density_filename.exists(), f"Missing density profiles: {density_filename}"
    
    # Load basis
    print(f"Loading basis from {basis_config}...")
    from compute_coefficients import load_basis_from_config_file
    basis = load_basis_from_config_file(str(basis_config))
    print(f"  Basis type: {type(basis).__name__}")
    
    # Load coefficients
    print(f"Loading coefficients from {coefs_file}...")
    coefs = pyEXP.coefs.Coefs.factory(str(coefs_file))
    times = coefs.Times()
    print(f"  Coefficient times (redshifts): {times}")
    
    # Load halo parameters
    halo_params_data = read_halo_params(str(halo_params))
    all_snaps = halo_params_data['snap']
    R200c_arr = halo_params_data['R200c']
    rho200c_arr = halo_params_data['M200c'] / (4/3. * np.pi * R200c_arr**3)
    
    # Build lookup dicts
    snap_arr = np.asarray(all_snaps, dtype=int)
    R200c_halo = {int(s): float(v) for s, v in zip(snap_arr, R200c_arr)}
    rho200c_halo = {int(s): float(v) for s, v in zip(snap_arr, rho200c_arr)}
    
    # Test snapshots
    test_snaps = [99, 50]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, snap in enumerate(test_snaps):
        print(f"\nProcessing snapshot {snap}...")
        
        # Read particle density profile
        r_part, rho_part = read_density_profile(str(density_filename), snap=snap)
        r_over_r200c = r_part / R200c_halo[snap]
        rho_part_norm = rho_part / rho200c_halo[snap]
        
        # Compute BFE density profile
        print(f"  Computing BFE density profile...")
        redshift = times[::-1][snap - 2]  # Snapshot index mapping
        rho_bfe = bfe_density_profiles(
            basis=basis,
            coefs=coefs,
            r_bins=r_over_r200c,
            time=redshift,
            theta_bins=20,
            phi_bins=40,
            statistic='weighted_mean',
        )
        rho_bfe_norm = np.abs(rho_bfe)# / rho200c_halo[snap]
        
        # Plot
        ax = axes[idx]
        ax.loglog(
            r_over_r200c,
            rho_part_norm,
            'k-',
            lw=1.5,
            label='Particle density',
            alpha=0.7,
        )
        ax.loglog(
            r_over_r200c,
            rho_bfe_norm,
            'C0-',
            lw=1.5,
            label='BFE density',
            alpha=0.8,
        )
        
        ax.set_xlim(0.5e-2, 1.4)
        ax.set_ylim(1e-3, 2e3)
        ax.set_xlabel(r'$r / r_{200c}$', fontsize=14)
        ax.set_ylabel(r'$\rho / \rho_{200c}$', fontsize=14)
        ax.set_title(f'Snapshot {snap} (z={redshift:.1f})', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        print(f"  ✓ Snapshot {snap} complete")
    
    plt.tight_layout()
    TESTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = TESTS_OUTPUT_DIR / f"test_density_profiles_21537_{nmax:02d}_{lmax:02d}.png"
    plt.savefig(plot_path, dpi=100)
    print(f"\nPlot saved to {plot_path}")
    plt.show()
    
    print("✓ Density profile test passed")


if __name__ == "__main__":
    test_density_profiles_two_snapshots(nmax=8, lmax=2)