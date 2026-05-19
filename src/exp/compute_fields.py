#!/usr/bin/env python3
"""
Compute BFE and KDE density fields on a 3-D Cartesian grid.

This script computes BFE (basis-function expansion) and KDE (kernel
density estimation) fields for a specified halo and writes them to HDF5
files.  It is designed to be run *before* ``make_dashboard.py``, which
reads the pre-computed fields and produces comparison figures.

The script checks for existing output files and skips snapshots whose
time keys already appear in the BFE HDF5 file or whose per-snapshot KDE
files already exist, making it safe to restart after a partial run.

Usage
-----
    python compute_fields.py --halo_id 108 --output_dir ./output
    python compute_fields.py --halo_id 108 --output_dir ./output --grid_bins 40
    python compute_fields.py --halo_id 108 --output_dir ./output --skip_kde
"""

import sys
import os
import argparse

import numpy as np
import h5py

# Add sibling directories to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "visuals"))
sys.path.append(os.path.join(_THIS_DIR, "../exp_pipeline/"))

from field_projections import FieldProjections
try:
    from field_io import (
        write_kde_density,
        merge_kde_density_files,
    )
except ModuleNotFoundError:
    from visuals.field_io import (
        write_kde_density,
        merge_kde_density_files,
    )
#from ios_nbody_sims import load_particle_data
#from compute_bfe_helpers import load_sheng24_exp_center
#from basis_utils import load_basis

import pyEXP

_EXP_ROOT = "/n/nyx3/garavito/projects/XMC-Atlas/scripts/exp_expansions"


# ------------------------------------------------------------------
# Core computation functions
# ------------------------------------------------------------------




def compute_bfe_fields(grid, basis, coefs, eval_times):
    """Compute BFE fields on a grid for given evaluation times.

    Parameters
    ----------
    grid : ndarray, shape (3, N, N, N)
        Stacked meshgrid arrays (x, y, z).
    basis : pyEXP basis object
        The spherical basis expansion.
    coefs : pyEXP coefs object
        Coefficient container.
    eval_times : array-like
        Snapshot times at which to evaluate.

    Returns
    -------
    dens_bfe_list : list of ndarray
        BFE density arrays, one per time in *eval_times*.
    FP : FieldProjections
        The field-projections helper (retains grid/mesh info).
    points : dict
        ``{time: {field_name: ndarray, ...}, ...}`` — full field
        dictionary returned by ``pyEXP.field.FieldGenerator.points``.
    """
    FP = FieldProjections(grid, basis, coefs, eval_times)

    print("Computing EXP fields...")
    points = FP.compute_fields_in_points()

    dens_bfe_list = []
    for t in eval_times:
        dens_bfe_list.append(FP.twod_field(points, t, "dens")[0])

    return dens_bfe_list, FP, points


def compute_BFE_fields_in_grid(halo_id, component, grid_range=(-100, 100), grid_bins=20,
                           basis_dir=None, coefs_dir=None):
    """Load basis/coefficients, build a grid, and compute BFE fields.

    Parameters
    ----------
    halo_id : int
        Halo model ID (e.g., 108).
    grid_range : tuple of float, optional
        ``(min, max)`` extent in kpc (default ``(-100, 100)``).
    grid_bins : int, optional
        Number of bins per axis (default 20).
    basis_dir : str, optional
        Absolute path to the basis directory
        (default ``/n/nyx3/garavito/projects/XMC-Atlas/exp_expansions/basis``).
    coefs_dir : str, optional
        Absolute path to the coefficients directory
        (default ``/n/nyx3/garavito/projects/XMC-Atlas/exp_expansions/coefficients``).

    Returns
    -------
    dens_bfe_list : list of ndarray
    FP : FieldProjections
    times : ndarray
    points : dict
    grid_arrays : list of ndarray
    """
    if basis_dir is None:
        basis_dir = os.path.join(_EXP_ROOT, "basis")
    if coefs_dir is None:
        coefs_dir = os.path.join(_EXP_ROOT, "coefficients")

    # Load basis (chdir needed — pyEXP reads cache files relative to cwd)
    cwd_save = os.getcwd()
    os.chdir(basis_dir)

    # Load coefficients
    # MWhalo coefs
    if component == 'MWhalo':
        config_name = os.path.join(basis_dir, f"basis_halo_{halo_id:04d}.yaml")
        coefs_file = os.path.join(coefs_dir, f"halo_{halo_id:04d}_coefficients_center.h5")
    elif component == 'lmc':
        config_name = os.path.join(basis_dir, f"basis_init_lmc_{halo_id:04d}.yaml")
        coefs_file = os.path.join(coefs_dir, f"lmc_init_{halo_id:04d}_coefficients.h5")
    
    print(f"Loading basis from {config_name}...")
    basis = load_basis(config_name)
    print("  Basis loaded")
    os.chdir(cwd_save)
    print(f"Loading coefficients from {coefs_file}...")
    coefs = pyEXP.coefs.Coefs.factory(coefs_file)
    times = coefs.Times()
    print(f"  Found {len(times)} time snapshots")

    power = coefs.Power()
    print(f"  Power range: {power}")

    # Create grid
    dbins = np.linspace(grid_range[0], grid_range[1], grid_bins)
    grid_arrays = np.meshgrid(dbins, dbins, dbins, indexing="ij")
    grid = np.stack(grid_arrays)
    print(f"Grid created: {grid_bins} x {grid_bins} x {grid_bins}")

    # Compute BFE fields
    print("\nComputing BFE fields...")
    dens_bfe_list, FP, points = compute_bfe_fields(grid, basis, coefs, times)
    print("Field points computed")

    return dens_bfe_list, FP, times, points, grid_arrays


# ------------------------------------------------------------------
# BFE field writing (with per-time skip logic)
# ------------------------------------------------------------------

def _existing_time_keys(filename):
    """Return the set of time-key strings already stored in *filename*."""
    if not os.path.isfile(filename):
        return set()
    with h5py.File(filename, "r") as f:
        return set(f.keys())


def compute_all_bfe_fields(halo_id, component, output_dir, grid_range=(-100, 100),
                           grid_bins=20, basis_dir=None, coefs_dir=None):
    """Compute BFE fields for every snapshot and write to HDF5.

    If the output file already exists, snapshots whose time keys are
    already present are skipped.

    Parameters
    ----------
    halo_id : int
    output_dir : str
    grid_range : tuple of float, optional
    grid_bins : int, optional
    basis_dir : str, optional
    coefs_dir : str, optional

    Returns
    -------
    dens_bfe_list : list of ndarray
    FP : FieldProjections
    times : ndarray
    grid_arrays : list of ndarray
    """
    if component == "MWhalo":
        fields_file = os.path.join(output_dir, f"halo_{halo_id:04d}_BFE_fields.h5")

    elif component == "lmc":
        fields_file = os.path.join(output_dir, f"lmc_init_{halo_id:04d}_BFE_fields.h5")

    # Compute all fields (needed in memory for dashboards later)
    dens_bfe_list, FP, times, points, grid_arrays = compute_BFE_fields_in_grid(
        halo_id, component, grid_range, grid_bins, basis_dir=basis_dir, coefs_dir=coefs_dir
    )

    # Determine which times still need writing
    existing = _existing_time_keys(fields_file)
    new_points = {
        t: fields for t, fields in points.items()
        if str(t) not in existing
    }

    if not new_points:
        print(f"All {len(times)} BFE snapshots already in {fields_file} — skipping write.")
    else:
        n_skip = len(points) - len(new_points)
        if n_skip:
            print(f"Skipping {n_skip} existing snapshots, writing {len(new_points)} new ones.")

        # Append new time groups to the file
        mode = "a" if os.path.isfile(fields_file) else "w"
        with h5py.File(fields_file, mode) as f:
            # Write grid shape once
            if "grid_shape" not in f.attrs:
                f.attrs["grid_shape"] = np.asarray(grid_arrays[0]).shape

            for time_key, fields_dict in new_points.items():
                grp = f.create_group(str(time_key))
                for field_name, data in fields_dict.items():
                    grp.create_dataset(
                        field_name,
                        data=np.asarray(data),
                        compression="gzip",
                        compression_opts=4,
                    )
        print(f"BFE fields written to {fields_file}")

    del points  # free memory
    return dens_bfe_list, FP, times, grid_arrays


# ------------------------------------------------------------------
# KDE field computation (with per-snapshot skip + final merge)
# ------------------------------------------------------------------

def compute_kde_density(pos, mass, grid, Ndens=64):
    """Compute KDE density at every grid point.

    Parameters
    ----------
    pos : ndarray, shape (Np, 3)
        Particle positions.
    mass : ndarray, shape (Np,)
        Particle masses.
    grid: ndarray, shape (3, N, N, N)
        Stacked meshgrid arrays (x, y, z).
        e.g.,:
        dbins = np.linspace(grid_range[0], grid_range[1], grid_bins)
        grid_arrays = np.meshgrid(dbins, dbins, dbins, indexing="ij")
        grid = np.stack(grid_arrays)
        
    Ndens : int, optional
        Number of nearest neighbours (default 64).

    Returns
    -------
    kd_dens : ndarray, shape (nbins, nbins, nbins)
    """
    assert np.shape(grid)[0] == 3
    assert np.shape(grid)[1] == np.shape(grid)[2] == np.shape(grid)[3]
    nbins = np.shape(grid)[1]
    mesh = np.zeros((nbins**3, 3))
    mesh[:, 0] = grid[0].flatten()
    mesh[:, 1] = grid[1].flatten()
    mesh[:, 2] = grid[2].flatten()
    
    kddens = pyEXP.util.KDdensity(mass=mass, pos=pos, Ndens=Ndens)
    kd_dens = np.zeros(nbins**3)
    for i in range(nbins**3):
        kd_dens[i] = kddens.getDensityAtPoint(
            mesh[i, 0], mesh[i, 1], mesh[i, 2]
            )
    return kd_dens.reshape(nbins, nbins, nbins)



def compute_all_kde_fields(halo_id, component, output_dir, sim_centers,
                           grid_range=(-100, 100), grid_bins=20,
                           basis_dir=None, coefs_dir=None,
                           suite="Sheng24", Ndens=64):
    """Compute KDE density for every snapshot and write per-snapshot HDF5.

    Builds its own grid and :class:`FieldProjections` internally so that
    it can run independently of the BFE computation.

    Existing per-snapshot files are detected and skipped.  After all
    snapshots are processed the individual files are merged into a single
    HDF5 via :func:`field_io.merge_kde_density_files`.

    Parameters
    ----------
    halo_id : int
    output_dir : str
    sim_centers : dict
        Must contain ``"mw_center"`` array.
    grid_range : tuple of float, optional
    grid_bins : int, optional
    basis_dir : str, optional
    coefs_dir : str, optional
    suite : str, optional
    Ndens : int, optional
    """
    if basis_dir is None:
        basis_dir = os.path.join(_EXP_ROOT, "basis")
    if coefs_dir is None:
        coefs_dir = os.path.join(_EXP_ROOT, "coefficients")

    # Load basis
    cwd_save = os.getcwd()
    os.chdir(basis_dir)

    if component == "MWhalo":
        config_name = os.path.join(basis_dir, f"basis_halo_{halo_id:04d}.yaml")
        # Load coefficients
        coefs_file = os.path.join(coefs_dir, f"halo_{halo_id:04d}_coefficients_center.h5")
        per_snap_pattern = os.path.join(output_dir, f"halo_{halo_id:04d}_kde_density_*.h5")
        merged_file = os.path.join(output_dir, f"halo_{halo_id:04d}_kde_density.h5")
        output_filename = f"lmc_init_{halo_id:04d}_kde_densit"
        compname = f"halo"

    elif component == "lmc":
        config_name = os.path.join(basis_dir, f"basis_init_lmc_{halo_id:04d}.yaml")
        # Load coefficients
        coefs_file = os.path.join(coefs_dir, f"lmc_init_{halo_id:04d}_coefficients.h5")
        per_snap_pattern = os.path.join(output_dir, f"lmc_init_{halo_id:04d}_kde_density_*.h5")
        merged_file = os.path.join(output_dir, f"lmc_init_{halo_id:04d}_kde_density.h5")
        compname = f"lmc_init"
    
    print(f"Loading basis from {config_name}...")
    basis = load_basis(config_name)
    os.chdir(cwd_save)

    # Load coefficients
    print(f"Loading coefficients from {coefs_file}...")
    coefs = pyEXP.coefs.Coefs.factory(coefs_file)
    times = coefs.Times()

    # Build grid and FieldProjections
    dbins = np.linspace(grid_range[0], grid_range[1], grid_bins)
    grid_arrays = np.meshgrid(dbins, dbins, dbins, indexing="ij")
    grid = np.stack(grid_arrays)
    FP = FieldProjections(grid, basis, coefs, times)

    assert len(times) == len(sim_centers["mw_center"]), \
        f"Number of times ({len(times)}) != number of centres ({len(sim_centers['mw_center'])})"
    for i in range(len(times)):
        snap_file = os.path.join(
            output_dir, f"{compname}_{halo_id:04d}_kde_density_{i:03d}.h5"
        )
        if os.path.isfile(snap_file):
            print(f"snapshot {i:03d} KDE already exists — skipping")
            continue

        print(f"Loading particle data for snapshot {i:03d}...")
        if component == "MWhalo":
            p = load_particle_data(
                f"/n/nyx3/garavito/XMC-Atlas-sims/Sheng/Model_{halo_id}",
                snapname="snapshot",
                components=[component],
                nsnap=i,
                suite=suite,
                quantities=["pos", "mass"],
            )
            pos = p[component]["pos"] - sim_centers["mw_center"][i]
            mass = p[component]["mass"]
        elif component == "lmc":
            p = load_particle_data(
                f"/n/nyx3/garavito/XMC-Atlas-sims/Sheng/Model_{halo_id}",
                snapname="snapshot",
                components=["LMChalo"],
                nsnap=i,
                suite=suite,
                quantities=["pos", "mass"],
            )
            pos = p["LMChalo"]["pos"] - sim_centers["lmc_center"][i]
            mass = p["LMChalo"]["mass"]

        print(f"  Computing KDE density (Ndens={Ndens})...")
        kd_dens = FP.kde_density(pos, mass, Ndens=Ndens)

        write_kde_density(
            kd_dens,
            filename=snap_file,
            grid_shape=kd_dens.shape,
            snapshot_name=f"snapshot_{i:03d}",
            Ndens=Ndens,
        )

    # Merge per-snapshot files into one
    print("\nMerging per-snapshot KDE files...")
    merge_kde_density_files(per_snap_pattern, merged_file)


# ------------------------------------------------------------------
# Top-level driver
# ------------------------------------------------------------------

def run(halo_id, output_dir, component, suite="Sheng24",
        grid_range=(-100, 100), grid_bins=20,
        skip_bfe=False, skip_kde=False, Ndens=64):
    """Compute and write BFE and/or KDE fields for a halo.

    Parameters
    ----------
    halo_id : int
    output_dir : str
    suite : str, optional
    grid_range : tuple, optional
    grid_bins : int, optional
    skip_bfe : bool, optional
        If True, skip the BFE field computation/writing.
    skip_kde : bool, optional
        If True, skip the KDE field computation/writing.
    Ndens : int, optional
        Number of KDE neighbours (default 64).
    """
    basis_dir = os.path.join(_EXP_ROOT, "basis")
    coefs_dir = os.path.join(_EXP_ROOT, "coefficients")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Basis directory:  {basis_dir}")
    print(f"Coefs directory:  {coefs_dir}")

    abs_output_dir = os.path.abspath(output_dir)

    # Load simulation centres
    print("Loading centre data...")
    sim_centers = load_sheng24_exp_center(
        origin_dir="suites/Sheng24/orbits",
        centers_filename="MW_LMC_orbits_iso.txt",
        sim_id=halo_id,
        return_vel=True,
    )
    print("  Centre data loaded")

    # --- BFE fields ---
    if not skip_bfe:
        print("\n=== Computing BFE fields ===")
        compute_all_bfe_fields(
            halo_id, component, abs_output_dir, grid_range, grid_bins,
            basis_dir=basis_dir, coefs_dir=coefs_dir,
        )

    # --- KDE fields ---
    if not skip_kde:
        print("\n=== Computing KDE fields ===")
        compute_all_kde_fields(
            halo_id, component, abs_output_dir, sim_centers,
            grid_range=grid_range, grid_bins=grid_bins,
            basis_dir=basis_dir, coefs_dir=coefs_dir,
            suite=suite, Ndens=Ndens,
        )

    print("\nField computation complete.")
    print(f"Output files in: {abs_output_dir}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute BFE and KDE density fields for a halo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_fields.py --halo_id 108 --output_dir ./output
  python compute_fields.py --halo_id 108 --output_dir ./output --grid_bins 40
  python compute_fields.py --halo_id 108 --output_dir ./output --skip_kde
  python compute_fields.py --halo_id 108 --output_dir ./output --Ndens 128
        """,
    )
    parser.add_argument("--halo_id", type=int, required=True,
                        help="Halo model ID (e.g. 108)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output HDF5 files")
    parser.add_argument("--component", type=str, required=True,
                        help="component (e.g. lmc, MWhalo, MWbulge)")
    parser.add_argument("--suite", type=str, default="Sheng24",
                        help="Simulation suite name (default: Sheng24)")
    parser.add_argument("--grid_range", type=float, nargs=2,
                        default=[-100, 100], metavar=("MIN", "MAX"),
                        help="Grid extent in kpc (default: -100 100)")
    parser.add_argument("--grid_bins", type=int, default=20,
                        help="Number of grid bins per axis (default: 20)")
    parser.add_argument("--skip_bfe", action="store_true",
                        help="Skip BFE field computation")
    parser.add_argument("--skip_kde", action="store_true",
                        help="Skip KDE field computation")
    parser.add_argument("--Ndens", type=int, default=64,
                        help="KDE neighbour count (default: 64)")
    args = parser.parse_args()

    run(
        halo_id=args.halo_id,
        output_dir=args.output_dir,
        component=args.component,
        suite=args.suite,
        grid_range=tuple(args.grid_range),
        grid_bins=args.grid_bins,
        skip_bfe=args.skip_bfe,
        skip_kde=args.skip_kde,
        Ndens=args.Ndens,
    )


if __name__ == "__main__":
    main()
