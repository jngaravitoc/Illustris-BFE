"""
Compute and save EXP coefficients from TNG halo particle data using a basis.

This script:
1. Loads a basis from a YAML config file
2. Loads halo particles from TNG simulation snapshots
3. Computes coefficients for each snapshot
4. Saves them to a new HDF5 file in _coefs_tmp/

Usage:
    python compute_coefficients.py <basis_config_file> <halo_params_file> <data_dir> [--nmax NMAX] [--lmax LMAX] [--snapshots S1 S2 ...]

Example:
    python compute_coefficients.py \
        src/tests/_basis_tmp/halo_21537_basis_config_16_08.yaml \
        data/tng35-3-dark/halo_21537_params.hdf5 \
        data/tng35-3-dark \
        --nmax 16 --lmax 8 --snapshots 17 21 25
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyEXP

sys.path.insert(0, str(Path(__file__).resolve().parent / "exp"))

from exp_coefficients import compute_exp_coefs
import data_ios


# TNG50-3-dark particle mass (in Msun)
MASS_TNG = 2.33443590182933*1e7


# Output directory for coefficients
COEFS_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "_coefs_tmp"


def load_basis_from_config_file(basis_config_file: str) -> object:
    """
    Load a basis from a YAML config file.
    
    Parameters
    ----------
    basis_config_file : str
        Path to basis config YAML file.
        
    Returns
    -------
    basis : pyEXP.basis.Basis
        The loaded basis object.
    """
    config_path = Path(basis_config_file).resolve()
    print(f"Loading basis from {config_path}...")
    # pyEXP expects YAML config content here, not a file path.
    with open(config_path, "r", encoding="utf-8") as f:
        basis_yaml = f.read()

    # pyEXP hangs on first use when modelname/cachename are absolute paths and
    # no cache exists.  Rewrite any absolute paths to bare filenames so pyEXP
    # resolves them relative to the YAML's directory (set via os.chdir below).
    import yaml as _yaml
    cfg = _yaml.safe_load(basis_yaml)
    params = cfg.get("parameters", {})
    for key in ("modelname", "cachename"):
        if key in params and os.path.isabs(str(params[key])):
            params[key] = os.path.basename(params[key])
    basis_yaml = _yaml.dump(cfg, default_flow_style=False, sort_keys=False)

    # Resolve model/cache filenames from the YAML's directory.
    cwd = Path.cwd()
    try:
        os.chdir(config_path.parent)
        basis = pyEXP.basis.Basis.factory(basis_yaml)
    finally:
        os.chdir(cwd)
    print(f"  Basis type: {type(basis).__name__}")
    return basis


def read_halo_params(halo_params_file: str) -> dict:
    """Load halo parameters from HDF5 file."""
    return data_ios.read_halo_params(halo_params_file)


def load_tng_halo_particles(particle_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load particle coordinates and masses from TNG snapshot.
    
    Parameters
    ----------
    particle_file : str
        Path to TNG HDF5 snapshot file.
        
    Returns
    -------
    coords : ndarray, shape (N, 3)
        Particle positions.
    masses : ndarray, shape (N,)
        Particle masses.
    """
    coords, marr = data_ios.read_tng_halo_particles(particle_file)
    return coords, marr


def compute_coefficients_for_snapshots(
    basis_config_file: str,
    halo_params_file: str,
    data_dir: str,
    snapshots: list[int],
    nmax: int = 8,
    lmax: int = 2,
    halo_id: int = 21537,
    sim: str = "tng35-3-dark",
    coefs_filename: str | None = None,
    output_dir: Path | None = None,
    covariance: bool = False,
    samplesz: int = 1,
) -> Path:
    """
    Compute and save coefficients for specified snapshots.
    
    Parameters
    ----------
    basis_config_file : str
        Path to basis config YAML file.
    halo_params_file : str
        Path to halo parameters HDF5 file.
    data_dir : str
        Directory containing TNG snapshot files.
    snapshots : list[int]
        List of snapshot numbers to process.
    nmax : int, optional
        Radial basis order (default: 8).
    lmax : int, optional
        Angular basis order (default: 2).
    halo_id : int, optional
        Halo ID (default: 21537).
    sim : str, optional
        Simulation name (default: "tng35-3-dark").
    coefs_filename : str, optional
        Output coefficients filename. If None, uses coefficients_{nmax}_{lmax}.h5.
    output_dir : Path, optional
        Directory to write the coefficients file. Defaults to COEFS_OUTPUT_DIR.
    covariance : bool, optional
        Whether to compute coefficient covariance (default: False).
    samplesz : int, optional
        Sample size passed to pyEXP covariance estimation (default: 1).

    Returns
    -------
    Path
        Full path to the output coefficients file.
    """
    
    # Create output directory
    out_dir = Path(output_dir) if output_dir is not None else COEFS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    basis = load_basis_from_config_file(basis_config_file)
    
    # Load halo parameters
    print(f"Loading halo parameters from {halo_params_file}...")
    halo_params = read_halo_params(halo_params_file)
    
    # Define unit system (TNG35-3-dark specific)
    units = [
        ('mass', 'Msun', MASS_TNG),
        ('length', 'kpc', 1.0),
        ('velocity', 'km/s', 1.0),
        ('G', 'mixed', 43007.1)
    ]
    
    # Extract halo properties
    all_snaps = halo_params['snap']
    M200c = halo_params['M200c']
    R200c = halo_params['R200c']
    c200c = halo_params['c200c']
    rho200c = halo_params['M200c'] / (4/3. * np.pi * halo_params['R200c']**3)

    # Create snapshot lookup dicts
    snap_arr = np.asarray(all_snaps, dtype=int)
    R200c_dict = {int(s): float(v) for s, v in zip(snap_arr, R200c)}
    rho200c_dict = {int(s): float(v) for s, v in zip(snap_arr, rho200c)}
    M200c_dict = {int(s): float(v) for s, v in zip(snap_arr, M200c)}
    
    # Output file for coefficients
    if coefs_filename is None:
        coefs_filename = f"coefficients_{nmax:02d}_{lmax:02d}.h5"
    coefs_file = out_dir / coefs_filename

    # Ensure each run starts from a clean output file.
    if coefs_file.exists():
        coefs_file.unlink()

    print(f"\nComputing coefficients for {len(snapshots)} snapshots...")
    print(f"Output file: {coefs_file}")
    
    # Snapshot filename template
    snap_basename = "galaxies_halo_{subfind_id}_tng50-3-dark_{snap:03d}.hdf5"
    
    # Process each snapshot
    for i, snap in enumerate(snapshots, 1):
        print(f"\n[{i}/{len(snapshots)}] Processing snapshot {snap}...")
        
        # Construct particle file path
        input_snap = os.path.join(
            data_dir,
            snap_basename.format(subfind_id=halo_id, snap=snap)
        )
        
        if not os.path.isfile(input_snap):
            print(f"  WARNING: File not found: {input_snap}")
            continue
        
        # Load particles (mass array is built from MASS_TNG to match notebook)
        coords, _ = load_tng_halo_particles(input_snap)
        masses = np.ones(len(coords), dtype=float) * MASS_TNG
        print(f"  Loaded {len(coords)} particles")
        
        # Normalize by halo properties
        halo_data = {
            'pos': coords / R200c_dict[snap],
            'mass': masses / (rho200c_dict[snap] * R200c_dict[snap]**3),
        }
        
        # Compute coefficients
        # The snapshot number is used as the time key so that coefs.Times()
        # returns plain snapshot numbers rather than redshifts.
        try:
            compute_exp_coefs(
                halo_data=halo_data,
                snap_time=float(snap),
                basis=basis,
                component='halo',
                coefs_file=str(coefs_file),
                unit_system=units,
                covariance=covariance,
                samplesz=samplesz,
            )
            print(f"  ✓ Coefficients computed and saved")
        except Exception as e:
            print(f"  ERROR: Failed to compute coefficients: {e}")
            continue
    
    print(f"\nDone! Coefficients saved to {coefs_file}")
    return coefs_file


def run_compute_coefficients_cli(args: list[str]) -> int:
    """Programmatic CLI entry point used by tests."""

    cmd = [sys.executable, str(Path(__file__).resolve())] + args
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compute and save EXP coefficients from TNG particle data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "basis_config_file",
        help="Path to basis config YAML file.",
    )
    parser.add_argument(
        "halo_params_file",
        help="Path to halo parameters HDF5 file.",
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing TNG snapshot files.",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=8,
        help="Radial basis order (default: 8).",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=2,
        help="Angular basis order (default: 2).",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        nargs="+",
        default=[17, 21, 25],
        help="Snapshot numbers to process (default: 17 21 25).",
    )
    parser.add_argument(
        "--halo-id",
        type=int,
        default=21537,
        help="Halo ID (default: 21537).",
    )
    parser.add_argument(
        "--sim",
        default="tng35-3-dark",
        help="Simulation name (default: tng35-3-dark).",
    )
    parser.add_argument(
        "--coefs-filename",
        default=None,
        help="Output coefficients filename inside _coefs_tmp (default: coefficients_XX_YY.h5).",
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.isfile(args.basis_config_file):
        print(f"Error: Basis config file not found: {args.basis_config_file}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(args.halo_params_file):
        print(f"Error: Halo params file not found: {args.halo_params_file}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.snapshots:
            snapshots = args.snapshots
        else:
            snapshots = []

        compute_coefficients_for_snapshots(
            basis_config_file=args.basis_config_file,
            halo_params_file=args.halo_params_file,
            data_dir=args.data_dir,
            snapshots=snapshots,
            nmax=args.nmax,
            lmax=args.lmax,
            halo_id=args.halo_id,
            sim=args.sim,
            coefs_filename=args.coefs_filename,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
