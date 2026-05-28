"""
Script to compute a basis from a density profile fit and halo parameters.

Usage:
    python compute_basis_from_fit.py <fit_file> <halo_params_file> [options]

Example:
    python compute_basis_from_fit.py \
        ../data/dimer_density_profile_fit.txt \
        ../data/tng35-3-dark/halo_21537_params.hdf5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add exp module to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "exp"))

from basis_helpers import compute_basis
import data_ios


def load_fit_data(fit_filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized density profile fit from text file.
    
    Parameters
    ----------
    fit_filename : str
        Path to text file with (r, rho) rows.
        
    Returns
    -------
    r_fit, rho_fit : ndarray
        Radius and density arrays.
    """
    data = np.loadtxt(fit_filename)
    # File format: first row is r values, second row is rho values
    r_fit = data[0]
    rho_fit = data[1]
    return r_fit, rho_fit


def load_halo_params(params_filename: str) -> dict:
    """Load halo parameters from HDF5 file.
    
    Parameters
    ----------
    params_filename : str
        Path to HDF5 file with halo properties.
        
    Returns
    -------
    params : dict
        Dictionary with parameter names and values.
    """
    return data_ios.read_halo_params(params_filename)


def compute_basis_from_fit(
    fit_filename: str,
    halo_params_filename: str,
    nmax: int = 8,
    lmax: int = 2,
    basis_path: str | None = None,
    basis_filename: str | None = None,
    covariance: bool = False,
    samplesz: int = 1,
) -> tuple:
    """
    Compute a basis from density profile fit and halo parameters.
    
    Parameters
    ----------
    fit_filename : str
        Path to density profile fit file.
    halo_params_filename : str
        Path to halo parameters HDF5 file.
    nmax : int, optional
        Radial basis order (default: 8).
    lmax : int, optional
        Angular basis order (default: 2).
    basis_path : str, optional
        Directory for basis output files. If None, uses a temp directory.
    basis_filename : str, optional
        Basis config filename. If None, generates one from nmax/lmax.
    covariance : bool, optional
        Whether to compute coefficient covariance (default: False).
    samplesz : int, optional
        Sample size passed to pyEXP covariance estimation (default: 1).

    Returns
    -------
    basis : pyEXP.basis.Basis
        The computed basis object.
    r_fit : ndarray
        Radius array from fit.
    rho_fit : ndarray
        Density array from fit.
    halo_params : dict
        Halo parameters loaded.
    """
    
    # Load data
    print(f"Loading density profile fit from {fit_filename}...")
    r_fit, rho_fit = load_fit_data(fit_filename)
    
    print(f"Loading halo parameters from {halo_params_filename}...")
    halo_params = load_halo_params(halo_params_filename)
    
    # Setup basis path
    if basis_path is None:
        basis_path = str(Path(__file__).resolve().parent / "_basis_tmp")
    Path(basis_path).mkdir(parents=True, exist_ok=True)
    
    if basis_filename is None:
        basis_filename = f"basis_config_{nmax:02d}_{lmax:02d}.yaml"
    
    # Configure basis parameters
    basis_params = {
        "rmin": float(r_fit[0]),
        "rmax": float(r_fit[-1]),
        "rmapping": 1.0,
        "Mtotal": 1.0,
        "nbins": 2000,
        "lmax": int(lmax),
        "nmax": int(nmax),
        "cachename": str(
            Path(basis_path) / f"basis_cache_{nmax:02d}_{lmax:02d}.txt"
        ),
        "modelname": str(Path(basis_path) / "basis_model.txt"),
        "basis_id": "sphereSL",
        "pcavar": covariance,
        "samplesz": samplesz,
        "totalCovar": covariance,
        "fullCovar": False, # For some particular tests we might need this
    }
    
    print(f"Computing basis with nmax={nmax}, lmax={lmax}...")
    print(f"  rmin={basis_params['rmin']:.6e}, rmax={basis_params['rmax']:.6f}")
    print(f"  Radial bins: {basis_params['nbins']}")
    
    basis = compute_basis(
        basis_params,
        r_fit,
        rho_fit,
        str(basis_path),
        basis_filename,
    )
    
    print(f"Basis computation complete.")
    print(f"  Basis type: {type(basis).__name__}")
    
    return basis, r_fit, rho_fit, halo_params


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compute a basis from a density profile fit and halo parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "fit_file",
        help="Path to density profile fit file (text with r, rho columns).",
    )
    parser.add_argument(
        "halo_params_file",
        help="Path to halo parameters HDF5 file.",
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
        "--basis-path",
        default=None,
        help="Directory for basis output files (default: creates _basis_tmp).",
    )
    parser.add_argument(
        "--basis-filename",
        default=None,
        help="Basis config filename (default: auto-generated).",
    )
    parser.add_argument(
        "--covariance",
        action="store_true",
        default=False,
        help="Compute coefficient covariance (default: False).",
    )
    parser.add_argument(
        "--samplesz",
        type=int,
        default=1,
        help="Sample size for pyEXP covariance estimation (default: 1).",
    )

    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.isfile(args.fit_file):
        print(f"Error: Fit file not found: {args.fit_file}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(args.halo_params_file):
        print(f"Error: Halo params file not found: {args.halo_params_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        basis, r_fit, rho_fit, halo_params = compute_basis_from_fit(
            args.fit_file,
            args.halo_params_file,
            nmax=args.nmax,
            lmax=args.lmax,
            basis_path=args.basis_path,
            basis_filename=args.basis_filename,
            covariance=args.covariance,
            samplesz=args.samplesz,
        )
        
        print("\nBasis computation successful!")
        print(f"  nmax={args.nmax}, lmax={args.lmax}")
        print(f"  Fit radii span: [{r_fit[0]:.6e}, {r_fit[-1]:.6f}]")
        print(f"  Halo params keys: {list(halo_params.keys())}")
        
    except Exception as e:
        print(f"Error during basis computation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
