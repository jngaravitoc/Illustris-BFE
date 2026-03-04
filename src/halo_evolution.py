#!/usr/bin/env python
"""
halo_evolution.py — Plot halo centre-of-mass position and M_200c evolution,
plus 2-D particle density projections for a sequence of TNG snapshots.

This script is derived from ``halo_evolution.ipynb`` and exposes the same
two figures via a command-line interface:

Figure 1 (notebook code-cell 6):
    Two-panel plot showing (left) the norm of the halo centre-of-mass
    position and (right) M_200c as a function of snapshot index.

Figure 2 (notebook code-cell 9/10):
    For every snapshot in the requested range, a two-panel 2-D histogram
    of the host and fuzz particle distributions.

Usage
-----
    python halo_evolution.py <DATAFILE_TEMPLATE> [--snap-start 79] [--nsnaps 20] [--outdir ./figures]

where ``DATAFILE_TEMPLATE`` is a format-string with a single ``{:03d}``
placeholder for the snapshot number, e.g.::

    ./halo2/galaxies_tng50-3-dark_{:03d}.hdf5

Bug-fixes applied relative to the original notebook
----------------------------------------------------
1. ``Mcrit200`` now builds the file path from the *snap* argument instead
   of relying on a stale global ``DATAFILE``.
2. Removed dead ``particle_mass`` double-assignment (kept only TNG50-3 value).
3. Removed unused ``halos_particle_selection`` variable.
4. All HDF5 files are opened inside context managers (``with`` blocks).
5. ``find_key_coordinates`` raises ``KeyError`` with a clear message when
   the expected datasets are not found, rather than failing with
   ``UnboundLocalError``.
6. File-path template is passed explicitly instead of being hard-coded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def com_halo(datafile_template: str, snap: int) -> np.ndarray:
    """Return the centre-of-mass position of the first group in *snap*.

    Parameters
    ----------
    datafile_template : str
        Format string for the HDF5 file path, e.g.
        ``'./halo2/galaxies_tng50-3-dark_{:03d}.hdf5'``.
    snap : int
        Snapshot number.

    Returns
    -------
    np.ndarray
        Shape-(3,) array with the (x, y, z) group position in kpc/h.
    """
    datafile = datafile_template.format(snap)
    with h5py.File(datafile, "r") as f:
        return np.array(f["catgrp_GroupPos"][0])


def mcrit200(datafile_template: str, snap: int) -> float:
    """Return M_Crit200 for the first group in *snap*.

    Parameters
    ----------
    datafile_template : str
        Format string for the HDF5 file path.
    snap : int
        Snapshot number.

    Returns
    -------
    float
        M_200c in 1e10 Msun/h (IllustrisTNG units).

    Notes
    -----
    **Bug-fix**: the original notebook always read from a global ``DATAFILE``
    instead of building the path from *snap*.
    """
    datafile = datafile_template.format(snap)
    with h5py.File(datafile, "r") as f:
        return float(np.array(f["catgrp_Group_M_Crit200"][0]))


def find_key_coordinates(f: h5py.File) -> tuple[str, str]:
    """Locate the host-particle and fuzz-particle coordinate datasets.

    Searches all top-level keys in *f* for names ending with
    ``'ptldm_Coordinates'`` (host) and ``'fuzz_Coordinates'`` (fuzz).

    Parameters
    ----------
    f : h5py.File
        An open HDF5 file handle.

    Returns
    -------
    tuple[str, str]
        ``(host_key, fuzz_key)``

    Raises
    ------
    KeyError
        If either key pattern is not found.
    """
    k_all: Optional[str] = None
    k_fuzz: Optional[str] = None

    for key in f.keys():
        if key.endswith("ptldm_Coordinates"):
            k_all = key
        elif key.endswith("fuzz_Coordinates"):
            k_fuzz = key

    if k_all is None:
        raise KeyError(
            "No dataset ending with 'ptldm_Coordinates' found in "
            f"{list(f.keys())}"
        )
    if k_fuzz is None:
        raise KeyError(
            "No dataset ending with 'fuzz_Coordinates' found in "
            f"{list(f.keys())}"
        )
    return k_all, k_fuzz


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_evolution(
    datafile_template: str,
    snap_start: int = 79,
    nsnaps: int = 20,
) -> plt.Figure:
    """Create a two-panel figure of COM position norm and M_200c vs snapshot.

    This reproduces **code-cell 6** (the plot in code-cell 8 of the full
    notebook numbering).

    Parameters
    ----------
    datafile_template : str
        Format string for the HDF5 file path.
    snap_start : int
        First snapshot number (inclusive).
    nsnaps : int
        Number of consecutive snapshots to process.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pos_com = np.zeros((nsnaps, 3))
    m_200c = np.zeros(nsnaps)
    snap_indices = np.arange(snap_start, snap_start + nsnaps)

    for j, snap in enumerate(snap_indices):
        pos_com[j] = com_halo(datafile_template, snap)
        m_200c[j] = mcrit200(datafile_template, snap)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].plot(snap_indices, np.linalg.norm(pos_com, axis=1))
    ax[0].set_xlabel("Snapshot")
    ax[0].set_ylabel(r"$|\mathbf{r}_{\rm COM}|$ [kpc/$h$]")
    ax[0].set_title("Centre-of-mass distance")

    ax[1].plot(snap_indices, m_200c)
    ax[1].set_xlabel("Snapshot")
    ax[1].set_ylabel(r"$M_{200c}$ [$10^{10}\,M_\odot/h$]")
    ax[1].set_title(r"$M_{200c}$ evolution")

    fig.tight_layout()
    return fig


def plot_halo_particles(
    datafile_template: str,
    snap: int,
    xlim: tuple[float, float] = (-400, 400),
    ylim: tuple[float, float] = (-400, 400),
    nbins: int = 200,
) -> plt.Figure:
    """Create a two-panel 2-D histogram of host and fuzz particles.

    This reproduces **code-cell 9** (the ``plot_halo_particles`` function
    in code-cell 11 of the full notebook numbering).

    Parameters
    ----------
    datafile_template : str
        Format string for the HDF5 file path.
    snap : int
        Snapshot number.
    xlim, ylim : tuple[float, float]
        Axis limits in kpc/h.
    nbins : int
        Number of histogram bins per axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TNG50-3 dark particle mass (Msun)
    # Reference: Table 1 in https://arxiv.org/pdf/1504.00362
    particle_mass = 4.8e8  # noqa: F841 – kept for reference

    datafile = datafile_template.format(snap)
    with h5py.File(datafile, "r") as f:
        coord_key, fuzz_key = find_key_coordinates(f)
        pos = np.array(f[coord_key])
        pos_fuzz = np.array(f[fuzz_key])

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    # Host particles
    ax[0].text(
        xlim[0] + 0.25 * (xlim[1] - xlim[0]),
        ylim[0] + 0.05 * (ylim[1] - ylim[0]),
        f"Npart={len(pos)}",
        color="white",
    )
    ax[0].hist2d(pos[:, 0], pos[:, 1], bins=nbins, norm=LogNorm(), cmap="twilight")
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_title(f"Host particles {snap:03d}")
    ax[0].set_xlabel("x [kpc/$h$]")
    ax[0].set_ylabel("y [kpc/$h$]")

    # Fuzz particles
    ax[1].text(
        xlim[0] + 0.25 * (xlim[1] - xlim[0]),
        ylim[0] + 0.05 * (ylim[1] - ylim[0]),
        f"Npart={len(pos_fuzz)}",
        color="white",
    )
    ax[1].hist2d(
        pos_fuzz[:, 0], pos_fuzz[:, 1], bins=nbins, norm=LogNorm(), cmap="twilight"
    )
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].set_title(f"Fuzz particles {snap:03d}")
    ax[1].set_xlabel("x [kpc/$h$]")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Plot halo evolution and particle density projections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "datafile",
        type=str,
        help=(
            "Format-string path to the HDF5 snapshots, e.g. "
            "'./halo2/galaxies_tng50-3-dark_{:03d}.hdf5'.  "
            "Must contain a single {:03d} placeholder for the snap number."
        ),
    )
    parser.add_argument(
        "--sim-name",
        type=str,
        default=None,
        help=(
            "Simulation name used in output filenames, e.g. 'tng50-3-dark'.  "
            "If not given, it is inferred from the datafile template."
        ),
    )
    parser.add_argument(
        "--snap-start",
        type=int,
        default=79,
        help="First snapshot index for the evolution plot (default: 79).",
    )
    parser.add_argument(
        "--nsnaps",
        type=int,
        default=20,
        help="Number of consecutive snapshots for the evolution plot (default: 20).",
    )
    parser.add_argument(
        "--particle-snap-start",
        type=int,
        default=None,
        help=(
            "First snapshot for particle projection plots.  "
            "Defaults to --snap-start."
        ),
    )
    parser.add_argument(
        "--particle-nsnaps",
        type=int,
        default=None,
        help=(
            "Number of snapshots for particle projection plots.  "
            "Defaults to --nsnaps."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./figures",
        help="Directory for saved figures (default: ./figures).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show(); useful for batch runs.",
    )
    return parser.parse_args()


def _infer_sim_name(datafile_template: str) -> str:
    """Try to extract a simulation name from the datafile template.

    Looks for a pattern like ``galaxies_<sim-name>_{:03d}.hdf5`` and returns
    the ``<sim-name>`` portion.  Falls back to ``'halo'`` if parsing fails.
    """
    import re
    basename = os.path.basename(datafile_template)
    m = re.match(r"galaxies_(.+?)_\{", basename)
    if m:
        return m.group(1)
    return "halo"


def main() -> None:
    """Entry point for the halo-evolution analysis script."""
    args = parse_args()

    sim_name = args.sim_name if args.sim_name else _infer_sim_name(args.datafile)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Optionally apply EXPtools style if available
    try:
        import EXPtools
        EXPtools.use_exptools_style(usetex=True)
    except ImportError:
        pass

    # --- Figure 1: evolution plot (notebook code-cell 6) ---
    fig_evo = plot_evolution(
        args.datafile,
        snap_start=args.snap_start,
        nsnaps=args.nsnaps,
    )
    evo_path = outdir / f"{sim_name}_halo_evolution.png"
    fig_evo.savefig(evo_path, dpi=150, bbox_inches="tight")
    print(f"Saved evolution figure → {evo_path}")

    # --- Figure 2: particle projections (notebook code-cell 9) ---
    p_start = args.particle_snap_start if args.particle_snap_start is not None else args.snap_start
    p_nsnaps = args.particle_nsnaps if args.particle_nsnaps is not None else args.nsnaps

    for snap in range(p_start, p_start + p_nsnaps):
        datafile = args.datafile.format(snap)
        if not os.path.isfile(datafile):
            print(f"  [skip] {datafile} not found")
            continue
        fig_part = plot_halo_particles(args.datafile, snap)
        part_path = outdir / f"{sim_name}_halo_particles_{snap:03d}.png"
        fig_part.savefig(part_path, dpi=150, bbox_inches="tight")
        plt.close(fig_part)
        print(f"  Saved particle figure → {part_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
