"""Extract subfind IDs and merger-tree data for MW-like halos across TNG suites.

This script iterates over a list of IllustrisTNG simulation names, runs
``hydrotools.extractGalaxyData`` for each one, and saves the resulting
subfind-ID tables into per-simulation subdirectories under a common
``output_path/data/`` tree.

Output layout::

    <output_path>/data/<sim_name>/galaxies_<sim_name>_099.hdf5   (hydrotools)
    <output_path>/data/<sim_name>/<sim_name>_halo_sample.txt     (this script)

Usage
-----
Adjust ``OUTPUT_PATH``, ``Mmin``, ``Mmax``, and ``SIMS`` as needed, then::

    python hydrotools_get_subfind_ids.py
"""

import os
import argparse
import numpy as np
import h5py
from hydrotools.core import interface as iface_run
from hydrotools.common import fields as common_fields


# ---------------------------------------------------------------------------
# Simulation catalogue
# ---------------------------------------------------------------------------

SIMS = [
    #'tng35',
    'tng35-2',
    'tng35-3',
    #'tng35-dark',
    #'tng75',
    'tng75-2',
    'tng75-3',
    'tng75-3-dark',
    #'tng75-dark',
]


def is_dark_sim(sim_name: str) -> bool:
    """Return True if *sim_name* is a dark-matter-only run."""
    return sim_name.endswith('-dark')


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def extract_galaxy(sim: str, snap_idx: int, output_dir: str,
                   Mmin: float, Mmax: float) -> str:
    """Run ``extractGalaxyData`` for a single simulation and snapshot.

    Parameters
    ----------
    sim : str
        Simulation suite name (e.g. ``'tng75-3-dark'``).
    snap_idx : int
        Snapshot index.
    output_dir : str
        Directory where hydrotools will write its HDF5 output.
    Mmin, Mmax : float
        Dark-matter mass selection range in solar masses.

    Returns
    -------
    str
        Path to the HDF5 file produced by hydrotools.
    """
    # Choose catalogue fields appropriate for dark vs. hydro runs
    if is_dark_sim(sim):
        catsh_fields = common_fields.default_catsh_fields_dark
    else:
        catsh_fields = common_fields.default_catsh_fields

    iface_run.extractGalaxyData(
        num_processes=12,
        machine_name='umdastro',
        sim=sim,
        snap_idx=snap_idx,
        no_snapshots=False,
        paranoid=True,
        verbose=False,
        mass_selection_type='and',
        output_path=output_dir,
        buffered_output=False,
        output_compression='gzip',
        Mdm_min=Mmin,
        Mdm_max=Mmax,
        extract_satellites=False,
        randomize_order=True,
        rank_by='random',
        n_max_extract=1,
        catsh_get=True,
        catsh_fields=catsh_fields,
        catgrp_get=True,
        catgrp_fields=common_fields.default_catgrp_fields_all,
        tree_get=True,
        tree_fields=['subfind_id', 'is_primary', 'SubhaloID'],
        ptldm_get=False,
        ptl_in_rad_get=True,
        save_ptl_sep=False,
        ptl_rad=1.5,
        ptl_rad_units='200m',
        profile_get=False,
        profile_fields=[],
    )

    datafile = os.path.join(output_dir, f'galaxies_{sim}_{snap_idx:03d}.hdf5')
    return datafile


def save_halo_sample(datafile: str, sim: str, output_dir: str,
                     Mmin: float, Mmax: float) -> None:
    """Read the hydrotools HDF5 output and write a human-readable text table.

    Parameters
    ----------
    datafile : str
        Path to the HDF5 file produced by ``extractGalaxyData``.
    sim : str
        Simulation suite name (used in the header / filename).
    output_dir : str
        Directory where the text file will be saved.
    Mmin, Mmax : float
        Mass range (written into the file header for provenance).
    """
    with h5py.File(datafile, 'r') as f:
        print(f"[{sim}] keys: {list(f.keys())}")

        subfind_ids = np.array(f['tree_subfind_id'])
        snaps       = np.array(f['info']['tree_snaps'])
        times       = np.array(f['info']['tree_t'])
        redshifts   = np.array(f['info']['tree_z'])

    nfields = len(subfind_ids)
    nsnaps  = len(snaps)

    header = (
        f"A MW-like halo in {sim}\n"
        f"mass range {Mmin:.2e}-{Mmax:.2e}\n"
        "snaps redshifts times"
    )

    data = np.zeros((nsnaps, nfields + 3))
    fmt_data = ("%d", "%.3e", "%.8e")

    data[:, 0] = snaps
    data[:, 1] = redshifts
    data[:, 2] = times

    for i in range(nfields):
        data[:, i + 3] = subfind_ids[i]
        fmt_data += ("%d",)
        header += f" {int(subfind_ids[i][-1])}"

    out_txt = os.path.join(output_dir, f'{sim}_halo_sample.txt')
    np.savetxt(out_txt, data, fmt=fmt_data, header=header)
    print(f"  -> Saved {out_txt}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract subfind IDs for MW-like halos across TNG suites.",
    )
    parser.add_argument(
        '--output-path', type=str, default='.',
        help="Root output directory.  Per-sim data goes into <output_path>/data/<sim>/  (default: '.')",
    )
    parser.add_argument(
        '--snap-idx', type=int, default=99,
        help="Snapshot index (default: 99).",
    )
    parser.add_argument(
        '--Mmin', type=float, default=1e12,
        help="Minimum DM halo mass in Msun (default: 1e12).",
    )
    parser.add_argument(
        '--Mmax', type=float, default=3e12,
        help="Maximum DM halo mass in Msun (default: 3e12).",
    )
    parser.add_argument(
        '--sims', nargs='+', default=None,
        help="Subset of simulations to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: loop over simulations, extract data, and save tables."""
    args = parse_args()

    sims = args.sims if args.sims else SIMS

    for sim in sims:
        print(f"\n{'='*60}")
        print(f"Processing {sim}  (snap {args.snap_idx})")
        print(f"{'='*60}")

        sim_dir = os.path.join(args.output_path, 'data', sim)
        os.makedirs(sim_dir, exist_ok=True)

        datafile = extract_galaxy(
            sim=sim,
            snap_idx=args.snap_idx,
            output_dir=sim_dir,
            Mmin=args.Mmin,
            Mmax=args.Mmax,
        )

        if not os.path.isfile(datafile):
            print(f"  [WARNING] Expected output not found: {datafile}")
            continue

        save_halo_sample(
            datafile=datafile,
            sim=sim,
            output_dir=sim_dir,
            Mmin=args.Mmin,
            Mmax=args.Mmax,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
