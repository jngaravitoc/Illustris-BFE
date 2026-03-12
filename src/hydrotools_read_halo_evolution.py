"""Read halo evolution data produced by hydrotools_get_subfind_ids.py.

For each simulation in the shared SIMS list, this script:

1. Reads the ``<sim>_halo_sample.txt`` subfind-ID table produced by
   ``hydrotools_get_subfind_ids.py``.
2. Calls ``extractGalaxyData`` snapshot-by-snapshot to retrieve particle
   and catalogue data for the tracked halo.
3. Optionally writes per-snapshot HDF5 files with particle coordinates,
   velocities, and halo-level quantities.

The directory layout mirrors ``hydrotools_get_subfind_ids.py``::

    <output_path>/data/<sim>/<sim>_halo_sample.txt      (input, from get_subfind_ids)
    <output_path>/data/<sim>/galaxies_<sim>_<snap>.hdf5  (hydrotools output)
    <output_path>/data/<sim>/processed/                  (optional post-processed files)

Usage
-----
    python hydrotools_read_halo_evolution.py [--output-path .] [--sims tng35-3-dark] \\
                                             [--snap-start 90] [--snap-end 99]
"""

import os
import argparse
import h5py
import numpy as np
from hydrotools.core import interface as iface_run
from hydrotools.common import fields as common_fields


# ---------------------------------------------------------------------------
# Simulation catalogue  (keep in sync with hydrotools_get_subfind_ids.py)
# ---------------------------------------------------------------------------

SIMS = [
    'tng35',
    'tng35-2',
    'tng35-3',
    'tng35-dark',
    'tng75',
    'tng75-2',
    'tng75-3',
    'tng75-3-dark',
    'tng75-dark',
]

# Particle masses in Msun for each resolution level



def is_dark_sim(sim_name: str) -> bool:
    """Return True if *sim_name* is a dark-matter-only run."""
    return sim_name.endswith('-dark')







# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Read halo evolution from hydrotools_get_subfind_ids.py output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output-path', type=str, default='.',
        help=(
            "Root output directory (same as used in hydrotools_get_subfind_ids.py).  "
            "Data is read/written under <output_path>/data/<sim>/  (default: '.')"
        ),
    )
    parser.add_argument(
        '--sims', nargs='+', default=None,
        help="Subset of simulations to process (default: all in SIMS).",
    )
    parser.add_argument(
        '--snap-start', type=int, default=90,
        help="First snapshot to extract (default: 90).",
    )
    parser.add_argument(
        '--snap-end', type=int, default=99,
        help="Last snapshot to extract, exclusive (default: 99).",
    )
    parser.add_argument(
        '--halo-col', type=int, default=0,
        help="Column index (0-based) of the halo to track in the subfind_ids table (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: loop over simulations and snapshots."""
    args = parse_args()
    sims = args.sims if args.sims else SIMS

    for sim in sims:
        print(f"\n{'='*60}")
        print(f"Processing {sim}")
        print(f"{'='*60}")

        sim_dir = os.path.join(args.output_path, 'data', sim)
        ids_file = os.path.join(sim_dir, f'{sim}_halo_sample.txt')

        if not os.path.isfile(ids_file):
            print(f"  [skip] {ids_file} not found — run hydrotools_get_subfind_ids.py first")
            continue

        table = load_subfind_ids(ids_file)
        snaps       = table["snaps"]
        redshifts   = table["redshifts"]
        times       = table["times"]
        subfind_ids = table["subfind_ids"]

        # Select which halo column to track
        if subfind_ids.ndim == 1:
            halo_ids = subfind_ids
        else:
            halo_ids = subfind_ids[:, args.halo_col]

        for snap_idx in range(args.snap_start, args.snap_end):
            if snap_idx >= len(snaps):
                print(f"  [skip] snap {snap_idx} out of range (max {len(snaps)-1})")
                break

            sid = int(halo_ids[snap_idx])
            print(f"  snap {snap_idx:3d}  z={redshifts[snap_idx]:.3f}  subfind_id={sid}")

            datafile = extract_snapshot(
                sim=sim,
                snap_idx=snap_idx,
                subfind_id=sid,
                output_dir=sim_dir,
            )

            if not os.path.isfile(datafile):
                print(f"    [WARNING] Expected output not found: {datafile}")
                continue

            # ----------------------------------------------------------
            # Optional: read back & post-process particle data here
            # ----------------------------------------------------------
            # with h5py.File(datafile, "r") as f:
            #     pdata = read_particle_data(f)
            #     coord_keys, vel_keys = find_particle_keys(f)
            #     ...

    print("\nDone.")


if __name__ == "__main__":
    main()
