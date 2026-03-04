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
PARTICLE_MASSES = {
    'tng35':        4.5e6,
    'tng35-2':      3.6e7,
    'tng35-3':      2.9e8,
    'tng35-dark':   5.4e6,
    'tng75':        4.5e6,
    'tng75-2':      3.6e7,
    'tng75-3':      2.9e8,
    'tng75-3-dark': 4.8e8,
    'tng75-dark':   5.4e6,
}


def is_dark_sim(sim_name: str) -> bool:
    """Return True if *sim_name* is a dark-matter-only run."""
    return sim_name.endswith('-dark')


def read_particle_data(h5file):
    """
    Read particle data from an HDF5 file, including optional fuzz and OSHS
    components in either coordinates or velocities.

    Parameters
    ----------
    h5file : h5py.File

    Returns
    -------
    data : dict
        Keys may include:
        - pos, vel
        - fuzz_pos, fuzz_vel
        - oshs_pos, oshs_vel
    """

    data = {
        "pos": [],
        "vel": [],
        "fuzz_pos": [],
        "fuzz_vel": [],
        "oshs_pos": [],
        "oshs_vel": [],
    }

    for key in h5file.keys():
        arr = np.array(h5file[key])

        if key.endswith("_Coordinates"):
            if "fuzz" in key:
                data["fuzz_pos"].append(arr)
            elif "oshs" in key:
                data["oshs_pos"].append(arr)
            else:
                data["pos"].append(arr)

        elif key.endswith("_Velocities"):
            if "fuzz" in key:
                data["fuzz_vel"].append(arr)
            elif "oshs" in key:
                data["oshs_vel"].append(arr)
            else:
                data["vel"].append(arr)

    # Stack where present, else None
    for k in data:
        if len(data[k]) > 0:
            data[k] = np.vstack(data[k])
        else:
            data[k] = None

    return data


def write_one_single_hdf5(
    filename, 
    snapshot,
    redshift,
    time,
    pos,
    vel,
    fuzz_pos=None,
    fuzz_vel=None,
    oshs_pos=None,
    oshs_vel=None,
):
    """
    Write particle data to a single HDF5 file with metadata.

    Metadata is written as file-level attributes.
    """

    with h5py.File(filename, "w") as f:

        # -------------------------
        # Metadata
        # -------------------------
        f.attrs["snapshot"] = snapshot
        f.attrs["redshift"] = redshift
        f.attrs["time"] = time

        # -------------------------
        # Main particles
        # -------------------------
        f.create_dataset("Coordinates", data=pos)
        f.create_dataset("Velocities", data=vel)

        # -------------------------
        # Optional components
        # -------------------------
        if fuzz_pos is not None:
            f.create_dataset("Fuzz_Coordinates", data=fuzz_pos)

        if fuzz_vel is not None:
            f.create_dataset("Fuzz_Velocities", data=fuzz_vel)

        if oshs_pos is not None:
            f.create_dataset("OSHS_Coordinates", data=oshs_pos)

        if oshs_vel is not None:
            f.create_dataset("OSHS_Velocities", data=oshs_vel)

def write_complementary_fields(
    filename,
    m200,
    r200,
    *,
    snapshot,
    redshift,
    time,
):
    """
    Write halo-level complementary fields to HDF5.
    """

    with h5py.File(filename, "w") as f:
        f.attrs["snapshot"] = snapshot
        f.attrs["redshift"] = redshift
        f.attrs["time"] = time

        f.create_dataset("Group_M_Crit200", data=m200)
        f.create_dataset("Group_R_Crit200", data=r200)



def find_particle_keys(h5file):
    """
    Find particle coordinate and velocity datasets in an HDF5 file.

    Parameters
    ----------
    h5file : h5py.File

    Returns
    -------
    coord_keys : list of str
        Dataset names ending in '_Coordinates'.
    vel_keys : list of str
        Dataset names ending in '_Velocities'.
    """

    coord_keys = []
    vel_keys = []

    found_fuzz = False
    found_oshs = False

    for key in h5file.keys():
        if key.endswith("_Coordinates"):
            coord_keys.append(key)
        elif key.endswith("_Velocities"):
            vel_keys.append(key)

        if "fuzz" in key:
            found_fuzz = True
        if "oshs" in key:
            found_oshs = True

    if found_fuzz:
        print("✓ Fuzz particles found")

    if found_oshs:
        print("✓ OSHS particles found")

    return coord_keys, vel_keys


def load_subfind_ids(ids_file: str) -> dict:
    """Load the subfind-ID table written by ``hydrotools_get_subfind_ids.py``.

    Parameters
    ----------
    ids_file : str
        Path to ``<sim>_halo_sample.txt``.

    Returns
    -------
    dict
        Keys: ``snaps``, ``redshifts``, ``times``, ``subfind_ids``
        (the latter is a 2-D array, one row per halo tracked).
    """
    data = np.loadtxt(ids_file)
    return {
        "snaps":       data[:, 0].astype(int),
        "redshifts":   data[:, 1],
        "times":       data[:, 2],
        "subfind_ids": data[:, 3:].astype(int),  # may have >1 halo column
    }


def extract_snapshot(
    sim: str,
    snap_idx: int,
    subfind_id: int,
    output_dir: str,
) -> str:
    """Run ``extractGalaxyData`` for one snapshot of the tracked halo.

    Parameters
    ----------
    sim : str
        Simulation name.
    snap_idx : int
        Snapshot index.
    subfind_id : int
        Subfind ID of the halo at this snapshot.
    output_dir : str
        Directory for hydrotools output files.

    Returns
    -------
    str
        Path to the output HDF5 file.
    """
    catsh_fields = (
        common_fields.default_catsh_fields_dark
        if is_dark_sim(sim)
        else common_fields.default_catsh_fields_all
    )

    iface_run.extractGalaxyData(
        num_processes=12,
        machine_name='umdastro',
        sim=sim,
        snap_idx=snap_idx,
        no_snapshots=False,
        paranoid=True,
        verbose=False,
        mass_selection_type='idxs',
        sh_idxs=[subfind_id],
        output_path=output_dir,
        buffered_output=False,
        output_compression='gzip',
        extract_satellites=True,
        n_max_extract=None,
        catsh_get=True,
        catsh_fields=catsh_fields,
        catgrp_get=True,
        catgrp_fields=common_fields.default_catgrp_fields_all,
        tree_get=True,
        tree_fields=['subfind_id', 'is_primary'],
        ptldm_get=False,
        ptl_rad_units='200m',
        profile_get=False,
        profile_fields=[],
    )

    return os.path.join(output_dir, f'galaxies_{sim}_{snap_idx:03d}.hdf5')


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
