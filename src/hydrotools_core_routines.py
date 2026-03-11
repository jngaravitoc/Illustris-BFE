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
    #'tng75',
    #'tng75-dark',
    #'tng75-2',
    'tng75-3',
    'tng75-3-dark',
    #'tng205',
    #'tng205-dark',
    #'tng205-2',
    'tng205-3',
    #'tng35',
    #'tng35-dark',
    #'tng35-2',
    'tng35-3',
    'tng35-3-dark',
]


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


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def extract_halo_ids(sim: str, snap_idx: int, output_dir: str,
                   Mmin: float, Mmax: float, ncores=1, ngalaxies=1) -> str:
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
    ncores : int
        Number of cores
    ngalaxies : int
        Number of galaxies to extract

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
        num_processes=ncores,
        machine_name='umdastro',
        sim=sim,
        snap_idx=snap_idx,
        no_snapshots=False,
        paranoid=True,
        verbose=False,
        mass_selection_type='and',
        output_path=output_dir,
        file_suffix="_ngal_{:02d}".format(ngalaxies),
        buffered_output=False,
        output_compression='gzip',
        Mdm_min=Mmin,
        Mdm_max=Mmax,
        extract_satellites=False,
        randomize_order=True,
        rank_by='random',
        n_max_extract=ngalaxies,
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



def extract_galaxy_from_subfind_id(
    sim: str,
    snap_idx: int,
    subfind_id: int,
    output_dir: str,
    ncores=12,
) -> str:
    """Run ``extractGalaxyData`` for one snapshot of the tracked halo.

    Parameters
    ----------
    sim : str
        Simulation name.
    snap_idx : int
        Snapshot index [0, 99].
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
        num_processes=ncores,
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


#----------------------------------------------------------
# Particle data routines
#----------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data handling routines
# ---------------------------------------------------------------------------


def save_halo_time_evolution(datafile: str, sim: str, output_dir: str,
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

    out_txt = os.path.join(output_dir, f'{sim}_halo_time_evol.txt')
    np.savetxt(out_txt, data, fmt=fmt_data, header=header)
    print(f"  -> Saved {out_txt}")


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



def _check_galaxy_file_exists(output_dir: str, sim: str, ngalaxies: int) -> bool:
    """
    Check whether a file exists in output_dir with the format:
    galaxies_<sim>_ngal_{:2d}.hdf5

    Parameters
    ----------
    output_dir : str
        Directory to check for the file.
    sim : str
        Simulation name.
    ngalaxies : int
        Number of galaxies (used in the suffix).

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    suffix = "_ngal_{:2d}".format(ngalaxies)
    filename = f"galaxies_{sim}{suffix}.hdf5"
    filepath = os.path.join(output_dir, filename)
    return os.path.isfile(filepath)


#--------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract subfind IDs for MW-like halos across TNG suites.",
    )
    parser.add_argument(
        '--output-path', type=str, default='/n/nyx3/garavito/projects/Illustris-BFE/data/',
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


def get_halo_ids(sim: str, ncores: int = 1, ngalaxies: int = 1) -> None:
    """
    Entry point: loop over simulations, extract data, and save tables.

    Parameters
    ----------
    ncores : int, optional
        Number of cores to use for extract_halo_ids (default: 1).
    ngalaxies : int, optional
        Number of galaxies to extract per simulation (default: 1).
    """
   
    print(f"\n{'='*60}")
    print(f"Processing {sim}  (snap {args.snap_idx})")
    print(f"{'='*60}")

    sim_dir = os.path.join(args.output_path, sim)
    os.makedirs(sim_dir, exist_ok=True)

    datafile = extract_halo_ids(
        sim=sim,
        snap_idx=args.snap_idx,
        output_dir=sim_dir,
        Mmin=args.Mmin,
        Mmax=args.Mmax,
        ncores=ncores,
        ngalaxies=ngalaxies,
    )

    if not os.path.isfile(datafile):
        print(f"  [WARNING] Expected output not found: {datafile}")
        #continue

    save_halo_time_evolution(
        datafile=datafile,
        sim=sim,
        output_dir=sim_dir,
        Mmin=args.Mmin,
        Mmax=args.Mmax,
    )


    print("\nDone.")



def get_halo_evol(sim, halo_subfind_ids_filename, output_dir_time_evol_file, Mmin, Mmax):
    save_halo_time_evolution(halo_subfind_ids_filename, 
                             sim, output_dir_time_evol_file,
                             Mmin, Mmax)
    time_evol_txt = os.path.join(output_dir_time_evol_file, f'{sim}_halo_time_evol.txt')
    time_evol_data = load_subfind_ids(time_evol_txt)
    subfind_ids = time_evol_data["subfind_ids"]
    print(subfind_ids)
    #extract_galaxy_from_subfind_id(sim, snap_idx=snap, subfind_id=subfind_ids, output_dir=, ncores=ncores)


#def plot_halo_evolution():

if __name__ == "__main__":
    # check if data exists for given sim
    # if not run get_halo_ids
    # TODO: have the sims loop here instead of inside the functions
    args = parse_args()
    sims = args.sims if args.sims else SIMS
    ngalaxies = 1

    suffix = "_ngal_{:02d}".format(ngalaxies)
 

    # Extract subfind ids
    for sim in sims:
        sim_dir = os.path.join(args.output_path, sim)

        file_exists = _check_galaxy_file_exists(sim_dir, sim, ngalaxies=ngalaxies)
        if file_exists == False:
            try:
                get_halo_ids(sim)
            except Exception as e:
                print(f"  [ERROR] Exception while processing {sim}: {e}")

    # Extract halo properties from subfind_ids
    sim = sims[0]
    filename = f"galaxies_{suffix}{sim}_099.hdf5"
    out_hdf5_filename = os.path.join(args.output_path, filename)
    get_halo_evol(sim, out_hdf5_filename, args.output_path, args.Mmin, args.Mmax)
