import numpy as np
import h5py


def read_tng_halo_particles(filename, suite='tng-35-3-dark'):
    """
    Read halo particle coordinates and group properties from a TNG HDF5 file.

    The function automatically detects the halo ID contained in the key
    'galaxy_XXXXXX_ptldm_Coordinates'.

    Parameters
    ----------
    filename : str
        Path to HDF5 file.

    Returns
    -------
    coords : ndarray, shape (N,3)
        Dark matter particle positions [kpc].
    M200c : float
        Halo M200c mass.
    R200c : float
        Halo R200c radius.
    halo_id : str
        Halo ID string extracted from file.
    """
    if suite == 'tng-35-3-dark':
        mass_tng = 2.33443590182933e7  # TNG50-3-dark DM particle mass [Msun]

    else:
        raise ValueError("Suite not implemented yet")

    with h5py.File(filename, 'r') as f:

        # Find coordinate key dynamically
        coord_key = None
        for key in f.keys():
            if key.startswith("galaxy_") and key.endswith("_ptldm_Coordinates"):
                coord_key = key
                break

        if coord_key is None:
            raise KeyError("No halo coordinate key found in file.")

        halo_id = coord_key.split('_')[1]

        coords = f[coord_key][:]
     
    marr = np.ones(len(coords)) * mass_tng
    return coords, marr 


def read_halo_params(h5file):
    """
    Read all datasets from a halo_params HDF5 file and return as a dictionary of numpy arrays.

    Parameters
    ----------
    h5file : str
        Path to the HDF5 file (e.g., 'halo_21537_params.hdf5').

    Returns
    -------
    params : dict
        Dictionary with keys as parameter names and values as numpy arrays.
    """
    params = {}
    with h5py.File(h5file, 'r') as f:
        for key in f.keys():
            params[key] = f[key][:]
    return params


def read_density_profile(filename, snap):
    """
    Read density profile from HDF5 file written by write_density_profile.

    Parameters
    ----------
    filename : str
        Path to HDF5 file.
    snap : int or float
        Snapshot number used in the group name.

    Returns
    -------
    radius : ndarray
        Array of radii [kpc].
    density : ndarray
        Array of densities [Msun/kpc^3].
    """
    group_name = f"halo_{snap:03d}"
    with h5py.File(filename, 'r') as f:
        grp = f[group_name]
        radius = grp["radius_kpc"][:]
        density = grp["density_Msun_kpc3"][:]
    return radius, density

