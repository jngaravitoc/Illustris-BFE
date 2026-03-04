"""
Compute density profiles and virial quantities from TNG dark halo particle catalogs.

Author: ChatGPT with the supporivsion of jngaravitoc
"""

import numpy as np
import h5py
from scipy.interpolate import splrep, BSpline

# Colossus
from colossus.cosmology import cosmology
from colossus.halo import mass_defs, concentration
from colossus.halo import profile_einasto

# ============================================================
# 1. READ HDF5 FILE
# ============================================================


def read_tng_halo_particles(filename):
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
        M200c = f['catgrp_Group_M_Crit200'][0]
        R200c = f['catgrp_Group_R_Crit200'][0]

    return coords, M200c, R200c, halo_id


# ============================================================
# 2. CREATE MASS ARRAY
# ============================================================

def build_mass_array(coords, mass_tng):
    """
    Create equal-mass particle array.

    Parameters
    ----------
    coords : ndarray (N,3)
    mass_tng : float
        Particle mass

    Returns
    -------
    mass : ndarray (N,)
    """
    return np.ones(len(coords)) * mass_tng


# ============================================================
# 3. RADIAL BINS
# ============================================================

def build_radial_bins(coords, nbins=200, rmin=0.01):
    """
    Construct radial bins based on particle distribution.

    Returns
    -------
    rbins : ndarray
    rho_halo : ndarray
    rmax : float
    """
    r = np.sqrt(np.sum(coords**2, axis=1))
    rmax = np.max(r)

    rbins = np.linspace(rmin, rmax, nbins+1)
    rho_halo = np.zeros((3, nbins))

    return rbins, rho_halo, rmax


# ============================================================
# 4. DENSITY PROFILE
# ============================================================

def empirical_density_profile(rbins, pos, mass, smooth_length=0.0):
    """
    Computes radial density profile.

    Parameters
    ----------
    rbins : ndarray
    pos : ndarray (N,3)
    mass : ndarray (N,)
    smooth_length : float

    Returns
    -------
    radius : ndarray
    density : ndarray
    """
    if len(pos) != len(mass):
        raise ValueError("pos and mass arrays must have the same length")

    r_p = np.sqrt(np.sum(pos**2, axis=1))

    V_shells = 4/3 * np.pi * (rbins[1:]**3 - rbins[:-1]**3)

    density, _ = np.histogram(r_p, bins=rbins, weights=mass)
    density = density / V_shells

    radius = 0.5 * (rbins[1:] + rbins[:-1])

    if smooth_length != 0.0:
        tck_s = splrep(radius, np.log10(density), s=smooth_length)
        density = 10**BSpline(*tck_s)(radius)

    return radius, density


# ============================================================
# 5. WRITE OUTPUT HDF5
# ============================================================

def write_density_profile(outfile, radius, density, halo_id):
    """
    Save density profile to HDF5 file.

    Parameters
    ----------
    outfile : str
    radius : ndarray
    density : ndarray
    halo_id : str
    """
    with h5py.File(outfile, 'w') as f:
        grp = f.create_group(f"halo_{halo_id}")
        grp.create_dataset("radius_kpc", data=radius)
        grp.create_dataset("density_Msun_kpc3", data=density)


# ============================================================
# 6. VIRIAL QUANTITIES (COLOSSUS)
# ============================================================

def compute_virial_quantities(M200c, R200c, redshift=0.0):
	"""
	Convert 200c quantities into virial quantities using Colossus.
	Returns
	-------
	Mvir : float
	Rvir : float
	cvir : float
	c200c : float
	"""

	cosmology.setCosmology('planck15')

	c200c = concentration.concentration(M200c, '200c', redshift)

	Mvir, Rvir, cvir = mass_defs.changeMassDefinition(
		M200c, c200c, redshift, '200c', 'vir'
	)

	return Mvir, Rvir, cvir, c200c


# ============================================================
# EINASTO PROFILE FROM VIRIAL QUANTITIES
# ============================================================

def build_einasto_profile(radius, Mvir, Rvir, cvir, redshift=0.0):
    """
    Construct Einasto density profile predicted by cosmology.

    Parameters
    ----------
    radius : ndarray
        Radii where density is evaluated [kpc]
    Mvir : float
        Virial mass [Msun]
    Rvir : float
        Virial radius [kpc]
    cvir : float
        Virial concentration
    redshift : float

    Returns
    -------
    rho_einasto : ndarray
        Einasto density profile evaluated at radius
    params : dict
        Einasto structural parameters (rhos, rs, alpha)
    """

    p_einasto = profile_einasto.EinastoProfile(
        M=Mvir,
        c=cvir,
        z=redshift,
        mdef='vir'
    )

    rho_einasto = p_einasto.density(radius)

    params = {}
    for key in p_einasto.par.keys():
        params[key] = p_einasto.par[key]

    return rho_einasto, params

# ============================================================
# MAIN DRIVER
# ============================================================

def write_density_profile(outfile, radius, density, halo_id, rho_einasto=None):
    """
    Save density profile to HDF5 file.
    """

    with h5py.File(outfile, 'w') as f:
        grp = f.create_group(f"halo_{halo_id}")
        grp.create_dataset("radius_kpc", data=radius)
        grp.create_dataset("density_Msun_kpc3", data=density)

        if rho_einasto is not None:
            grp.create_dataset("einasto_density_Msun_kpc3", data=rho_einasto)



def process_halo(input_file, output_file, mass_tng):
    """
    Full pipeline execution.
    """

    # Read data
    coords, M200c, R200c, halo_id = read_tng_halo_particles(input_file)

    # Mass array
    mass = build_mass_array(coords, mass_tng)

    # Bins
    rbins, _, _ = build_radial_bins(coords)

    # Density profile
    radius, density = empirical_density_profile(rbins, coords, mass)

    # Save
    write_density_profile(output_file, radius, density, halo_id)

    # Virial quantities
    Mvir, Rvir, cvir, c200c = compute_virial_quantities(M200c, R200c)
 	
    # Build Einasto halo
    rho_einasto, einasto_params = build_einasto_profile(radius, Mvir, Rvir, cvir)

    #
    write_density_profile(output_file, radius, density, halo_id, rho_einasto)
    ("\nEinasto parameters:")
    for k,v in einasto_params.items():
        print(f"{k} = {v}")

    print("\nHalo:", halo_id)
    print("M200c =", M200c)
    print("R200c =", R200c)
    print("c200c =", c200c)
    print("Mvir =", Mvir)
    print("Rvir =", Rvir)
    print("cvir =", cvir)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    input_file = "../galaxies_tng50-3-dark_099.hdf5"
    output_file = "halo_density_profile.hdf5"

    # Example TNG50-3 DM particle mass (Msun)
    mass_tng = 4.8e8

    process_halo(input_file, output_file, mass_tng)

