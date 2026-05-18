"""
Functions adapted from Hayden Foote UofA (2025)
"""
import numpy as np
from scipy.spatial import KDTree
import pyEXP


def mise(field_bfe, field_target, axis=None):
    """Mean integrated squared error."""
    return np.mean((field_bfe - field_target) ** 2, axis=axis)


def mirse(field_bfe, field_target, axis=None):
    """Mean integrated relative squared error."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.mean((field_bfe / field_target - 1.0) ** 2, axis=axis)


def goodness_of_fit(field_bfe, field_target):
    """Chi-square–like goodness-of-fit statistic."""
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((field_bfe - field_target) ** 2 / field_target)
    return chi2


def relative_error(field_bfe, field_target):
    """
    Compute the pointwise relative error between two fields.

    Parameters
    ----------
    field_bfe : array_like
        Reconstructed or model field values.
    field_target : array_like
        True or target field values.

    Returns
    -------
    err : ndarray
        Element-wise relative error:
        |field_bfe - field_target| / field_target.
    """
    return np.abs(field_bfe - field_target) / field_target


def bfe_coef_profiles(basis, coefs, edges, times=0.0, field='dens'):
    field_spherical_exp = np.zeros(len(edges))
    for r in range(len(edges)):
        xyz_sph = spherical_grid(gridspecs= {'theta_bins': 10, 'phi_bins' : 20}, rgrid = edges[r])
        fields_grid = pyEXP.field.FieldGenerator([times], xyz_sph)
        dens = fields_grid.points(basis, coefs)[times][field]
        field_spherical_exp[r] = np.mean(dens)
    return field_spherical_exp


def kd3_point_density(pos, mass, points, k_max=100):
    """
    k-nearest-neighbor density estimate at arbitrary points.
    """
    tree = KDTree(pos)

    dist, idx = tree.query(points, k=k_max)
    dist = dist[:, -1]  # distance to k-th neighbor

    m_enc = np.sum(mass[idx], axis=1)
    vol = (4.0 / 3.0) * np.pi * dist**3

    return m_enc / vol

def radial_equal_mass_bins(pos, mass, nbins):
    """
    Compute a mass-weighted radial density profile using radial bins 
    that each contain equal total mass.

    Parameters
    ----------
    pos : (N, 3) array
        Cartesian particle positions.
    mass : (N,) array
        Particle masses.
    nbins : int
        Number of equal-mass radial bins.

    Returns
    -------
    r_center : (nbins,) array
        Mean radius of particles in each bin.
    rho : (nbins,) array
        Mass density in each bin (mass / shell volume).
    Mbin : float
        Target mass per bin (same for each bin).
    edges : (nbins+1,) array
        Radial edges that define each equal-mass bin.
    """

    # Compute radial distances
    r = np.linalg.norm(pos, axis=1)

    # Sort by radius
    idx = np.argsort(r)
    r_sorted = r[idx]
    m_sorted = mass[idx]

    # Cumulative mass profile
    m_cum = np.cumsum(m_sorted)
    Mtot = m_cum[-1]

    # Equal-mass target
    Mbin = Mtot / nbins

    # Bin edges in cumulative mass
    target_cum = np.linspace(0, Mtot, nbins + 1)

    # Convert cumulative mass targets → radial bin edges
    edges = np.interp(target_cum, m_cum, r_sorted)

    # Compute per-bin density
    r_center = np.zeros(nbins)
    rho = np.zeros(nbins)

    for i in range(nbins):
        # Select particles in this bin
        mask = (r_sorted >= edges[i]) & (r_sorted < edges[i+1])
        m_bin = m_sorted[mask]

        # Avoid empty bin (should not happen if masses > 0)
        if len(m_bin) == 0:
            r_center[i] = 0.5 * (edges[i] + edges[i+1])
            rho[i] = np.nan
            continue

        # Mean radius
        r_center[i] = np.mean(r_sorted[mask])

        # Shell volume (spherical)
        vol = (4.0/3.0) * np.pi * (edges[i+1]**3 - edges[i]**3)

        rho[i] = m_bin.sum() / vol

    return r_center, rho, Mbin, edges

def projected_map(pos, q, axis="z", bins=100, range=None):
    """
    Compute a 2D histogram (projection) of a 3D particle distribution 
    onto the plane perpendicular to a chosen axis. Each pixel stores 
    the mean of q for particles that fall into that bin.

    Parameters
    ----------
    pos : (N, 3) array
        Cartesian positions of particles.
    q : (N,) array
        Scalar values (e.g., density) to average inside each bin.
    axis : {"x","y","z"}, optional
        Axis along which to project the distribution. 
        - "z" → output is (x, y)
        - "y" → output is (x, z)
        - "x" → output is (y, z)
    bins : int or [int, int], optional
        Number of bins along each projected dimension.
    range : [[min_x, max_x], [min_y, max_y]] or None
        Data range for histogram. If None, computed from the data.

    Returns
    -------
    mean_q : (bins_x, bins_y) array
        Mean q value in each pixel. Empty bins are np.nan.
    counts : (bins_x, bins_y) array
        Number of particles in each pixel.
    xedges, yedges : arrays
        Histogram bin edges along each projected dimension.
    """

    pos = np.asarray(pos)
    q = np.asarray(q)

    if axis == "z":
        X, Y = pos[:, 0], pos[:, 1]
    elif axis == "y":
        X, Y = pos[:, 0], pos[:, 2]
    elif axis == "x":
        X, Y = pos[:, 1], pos[:, 2]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # Compute sum of q per bin
    qsum, xedges, yedges = np.histogram2d(
        X, Y, bins=bins, range=range, weights=q
    )

    # Number of particles per bin
    counts, _, _ = np.histogram2d(
        X, Y, bins=[xedges, yedges], range=range
    )

    # Compute mean per bin, avoid division by zero
    mean_q = np.divide(
        qsum, counts, out=np.full_like(qsum, np.nan), where=counts > 0
    )

    return mean_q, counts, xedges, yedges



def halo_radial_bfe_fidelity(particles_pos, particles_masses, basis, coefs, edges, times, field_type, gof_metric, k_max=200):
    """
    Parameters:
    -----------

    Returns:
    -------
    """ 
    
    radial_bins = len(edges)
    xyz_sph_3d = np.zeros((radial_bins, theta_bins, phi_bins))
    for r in range(len(edges)):
        xyz_sph_3d[r] = spherical_grid(gridspecs= {'theta_bins': theta_bins, 'phi_bins' : phi_bins}, rgrid = edges[r])
        fields_grid = pyEXP.field.FieldGenerator([times], xyz_sph_3d)
        bfe_field_grid = fields_grid.points(basis, coefs)[times][field_type]
        if field_type=='dens':
            simulated_field_grid = kd3_point_density(particles_pos, particles_masses, xyz_sph_3d, k_max=k_max)
        elif field_type == 'potential':
            # implement this
            raise ValueError("Potential not implemented yet")
    
    surface_metric_proj = {}

    for metric  in gof_metric:
        if metric == 'mise':
            metric_field = mise(bfe_field_grid, simulated_field_grid)
        if metric == 'mirse':
            metric_field = mirse(bfe_field_grid, simulated_field_grid)
        elif metric == 'gof':
            metric_field = gof(bfe_field_grid, simulated_field_grid)
        elif metric == 'relative_error':
            metric_field = relative_error(bfe_field_grid, simulated_field_grid)
        surface_metric_proj[metric] = projected_map(xyz_sph_3d, metric_field)
    print('return surface density metric for:', surface_metric_proj.keys())
    return surface_metric_proj


def halo_bfe_fidelity(
        particles_pos,
        particles_masses,
        basis,
        coefs,
        edges,
        theta_bins,
        phi_bins,
        times=0.0,
        field_type='dens',
        gof_metric=None,
        k_max=200,
        projected_map_axis='z'
    ):
    """
    Compute goodness-of-fit metrics between a BFE-reconstructed field and a
    particle-simulated field on a spherical grid, optionally projecting the
    result onto a 2D map.

    Parameters
    ----------
    particles_pos : ndarray, shape (N, 3)
        Cartesian coordinates of particles.
    particles_masses : ndarray, shape (N,)
        Mass of each particle.
    basis : object
        BFE basis object used by `pyEXP`.
    coefs : dict or ndarray
        Expansion coefficients of the BFE model.
    edges : array_like
        Radial grid edges where the spherical field is evaluated.
    theta_bins : int
        Number of bins in polar angle.
    phi_bins : int
        Number of bins in azimuthal angle.
    times : float or sequence, optional
        Time(s) for the BFE evaluation. Default is 0.0.
    field_type : {'dens', 'potential'}
        Field to compute: density or potential.
    gof_metric : list of str, optional
        Goodness-of-fit metrics to compute. Options:
        ['mise', 'mirse', 'gof', 'relative_error'].
    k_max : int, optional
        Max neighbor parameter for the kD-tree density estimator.
    projected_map_axis : {'x', 'y', 'z'}, optional
        Axis along which to project the 3D field into 2D.

    Returns
    -------
    surface_metric_proj : dict
        Dictionary mapping metric names to their projected 2D maps.

    Notes
    -----
    Requires the following helper functions to exist in scope:
    - spherical_grid
    - kd3_point_density
    - mise
    - mirse
    - goodness_of_fit
    - relative_error
    - projected_map
    """

    if gof_metric is None:
        gof_metric = ['mise', 'mirse', 'gof', 'relative_error']

    radial_bins = len(edges)

    # -----------------------------------------------------------
    # Build spherical sampling grid
    # -----------------------------------------------------------
    xyz_sph_3d = np.zeros((radial_bins, theta_bins, phi_bins, 3))
    for i, r in enumerate(edges):
        xyz_sph_3d[i] = spherical_grid(
            gridspecs={'theta_bins': theta_bins, 'phi_bins': phi_bins},
            rgrid=r
        )

    # -----------------------------------------------------------
    # BFE field on the grid
    # -----------------------------------------------------------
    fields_grid = pyEXP.field.FieldGenerator([times], xyz_sph_3d)
    bfe_field_grid = fields_grid.points(basis, coefs)[times][field_type]

    # -----------------------------------------------------------
    # Simulated (particle-based) field
    # -----------------------------------------------------------
    if field_type == 'dens':
        simulated_field_grid = kd3_point_density(
            particles_pos,
            particles_masses,
            xyz_sph_3d,
            k_max=k_max
        )
    elif field_type == 'potential':
        raise NotImplementedError("Potential field not yet implemented.")
    else:
        raise ValueError(f"Invalid field_type '{field_type}'.")

    # -----------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------
    surface_metric_proj = {}

    for metric in gof_metric:
        if metric == 'mise':
            metric_field = mise(bfe_field_grid, simulated_field_grid)[0]
        elif metric == 'mirse':
            metric_field = mise(bfe_field_grid, simulated_field_grid)[1]
        elif metric == 'gof':
            metric_field = goodness_of_fit(bfe_field_grid, simulated_field_grid)
        elif metric == 'relative_error':
            metric_field = relative_error(bfe_field_grid, simulated_field_grid)
        else:
            raise ValueError(f"Unknown metric '{metric}'")

        surface_metric_proj[metric] = projected_map(
            xyz_sph_3d,
            metric_field,
            axis=projected_map_axis
        )

    print("Returning surface metric maps for:", list(surface_metric_proj.keys()))
    return surface_metric_proj
