import os
import sys
import time
from pathlib import Path
import numpy as np
import logging
import pyEXP

def compute_exp_coefs(halo_data, snap_time, basis, component, coefs_file, unit_system, covariance=True, samplesz=1, **kwargs):
    """
    Compute EXP basis function coefficients for a halo snapshot.

    Parameters
    ----------
    halo_data : dict
        Dictionary containing particle data with keys:
        - 'mass': array of particle masses
        - 'pos': array of particle positions (shape: N×3)
    snap_time : float
        Snapshot time value.
    basis : pyEXP.basis.Basis
        Initialized EXP basis object.
    component : str
        Component name for the coefficient container.
    coefs_file : str
        Path to output HDF5 file for coefficients.
    unit_system : str or dict
        Unit system specification for the coefficients.
    covariance : bool, optional
        If True, compute and write coefficient covariance matrix. Default is True.
    **kwargs
        Additional keyword arguments (unused).

    Notes
    -----
    - Creates coefficients from particle data using the EXP basis.
    - Optionally computes coefficient covariance if covariance=True.
    - If coefs_file exists, extends it; otherwise creates a new file.
    - Logs execution time and number of particles processed.

    Examples
    --------
    Define a unit system:

    >>> units = [('mass', 'Msun', mass_tng),
    ...          ('length', 'kpc', 1.0),
    ...          ('velocity', 'km/s', 1.0),
    ...          ('G', 'mixed', 43007.1)]


    """

    # Compute coefficients
    start_time = time.time()

    # enableCoefCovariance must be called BEFORE createFromArray so that pyEXP
    # accumulates the subsample covariance during the array expansion.
    if covariance == True:
        basis.enableCoefCovariance(True, samplesz)

    coef = basis.createFromArray(halo_data['mass'], halo_data['pos'], snap_time)
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)
    
    coefs.setUnits(unit_system)

    # Test this and add unit tests
    if covariance == True:
        # runtag is added in case one wants to sample the particle data and compute
        # the covariance for each sample. For now, we will only sample once.
        print("    computing covariance")
        runtag = 'cov'
        coefs_dir = os.path.dirname(os.path.abspath(coefs_file))
        cwd_orig = os.getcwd()
        try:
            os.chdir(coefs_dir)
            basis.writeCoefCovariance(Path(coefs_file).stem, runtag, snap_time)
        finally:
            os.chdir(cwd_orig)

    if os.path.exists(coefs_file):
        coefs.ExtendH5Coefs(coefs_file)
    else:
        coefs.WriteH5Coefs(coefs_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("> Done computing coefficients for snap {}*".format(snap_time))
    
    nparticles = len(halo_data['mass'])
    logging.info(f"Coefficients for snapshot t={snap_time}\
                  and nparticles={nparticles} computed \
                  in: {elapsed_time:.2f} s\n")

def compute_exp_coefs_parallel(
    particle_data,
    basis,
    component,
    coefs_file,
    unit_system,
    **kwargs
):
    """
    Compute EXP coefficients in an MPI-safe manner.

    All MPI ranks participate in coefficient construction.
    Only rank 0 performs HDF5 I/O.
    """

    # --------------------------------------------------
    # MPI setup
    # --------------------------------------------------
    from mpi4py import MPI
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank    = world_comm.Get_rank()

    # --------------------------------------------------
    # Start timing (global sync)
    # --------------------------------------------------
    world_comm.Barrier()
    start_time = time.time()

    # --------------------------------------------------
    # Coefficient construction (MPI-parallel inside C++)
    # --------------------------------------------------

    coef = basis.createFromArray(
        particle_data["mass"],
        particle_data["pos"],
        time=particle_data["snapshot_time"],
    )

    if my_rank == 0:
        logging.info("Created EXP coef object")

    # --------------------------------------------------
    # Coefs container logic (null-pointer workaround)
    # --------------------------------------------------
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)

    if my_rank == 0:
        logging.info("Added coef to container")

    # --------------------------------------------------
    # MPI-safe HDF5 output (rank 0 only)
    # --------------------------------------------------
    
    if my_rank == 0:
        coefs.setUnits(unit_system)
        if os.path.exists(coefs_file):
            coefs.ExtendH5Coefs(coefs_file)
            logging.info(f"Extended HDF5 file: {coefs_file}")
        else:
            coefs.WriteH5Coefs(coefs_file)
            logging.info(f"Created HDF5 file: {coefs_file}")

    # --------------------------------------------------
    # Final synchronization and timing
    # --------------------------------------------------
    world_comm.Barrier()
    end_time = time.time()

    elapsed_time = end_time - start_time
    nparticles = len(particle_data["mass"])
    snap_time = particle_data["snapshot_time"]
    if my_rank == 0:
        logging.info(
            f"Coefficients for snapshot t={snap_time} "
            f"with nparticles={nparticles} computed in "
            f"{elapsed_time:.2f} s"
        )

