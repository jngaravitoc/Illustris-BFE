import os
import sys
import time
import numpy as np
import logging
from mpi4py import MPI
import pyEXP

def compute_exp_coefs(halo_data, snap_time, basis, component, coefs_file, unit_system, **kwargs):
    # Compute coefficients
    start_time = time.time()
    coef = basis.createFromArray(halo_data['mass'], halo_data['pos'], snap_time)
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)
    
    #TODO: move unit systems to another function or a file
    coefs.setUnits(unit_system)

    if os.path.exists(coefs_file):
        coefs.ExtendH5Coefs(coefs_file)
    else:
        coefs.WriteH5Coefs(coefs_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("> Done computing coefficients*")
    
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

