from hydrotools.core import interface as iface_run
from hydrotools.common import simulations as common_sims
from hydrotools.common import fields as common_fields



"""
 from Beneidkt:
 -> hydrotools command that creates a file with the DM particle coordinates of 5 random halo


oshs: particles in other subhalos in the same group
fuzz: members of the FOF group that are not bound to any subhalo


"""


iface_run.extractGalaxyData(
    num_processes = 12, 
    machine_name='umdastro',
    sim = 'tng35-3', 
    snap_idx = 99, 
    no_snapshots = False, 
    paranoid = True, 
    verbose = False,
    mass_selection_type = 'and',
    output_path = None, 
    buffered_output = False, 
    output_compression = 'gzip',
    Mdm_min = 1E12, 
    Mdm_max = 3E12, 
    extract_satellites = False,
    randomize_order = True, 
    rank_by = 'random', 
    n_max_extract = 20,
    catsh_get = True, 
    catsh_fields = common_fields.default_catsh_fields_dark, 
    catgrp_get = True, 
    catgrp_fields = common_fields.default_catgrp_fields_all,
    tree_get = True, 
    tree_fields = ['subfind_id', 'is_primary'],
    ptldm_get = True, 
    ptldm_fields = ['Coordinates', 'Velocities', 'oshs_Coordinates', 'fuzz_Coordinates'], 
    ptl_in_rad_get = True, 
    save_ptl_sep = True, 
    ptl_rad = 1.5, 
    ptl_rad_units = '200m',
    profile_get = False, 
    profile_fields = [])
