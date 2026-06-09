import os
from pathlib import Path

import h5py
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

import pyEXP

if __name__ == "__main__":

    # Define data paths:
    CATALOG_FILE = '/home/garavito/codes/hydrotools/tutorials/mwm31s_hostcatalog.hdf5'
    OUTPUT_DIR   = '/n/nyx3/garavito/projects/Illustris-BFE/data/mwm31'

    # Load subfind halo IDs
    with h5py.File(CATALOG_FILE, 'r') as f:
        subhalo_ids = f['SubfindID'][:].tolist()
    
    #tutorial_dir = Path.cwd()

    print('Tutorial directory:', tutorial_dir)
    print('Output directory  :', output_dir)

    output_file = OUTPUT_DIR / f'galaxies{subhalo_ids}_tng50_099.hdf5'
    
    for subhalo_id in subhalo_ids:
        with h5py.File(output_file, 'r') as f:
            bfe_keys = sorted([k for k in f.keys() if k.startswith('bfe_')])
            print('BFE keys:')
            for k in bfe_keys:
                print('  ', k, f[k].shape)

            R200m = float(f['catgrp_R200m'][0])
            M200m = float(f['catgrp_M200m'][0])
            rho200m = M200m / (4.0 / 3.0 * np.pi * R200m**3)

            bfe_bins_log10 = f['bfe_bins'][0]
            bfe_rho = np.abs(f['bfe_dkm_rho'][0])
            bfe_pot = f['bfe_dkm_pot'][0]
            bfe_rforce = f['bfe_dkm_rforce'][0]
            bfe_rho_3d = f['bfe_dkm_3d_rho'][0]
            kde_rho_3d = f['bfe_dkm_3d_rho_kde'][0]
            grid_axis = f['bfe_dkm_3d_rho_grid'][0]

            prof_r = f['profile_bin_mids'][0] / R200m
            prof_rho = f['profile_dkm_rho_3d'][0] / rho200m

            mise = np.asarray(f['bfe_dkm_mise'][0]).squeeze()
            mirse = np.asarray(f['bfe_dkm_mirse'][0]).squeeze()
            gof = np.asarray(f['bfe_dkm_gof'][0]).squeeze()

        bfe_r = 10.0**bfe_bins_log10
        print('MISE :', mise)
        print('MIRSE:', mirse)
        print('GOF  :', gof)
        print('3D BFE density shape:', bfe_rho_3d.shape)
        print('3D KDE density shape:', kde_rho_3d.shape)
        print('3D grid axis shape  :', grid_axis.shape)

        mid_idx = grid_axis.size // 2
        x_grid, y_grid = np.meshgrid(grid_axis, grid_axis, indexing='xy')
        kde_slice = np.asarray(kde_rho_3d[:, :, mid_idx], dtype=float)
        bfe_slice = np.asarray(bfe_rho_3d[:, :, mid_idx], dtype=float)

        kde_positive = kde_slice[kde_slice > 0.0]
        kde_floor = max(np.nanmin(kde_positive), 1.0e-12) if kde_positive.size else 1.0e-12
        bfe_abs = np.abs(bfe_slice)
        bfe_positive = bfe_abs[bfe_abs > 0.0]
        bfe_floor = max(np.nanmin(bfe_positive), 1.0e-12) if bfe_positive.size else 1.0e-12

        kde_log = np.log10(np.clip(kde_slice, kde_floor, None))
        bfe_log = np.log10(np.clip(bfe_abs, bfe_floor, None))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        kde_plot = axes[0].contourf(
            x_grid,
            y_grid,
            kde_log,
            levels=40,
            cmap='twilight',
        )
        axes[0].set_title('KDE density mid-plane')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_aspect('equal')
        fig.colorbar(kde_plot, ax=axes[0], label=r'$\log_{10}\,\rho_{\rm KDE}$')

        bfe_plot = axes[1].contourf(
            x_grid,
            y_grid,
            bfe_log,
            levels=40,
            cmap='twilight',
            vmin=-4, vmax=1.6,
        )
        axes[1].set_title('BFE density mid-plane')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_aspect('equal')
        fig.colorbar(bfe_plot, ax=axes[1], label=r'$\log_{10}\,|\rho_{\rm BFE}|$')
        plt.savefig('test_contour_fields.png')
        plt.show()

