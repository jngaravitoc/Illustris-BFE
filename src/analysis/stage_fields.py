"""
Stage 4 — Density fields (BFE and KDE).

Responsibility: compute BFE and KDE density fields on a 3-D Cartesian grid
for every requested snapshot and write one HDF5 file per snapshot to:
    BFE  →  {data_root}/{sim}/{halo_id}/fields/bfe/
    KDE  →  {data_root}/{sim}/{halo_id}/fields/kde/

Per-snapshot files that already exist are skipped, so the stage is safe to
restart after a partial run.

Requires
--------
* Stage 3 output: config.coefficients_file()
* Stage 2 output: config.basis_config_file()
* Particle snapshots: config.particles_file(snap)
* config.halo_params_file  (for R200c / rho200c normalisation)
* Time-evolution file: {data_root}/{sim}/{sim}_halo_time_evol.txt

run() contract
--------------
    outputs = stage_fields.run(config, skip_kde=False, skip_bfe=False)
    outputs["bfe_files"]   # dict[int, Path] — {snap: path}
    outputs["kde_files"]   # dict[int, Path] — {snap: path}
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pyEXP

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.pipeline_config import PipelineConfig
from compute_coefficients import load_basis_from_config_file
from exp.compute_fields import compute_bfe_fields, compute_kde_density
from exp.data_ios import read_halo_params, read_tng_halo_particles
from visuals.field_io import write_fields, write_kde_density


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_grid(config: PipelineConfig) -> np.ndarray:
    """Build the normalised 3-D Cartesian grid specified by the config."""
    lo, hi = config.grid_range
    dbins = np.linspace(lo, hi, config.grid_bins)
    grid_arrays = np.meshgrid(dbins, dbins, dbins, indexing="ij")
    return np.stack(grid_arrays)


def _halo_params_by_snap(config: PipelineConfig) -> dict[str, dict]:
    """
    Return per-snapshot halo parameters (R200c, rho200c) as nested dicts.

    Returns
    -------
    dict with keys "R200c" and "rho200c", each mapping snap -> float.
    """
    params = read_halo_params(str(config.halo_params_file))

    snap_arr = np.asarray(params["snap"], dtype=int)
    M200c_arr = np.asarray(params["M200c"], dtype=float)
    R200c_arr = np.asarray(params["R200c"], dtype=float)
    rho200c_arr = M200c_arr / (4.0 / 3.0 * np.pi * R200c_arr ** 3)

    R200c = {int(s): float(v) for s, v in zip(snap_arr, R200c_arr)}
    rho200c = {int(s): float(v) for s, v in zip(snap_arr, rho200c_arr)}
    return {"R200c": R200c, "rho200c": rho200c}


# ------------------------------------------------------------------
# BFE fields
# ------------------------------------------------------------------

def _compute_bfe_fields(config: PipelineConfig, missing_snaps: list[int]) -> dict[int, Path]:
    """
    Compute BFE density fields for the requested snapshots and write HDF5 files.

    BFE fields for all missing snapshots are evaluated in a single
    FieldGenerator call (most efficient).

    Returns
    -------
    dict[snap, Path]  — paths to the written files.
    """
    # Time keys in the coefficients file are the snapshot numbers (floats).
    eval_times = [float(s) for s in missing_snaps]

    # Load basis and coefficients
    print(f"[fields/bfe] Loading basis from {config.basis_config_file()}")
    basis = load_basis_from_config_file(str(config.basis_config_file()))

    print(f"[fields/bfe] Loading coefficients from {config.coefficients_file()}")
    coefs = pyEXP.coefs.Coefs.factory(str(config.coefficients_file()))

    grid = _build_grid(config)
    field_shape = (config.grid_bins, config.grid_bins, config.grid_bins)

    print(
        f"[fields/bfe] Computing fields for {len(missing_snaps)} snapshots "
        f"on a {config.grid_bins}³ grid …"
    )
    _, _FP, points = compute_bfe_fields(grid, basis, coefs, eval_times)

    # Write one HDF5 per snapshot
    bfe_dir = config.output_dir("fields") / "bfe"
    bfe_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for snap, t in zip(missing_snaps, eval_times):
        out_file = config.bfe_fields_file(snap)
        # write_fields expects the full points dict, but we write only this time
        write_fields({t: points[t]}, str(out_file), field_shape=field_shape)
        written[snap] = out_file
        print(f"[fields/bfe]   snap {snap:3d} → {out_file.name}")

    return written


def _compute_kde_fields(config: PipelineConfig, missing_snaps: list[int]) -> dict[int, Path]:
    """
    Compute KDE density fields for the requested snapshots and write HDF5 files.

    Each snapshot is processed independently (particles must be loaded and
    normalised per snapshot).

    Returns
    -------
    dict[snap, Path]  — paths to the written files.
    """
    halo_p = _halo_params_by_snap(config)
    R200c_map = halo_p["R200c"]
    rho200c_map = halo_p["rho200c"]
    grid = _build_grid(config)

    kde_dir = config.output_dir("fields") / "kde"
    kde_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for i, snap in enumerate(missing_snaps, 1):
        particles_file = config.particles_file(snap)
        if not particles_file.exists():
            print(
                f"[fields/kde]   WARNING snap {snap:3d} — particle file not found, "
                f"skipping: {particles_file}"
            )
            continue

        print(
            f"[fields/kde]   [{i}/{len(missing_snaps)}] snap {snap:3d} — "
            f"loading {particles_file.name}"
        )

        coords, masses = read_tng_halo_particles(str(particles_file))

        R200c = R200c_map[snap]
        rho200c = rho200c_map[snap]
        pos_norm = coords / R200c
        mass_norm = masses / (rho200c * R200c ** 3)

        kd_dens = compute_kde_density(pos_norm, mass_norm, grid)

        out_file = config.kde_fields_file(snap)
        grid_shape = (config.grid_bins, config.grid_bins, config.grid_bins)
        write_kde_density(
            kd_dens,
            str(out_file),
            grid_shape=grid_shape,
            snapshot_name=f"snap_{snap:03d}",
        )
        written[snap] = out_file
        print(f"[fields/kde]   snap {snap:3d} → {out_file.name}")

    return written


# ------------------------------------------------------------------
# Public interface
# ------------------------------------------------------------------

def run(
    config: PipelineConfig,
    skip_bfe: bool = False,
    skip_kde: bool = False,
) -> dict:
    """
    Compute BFE and KDE density fields for all snapshots in the config.

    Only snapshots whose output files are missing are (re-)computed.

    Parameters
    ----------
    config : PipelineConfig
    skip_bfe : bool
        If True, skip BFE field computation entirely.
    skip_kde : bool
        If True, skip KDE field computation entirely.

    Returns
    -------
    dict with keys:
        "bfe_files" : dict[int, Path]  — {snap: path} (empty if skip_bfe)
        "kde_files" : dict[int, Path]  — {snap: path} (empty if skip_kde)

    Raises
    ------
    FileNotFoundError
        If required input files (basis, coefficients, halo params) are missing.
    """
    _check_required_inputs(config, skip_bfe)

    # --- BFE ---
    bfe_files: dict[int, Path] = {}
    if not skip_bfe:
        already = {s: config.bfe_fields_file(s) for s in config.snapshots if config.bfe_fields_file(s).exists()}
        missing = [s for s in config.snapshots if not config.bfe_fields_file(s).exists()]
        if already:
            print(f"[fields/bfe] {len(already)} snapshot(s) already exist, skipping those.")
        if missing:
            bfe_files = _compute_bfe_fields(config, missing)
        bfe_files.update(already)

    # --- KDE ---
    kde_files: dict[int, Path] = {}
    if not skip_kde:
        already = {s: config.kde_fields_file(s) for s in config.snapshots if config.kde_fields_file(s).exists()}
        missing = [s for s in config.snapshots if not config.kde_fields_file(s).exists()]
        if already:
            print(f"[fields/kde] {len(already)} snapshot(s) already exist, skipping those.")
        if missing:
            print(f"[fields/kde] Computing KDE fields for {len(missing)} snapshot(s) …")
            kde_files = _compute_kde_fields(config, missing)
        kde_files.update(already)

    return {"bfe_files": bfe_files, "kde_files": kde_files}


def _check_required_inputs(config: PipelineConfig, need_bfe: bool) -> None:
    """Raise FileNotFoundError for any missing required input files."""
    if need_bfe or True:   # halo_params always needed (for KDE normalisation)
        if not config.halo_params_file.exists():
            raise FileNotFoundError(
                f"[fields] Halo params file not found: {config.halo_params_file}"
            )
    if not need_bfe:
        return
    if not config.basis_config_file().exists():
        raise FileNotFoundError(
            f"[fields] Basis config not found: {config.basis_config_file()}\n"
            "  Run stage_basis first."
        )
    if not config.coefficients_file().exists():
        raise FileNotFoundError(
            f"[fields] Coefficients file not found: {config.coefficients_file()}\n"
            "  Run stage_coefficients first."
        )


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 4 — compute BFE and KDE density fields.",
    )
    parser.add_argument("config", help="Path to pipeline YAML config file.")
    parser.add_argument("--skip-bfe", action="store_true", help="Skip BFE field computation.")
    parser.add_argument("--skip-kde", action="store_true", help="Skip KDE field computation.")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    outputs = run(cfg, skip_bfe=args.skip_bfe, skip_kde=args.skip_kde)
    print(outputs)
