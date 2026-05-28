"""
Stage 3 — BFE coefficients.

Responsibility: compute pyEXP BFE coefficients for every requested snapshot
and write them to a single HDF5 file at:
    {data_root}/{sim}/{halo_id}/coefficients/halo_{halo_id}_coefficients_{nmax:02d}_{lmax:02d}.h5

The stage skips computation if the coefficients file already exists.

Requires
--------
* Stage 2 output: config.basis_config_file()
* Particle snapshots:  config.particles_file(snap) for each snap in config.snapshots
* config.halo_params_file

run() contract
--------------
    outputs = stage_coefficients.run(config)
    outputs["coefficients_file"]  # Path to the HDF5 coefficients file
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.pipeline_config import PipelineConfig
from compute_coefficients import compute_coefficients_for_snapshots


def run(config: PipelineConfig) -> dict:
    """
    Compute BFE coefficients for all snapshots in the config.

    Skips computation if the coefficients HDF5 file already exists.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    dict with key:
        "coefficients_file" : Path — HDF5 file containing all coefficients.

    Raises
    ------
    FileNotFoundError
        If the basis config YAML (from stage_basis) does not exist.
    """
    coefs_file = config.coefficients_file()

    # Skip if already computed
    if coefs_file.exists():
        print(f"[coefficients] Already exists, skipping: {coefs_file}")
        return {"coefficients_file": coefs_file}

    # Check required inputs
    basis_config = config.basis_config_file()
    if not basis_config.exists():
        raise FileNotFoundError(
            f"[coefficients] Basis config not found: {basis_config}\n"
            "  Run stage_basis first."
        )
    if not config.halo_params_file.exists():
        raise FileNotFoundError(
            f"[coefficients] Halo params file not found: {config.halo_params_file}"
        )

    # Create output directory
    coefs_dir = config.output_dir("coefficients")
    coefs_dir.mkdir(parents=True, exist_ok=True)

    # data_dir is the directory that contains the particle snapshots;
    # for TNG this is data_root/{sim}/halo_{halo_id}/particle_data/
    data_dir = str(
        config.data_root / config.sim / f"halo_{config.halo_id}" / "particle_data"
    )

    print(
        f"[coefficients] Computing for {len(config.snapshots)} snapshots "
        f"(nmax={config.nmax}, lmax={config.lmax}) → {coefs_dir}"
    )

    compute_coefficients_for_snapshots(
        basis_config_file=str(basis_config),
        halo_params_file=str(config.halo_params_file),
        data_dir=data_dir,
        snapshots=config.snapshots,
        nmax=config.nmax,
        lmax=config.lmax,
        halo_id=config.halo_id,
        sim=config.sim,
        coefs_filename=coefs_file.name,
        output_dir=coefs_dir,
        covariance=config.covariance,
    )

    if not coefs_file.exists():
        raise RuntimeError(
            f"[coefficients] compute_coefficients_for_snapshots completed "
            f"but output not found: {coefs_file}"
        )

    print(f"[coefficients] Done: {coefs_file}")
    return {"coefficients_file": coefs_file}


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 3 — compute BFE coefficients from particle snapshots.",
    )
    parser.add_argument("config", help="Path to pipeline YAML config file.")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    outputs = run(cfg)
    print(outputs)
