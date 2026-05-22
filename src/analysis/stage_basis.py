"""
Stage 2 — Basis construction.

Responsibility: compute a pyEXP spherical basis from a density profile fit
and write the basis config YAML, cache, and model files to:
    {data_root}/{sim}/{halo_id}/basis/

The stage skips computation if the basis YAML config already exists.

Requires
--------
* config.profile_fit_file  — text file with (r, rho) normalised density fit
* config.halo_params_file  — HDF5 file with halo properties (R200c, M200c, …)

run() contract
--------------
    outputs = stage_basis.run(config)
    outputs["basis_config_file"]  # Path — the pyEXP basis YAML
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.pipeline_config import PipelineConfig
from compute_basis_from_fit import compute_basis_from_fit


def run(config: PipelineConfig) -> dict:
    """
    Compute the pyEXP basis from a density profile fit.

    Skips computation only if both the basis config YAML *and* the cache file
    already exist.  If the YAML exists but the cache is missing (e.g. a
    previous run was interrupted inside pyEXP.basis.Basis.factory), the
    basis is recomputed so the cache is written before stage_coefficients
    tries to load it.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    dict with key:
        "basis_config_file" : Path — pyEXP basis YAML config file.

    Raises
    ------
    FileNotFoundError
        If config.profile_fit_file or config.halo_params_file does not exist.
    """
    basis_config = config.basis_config_file()
    basis_dir = config.output_dir("basis")
    basis_cache = basis_dir / f"basis_cache_{config.basis_tag()}.txt"

    # Skip only if both the YAML and the cache already exist.
    # If the YAML exists but the cache is missing a previous run was
    # interrupted inside pyEXP.basis.Basis.factory; recompute in that case.
    if basis_config.exists() and basis_cache.exists():
        print(f"[basis] Already exists, skipping: {basis_config}")
        return {"basis_config_file": basis_config}

    if basis_config.exists() and not basis_cache.exists():
        print(
            f"[basis] YAML exists but cache is missing — recomputing: {basis_config}"
        )

    # Check required inputs
    if config.profile_fit_file is None:
        raise FileNotFoundError(
            "[basis] config.profile_fit_file is not set.  "
            "Set 'profile_fit_file' in your YAML config."
        )
    if not config.profile_fit_file.exists():
        raise FileNotFoundError(
            f"[basis] Profile fit file not found: {config.profile_fit_file}"
        )
    if not config.halo_params_file.exists():
        raise FileNotFoundError(
            f"[basis] Halo params file not found: {config.halo_params_file}"
        )

    # Create output directory
    basis_dir.mkdir(parents=True, exist_ok=True)

    # Name the basis config file after the halo and expansion order
    basis_filename = basis_config.name  # e.g. halo_21537_basis_config_08_02.yaml

    print(
        f"[basis] Computing basis (nmax={config.nmax}, lmax={config.lmax}) "
        f"→ {basis_dir}"
    )

    compute_basis_from_fit(
        fit_filename=str(config.profile_fit_file),
        halo_params_filename=str(config.halo_params_file),
        nmax=config.nmax,
        lmax=config.lmax,
        basis_path=str(basis_dir),
        basis_filename=str(basis_config),   # full absolute path for the YAML
    )

    if not basis_config.exists():
        raise RuntimeError(
            f"[basis] compute_basis_from_fit completed but output not found: {basis_config}"
        )

    print(f"[basis] Done: {basis_config}")
    return {"basis_config_file": basis_config}


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2 — compute the pyEXP basis from a density profile fit.",
    )
    parser.add_argument("config", help="Path to pipeline YAML config file.")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    outputs = run(cfg)
    print(outputs)
