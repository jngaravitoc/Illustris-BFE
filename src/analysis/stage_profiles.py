"""
Stage 1 — Density profiles.

Responsibility: locate or load the halo density profile HDF5 file for a run.

The stage first checks the canonical pipeline output location:
    {data_root}/{sim}/{halo_id}/profiles/halo_{halo_id}_density_profiles.hdf5

If that file does not exist, it falls back to the flat data layout that is
typically produced by the pre-processing scripts:
    {data_root}/{sim}/halo_{halo_id}_density_profiles.hdf5

If neither location holds the file, a FileNotFoundError is raised with
instructions on how to produce it.

run() contract
--------------
    outputs = stage_profiles.run(config)
    outputs["profiles_file"]  # Path to the profiles HDF5 file
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.pipeline_config import PipelineConfig
from exp.data_ios import read_density_profile  # noqa: F401 — re-exported for callers


def _canonical_profiles_file(config: PipelineConfig) -> Path:
    """Return the canonical pipeline output path for the profiles file."""
    name = f"halo_{config.halo_id}_density_profiles.hdf5"
    return config.output_dir("profiles") / name


def _flat_profiles_file(config: PipelineConfig) -> Path:
    """Return the legacy flat-layout path for the profiles file."""
    name = f"halo_{config.halo_id}_density_profiles.hdf5"
    return config.data_root / config.sim / name


def run(config: PipelineConfig) -> dict:
    """
    Locate the density profiles HDF5 file for this halo.

    Checks the canonical pipeline output path first, then the flat legacy
    layout.  Does not compute profiles from particles — that step requires
    separate pre-processing.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    dict with key:
        "profiles_file" : Path — path to the density profiles HDF5 file.

    Raises
    ------
    FileNotFoundError
        If the profiles file is not found at either expected location.
    """
    canonical = _canonical_profiles_file(config)
    flat = _flat_profiles_file(config)

    if canonical.exists():
        print(f"[profiles] Found (canonical): {canonical}")
        return {"profiles_file": canonical}

    if flat.exists():
        print(f"[profiles] Found (flat layout): {flat}")
        return {"profiles_file": flat}

    raise FileNotFoundError(
        f"[profiles] Density profiles file not found for halo {config.halo_id}.\n"
        f"  Checked (canonical) : {canonical}\n"
        f"  Checked (flat)      : {flat}\n"
        "  Run the density-profile pre-processing script first, then retry."
    )


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1 — locate the density profiles HDF5 file.",
    )
    parser.add_argument("config", help="Path to pipeline YAML config file.")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    outputs = run(cfg)
    print(outputs)
