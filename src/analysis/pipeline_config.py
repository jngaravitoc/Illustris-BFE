"""
PipelineConfig: dataclass-based configuration for the density-field analysis pipeline.

Load from a YAML file with PipelineConfig.from_yaml(path).
Query output paths with config.output_dir(stage) or the per-file helpers such as
config.basis_config_file(), config.coefficients_file(), etc.

Example
-------
    config = PipelineConfig.from_yaml("src/analysis/example_config.yaml")
    print(config.basis_config_file())   # data/tng35-3-dark/halo_21537/basis/halo_21537_basis_config_08_02.yaml
    print(config.particles_file(99))    # data/tng35-3-dark/halo_21537/particle_data/galaxies_halo_21537_tng50-3-dark_099.hdf5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import yaml


# ------------------------------------------------------------------
# Default values
# ------------------------------------------------------------------

DEFAULT_STAGES: dict[str, bool] = {
    "profiles": True,
    "basis": True,
    "coefficients": True,
    "fields": True,
    "metrics": True,
}


def _default_data_root() -> Path:
    """Return $ILLUSTRIS_BFE/data, or cwd/data if the env variable is not set."""
    illustris_bfe = os.environ.get("ILLUSTRIS_BFE")
    if illustris_bfe:
        return Path(illustris_bfe) / "data"
    return Path.cwd() / "data"


# ------------------------------------------------------------------
# Config dataclass
# ------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Configuration for a single-halo density-field pipeline run.

    Parameters
    ----------
    sim : str
        Simulation identifier, e.g. "tng35-3-dark".
    halo_id : int
        Halo identifier, e.g. 21537.
    nmax : int | list[int]
        Maximum radial order for BFE expansion. May be a scalar or a list.
    lmax : int | list[int]
        Maximum angular order for BFE expansion. May be a scalar or a list.
    snapshots : list of int
        Snapshot numbers to process, e.g. [17, 21, 25, 33, 50, 99].
    grid_range : tuple of float
        Spatial extent of the density-field grid in normalized units, e.g. (-1.1, 1.1).
    grid_bins : int
        Number of grid bins per axis, e.g. 50.
    spatial_axis : int
        Spatial axis index (0, 1, 2) used by stages that collapse 3-D fields.
        Defaults to 2.
    metrics_generate_all_snapshot_maps : bool
        If True, stage_metrics will also generate 2-D map figures for all
        snapshots in the metrics HDF5. Defaults to False.
    stages : dict[str, bool]
        Which pipeline stages to run.  Keys: profiles, basis, coefficients, fields, metrics.
    data_root : Path
        Root directory for all data.  Defaults to $ILLUSTRIS_BFE/data.
    halo_params_file : Path
        Path to the halo parameters HDF5 file.  If relative, resolved from data_root.
        Defaults to data_root/{sim}/halo_{halo_id}_params.hdf5.
    profile_fit_file : Path or None
        Path to the density profile fit text file used by stage_basis.
        If None, stage_profiles is expected to produce it.
    particles_file_pattern : str
        Format string for the raw particle snapshot path relative to data_root.
        Placeholders: {halo_id} and {snap} (zero-padded to 3 digits via :03d).
        Example: "tng35-3-dark/galaxies_halo_{halo_id}_tng50-3-dark_{snap:03d}.hdf5"
    """

    # Required fields
    sim: str
    halo_id: int
    nmax: int | list[int]
    lmax: int | list[int]
    # After __post_init__ this is always list[int].
    # In the YAML file you may write  snapshots: all  to auto-discover every
    # snapshot that has a particle file on disk.
    snapshots: list[int] | str

    # Optional fields with defaults
    grid_range: tuple[float, float] = (-1.1, 1.1)
    grid_bins: int = 50
    spatial_axis: int = 2
    metrics_generate_all_snapshot_maps: bool = False
    covariance: bool = False
    stages: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_STAGES))
    data_root: Path = field(default_factory=_default_data_root)
    halo_params_file: Optional[Path] = None
    profile_fit_file: Optional[Path] = None
    particles_file_pattern: str = (
        "{sim}/halo_{halo_id}/particle_data/galaxies_halo_{halo_id}_tng50-3-dark_{snap:03d}.hdf5"
    )

    def __post_init__(self) -> None:
        # Normalise data_root to a Path
        self.data_root = Path(self.data_root)

        # Validate nmax/lmax and supported scalar/list combinations.
        self._validate_expansion_orders()

        # Expand snapshots: all  →  every snapshot that has a particle file on disk
        snapshots = self.snapshots
        if isinstance(snapshots, str) and snapshots.strip().lower() == "all":
            self.snapshots = self._discover_snapshots()

        # Resolve halo_params_file: default to data_root/{sim}/halo_{id}/halo_{id}_params.hdf5
        if self.halo_params_file is None:
            self.halo_params_file = (
                self.data_root / self.sim
                / f"halo_{self.halo_id}"
                / f"halo_{self.halo_id}_params.hdf5"
            )
        else:
            p = Path(self.halo_params_file)
            self.halo_params_file = p if p.is_absolute() else self.data_root / p

        # Resolve profile_fit_file if provided
        if self.profile_fit_file is not None:
            p = Path(self.profile_fit_file)
            self.profile_fit_file = p if p.is_absolute() else self.data_root / p

        # Validate spatial axis for 3-D field reductions.
        if not isinstance(self.spatial_axis, int) or self.spatial_axis not in (0, 1, 2):
            raise ValueError(
                f"spatial_axis must be an integer in (0, 1, 2). Got {self.spatial_axis!r}."
            )

        if not isinstance(self.metrics_generate_all_snapshot_maps, bool):
            raise ValueError(
                "metrics_generate_all_snapshot_maps must be a boolean. "
                f"Got {self.metrics_generate_all_snapshot_maps!r}."
            )

    # ------------------------------------------------------------------
    # Snapshot discovery (used when snapshots: all is set in the YAML)
    # ------------------------------------------------------------------

    def _discover_snapshots(self) -> list[int]:
        """
        Return sorted snapshot numbers for which a particle file exists on disk.

        Works by substituting sim and halo_id into particles_file_pattern,
        globbing for matching files, then extracting the snapshot number from
        each filename with a regex.
        """
        import re

        # Substitute the known values so only {snap...} remains as a placeholder
        pat = (
            self.particles_file_pattern
            .replace("{sim}", self.sim)
            .replace("{halo_id}", str(self.halo_id))
        )

        # Glob pattern: replace {snap...} with a wildcard
        glob_pat = re.sub(r"\{snap[^}]*\}", "*", pat)

        # Regex: escape literal parts, join with a digit-capturing group
        parts = re.split(r"\{snap[^}]*\}", pat)
        regex = re.compile(r"(\d+)".join(re.escape(p) for p in parts) + "$")

        snapshots = []
        for f in sorted(self.data_root.glob(glob_pat)):
            rel = str(f.relative_to(self.data_root))
            m = regex.match(rel)
            if m:
                snapshots.append(int(m.group(1)))

        if not snapshots:
            raise FileNotFoundError(
                f"snapshots: all — no particle files found for halo {self.halo_id}.\n"
                f"  data_root : {self.data_root}\n"
                f"  glob pattern: {glob_pat}"
            )

        return snapshots

    # ------------------------------------------------------------------
    # Directory and path helpers
    # ------------------------------------------------------------------

    def halo_dir(self) -> Path:
        """Root directory for this halo: {data_root}/{sim}/halo_{halo_id}/."""
        return self.data_root / self.sim / f"halo_{self.halo_id}"

    def output_dir(self, stage: str) -> Path:
        """Output subdirectory for a pipeline stage: {halo_dir()}/{stage}/."""
        return self.halo_dir() / stage

    def basis_tag(self) -> str:
        """Short string encoding the expansion order, e.g. '08_02' for nmax=8, lmax=2."""
        if not isinstance(self.nmax, int) or not isinstance(self.lmax, int):
            raise ValueError(
                "basis_tag() requires scalar nmax/lmax. "
                "Use config.with_orders(nmax, lmax) for list-valued configs."
            )
        return f"{self.nmax:02d}_{self.lmax:02d}"

    def order_pairs(self) -> list[tuple[int, int]]:
        """
        Return the ordered list of (nmax, lmax) pairs to run.

        For scalar values this returns a single pair.
        """
        if isinstance(self.nmax, int) and isinstance(self.lmax, int):
            return [(self.nmax, self.lmax)]
        return list(zip(self.nmax, self.lmax))

    def is_multi_order(self) -> bool:
        """Return True when more than one (nmax, lmax) pair is configured."""
        return len(self.order_pairs()) > 1

    def with_orders(self, nmax: int, lmax: int) -> "PipelineConfig":
        """Return a scalar-order copy of this config for one (nmax, lmax) pair."""
        return replace(
            self,
            nmax=nmax,
            lmax=lmax,
            snapshots=list(self.snapshots),
            stages=dict(self.stages),
        )

    def basis_config_file(self) -> Path:
        """YAML config file that pyEXP uses to load the basis."""
        name = f"halo_{self.halo_id}_basis_config_{self.basis_tag()}.yaml"
        return self.output_dir("basis") / name

    def coefficients_file(self) -> Path:
        """HDF5 file containing BFE coefficients for all requested snapshots."""
        name = f"halo_{self.halo_id}_coefficients_{self.basis_tag()}.h5"
        return self.output_dir("coefficients") / name

    def bfe_fields_file(self, snap: int) -> Path:
        """HDF5 file for BFE density fields at a given snapshot."""
        name = f"halo_{self.halo_id}_bfe_density_{self.basis_tag()}_snap_{snap:03d}.h5"
        return self.output_dir("fields") / "bfe" / name

    def kde_fields_file(self, snap: int) -> Path:
        """HDF5 file for the KDE density field at a given snapshot."""
        name = f"halo_{self.halo_id}_kde_density_field_snap_{snap:03d}.h5"
        return self.output_dir("fields") / "kde" / name

    def metrics_file(self) -> Path:
        """HDF5 file for MISE/MIRSE metrics for all snapshots."""
        name = f"halo_{self.halo_id}_metrics_{self.basis_tag()}.h5"
        return self.output_dir("metrics") / name

    def particles_file(self, snap: int) -> Path:
        """Full path to the raw particle snapshot file for a given snapshot number."""
        rel = self.particles_file_pattern.format(
            sim=self.sim, halo_id=self.halo_id, snap=snap
        )
        return self.data_root / rel

    # ------------------------------------------------------------------
    # YAML serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """
        Load a PipelineConfig from a YAML file.

        If data_root is given as a relative path in the YAML, it is resolved
        relative to the directory that contains the YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML config file.

        Returns
        -------
        PipelineConfig
        """
        path = Path(path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # YAML key is "simulation"; dataclass field is "sim"
        data["sim"] = data.pop("simulation")

        # Resolve a relative data_root against the YAML file's directory
        if "data_root" in data and data["data_root"] is not None:
            data_root = Path(data["data_root"])
            if not data_root.is_absolute():
                data_root = (path.parent / data_root).resolve()
            data["data_root"] = data_root

        # YAML lists become Python lists; convert grid_range to a tuple
        if "grid_range" in data:
            data["grid_range"] = tuple(data["grid_range"])

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save this config to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "simulation": self.sim,
            "halo_id": self.halo_id,
            "nmax": self.nmax,
            "lmax": self.lmax,
            "snapshots": self.snapshots,
            "grid_range": list(self.grid_range),
            "grid_bins": self.grid_bins,
            "spatial_axis": self.spatial_axis,
            "metrics_generate_all_snapshot_maps": self.metrics_generate_all_snapshot_maps,
            "covariance": self.covariance,
            "stages": self.stages,
            "data_root": str(self.data_root),
            "halo_params_file": str(self.halo_params_file),
            "profile_fit_file": (
                str(self.profile_fit_file) if self.profile_fit_file is not None else None
            ),
            "particles_file_pattern": self.particles_file_pattern,
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        enabled = [s for s, on in self.stages.items() if on]
        return (
            f"PipelineConfig("
            f"sim={self.sim!r}, halo_id={self.halo_id}, "
            f"nmax={self.nmax}, lmax={self.lmax}, "
            f"snapshots={self.snapshots}, "
            f"stages={enabled})"
        )

    def _validate_expansion_orders(self) -> None:
        """Validate nmax/lmax values and scalar/list compatibility."""
        nmax_is_int = isinstance(self.nmax, int)
        lmax_is_int = isinstance(self.lmax, int)
        nmax_is_list = isinstance(self.nmax, list)
        lmax_is_list = isinstance(self.lmax, list)

        if nmax_is_int and lmax_is_int:
            if self.nmax < 0 or self.lmax < 0:
                raise ValueError("nmax and lmax must be non-negative integers")
            return

        if nmax_is_list and lmax_is_list:
            if len(self.nmax) == 0 or len(self.lmax) == 0:
                raise ValueError("nmax and lmax lists must not be empty")
            if len(self.nmax) != len(self.lmax):
                raise ValueError(
                    "nmax and lmax lists must have the same number of elements"
                )
            if any((not isinstance(v, int) or v < 0) for v in self.nmax):
                raise ValueError("nmax list must contain only non-negative integers")
            if any((not isinstance(v, int) or v < 0) for v in self.lmax):
                raise ValueError("lmax list must contain only non-negative integers")
            return

        raise ValueError(
            "nmax and lmax must both be integers or both be lists of integers"
        )
