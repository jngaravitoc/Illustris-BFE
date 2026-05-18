"""Helpers for the halo 21537 BFE profile export used by the notebook plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_exp_profiles_table(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an exported BFE profile table and parse its metadata header.

    The file format is a commented header followed by 19 whitespace-delimited
    numeric columns:

    1. r_over_200c
    2-4. rho_part_over_rho200c, rho_bfe_over_rho200c, rel_diff for the first z
    5-7. rho_part_over_rho200c, rho_bfe_over_rho200c, rel_diff for the second z
    ...
    17-19. rho_part_over_rho200c, rho_bfe_over_rho200c, rel_diff for the sixth z
    """

    path = Path(path)
    metadata: dict[str, Any] = {}
    comments: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                comments.append(line.lstrip("#").strip())
                continue
            break

    for line in comments:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if key in {"halo_id", "nmax", "lmax"}:
            metadata[key] = int(value)
        elif key in {"basis_name", "basis"}:
            metadata["basis_name"] = value
        elif key in {"z_values", "z_list"}:
            metadata["z_values"] = np.fromstring(value.strip("[]"), sep=",")
        elif key in {"snapshots", "snaps", "snap_list"}:
            metadata["snapshots"] = np.fromstring(value.strip("[]"), sep=",", dtype=int)
        elif key in {"rho_c_values", "rho_c", "rho200c_values"}:
            metadata["rho_c_values"] = np.fromstring(value.strip("[]"), sep=",")
        elif key in {"r200c_values", "r200c"}:
            metadata["r200c_values"] = np.fromstring(value.strip("[]"), sep=",")

    data = np.loadtxt(path, comments="#")
    return data, metadata


def get_profile_column_names() -> list[str]:
    """Return the 19 expected column names for the exported table."""

    columns = ["r_over_200c"]
    for idx in range(6):
        columns.extend(
            [
                f"rho_part_over_rho200c_z{idx}",
                f"rho_bfe_over_rho200c_z{idx}",
                f"rel_diff_z{idx}",
            ]
        )
    return columns
