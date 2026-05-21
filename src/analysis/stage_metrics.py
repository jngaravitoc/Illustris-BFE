"""
Stage 5 — Fidelity metrics (MISE / MIRSE).

Responsibility: for every requested snapshot, read the BFE and KDE density
fields produced by stage_fields, compute full MISE and MIRSE maps, save the
metric matrices to a single HDF5 file per halo configuration, and produce a
summary figure.

Outputs
-------
Metrics HDF5:
    {data_root}/{sim}/{halo_id}/metrics/halo_{halo_id}_metrics_{nmax:02d}_{lmax:02d}.h5
    Datasets: snapshots, redshift, mise, mirse

Summary figure:
    {data_root}/{sim}/{halo_id}/figures/halo_{halo_id}_metrics_{nmax:02d}_{lmax:02d}.pdf

The stage skips computation if the metrics HDF5 file already exists.

run() contract
--------------
    outputs = stage_metrics.run(config)
    outputs["metrics_file"]   # Path to the metrics HDF5 file
    outputs["figure_file"]    # Path to the PDF figure (None if matplotlib fails)
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.pipeline_config import PipelineConfig
from visuals.field_io import read_bfe_fields, read_kde_density


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_MPLSTYLE = _SRC_DIR / "illustris_bfe.mplstyle"


def _snap_to_time_map(config: PipelineConfig) -> dict[int, float]:
    """Return {snapshot: redshift} from the time-evolution file."""
    time_evol_file = config.halo_dir() / f"{config.sim}_halo_time_evol.txt"
    if not time_evol_file.exists():
        raise FileNotFoundError(
            f"[metrics] Time-evolution file not found: {time_evol_file}"
        )
    sim_snap, sim_z, _, _ = np.loadtxt(time_evol_file, skiprows=4, unpack=True)
    return {int(s): float(z) for s, z in zip(sim_snap, sim_z)}


def _compute_metrics(
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BFE and KDE fields for every snapshot and compute full MISE/MIRSE maps.

    Returns
    -------
    snaps : ndarray, shape (n,)          — snapshot numbers
    mise_vals : ndarray, shape (n, ...)  — MISE map per snapshot
    mirse_vals : ndarray, shape (n, ...) — MIRSE map per snapshot
    """
    snaps_out, mise_out, mirse_out = [], [], []

    for snap in config.snapshots:
        bfe_file = config.bfe_fields_file(snap)
        kde_file = config.kde_fields_file(snap)

        if not bfe_file.exists():
            print(f"[metrics]   WARNING snap {snap:3d} — BFE file missing, skipping.")
            continue
        if not kde_file.exists():
            print(f"[metrics]   WARNING snap {snap:3d} — KDE file missing, skipping.")
            continue

        # Read BFE density — time key is the snapshot number (stored as float)
        t = float(snap)
        dens_bfe = read_bfe_fields(str(bfe_file), "dens", t)

        # Read KDE density (reshaped to 3-D by read_kde_density)
        kd_dens, _ = read_kde_density(str(kde_file))

        # Compute full 3-D pointwise metric fields (no spatial reduction).
        diff = np.asarray(dens_bfe, dtype=float) - np.asarray(kd_dens, dtype=float)
        mise_arr = diff ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.asarray(dens_bfe, dtype=float) / np.asarray(kd_dens, dtype=float) - 1.0
            mirse_arr = rel ** 2

        snaps_out.append(snap)
        mise_out.append(mise_arr)
        mirse_out.append(mirse_arr)

        print(
            f"[metrics]   snap {snap:3d}  "
            f"stored MISE/MIRSE maps with shape {mise_arr.shape}"
        )

    return (
        np.asarray(snaps_out, dtype=int),
        np.asarray(mise_out, dtype=float),
        np.asarray(mirse_out, dtype=float),
    )


def _write_metrics(
    config: PipelineConfig,
    snaps: np.ndarray,
    mise_vals: np.ndarray,
    mirse_vals: np.ndarray,
    redshift: np.ndarray,
) -> Path:
    """Write metric arrays to the canonical HDF5 file."""
    out_file = config.metrics_file()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_file), "w") as f:
        f.attrs["halo_id"] = config.halo_id
        f.attrs["sim"] = config.sim
        f.attrs["nmax"] = config.nmax
        f.attrs["lmax"] = config.lmax

        f.create_dataset("snapshots", data=snaps)
        f.create_dataset("redshift", data=redshift)
        f.create_dataset("mise", data=mise_vals)
        f.create_dataset("mirse", data=mirse_vals)

    print(f"[metrics] Metrics saved: {out_file}")
    return out_file


def _read_metrics(metrics_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read snapshots, redshift, and full MISE/MIRSE matrices from metrics HDF5."""
    with h5py.File(str(metrics_file), "r") as f:
        snaps = np.asarray(f["snapshots"], dtype=int)
        redshift = np.asarray(f["redshift"], dtype=float)
        mise_vals = np.asarray(f["mise"], dtype=float)
        mirse_vals = np.asarray(f["mirse"], dtype=float)
    return snaps, redshift, mise_vals, mirse_vals


def _summarize_metric_matrices(
    mise_vals: np.ndarray,
    mirse_vals: np.ndarray,
    spatial_axis: int = 2,
    return_2d: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-snapshot mean and variance from full MISE/MIRSE matrices.

    The requested `spatial_axis` (0, 1, or 2) is collapsed first.

    If `return_2d=False` (default), this returns per-snapshot 1-D summaries
    computed over the remaining spatial dimensions.

    If `return_2d=True`, this returns per-snapshot 2-D maps where each map is
    the mean or variance along the chosen `spatial_axis`.

    Returns
    -------
    mise_mean, mirse_mean, mise_var, mirse_var : ndarray
        If `return_2d=False`: shape (n_snapshots,)
        If `return_2d=True`:  shape (n_snapshots, ny, nx) (or analogous 2-D shape
        from the remaining spatial axes)

    Examples
    --------
    # Per-snapshot scalar summaries (default behavior)
    mise_mean, mirse_mean, mise_var, mirse_var = _summarize_metric_matrices(
        mise_vals,
        mirse_vals,
        spatial_axis=2,
    )

    # Per-snapshot 2-D summary maps after collapsing axis=2
    mise_mean_2d, mirse_mean_2d, mise_var_2d, mirse_var_2d = _summarize_metric_matrices(
        mise_vals,
        mirse_vals,
        spatial_axis=2,
        return_2d=True,
    )
    """
    if mise_vals.shape != mirse_vals.shape:
        raise ValueError(
            f"MISE and MIRSE arrays must have the same shape. "
            f"Got {mise_vals.shape} vs {mirse_vals.shape}."
        )
    # Backward compatibility: legacy files may already store per-snapshot scalars.
    if mise_vals.ndim == 1:
        if return_2d:
            raise ValueError(
                "Cannot return 2-D maps from 1-D legacy metrics arrays. "
                "Recompute metrics to store full per-snapshot matrices."
            )
        n_snapshots = mise_vals.shape[0]
        return (
            np.asarray(mise_vals, dtype=float).reshape(n_snapshots),
            np.asarray(mirse_vals, dtype=float).reshape(n_snapshots),
            np.zeros(n_snapshots, dtype=float),
            np.zeros(n_snapshots, dtype=float),
        )

    if mise_vals.ndim < 2:
        raise ValueError(
            f"Expected metric arrays with snapshot + spatial dimensions. "
            f"Got ndim={mise_vals.ndim}."
        )

    n_spatial_dims = mise_vals.ndim - 1
    if not isinstance(spatial_axis, int) or not (0 <= spatial_axis < n_spatial_dims):
        raise ValueError(
            f"spatial_axis must be in [0, {n_spatial_dims - 1}] for arrays with "
            f"shape {mise_vals.shape}. Got {spatial_axis}."
        )

    reduce_axis = spatial_axis + 1  # offset by snapshot axis at position 0
    # Per-snapshot 2-D maps after collapsing the selected spatial axis.
    mise_mean_2d = np.mean(mise_vals, axis=reduce_axis)
    mirse_mean_2d = np.mean(mirse_vals, axis=reduce_axis)
    mise_var_2d = np.var(mise_vals, axis=reduce_axis)
    mirse_var_2d = np.var(mirse_vals, axis=reduce_axis)

    if return_2d:
        return (
            np.asarray(mise_mean_2d, dtype=float),
            np.asarray(mirse_mean_2d, dtype=float),
            np.asarray(mise_var_2d, dtype=float),
            np.asarray(mirse_var_2d, dtype=float),
        )

    remaining_axes = tuple(range(1, mise_mean_2d.ndim))
    if remaining_axes:
        mise_mean = np.mean(mise_mean_2d, axis=remaining_axes)
        mirse_mean = np.mean(mirse_mean_2d, axis=remaining_axes)
        mise_var = np.var(mise_mean_2d, axis=remaining_axes)
        mirse_var = np.var(mirse_mean_2d, axis=remaining_axes)
    else:
        # Degenerate case: only one spatial dimension after collapse.
        mise_mean = mise_mean_2d
        mirse_mean = mirse_mean_2d
        mise_var = np.zeros_like(mise_mean_2d)
        mirse_var = np.zeros_like(mirse_mean_2d)

    # Enforce 1-D per-snapshot outputs.
    n_snapshots = mise_vals.shape[0]
    mise_mean = np.asarray(mise_mean, dtype=float).reshape(n_snapshots)
    mirse_mean = np.asarray(mirse_mean, dtype=float).reshape(n_snapshots)
    mise_var = np.asarray(mise_var, dtype=float).reshape(n_snapshots)
    mirse_var = np.asarray(mirse_var, dtype=float).reshape(n_snapshots)

    return mise_mean, mirse_mean, mise_var, mirse_var




def _make_figure(
    config: PipelineConfig,
    redshift: np.ndarray,
    mise_mean: np.ndarray,
    mirse_mean: np.ndarray,
    mise_var: np.ndarray,
    mirse_var: np.ndarray,
) -> Path | None:
    """
    Generate and save a four-panel MISE / MIRSE summary figure.

    Top row: per-snapshot means (MISE, MIRSE).
    Bottom row: per-snapshot variances (MISE, MIRSE).

    Returns the figure path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[metrics] matplotlib not available — skipping figure.")
        return None

    figures_dir = config.output_dir("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_name = f"halo_{config.halo_id}_metrics_{config.basis_tag()}.pdf"
    fig_path = figures_dir / fig_name

    style = str(_MPLSTYLE) if _MPLSTYLE.exists() else "default"
    with plt.style.context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        eps = np.finfo(float).tiny

        # --- Mean MISE ---
        ax = axes[0, 0]
        ax.semilogy(
            redshift,
            np.maximum(mise_mean, eps),
            "o-",
            color="tab:blue",
            label=f"nmax={config.nmax}, lmax={config.lmax}",
        )
        ax.set_title("Mean MISE")
        ax.set_ylabel("Value")
        ax.legend(fontsize=10)
        ax.invert_xaxis()

        # --- Mean MIRSE ---
        ax = axes[0, 1]
        ax.semilogy(
            redshift,
            np.maximum(mirse_mean, eps),
            "s-",
            color="tab:orange",
            label=f"nmax={config.nmax}, lmax={config.lmax}",
        )
        ax.set_title("Mean MIRSE")
        ax.legend(fontsize=10)
        ax.invert_xaxis()

        # --- Variance MISE ---
        ax = axes[1, 0]
        ax.semilogy(
            redshift,
            np.maximum(mise_var, eps),
            "o-",
            color="tab:green",
            label=f"nmax={config.nmax}, lmax={config.lmax}",
        )
        ax.set_title("Variance MISE")
        ax.set_ylabel("Value")
        ax.set_xlabel("Redshift $z$")
        ax.legend(fontsize=10)
        ax.invert_xaxis()

        # --- Variance MIRSE ---
        ax = axes[1, 1]
        ax.semilogy(
            redshift,
            np.maximum(mirse_var, eps),
            "s-",
            color="tab:red",
            label=f"nmax={config.nmax}, lmax={config.lmax}",
        )
        ax.set_title("Variance MIRSE")
        ax.set_xlabel("Redshift $z$")
        ax.legend(fontsize=10)
        ax.invert_xaxis()

        fig.suptitle(
            rf"Halo {config.halo_id} — {config.sim}  "
            rf"($n_{{max}}={config.nmax}$, $l_{{max}}={config.lmax}$)"
        )
        fig.tight_layout()
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)

    print(f"[metrics] Figure saved: {fig_path}")
    return fig_path


def make_snapshot_maps_figure(
    config: PipelineConfig,
    snapshots: int | list[int] | tuple[int, ...],
) -> dict[int, Path] | None:
    """
    Create per-snapshot 2x2 map figures from stored fidelity metric matrices.

    Each figure contains 4 subplots:
    - MISE mean map (collapsed along config.spatial_axis)
    - MIRSE mean map (collapsed along config.spatial_axis)
    - MISE variance map (along config.spatial_axis)
    - MIRSE variance map (along config.spatial_axis)

    Parameters
    ----------
    config : PipelineConfig
    snapshots : int or list/tuple of int
        Snapshot(s) to plot. If multiple snapshots are provided, one figure is
        generated per snapshot.

    Returns
    -------
    dict[int, Path] | None
        Mapping {snapshot: figure_path}, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[metrics] matplotlib not available — skipping snapshot map figures.")
        return None

    metrics_file = config.metrics_file()
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"[metrics] Metrics file not found: {metrics_file}. Run stage_metrics first."
        )

    snaps_all, redshift_all, mise_vals, mirse_vals = _read_metrics(metrics_file)
    if mise_vals.ndim < 3:
        raise ValueError(
            "Snapshot map figures require matrix-valued metrics in the HDF5 file. "
            f"Found mise ndim={mise_vals.ndim}. Re-run stage_metrics to regenerate "
            "the metrics file with full per-snapshot matrices."
        )
    mise_mean_2d, mirse_mean_2d, mise_var_2d, mirse_var_2d = _summarize_metric_matrices(
        mise_vals,
        mirse_vals,
        spatial_axis=config.spatial_axis,
        return_2d=True,
    )

    snap_to_index = {int(s): i for i, s in enumerate(snaps_all)}
    if isinstance(snapshots, int):
        requested_snaps = [snapshots]
    else:
        requested_snaps = [int(s) for s in snapshots]

    missing = [s for s in requested_snaps if s not in snap_to_index]
    if missing:
        raise ValueError(
            f"[metrics] Requested snapshot(s) not found in metrics file: {missing}. "
            f"Available snapshots: {list(map(int, snaps_all))}."
        )

    figures_dir = config.output_dir("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    style = str(_MPLSTYLE) if _MPLSTYLE.exists() else "default"
    out_paths: dict[int, Path] = {}

    with plt.style.context(style):
        for snap in requested_snaps:
            idx = snap_to_index[snap]
            z = float(redshift_all[idx])
            x_min, x_max = float(config.grid_range[0]), float(config.grid_range[1])
            extent = (x_min, x_max, x_min, x_max)

            fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

            maps = [
                (mise_mean_2d[idx], "MISE (mean along axis)", "viridis"),
                (mirse_mean_2d[idx], "MIRSE (mean along axis)", "magma"),
                (mise_var_2d[idx], "MISE variance (along axis)", "cividis"),
                (mirse_var_2d[idx], "MIRSE variance (along axis)", "plasma"),
            ]

            for ax, (arr, ttl, cmap) in zip(axs.flatten(), maps):
                im = ax.imshow(
                    np.asarray(arr),
                    origin="lower",
                    cmap=cmap,
                    aspect="equal",
                    extent=extent,
                )
                ax.set_title(ttl)
                ax.set_xlabel(r"$x / r_{200c}$")
                ax.set_ylabel(r"$y / r_{200c}$")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(
                rf"Halo {config.halo_id} — {config.sim}  "
                rf"snap={snap:03d}, z={z:.3f}, axis={config.spatial_axis}"
            )

            nmax_label = f"{config.nmax:02d}" if isinstance(config.nmax, int) else str(config.nmax)
            lmax_label = f"{config.lmax:02d}" if isinstance(config.lmax, int) else str(config.lmax)
            fig_name = (
                f"halo_{config.halo_id}_nmax_{nmax_label}_lmax_{lmax_label}"
                f"_axis_projection_{config.spatial_axis}_snapshot_{snap:03d}_metrics_maps.pdf"
            )
            fig_path = figures_dir / fig_name
            fig.savefig(str(fig_path), dpi=150)
            plt.close(fig)

            out_paths[snap] = fig_path
            print(f"[metrics] Snapshot map figure saved: {fig_path}")

    return out_paths


# ------------------------------------------------------------------
# Public interface
# ------------------------------------------------------------------

def run(config: PipelineConfig, generate_all_snapshot_maps: bool = False) -> dict:
    """
    Compute MISE and MIRSE for all available snapshots and save results.

    Skips computation if the metrics HDF5 file already exists.

    Parameters
    ----------
    config : PipelineConfig
    generate_all_snapshot_maps : bool, optional
        If True, generate 2-D metric map figures for all snapshots available in
        the metrics HDF5 file. Default is False.

    Returns
    -------
    dict with keys:
        "metrics_file" : Path        — HDF5 file with metric arrays.
        "figure_file"  : Path | None — PDF summary figure (None if skipped).
        "map_figure_files" : dict[int, Path] | None — per-snapshot map figures
            (included only when generate_all_snapshot_maps is True).
    """
    metrics_file = config.metrics_file()

    if metrics_file.exists():
        print(f"[metrics] Already exists, skipping: {metrics_file}")
        snaps_all, redshift, mise_vals, mirse_vals = _read_metrics(metrics_file)

        # If map figures are requested, legacy 1-D metrics files must be upgraded.
        if generate_all_snapshot_maps and mise_vals.ndim < 3:
            print(
                "[metrics] Existing metrics file uses legacy 1-D arrays; "
                "recomputing to store full matrix-valued metrics for map figures."
            )

            snaps_new, mise_vals_new, mirse_vals_new = _compute_metrics(config)
            if len(snaps_new) == 0:
                raise RuntimeError(
                    "[metrics] No snapshots could be processed while upgrading "
                    "legacy metrics file. Make sure stage_fields has run successfully."
                )

            snap_to_z = _snap_to_time_map(config)
            redshift_new = np.asarray([snap_to_z[s] for s in snaps_new], dtype=float)
            metrics_file = _write_metrics(
                config,
                snaps_new,
                mise_vals_new,
                mirse_vals_new,
                redshift_new,
            )

            snaps_all, redshift, mise_vals, mirse_vals = _read_metrics(metrics_file)

        mise_mean, mirse_mean, mise_var, mirse_var = _summarize_metric_matrices(
            mise_vals,
            mirse_vals,
            spatial_axis=config.spatial_axis,
        )
        fig_path = _make_figure(
            config,
            redshift,
            mise_mean,
            mirse_mean,
            mise_var,
            mirse_var,
        )
        run_outputs = {
            "metrics_file": metrics_file,
            "figure_file": fig_path,
        }
        if generate_all_snapshot_maps:
            run_outputs["map_figure_files"] = make_snapshot_maps_figure(
                config,
                snapshots=[int(s) for s in snaps_all],
            )
        return run_outputs

    print(f"[metrics] Computing MISE / MIRSE for {len(config.snapshots)} snapshots …")

    snaps, mise_vals, mirse_vals = _compute_metrics(config)

    if len(snaps) == 0:
        raise RuntimeError(
            "[metrics] No snapshots could be processed.  "
            "Make sure stage_fields has run successfully."
        )

    # Load redshifts from the time-evolution file — used only for the figure's x-axis.
    snap_to_z = _snap_to_time_map(config)
    redshift = np.asarray([snap_to_z[s] for s in snaps], dtype=float)

    metrics_file = _write_metrics(
        config,
        snaps,
        mise_vals,
        mirse_vals,
        redshift,
    )

    # Read back from the fidelity HDF5 and summarize across spatial dimensions.
    _, redshift_h5, mise_vals_h5, mirse_vals_h5 = _read_metrics(metrics_file)
    mise_mean, mirse_mean, mise_var, mirse_var = _summarize_metric_matrices(
        mise_vals_h5,
        mirse_vals_h5,
        spatial_axis=config.spatial_axis,
    )

    fig_path = _make_figure(
        config,
        redshift_h5,
        mise_mean,
        mirse_mean,
        mise_var,
        mirse_var,
    )

    run_outputs = {"metrics_file": metrics_file, "figure_file": fig_path}
    if generate_all_snapshot_maps:
        run_outputs["map_figure_files"] = make_snapshot_maps_figure(
            config,
            snapshots=[int(s) for s in snaps],
        )

    return run_outputs


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 5 — compute MISE/MIRSE metrics and generate figure.",
    )
    parser.add_argument("config", help="Path to pipeline YAML config file.")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    run_result = run(cfg)
    print(run_result)
