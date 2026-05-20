#!/usr/bin/env python
"""
run_pipeline.py — CLI entry point for the Illustris BFE analysis pipeline.

Usage
-----
    # Run all enabled stages for a single halo config
    python src/run_pipeline.py config.yaml

    # Run specific stages only
    python src/run_pipeline.py config.yaml --stages basis coefficients

    # Validate config + print resolved paths without running anything
    python src/run_pipeline.py config.yaml --dry-run

    # Override the halo_id in the config (useful for testing)
    python src/run_pipeline.py config.yaml --halo-id 99999

    # Run in parallel over a CSV of halo IDs using the same base config
    python src/run_pipeline.py config.yaml --batch data/TNG_parameters.csv --parallel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python src/run_pipeline.py` without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from analysis.pipeline_config import PipelineConfig
from analysis.pipeline import HaloPipeline, STAGE_ORDER
from analysis.batch import BatchRunner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Run the Illustris BFE density-field analysis pipeline.",
    )
    p.add_argument(
        "config",
        metavar="CONFIG",
        help="Path to a pipeline YAML config file.",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        choices=STAGE_ORDER,
        default=None,
        help=(
            "Stages to run (default: all stages enabled in the config).  "
            f"Choices: {STAGE_ORDER}."
        ),
    )
    p.add_argument(
        "--halo-id",
        type=int,
        default=None,
        metavar="ID",
        help="Override the halo_id in the config.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and planned stages, then exit without running anything.",
    )
    p.add_argument(
        "--batch",
        metavar="CSV",
        default=None,
        help=(
            "Path to a CSV file with a 'halo_id' column.  When provided, the pipeline "
            "is run for every halo in the CSV using the base config as a template."
        ),
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="(Batch mode only) Process halos in parallel using multiprocessing.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="(Batch mode only) Number of parallel worker processes (default: cpu_count).",
    )
    return p


def _dry_run(config: PipelineConfig, stages: list[str]) -> None:
    """Print config summary and resolved paths, then exit."""
    print("\n=== DRY RUN ===")
    print(f"  sim          : {config.sim}")
    print(f"  halo_id      : {config.halo_id}")
    if config.is_multi_order():
        print(f"  nmax / lmax  : paired lists {config.order_pairs()}")
    else:
        print(f"  nmax / lmax  : {config.nmax} / {config.lmax}")
    print(f"  snapshots    : {config.snapshots}")
    print(f"  grid         : {config.grid_bins}³, range={config.grid_range}")
    print(f"  data_root    : {config.data_root}")
    print(f"\n  Stages to run: {stages}")
    print(f"\n  Resolved paths:")
    print(f"    halo_dir           : {config.halo_dir()}")
    for nmax, lmax in config.order_pairs():
        cfg = config.with_orders(nmax=nmax, lmax=lmax)
        print(f"    [{cfg.basis_tag()}] basis_config_file  : {cfg.basis_config_file()}")
        print(f"    [{cfg.basis_tag()}] coefficients_file  : {cfg.coefficients_file()}")
        for snap in cfg.snapshots:
            print(f"    [{cfg.basis_tag()}] bfe_fields snap {snap:3d}: {cfg.bfe_fields_file(snap)}")
        print(f"    [{cfg.basis_tag()}] metrics_file       : {cfg.metrics_file()}")
    print("\n=== END DRY RUN ===\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    config = PipelineConfig.from_yaml(config_path)

    # Apply CLI overrides
    if args.halo_id is not None:
        config.halo_id = args.halo_id

    stages = args.stages if args.stages is not None else [
        s for s in STAGE_ORDER if config.stages.get(s, False)
    ]

    if args.dry_run:
        _dry_run(config, stages)
        return 0

    if args.batch:
        batch_csv = Path(args.batch)
        if not batch_csv.exists():
            parser.error(f"Batch CSV file not found: {batch_csv}")

        runner = BatchRunner.from_csv(
            batch_csv,
            sim=config.sim,
            nmax=config.nmax,
            lmax=config.lmax,
            snapshots=config.snapshots,
            data_root=config.data_root,
            grid_range=config.grid_range,
            grid_bins=config.grid_bins,
            stages=config.stages,
            halo_params_file=config.halo_params_file,
            profile_fit_file=config.profile_fit_file,
            particles_file_pattern=config.particles_file_pattern,
        )
        results = runner.run_all(
            stages=stages,
            parallel=args.parallel,
            workers=args.workers,
        )
        n_failed = sum(1 for ok in results.values() if not ok)
        return 1 if n_failed else 0

    # Single-halo mode
    if config.is_multi_order():
        print(f"[run_pipeline] Running paired orders: {config.order_pairs()}")
    pipe = HaloPipeline(config)
    pipe.run_stages(stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
