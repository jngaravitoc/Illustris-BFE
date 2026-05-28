"""
BatchRunner — run the pipeline over a collection of halos.

Usage
-----
    from analysis.batch import BatchRunner
    from analysis.pipeline_config import PipelineConfig

    # Build from explicit config objects
    configs = [PipelineConfig.from_yaml(p) for p in yaml_files]
    runner  = BatchRunner(configs)
    runner.run_all()

    # Build from a CSV file of halo IDs
    runner = BatchRunner.from_csv(
        "data/TNG_parameters.csv",
        sim="tng35-3-dark",
        nmax=8,
        lmax=2,
        snapshots=[17, 21, 25, 33, 50, 99],
    )
    runner.run_all(stages=["basis", "coefficients"], parallel=True)
"""

from __future__ import annotations

import multiprocessing
import traceback
from pathlib import Path
from typing import Optional

from analysis.pipeline_config import PipelineConfig
from analysis.pipeline import HaloPipeline, STAGE_ORDER


def _run_halo(args: tuple) -> tuple[int, bool, str]:
    """
    Top-level function (picklable) used by the multiprocessing pool.

    Returns
    -------
    (halo_id, success, error_message)
    """
    config, stages = args
    try:
        pipe = HaloPipeline(config)
        pipe.run_stages(stages)
        return (config.halo_id, True, "")
    except Exception:  # noqa: BLE001
        return (config.halo_id, False, traceback.format_exc())


class BatchRunner:
    """
    Run the analysis pipeline over a list of halos.

    Parameters
    ----------
    configs : list[PipelineConfig]
        One config per halo.
    """

    def __init__(self, configs: list[PipelineConfig]) -> None:
        if not configs:
            raise ValueError("configs list must not be empty")
        self.configs = configs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_all(
        self,
        stages: Optional[list[str]] = None,
        parallel: bool = False,
        workers: int = 0,
    ) -> dict[int, bool]:
        """
        Run the pipeline for every halo.

        Parameters
        ----------
        stages : list[str] | None
            Subset of stages to run.  ``None`` means run all enabled stages.
        parallel : bool
            If True, use a ``multiprocessing.Pool`` to process halos in
            parallel.  Each halo runs all its stages sequentially within a
            single worker process.
        workers : int
            Number of worker processes.  0 (default) means
            ``multiprocessing.cpu_count()``.

        Returns
        -------
        dict mapping halo_id → True (success) / False (failed).
        """
        run_stages = stages if stages is not None else STAGE_ORDER
        args = [(cfg, run_stages) for cfg in self.configs]

        if parallel:
            nworkers = workers if workers > 0 else multiprocessing.cpu_count()
            nworkers = min(nworkers, len(self.configs))
            print(
                f"[batch] Running {len(self.configs)} halo(s) "
                f"with {nworkers} worker(s) …"
            )
            with multiprocessing.Pool(processes=nworkers) as pool:
                results = pool.map(_run_halo, args)
        else:
            results = [_run_halo(a) for a in args]

        return self._report(results)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        sim: str,
        nmax: int | list[int],
        lmax: int | list[int],
        snapshots: list[int] | str = "all",
        halo_id_column: str = "halo_id",
        data_root: Optional[str | Path] = None,
        **config_kwargs,
    ) -> "BatchRunner":
        """
        Build a BatchRunner from a CSV file that contains halo IDs.

        Parameters
        ----------
        csv_path : path-like
            CSV file.  Must contain a column named *halo_id_column*.
        sim : str
            Simulation name (e.g. ``"tng35-3-dark"``).
        nmax, lmax : int | list[int]
            Basis order. Can be scalar values or same-length paired lists.
        snapshots : list[int] | "all"
            Snapshot list shared by every halo (or ``"all"`` to discover).
        halo_id_column : str
            Name of the column holding halo IDs (default ``"halo_id"``).
        data_root : path-like | None
            Passed to PipelineConfig.  Defaults to ``$ILLUSTRIS_BFE/data``.
        **config_kwargs
            Extra keyword arguments forwarded to ``PipelineConfig``.

        Returns
        -------
        BatchRunner
        """
        import csv

        csv_path = Path(csv_path)
        halo_ids: list[int] = []
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if halo_id_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Column '{halo_id_column}' not found in {csv_path}. "
                    f"Available columns: {reader.fieldnames}"
                )
            for row in reader:
                halo_ids.append(int(row[halo_id_column]))

        configs = [
            PipelineConfig(
                sim=sim,
                halo_id=hid,
                nmax=nmax,
                lmax=lmax,
                snapshots=snapshots,
                data_root=data_root,
                **config_kwargs,
            )
            for hid in halo_ids
        ]
        return cls(configs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _report(results: list[tuple[int, bool, str]]) -> dict[int, bool]:
        successes = [r for r in results if r[1]]
        failures = [r for r in results if not r[1]]

        print(f"\n{'='*60}")
        print(f"  Batch complete: {len(successes)} OK, {len(failures)} FAILED")
        if failures:
            print(f"{'='*60}")
            for halo_id, _, tb in failures:
                print(f"\n  --- halo {halo_id} FAILED ---")
                print(tb)
        print(f"{'='*60}\n")

        return {halo_id: ok for halo_id, ok, _ in results}
