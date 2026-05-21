"""
HaloPipeline — single-halo orchestrator.

Runs the five analysis stages in dependency order:
    profiles → basis → coefficients → fields → metrics

Usage
-----
    from analysis.pipeline_config import PipelineConfig
    from analysis.pipeline import HaloPipeline

    config = PipelineConfig.from_yaml("example_config.yaml")
    pipe   = HaloPipeline(config)
    pipe.run_all()
"""

from __future__ import annotations

import time

from analysis.pipeline_config import PipelineConfig
from analysis import (
    stage_profiles,
    stage_basis,
    stage_coefficients,
    stage_fields,
    stage_metrics,
)

# Canonical stage order and the module that implements each stage.
STAGE_ORDER: list[str] = [
    "profiles",
    "basis",
    "coefficients",
    "fields",
    "metrics",
]

_STAGE_MODULES = {
    "profiles": stage_profiles,
    "basis": stage_basis,
    "coefficients": stage_coefficients,
    "fields": stage_fields,
    "metrics": stage_metrics,
}


class HaloPipeline:
    """
    Orchestrator for a single halo's analysis pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Fully populated configuration object.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, dict]:
        """
        Run every stage that is enabled in config.stages, in dependency order.

        Returns
        -------
        dict  mapping stage name → outputs dict returned by the stage.
        """
        enabled = [s for s in STAGE_ORDER if self.config.stages.get(s, False)]
        return self.run_stages(enabled)

    def run_stages(self, names: list[str]) -> dict[str, dict]:
        """
        Run a specific subset of stages in dependency order.

        Stages not present in *names* are skipped (even if enabled in config).
        The subset is sorted into canonical dependency order automatically.

        Parameters
        ----------
        names : list[str]
            Stage names to run, e.g. ``["basis", "coefficients"]``.

        Returns
        -------
        dict  mapping stage name → outputs dict returned by the stage.
        """
        unknown = set(names) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"Unknown stage(s): {unknown}. Valid: {STAGE_ORDER}")

        if self.config.is_multi_order():
            return self._run_multi_order(names)

        ordered = [s for s in STAGE_ORDER if s in names]
        self._print_header(ordered)

        for stage in ordered:
            self._run_one(stage)

        self._print_summary()
        return dict(self._results)

    def _run_multi_order(self, names: list[str]) -> dict[str, dict]:
        """Run the requested stages for every configured (nmax, lmax) pair."""
        aggregated: dict[str, dict] = {}
        print(
            f"\n{'#'*60}\n"
            f"  HaloPipeline — multi-order run for halo {self.config.halo_id} "
            f"({self.config.sim})\n"
            f"  Order pairs: {self.config.order_pairs()}\n"
            f"{'#'*60}"
        )
        for nmax, lmax in self.config.order_pairs():
            pair_cfg = self.config.with_orders(nmax=nmax, lmax=lmax)
            pair_tag = pair_cfg.basis_tag()
            pair_pipe = HaloPipeline(pair_cfg)
            aggregated[pair_tag] = pair_pipe.run_stages(names)
        return aggregated

    def run_stage(self, name: str) -> dict:
        """
        Run a single stage by name and return its outputs dict.

        Parameters
        ----------
        name : str
            One of the stage names in ``STAGE_ORDER``.
        """
        if name not in STAGE_ORDER:
            raise ValueError(f"Unknown stage '{name}'. Valid: {STAGE_ORDER}")
        self._run_one(name)
        return self._results[name]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(self, name: str) -> None:
        """Run one stage, record elapsed time, and store outputs."""
        mod = _STAGE_MODULES[name]
        print(f"\n{'='*60}")
        print(f"  Stage: {name}  |  halo {self.config.halo_id}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        if name == "metrics":
            outputs = mod.run(
                self.config,
                generate_all_snapshot_maps=self.config.metrics_generate_all_snapshot_maps,
            )
        else:
            outputs = mod.run(self.config)
        elapsed = time.perf_counter() - t0
        self._results[name] = {**outputs, "_elapsed_s": elapsed}
        print(f"  [{name}] done in {elapsed:.1f}s")

    def _print_header(self, stages: list[str]) -> None:
        print(
            f"\n{'#'*60}\n"
            f"  HaloPipeline — halo {self.config.halo_id} "
            f"({self.config.sim})\n"
            f"  nmax={self.config.nmax}  lmax={self.config.lmax}  "
            f"snapshots={self.config.snapshots}\n"
            f"  Stages to run: {stages}\n"
            f"{'#'*60}"
        )

    def _print_summary(self) -> None:
        total = sum(v["_elapsed_s"] for v in self._results.values())
        print(f"\n{'#'*60}")
        print(f"  Summary — halo {self.config.halo_id}")
        print(f"{'#'*60}")
        for stage in STAGE_ORDER:
            if stage not in self._results:
                continue
            elapsed = self._results[stage]["_elapsed_s"]
            print(f"  {stage:<15}  {elapsed:6.1f}s")
        print(f"  {'TOTAL':<15}  {total:6.1f}s")
        print(f"{'#'*60}\n")
