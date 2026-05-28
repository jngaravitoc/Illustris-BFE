"""
Tests for the HaloPipeline orchestrator and BatchRunner.

Strategy
--------
Heavy computation (BFE/KDE fields from particles) is already covered by
test_fields.py.  These tests focus on:

  1. PipelineConfig round-trip serialisation.
  2. Stage skip (idempotency) logic — pre-populate expected outputs,
     re-run stage, verify nothing was recomputed.
  3. stage_metrics end-to-end using pre-computed field fixtures.
  4. HaloPipeline and BatchRunner orchestration with the metrics stage.

All outputs are written to per-test folders under
src/tests/_temp_tests_outputs/test_pipeline, keeping the repository clean.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
TEST_DATA = REPO_ROOT / "src" / "tests" / "data"
REAL_DATA = REPO_ROOT / "data"
TESTS_OUTPUT_DIR = Path(__file__).resolve().parent / "_temp_tests_outputs" / "test_pipeline"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.pipeline_config import PipelineConfig
from analysis.pipeline import HaloPipeline, STAGE_ORDER
from analysis.batch import BatchRunner
from analysis import stage_basis, stage_coefficients, stage_fields, stage_metrics
import run_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NMAX, LMAX = 8, 2
HALO_ID = 21537
SIM = "tng35-3-dark"
SNAPSHOTS = [50, 99]


def _output_root(test_name: str) -> Path:
    """Return a stable per-test output directory under src/tests/_temp_tests_outputs."""
    out = TESTS_OUTPUT_DIR / test_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_config(output_root: Path, snapshots: list[int] = SNAPSHOTS) -> PipelineConfig:
    """Return a PipelineConfig whose data_root is a test output folder."""
    return PipelineConfig(
        sim=SIM,
        halo_id=HALO_ID,
        nmax=NMAX,
        lmax=LMAX,
        snapshots=snapshots,
        data_root=output_root,
        halo_params_file=REAL_DATA / SIM / f"halo_{HALO_ID}" / f"halo_{HALO_ID}_params.hdf5",
        profile_fit_file=TEST_DATA / f"halo_{HALO_ID}_normalized_density_profile_fit.txt",
        stages={s: True for s in STAGE_ORDER},
    )


def _make_multi_order_config(
    output_root: Path,
    nmax_list: list[int],
    lmax_list: list[int],
    snapshots: list[int] = SNAPSHOTS,
) -> PipelineConfig:
    """Return a list-order PipelineConfig whose pairs are zipped by index."""
    return PipelineConfig(
        sim=SIM,
        halo_id=HALO_ID,
        nmax=nmax_list,
        lmax=lmax_list,
        snapshots=snapshots,
        data_root=output_root,
        halo_params_file=REAL_DATA / SIM / f"halo_{HALO_ID}" / f"halo_{HALO_ID}_params.hdf5",
        profile_fit_file=TEST_DATA / f"halo_{HALO_ID}_normalized_density_profile_fit.txt",
        stages={s: True for s in STAGE_ORDER},
    )


def _populate_basis(config: PipelineConfig) -> None:
    """Copy test-fixture basis files to the expected location for this test run."""
    dest_dir = config.output_dir("basis")
    dest_dir.mkdir(parents=True, exist_ok=True)

    tag = config.basis_tag()
    for fname in [
        f"halo_{HALO_ID}_basis_config_{tag}.yaml",
        f"halo_{HALO_ID}_basis_cache_{tag}.txt",
        f"halo_{HALO_ID}_model.txt",
    ]:
        src = TEST_DATA / fname
        if src.exists():
            shutil.copy(src, dest_dir / fname)

    # The basis YAML references cache/model files via relative paths; for
    # tests we just need the YAML to exist.
    assert config.basis_config_file().exists()


def _populate_coefficients(config: PipelineConfig) -> None:
    """Copy test-fixture coefficients file to the expected location for this test run."""
    dest_dir = config.output_dir("coefficients")
    dest_dir.mkdir(parents=True, exist_ok=True)

    tag = config.basis_tag()
    src = TEST_DATA / f"halo_{HALO_ID}_coefficients_{tag}.h5"
    shutil.copy(src, config.coefficients_file())
    assert config.coefficients_file().exists()


def _populate_fields(config: PipelineConfig) -> None:
    """Copy test-fixture BFE and KDE field files to the expected locations."""
    tag = config.basis_tag()

    bfe_dir = config.output_dir("fields") / "bfe"
    kde_dir = config.output_dir("fields") / "kde"
    bfe_dir.mkdir(parents=True, exist_ok=True)
    kde_dir.mkdir(parents=True, exist_ok=True)

    for snap in config.snapshots:
        bfe_src = TEST_DATA / f"halo_{HALO_ID}_bfe_density_{tag}_snap_{snap:03d}.h5"
        kde_src = TEST_DATA / f"halo_{HALO_ID}_kde_density_field_snap_{snap:03d}.h5"
        assert bfe_src.exists(), f"Missing BFE fixture: {bfe_src}"
        assert kde_src.exists(), f"Missing KDE fixture: {kde_src}"
        shutil.copy(bfe_src, config.bfe_fields_file(snap))
        shutil.copy(kde_src, config.kde_fields_file(snap))


def _populate_time_evol(config: PipelineConfig) -> None:
    """Copy the time-evolution file needed by stage_metrics."""
    src = REAL_DATA / SIM / f"halo_{HALO_ID}" / f"{SIM}_halo_time_evol.txt"
    dest = config.halo_dir()
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest / f"{SIM}_halo_time_evol.txt")


def _populate_fields_for_all_orders(config: PipelineConfig) -> None:
    """Copy BFE/KDE fixtures for every configured (nmax, lmax) pair."""
    for nmax, lmax in config.order_pairs():
        _populate_fields(config.with_orders(nmax=nmax, lmax=lmax))


# ---------------------------------------------------------------------------
# 1. Config round-trip
# ---------------------------------------------------------------------------

def test_config_roundtrip() -> None:
    """PipelineConfig serialises to YAML and deserialises identically."""
    src_yaml = REPO_ROOT / "src" / "analysis" / "example_config.yaml"
    config_a = PipelineConfig.from_yaml(src_yaml)

    out_yaml = _output_root("config_roundtrip") / "config_rt.yaml"
    config_a.to_yaml(out_yaml)
    config_b = PipelineConfig.from_yaml(out_yaml)

    assert config_b.sim == config_a.sim
    assert config_b.halo_id == config_a.halo_id
    assert config_b.nmax == config_a.nmax
    assert config_b.lmax == config_a.lmax
    assert config_b.snapshots == config_a.snapshots
    assert config_b.grid_bins == config_a.grid_bins
    assert config_b.grid_range == config_a.grid_range
    assert config_b.stages == config_a.stages
    assert config_b.covariance == config_a.covariance


# ---------------------------------------------------------------------------
# 2. CLI dry-run
# ---------------------------------------------------------------------------

def test_dry_run_cli() -> None:
    """run_pipeline --dry-run exits 0 and prints resolved paths."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_pipeline.main([
            str(REPO_ROOT / "src" / "analysis" / "example_config.yaml"),
            "--stages", "basis", "coefficients",
            "--dry-run",
        ])

    out = buf.getvalue()
    assert rc == 0
    assert "DRY RUN" in out
    assert "basis_config_file" in out
    assert "coefficients_file" in out


# ---------------------------------------------------------------------------
# 3. Stage skip (idempotency) tests
# ---------------------------------------------------------------------------

def test_stage_basis_skip() -> None:
    """stage_basis skips computation when its output already exists."""
    config = _make_config(_output_root("stage_basis_skip"))
    _populate_basis(config)

    mtime_before = config.basis_config_file().stat().st_mtime
    result = stage_basis.run(config)
    mtime_after = config.basis_config_file().stat().st_mtime

    assert result["basis_config_file"] == config.basis_config_file()
    assert mtime_after == mtime_before, "basis file was unexpectedly re-written"


def test_stage_coefficients_skip() -> None:
    """stage_coefficients skips computation when its output already exists."""
    config = _make_config(_output_root("stage_coefficients_skip"))
    _populate_coefficients(config)

    mtime_before = config.coefficients_file().stat().st_mtime
    result = stage_coefficients.run(config)
    mtime_after = config.coefficients_file().stat().st_mtime

    assert result["coefficients_file"] == config.coefficients_file()
    assert mtime_after == mtime_before, "coefficients file was unexpectedly re-written"


def test_stage_fields_all_skip() -> None:
    """stage_fields skips all snapshots when all output files already exist."""
    config = _make_config(_output_root("stage_fields_all_skip"))
    _populate_basis(config)
    _populate_coefficients(config)
    _populate_fields(config)

    mtimes_before = {
        snap: config.bfe_fields_file(snap).stat().st_mtime
        for snap in config.snapshots
    }

    result = stage_fields.run(config)

    # All files should be returned and none should have been rewritten.
    for snap in config.snapshots:
        assert snap in result["bfe_files"]
        assert snap in result["kde_files"]
        assert config.bfe_fields_file(snap).stat().st_mtime == mtimes_before[snap], (
            f"BFE field for snap {snap} was unexpectedly re-written"
        )


# ---------------------------------------------------------------------------
# 4. stage_metrics end-to-end with fixtures
# ---------------------------------------------------------------------------

def test_stage_metrics_creates_hdf5() -> None:
    """stage_metrics reads BFE/KDE fixtures and writes a valid metrics HDF5."""
    config = _make_config(_output_root("stage_metrics_creates_hdf5"))
    _populate_fields(config)
    _populate_time_evol(config)

    result = stage_metrics.run(config)

    metrics_file = result["metrics_file"]
    assert metrics_file.exists()

    with h5py.File(metrics_file, "r") as f:
        for key in ("snapshots", "redshift", "mise", "mirse"):
            assert key in f, f"Missing dataset '{key}' in metrics HDF5"
        snaps = np.asarray(f["snapshots"], dtype=int)
        assert set(snaps).issubset(set(config.snapshots))


def test_stage_metrics_idempotency() -> None:
    """Running stage_metrics twice does not rewrite the metrics HDF5."""
    config = _make_config(_output_root("stage_metrics_idempotency"))
    _populate_fields(config)
    _populate_time_evol(config)

    stage_metrics.run(config)
    mtime_first = config.metrics_file().stat().st_mtime

    stage_metrics.run(config)
    mtime_second = config.metrics_file().stat().st_mtime

    assert mtime_second == mtime_first, "metrics file was unexpectedly re-written on second run"


# ---------------------------------------------------------------------------
# 5. HaloPipeline orchestration
# ---------------------------------------------------------------------------

def test_pipeline_single_stage() -> None:
    """HaloPipeline.run_stages([metrics]) produces the metrics file."""
    config = _make_config(_output_root("pipeline_single_stage"))
    _populate_fields(config)
    _populate_time_evol(config)

    pipe = HaloPipeline(config)
    outputs = pipe.run_stages(["metrics"])

    assert "metrics" in outputs
    assert outputs["metrics"]["metrics_file"].exists()
    assert outputs["metrics"]["_elapsed_s"] >= 0.0


def test_pipeline_run_all_with_skip() -> None:
    """HaloPipeline.run_all() skips stages whose outputs already exist."""
    config = _make_config(_output_root("pipeline_run_all_with_skip"))

    # Pre-populate everything so all compute stages skip.
    _populate_basis(config)
    _populate_coefficients(config)
    _populate_fields(config)
    _populate_time_evol(config)

    # stage_profiles will raise because no profiles file exists.
    # Only run from basis onwards to keep the test self-contained.
    pipe = HaloPipeline(config)
    outputs = pipe.run_stages(["basis", "coefficients", "fields", "metrics"])

    for stage in ["basis", "coefficients", "fields", "metrics"]:
        assert stage in outputs


def test_pipeline_unknown_stage_raises() -> None:
    """HaloPipeline.run_stages raises ValueError for unknown stage names."""
    config = PipelineConfig(
        sim=SIM, halo_id=HALO_ID, nmax=NMAX, lmax=LMAX,
        snapshots=SNAPSHOTS,
    )
    pipe = HaloPipeline(config)
    with pytest.raises(ValueError, match="Unknown stage"):
        pipe.run_stages(["nonexistent"])


def test_pipeline_stage_order_enforced() -> None:
    """HaloPipeline runs requested stages in canonical order regardless of input order."""
    config = _make_config(_output_root("pipeline_stage_order_enforced"))
    _populate_fields(config)
    _populate_time_evol(config)

    pipe = HaloPipeline(config)
    # Pass stages in reverse order — should still run in canonical order.
    outputs = pipe.run_stages(["metrics", "fields"])
    assert list(outputs.keys()) == ["fields", "metrics"]


def test_pipeline_multi_order_runs_pairwise() -> None:
    """HaloPipeline runs all configured (nmax, lmax) pairs and writes metrics for each."""
    config = _make_multi_order_config(
        _output_root("pipeline_multi_order_runs_pairwise"),
        nmax_list=[8, 16],
        lmax_list=[2, 8],
    )
    _populate_fields_for_all_orders(config)
    _populate_time_evol(config)

    pipe = HaloPipeline(config)
    outputs = pipe.run_stages(["metrics"])

    assert set(outputs.keys()) == {"08_02", "16_08"}
    assert outputs["08_02"]["metrics"]["metrics_file"].exists()
    assert outputs["16_08"]["metrics"]["metrics_file"].exists()


def test_pipeline_config_multi_order_length_mismatch_raises() -> None:
    """PipelineConfig rejects nmax/lmax lists when their lengths differ."""
    with pytest.raises(ValueError, match="same number of elements"):
        _make_multi_order_config(
            _output_root("pipeline_config_multi_order_length_mismatch_raises"),
            nmax_list=[8, 16],
            lmax_list=[2],
        )


# ---------------------------------------------------------------------------
# 6. BatchRunner
# ---------------------------------------------------------------------------

def test_batch_runner_single_halo() -> None:
    """BatchRunner with one halo returns {halo_id: True} on success."""
    config = _make_config(_output_root("batch_runner_single_halo"))
    _populate_fields(config)
    _populate_time_evol(config)

    runner = BatchRunner([config])
    results = runner.run_all(stages=["metrics"])

    assert results == {HALO_ID: True}


def test_batch_runner_reports_failure() -> None:
    """BatchRunner captures exceptions and marks halo as failed (not raised)."""
    # Config with no data at all — stage_metrics will fail because field files
    # are missing.
    config = _make_config(_output_root("batch_runner_reports_failure"))

    runner = BatchRunner([config])
    results = runner.run_all(stages=["metrics"])

    assert results == {HALO_ID: False}


def test_batch_runner_empty_raises() -> None:
    """BatchRunner raises ValueError when constructed with an empty list."""
    with pytest.raises(ValueError, match="empty"):
        BatchRunner([])


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
