"""Integration test for coefficient computation against reference output."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pyEXP

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src/tests/data"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compute_coefficients import run_compute_coefficients_cli, compute_coefficients_for_snapshots
from exp.data_ios import read_halo_params  


TEST_MODE_ENV = "ILLUSTRIS_BFE_COEFS_TEST_MODE"


def _print_basis_yaml_debug(basis_config: Path) -> None:
    """Print a short debug preview of the basis YAML used by this test."""

    print(f"[debug] basis config path: {basis_config}")
    with open(basis_config, "r", encoding="utf-8") as f:
        first_lines = [next(f, "").rstrip("\n") for _ in range(6)]
    print("[debug] basis config first lines:")
    for line in first_lines:
        print(f"[debug]   {line}")

def _compare_stanzas(coefs_obj, coefs_tests) -> None:
    """Compare pyEXP coefficient stanzas, supporting known method-name variants."""

    compare_fn = getattr(coefs_obj, "CompareStraznas", None)
    if compare_fn is None:
        compare_fn = getattr(coefs_obj, "CompareStanzas", None)
    if compare_fn is None:
        raise AttributeError("pyEXP Coefs object has no stanza-compare method")

    result = compare_fn(coefs_tests)
    if isinstance(result, bool):
        assert result, "Coefficient stanza comparison returned False"


def test_compute_coefficients(nmax: int = 8, lmax: int = 2) -> None:
    """Smoke test for direct API usage of compute_coefficients_for_snapshots."""

    basis_config = DATA_DIR / f"halo_21537_basis_config_{nmax:02d}_{lmax:02d}.yaml"
    halo_params = REPO_ROOT / "data" / "tng35-3-dark" / "halo_21537_params.hdf5"
    data_dir = REPO_ROOT / "data" / "tng35-3-dark"

    assert basis_config.exists(), f"Missing basis config: {basis_config}"
    assert halo_params.exists(), f"Missing halo params file: {halo_params}"
    assert data_dir.exists(), f"Missing data dir: {data_dir}"

    mode = os.environ.get(TEST_MODE_ENV, "full").strip().lower()
    assert mode in {"full", "lite"}, f"{TEST_MODE_ENV} must be 'full' or 'lite'"

    halo_params_data = read_halo_params(str(halo_params))
    snapshots = [int(s) for s in halo_params_data["snap"]]
    if mode == "lite":
        snapshots = snapshots[::10]

    output_name = f"test_smoke_halo_21537_coefficients_{nmax:02d}_{lmax:02d}.h5"
    output_path = REPO_ROOT / "_coefs_tmp" / output_name
    if output_path.exists():
        output_path.unlink()

    coefs_file = compute_coefficients_for_snapshots(
        basis_config_file=str(basis_config),
        halo_params_file=str(halo_params),
        data_dir=str(data_dir),
        snapshots=snapshots,
        nmax=nmax,
        lmax=lmax,
        coefs_filename=output_name,
    )

    assert coefs_file == output_path
    assert output_path.exists(), f"Expected output not found: {output_path}"

    coefs_tests = pyEXP.coefs.Coefs.factory(str(output_path))
    times = coefs_tests.Times()
    assert len(times) >= 1, "Coefficient file was created but contains no snapshots"

    return coefs_tests

def test_coefficients_computation_matches_reference(nmax: int=8, lmax: int=2) -> None:
    tests_dir = Path(__file__).resolve().parent

    obj_coefs_filename =  tests_dir / f"data/halo_21537_coefficients_{nmax:02d}_{lmax:02d}.h5"
    coefs_obj = pyEXP.coefs.Coefs.factory(str(obj_coefs_filename))

    coefs_tests = test_compute_coefficients(nmax, lmax)

    coefs_obj.CompareStanzas(coefs_tests)


if __name__ == "__main__":
    # Optional local shortcut: python test_coefficients_computation.py smoke
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() in {"lite", "full"}:
        os.environ[TEST_MODE_ENV] = sys.argv[1].strip().lower()

    test_coefficients_computation_matches_reference()
    print("Coefficient comparison test passed.")
