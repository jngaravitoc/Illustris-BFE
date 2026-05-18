"""Integration test for coefficient computation against reference output."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pyEXP

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compute_coefficients import run_compute_coefficients_cli  # noqa: E402
from exp.data_ios import read_halo_params  # noqa: E402


TEST_MODE_ENV = "ILLUSTRIS_BFE_COEFS_TEST_MODE"


def _print_basis_yaml_debug(basis_config: Path) -> None:
    """Print a short debug preview of the basis YAML used by this test."""

    print(f"[debug] basis config path: {basis_config}")
    with open(basis_config, "r", encoding="utf-8") as f:
        first_lines = [next(f, "").rstrip("\n") for _ in range(6)]
    print("[debug] basis config first lines:")
    for line in first_lines:
        print(f"[debug]   {line}")


def _get_reference_coefficients_path() -> Path:
    """Return the existing reference coefficients path in src/tests."""

    tests_dir = Path(__file__).resolve().parent
    candidates = [
        tests_dir / "halo_21537_coefficients_16_08.h5",
        tests_dir / "halo_21537_coefficints_16_08.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Reference coefficients file not found in src/tests")


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


def test_coefficients_computation_matches_reference() -> None:
    tests_dir = Path(__file__).resolve().parent

    basis_config = tests_dir / "_basis_tmp" / "halo_21537_basis_config_16_08.yaml"
    halo_params = REPO_ROOT / "data" / "tng35-3-dark" / "halo_21537_params.hdf5"
    data_dir = REPO_ROOT / "data" / "tng35-3-dark"

    assert basis_config.exists(), f"Missing basis config: {basis_config}"
    assert halo_params.exists(), f"Missing halo params file: {halo_params}"
    assert data_dir.exists(), f"Missing data dir: {data_dir}"

    _print_basis_yaml_debug(basis_config)

    reference_coefs = _get_reference_coefficients_path()

    mode = os.environ.get(TEST_MODE_ENV, "full").strip().lower()
    assert mode in {"full", "smoke"}, f"{TEST_MODE_ENV} must be 'full' or 'smoke'"

    # Use the same snap list as halo_params so reference and test outputs match in coverage.
    halo_params_data = read_halo_params(str(halo_params))
    snaps = halo_params_data["snap"]
    if mode == "smoke":
        snaps = snaps[:10]

    snap_args = [str(int(s)) for s in snaps]

    output_name = "test_halo_21537_coefficients_16_08.h5"
    output_path = REPO_ROOT / "_coefs_tmp" / output_name
    if output_path.exists():
        output_path.unlink()

    rc = run_compute_coefficients_cli(
        [
            str(basis_config),
            str(halo_params),
            str(data_dir),
            "--nmax",
            "16",
            "--lmax",
            "8",
            "--coefs-filename",
            output_name,
            "--snapshots",
            *snap_args,
        ]
    )
    assert rc == 0, "compute_coefficients.py returned a non-zero exit code"
    assert output_path.exists(), f"Expected output not found: {output_path}"

    coefs_tests = pyEXP.coefs.Coefs.factory(str(output_path))

    if mode == "smoke":
        times = coefs_tests.Times()
        assert len(times) >= 1, "Smoke run produced empty coefficients"
        return

    coefs_obj = pyEXP.coefs.Coefs.factory(str(reference_coefs))

    _compare_stanzas(coefs_obj, coefs_tests)


if __name__ == "__main__":
    # Optional local shortcut: python test_coefficients_computation.py smoke
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() in {"smoke", "full"}:
        os.environ[TEST_MODE_ENV] = sys.argv[1].strip().lower()

    test_coefficients_computation_matches_reference()
    print("Coefficient comparison test passed.")
