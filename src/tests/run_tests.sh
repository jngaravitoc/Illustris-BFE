#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN_DEFAULT="/home/ngc/Work/research/codes/environments/pyexp310/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

MODE="${1:-lite}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [mode]

Modes:
  lite        Run basis test + coefficient lite test (default)
  full        Run basis test + full coefficient comparison test
  basis       Run only basis test
  coeff-lite  Run only coefficient lite test
  coeff-full  Run only full coefficient comparison test
  profile     Run only exo/exp density profile test

Env vars:
  PYTHON_BIN  Override Python interpreter (default: $PYTHON_BIN_DEFAULT)

Examples:
  $(basename "$0")
  $(basename "$0") full
  PYTHON_BIN=python3 $(basename "$0") basis
EOF
}

run_basis_test() {
  echo "[run] basis test"
  "$PYTHON_BIN" "$REPO_ROOT/src/tests/test_basis.py"
}

run_coeff_lite() {
  echo "[run] coefficients lite test"
  (
    cd "$REPO_ROOT/src/tests"
    ILLUSTRIS_BFE_COEFS_TEST_MODE=lite "$PYTHON_BIN" test_coefficients_computation.py
  )
}

run_coeff_full() {
  echo "[run] coefficients full test"
  (
    cd "$REPO_ROOT/src/tests"
    "$PYTHON_BIN" test_coefficients_computation.py
  )
}

run_exo_density_profile_test() {
  echo "[run] exo/exp density profile test"
  (
    cd "$REPO_ROOT/src/tests"
    if [[ -f test_exo_density_profile.py ]]; then
      "$PYTHON_BIN" test_exo_density_profile.py
    else
      "$PYTHON_BIN" test_exp_density_profiles.py
    fi
  )
}

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODE" in
  lite)
    run_basis_test
    run_coeff_lite
    ;;
  full)
    run_basis_test
    run_coeff_full
    ;;
  basis)
    run_basis_test
    ;;
  coeff-lite)
    run_coeff_lite
    ;;
  coeff-full)
    run_coeff_full
    ;;
  profile|exp-density)
    run_exo_density_profile_test
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

echo "[done] test run completed"
