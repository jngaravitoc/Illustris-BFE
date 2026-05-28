# Changelog

- Update notebook style path after repo reorganization (`../src/illustris_bfe.mplstyle`)
- Add basis density comparison in `src/tests/test_basis.py` for `getBasis(1e-2, 1.2, numr=400)`
- Create `src/compute_basis_from_fit.py` (defaults: `nmax=8`, `lmax=2`)
- Create/update `src/compute_coefficients.py` to load basis from YAML and write coefficients to `_coefs_tmp/`
- Add `src/tests/test_coefficients_computation.py` with full reference comparison and smoke mode
- Streamline end-to-end tests for halo 21537 — `846f132` (2026-05-19)
- Compute a grid of basis as a function of nmax and lmax for halo 21537
- Evaluate MISE and decide on expansion order — `6633c25` (2026-05-18)
- Implement coeffs variance calculations — `9bec4b8` (2026-05-28); tests `343fc37` (2026-05-28)
- Add `samplesz` parameter throughout the pipeline (`PipelineConfig`, `stage_coefficients`, `compute_coefficients_for_snapshots`) — `7348118` (2026-05-28)
- Fix covariance file naming: output is now `coefcovar.{coefs_stem}.cov.h5`, mirroring the coefficients filename (2026-05-28)
- Fix `writeCoefCovariance` call: was incorrectly passing the full coefs file path instead of the stem (2026-05-28)
- Add dedicated `covariance` test mode to `run_tests.sh` and `test_coefficients_computation.py` (2026-05-28)
