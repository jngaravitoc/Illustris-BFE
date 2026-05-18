## Done

- Update notebook style path after repo reorganization (`../src/illustris_bfe.mplstyle`)
- Add basis density comparison in `src/tests/test_basis.py` for `getBasis(1e-2, 1.2, numr=400)`
- Create `src/compute_basis_from_fit.py` (defaults: `nmax=8`, `lmax=2`)
- Create/update `src/compute_coefficients.py` to load basis from YAML and write coefficients to `_coefs_tmp/`
- Add `src/tests/test_coefficients_computation.py` with full reference comparison and smoke mode

## Remaining

- Check particle mass table
- Compute density profile for host particles and host+fuzz
- Plot time-evolving normalized density profile
- Fit density profiles
