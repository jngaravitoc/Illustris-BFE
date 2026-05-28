import os
import yaml
import pyEXP
from makemodel import make_model
from pathlib import Path


def make_basis_config(basis_id, parameters):
    """
    Generate a YAML configuration string for a pyEXP basis from arbitrary parameters.

    Parameters
    ----------
    basis_id : str
        Basis identifier (e.g. "sphereSL").
    parameters : dict
        Dictionary of basis parameters.

    Returns
    -------
    config : str
        YAML-formatted configuration string.
    """
    config_dict = {
        "id": basis_id,
        "parameters": parameters,
    }

    return yaml.dump(
        config_dict,
        default_flow_style=False,
        sort_keys=False,
    )

def make_basis(R, D, Mtotal, model_output, basis_params, basis_filename=None):
    """
    Construct a basis from a given radial density profile.

    Parameters
    ----------
    R : array_like
        Radial grid points (e.g., radii at which density `D` is defined).
    D : array_like
        Density values corresponding to each radius in `R`.
    Mtotal : float, optional
        Total mass normalization (default is 1.0).
    basis_params : dict 
        basis parameters e.g., basis_id, nmax, lmax

    Returns
    -------
    basis : pyEXP.basis.Basis
        A basis object initialized with the given density model.

    Notes
    -----
    - This function wraps `makemodel.makemodel` to generate a model from 
      the supplied density profile and total mass.
    - It then builds a basis either spherical (`sphereSL`) or cylindrical using `EXPtools.make_config`
      and returns the corresponding `pyEXP` basis object.
    """

    modelname = basis_params["modelname"]
    output_filename = (
        modelname if os.path.isabs(modelname) else os.path.join(model_output, modelname)
    )

    R, D, _, _ = make_model(
        R,
        D,
        Mtotal=Mtotal,
        output_filename=output_filename,
    )
    basis_params = dict(basis_params)
    basis_id = basis_params["basis_id"]
    basis_params.pop("basis_id")
    config = make_basis_config(
        basis_id = basis_id,
        parameters = basis_params,
    )
    
    if basis_filename:
        with open(basis_filename, "w", encoding="utf-8") as f:
            f.write(config)
    return config


def compute_basis(basis_params, r_basis, rho_basis, basis_path, basis_filename):
    """
    Compute and initialize a pyEXP basis object from a fitted density profile.

    Parameters
    ----------
    basis_params : dict
        Dictionary containing basis configuration and fit parameters. Must include:
            - 'rmin', 'rmax', 'nbins', 'lmax', 'nmax', 'rmapping', 'cachename', 'modelname'
            - Any additional parameters required by fit_density_profile and make_basis
    basis_path : str
        Directory where basis files and cache will be stored.
    basis_filename : str
        Name of the YAML file to write the basis configuration.

    Returns
    -------
    basis : pyEXP.basis.Basis
        Initialized basis object.

    Notes
    -----
    - Assumes fit_density_profile and fit_params are defined elsewhere.
    - Changes working directory to basis_path during basis construction.
    - Restores original working directory before returning.
    """

    # Type enforcement and defaults
    required_floats = ['rmin', 'rmax', 'rmapping', 'Mtotal']
    required_ints = ['nbins', 'lmax', 'nmax']
    required_strs = ['cachename', 'modelname']

    # Set default for basis_id
    basis_id = basis_params.get('basis_id', 'sphereSL')
    if not isinstance(basis_id, str):
        raise TypeError("basis_id must be a string")
    basis_params['basis_id'] = basis_id

    for key in required_floats:
        if key not in basis_params:
            raise KeyError(f"Missing required float parameter: {key}")
        basis_params[key] = float(basis_params[key])

    for key in required_ints:
        if key not in basis_params:
            raise KeyError(f"Missing required int parameter: {key}")
        basis_params[key] = int(basis_params[key])

    for key in required_strs:
        if key not in basis_params:
            raise KeyError(f"Missing required string parameter: {key}")
        basis_params[key] = str(basis_params[key])

    rmin = basis_params['rmin']
    rmax = basis_params['rmax']
    nbins_basis = basis_params['nbins']
    lmax = basis_params['lmax']
    nmax = basis_params['nmax']
    rmapping = basis_params['rmapping']
    cachename = basis_params['cachename']
    modelname = basis_params['modelname']

    #r_basis = np.linspace(rmin, rmax, nbins_basis)
    #rho_fit = fit_density_profile(r_basis, *fit_params)  # fit_density_profile and fit_params must be defined

    # Use only basenames in the YAML so pyEXP resolves them relative to the
    # working directory (basis_path).  Absolute paths in the YAML can cause
    # pyEXP to hang on first use when no cache exists.
    known_keys = {
        'rmin', 'rmax', 'nbins', 'lmax', 'nmax', 'rmapping',
        'cachename', 'modelname', 'basis_id', 'Mtotal',
    }
    extra_params = {k: v for k, v in basis_params.items() if k not in known_keys}

    basis_config = {
        "basis_id": basis_params.get("basis_id", "sphereSL"),
        "numr": nbins_basis,
        "rmin": rmin,
        "rmax": rmax,
        "Lmax": lmax,
        "nmax": nmax,
        "rmapping": rmapping,
        "modelname": os.path.basename(modelname),
        "cachename": os.path.basename(cachename),
        **extra_params,
    }

    # Use the provided total mass normalization.
    Mtotal = basis_params.get("Mtotal", 1.0)

    bconfig = make_basis(
        r_basis,
        rho_basis,
        Mtotal=Mtotal,
        model_output=basis_path,
        basis_params=basis_config,
        basis_filename=os.path.join(basis_path, basis_filename)
    )

    cwd_path = os.getcwd()
    os.chdir(basis_path)
    basis = pyEXP.basis.Basis.factory(bconfig)
    os.chdir(cwd_path)
    return basis

def _validate_cache(cache_path: Path, params: dict) -> None:
    """
    Validate that the cache file is consistent with the YAML config parameters.

    Raises
    ------
    ValueError
        If any parameter stored in the cache does not match the YAML config.
    FileNotFoundError
        If the cache file does not exist.
    """
    import h5py

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            "Run build_basis() to create it."
        )

    checks = {
        "model":    ("modelname", str,   lambda a, b: Path(a).name == Path(b).name),
        "nmax":     ("nmax",      int,   lambda a, b: int(a) == int(b)),
        "lmax":     ("lmax",      int,   lambda a, b: int(a) == int(b)),
        "numr":     ("numr",      int,   lambda a, b: int(a) == int(b)),
        "rmapping": ("rmapping",  float, lambda a, b: abs(float(a) - float(b)) < 1e-10),
        "rmin":     ("rmin",      float, lambda a, b: abs(float(a) - float(b)) < 1e-6),
        "rmax":     ("rmax",      float, lambda a, b: abs(float(a) - float(b)) < 1e-6),
    }

    with h5py.File(cache_path, "r") as f:
        cache_attrs = dict(f.attrs)

    mismatches = []
    for cache_key, (yaml_key, _, eq) in checks.items():
        if cache_key not in cache_attrs or yaml_key not in params:
            continue
        cache_val = cache_attrs[cache_key]
        yaml_val  = params[yaml_key]
        if not eq(cache_val, yaml_val):
            mismatches.append(
                f"  {yaml_key}: YAML={yaml_val!r}  vs  cache={cache_val!r}"
            )

    if mismatches:
        raise ValueError(
            f"Cache file '{cache_path.name}' does not match the YAML config:\n"
            + "\n".join(mismatches)
            + "\nDelete or regenerate the cache with build_basis()."
        )


def load_basis(basis_config_file: str) -> object:
    """
    Load a basis from a YAML config file.

    Validates the cache against the config before calling pyEXP to avoid a
    silent, slow recomputation caused by a stale or mismatched cache.

    Parameters
    ----------
    basis_config_file : str
        Path to basis config YAML file.

    Returns
    -------
    basis : pyEXP.basis.Basis
        The loaded basis object.
    """
    import yaml as _yaml

    config_path = Path(basis_config_file).resolve()
    print(f"Loading basis from {config_path}...")

    with open(config_path, "r", encoding="utf-8") as f:
        basis_yaml = f.read()

    cfg = _yaml.safe_load(basis_yaml)
    params = cfg.get("parameters", {})

    # Rewrite any absolute paths to bare filenames so pyEXP resolves them
    # relative to the working directory set via os.chdir below.
    for key in ("modelname", "cachename"):
        if key in params and os.path.isabs(str(params[key])):
            params[key] = os.path.basename(params[key])
    basis_yaml = _yaml.dump(cfg, default_flow_style=False, sort_keys=False)

    # Validate cache before handing off to pyEXP (avoids silent recomputation).
    cache_name = params.get("cachename", "")
    cache_path = config_path.parent / cache_name
    _validate_cache(cache_path, params)

    # Resolve model/cache filenames from the YAML's directory.
    cwd = Path.cwd()
    try:
        os.chdir(config_path.parent)
        basis = pyEXP.basis.Basis.factory(basis_yaml)
    finally:
        os.chdir(cwd)
    print(f"  Basis type: {type(basis).__name__}")
    return basis
