#!/usr/bin/env python3
"""
HDF5 I/O utilities for BFE field dictionaries.
"""

import glob
import os

import numpy as np
import h5py


def write_fields(points, filename, grid=None, field_shape=None):
    """
    Write a ``fields.points`` dictionary to an HDF5 file.

    The dictionary is expected to have the structure returned by
    ``pyEXP.field.FieldGenerator.points``::

        {
            time_0: {field_name: ndarray, ...},
            time_1: {field_name: ndarray, ...},
            ...
        }

    Each time snapshot is stored as a separate HDF5 group named after its
    time value, and every field array is stored as a dataset inside that
    group.

    Parameters
    ----------
    points : dict
        Nested dictionary ``{time: {field_name: ndarray, ...}, ...}``.
    filename : str
        Path to the output HDF5 file.
    grid : array-like, optional
        The grid used to compute the fields (e.g. one array from
        ``np.meshgrid``).  If provided, its shape is stored in the file
        header attribute ``grid_shape``.
    field_shape : tuple of int, optional
        Explicit shape to record in the file header attribute
        ``field_shape``.  Ignored when *grid* is also given.
    """
    with h5py.File(filename, "w") as f:
        # Store grid / field shape metadata in root attributes
        if grid is not None:
            f.attrs["grid_shape"] = np.asarray(grid).shape
        elif field_shape is not None:
            f.attrs["field_shape"] = np.asarray(field_shape)

        for time, fields_dict in points.items():
            grp = f.create_group(str(time))
            for field_name, data in fields_dict.items():
                grp.create_dataset(
                    field_name,
                    data=np.asarray(data),
                    compression="gzip",
                    compression_opts=4,
                )
    print(f"Fields written to {filename}")


def read_bfe_fields(filename, field, time):
    """
    Read a single field array from an HDF5 file written by :func:`write_fields`.

    If the file contains a ``grid_shape`` or ``field_shape`` attribute the
    returned array is automatically reshaped to that shape.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    field : str
        Name of the field dataset to read (e.g. ``'dens'``, ``'potl'``).
    time : float or str
        Time snapshot key.  Converted to ``str`` for the HDF5 group lookup.

    Returns
    -------
    data : numpy.ndarray
        The field array, reshaped to the stored grid/field shape when
        available.
    """
    with h5py.File(filename, "r") as f:
        data = np.array(f[str(time)][field])

        if "grid_shape" in f.attrs:
            data = data.reshape(f.attrs["grid_shape"])
        elif "field_shape" in f.attrs:
            data = data.reshape(f.attrs["field_shape"])

    return data


def write_kde_density(kd_dens, filename, grid_shape, snapshot_name, Ndens=64):
    """
    Write a KDE density field to an HDF5 file.

    Parameters
    ----------
    kd_dens : ndarray, shape (nbins, nbins, nbins)
        KDE density array returned by
        :meth:`~field_projections.FieldProjections.kde_density`.
    filename : str
        Path to the output HDF5 file.
    grid_shape : tuple of int
        Shape of the 3-D grid, e.g. ``(nbins, nbins, nbins)``.
    snapshot_name : str
        Identifier of the snapshot used to compute the density field.
    Ndens : int, optional
        Number of nearest neighbours used in the KDE (default 64).
    """
    with h5py.File(filename, "w") as f:
        f.attrs["grid_shape"] = np.asarray(grid_shape)
        f.attrs["snapshot_name"] = snapshot_name
        f.attrs["Ndens"] = Ndens

        f.create_dataset(
            "kde_density",
            data=np.asarray(kd_dens),
            compression="gzip",
            compression_opts=4,
        )
    print(f"KDE density written to {filename}")


def read_kde_density(filename):
    """
    Read a KDE density field from an HDF5 file written by
    :func:`write_kde_density`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    kd_dens : ndarray
        The density array, reshaped to the stored ``grid_shape``.
    attrs : dict
        Header attributes (``grid_shape``, ``snapshot_name``, ``Ndens``).
    """
    with h5py.File(filename, "r") as f:
        data = np.array(f["kde_density"])
        attrs = {key: f.attrs[key] for key in f.attrs}

        if "grid_shape" in attrs:
            data = data.reshape(attrs["grid_shape"])

    return data, attrs


def merge_kde_density_files(pattern, output_filename):
    """Merge per-snapshot KDE density HDF5 files into a single file.

    Each input file is expected to have been written by
    :func:`write_kde_density` and must contain a ``kde_density`` dataset
    and header attributes (``grid_shape``, ``snapshot_name``, ``Ndens``).

    The merged file stores one HDF5 group per snapshot (named after the
    ``snapshot_name`` attribute) containing the density dataset.  Common
    header attributes (``grid_shape``, ``Ndens``) are stored as root-level
    attributes.

    Parameters
    ----------
    pattern : str
        Glob pattern matching the per-snapshot files, e.g.
        ``"output/halo_0100_kde_density_*.h5"``.
    output_filename : str
        Path to the merged output HDF5 file.

    Examples
    --------
    >>> merge_kde_density_files(
    ...     "output/halo_0100_kde_density_*.h5",
    ...     "output/halo_0100_kde_density.h5",
    ... )
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    with h5py.File(output_filename, "w") as fout:
        for i, filepath in enumerate(files):
            data, attrs = read_kde_density(filepath)
            snapshot_name = str(attrs.get("snapshot_name", f"snapshot_{i:03d}"))

            # Write root-level attributes once (from first file)
            if i == 0:
                fout.attrs["grid_shape"] = np.asarray(attrs["grid_shape"])
                fout.attrs["Ndens"] = attrs["Ndens"]
                fout.attrs["n_snapshots"] = len(files)

            grp = fout.create_group(snapshot_name)
            grp.attrs["snapshot_name"] = snapshot_name
            grp.attrs["source_file"] = os.path.basename(filepath)
            grp.create_dataset(
                "kde_density",
                data=data,
                compression="gzip",
                compression_opts=4,
            )

    print(f"Merged {len(files)} KDE density files into {output_filename}")


def read_merged_kde_density(filename, snapshot=None):
    """Read density arrays from a merged KDE density HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the merged HDF5 file created by
        :func:`merge_kde_density_files`.
    snapshot : str or int or None, optional
        If ``None`` (default), return all snapshots as a dict.
        If a string, return the density array for that snapshot group.
        If an int, return the density array for the *n*-th group
        (sorted alphabetically).

    Returns
    -------
    data : ndarray or dict
        If *snapshot* is given, a single density array.  Otherwise a
        dictionary ``{snapshot_name: ndarray, ...}``.
    attrs : dict
        Root-level header attributes.
    """
    with h5py.File(filename, "r") as f:
        attrs = {key: f.attrs[key] for key in f.attrs}
        grid_shape = tuple(attrs.get("grid_shape", ()))
        groups = sorted([k for k in f.keys()])

        if snapshot is not None:
            if isinstance(snapshot, int):
                snapshot = groups[snapshot]
            snapshot = str(snapshot)
            data = np.array(f[snapshot]["kde_density"])
            if grid_shape:
                data = data.reshape(grid_shape)
            return data, attrs

        data = {}
        for name in groups:
            arr = np.array(f[name]["kde_density"])
            if grid_shape:
                arr = arr.reshape(grid_shape)
            data[name] = arr

    return data, attrs
