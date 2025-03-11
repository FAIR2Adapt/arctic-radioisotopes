import numpy as np
import xarray as xr
import xdggs  # Discrete global grid systems in x-array
from xarray_healpy import HealpyGridInfo, HealpyRegridder
# Development versions of xdggs and xarray_healpy can be installed with:
# pip install git+https://github.com/IAOCEA/xarray-healpy.git git+https://github.com/xarray-contrib/xdggs.git


def load_grid_vertex(grid_file: str, grid_type: str = "p"):
    """
    Load vertex data of BLOM ocean grid.

    :param grid_file: Path to the grid file.
    :param grid_type: Grid type (e.g., 'p').

    :return: Tuple of latitude, longitude, and extended vertices (clat, clon).
    """
    with xr.open_dataset(grid_file) as grid:
        lat = grid[grid_type + "lat"].data
        lon = grid[grid_type + "lon"].data
        clat = grid[grid_type + "clat"].data
        clon = grid[grid_type + "clon"].data

    dims = [x + 1 for x in list(lat.shape)]
    clat_new = np.zeros(dims)
    clon_new = np.zeros(dims)
    clat_new[:-1, :-1] = clat[0, :, :]
    clon_new[:-1, :-1] = clon[0, :, :]
    clat_new[0:-1, -1] = clat[2, :, -1]
    clat_new[-1, 0:-1] = clat[2, -1, :]
    clon_new[0:-1, -1] = clon[2, :, -1]
    clon_new[-1, 0:-1] = clon[2, -1, :]
    clat_new[-1, -1] = clat[2, -1, -1]
    clon_new[-1, -1] = clon[2, -1, -1]

    return lat, lon, clat_new, clon_new


def standardize_variable_names(ds: xr.DataArray):
    """
    Standardize the names of longitude and latitude

    :param ds: xarray data

    :return: Data with standardized names for longitude and latitude
    """

    required_vars = {"lat", "lon"}
    missing_vars = required_vars - set(ds.coords) - set(ds.data_vars)

    if missing_vars:
        raise ValueError(f"Missing required coordinate(s): {missing_vars}")

    ds = ds.rename_vars({"lat": "latitude", "lon": "longitude"})
    ds.latitude.attrs["standard_name"] = "latitude"
    ds.longitude.attrs["standard_name"] = "longitude"
    return ds


def center_longitude(ds: object, center: int = 0):
    """
    Shift longitude coordinates of a dataset to either [-180, 180] or [0, 360] range.

    :param ds: Dataset with a 'longitude' coordinate.
    :param center: If True, shift to [-180, 180]; otherwise, to [0, 360].

    :return: Dataset with updated longitude coordinates.
    """

    if not hasattr(ds, "longitude"):
        raise ValueError("Dataset must have a 'longitude' coordinate.")

    if center not in (0, 180):
        raise ValueError("center must be 0 (for [-180, 180]) or 180 (for [0, 360]).")

    if center == 0:
        centered = (ds.longitude + 180) % 360 - 180
    else:  # center == 180
        centered = (ds.longitude - 180) % 360 + 180

    return ds.assign_coords(longitude=centered)


def regrid_to_dggs(
    dr: xr.DataArray,
    nside: int,
    min_vertices: int,
    method: int = "bilinear",
    mask: xr.DataArray = None,
):
    """
    Regrid data from PlateCarree to Healpy DGGS

    :param dr: xarray data on a PlateCarree projection
    :param nside: Each side of the original 12 faces in Healpix is divided into nside parts
    :param method: Regridding method, defaults to bilinear
    :param mask: Mask showing where data is

    :return: Data regridded to Healpy DGGS
    """

    healpy_grid_level = int(np.log2(nside))  # Healpix level

    # Define the target Healpix grid
    grid = HealpyGridInfo(level=healpy_grid_level)

    target_grid = (
        grid.target_grid(dr)
        .pipe(center_longitude, 0)
        .drop_attrs(deep=False)
        .dggs.decode(
            {"grid_name": "healpix", "nside": nside, "indexing_scheme": "nested"}
        )
    )

    # The default mask is no mask
    interpolation_kwargs = (
        {"min_vertices": min_vertices}
        if mask is None
        else {"mask": mask, "min_vertices": min_vertices}
    )

    # Create the regridder: Compute interpolation weights for regridding diff data
    regridder = HealpyRegridder(
        dr[["longitude", "latitude"]].compute(),
        target_grid,
        method=method,
        interpolation_kwargs=interpolation_kwargs,
    )

    return regridder.regrid_ds(dr).pipe(xdggs.decode)
