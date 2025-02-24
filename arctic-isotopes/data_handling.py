import numpy as np
import xarray as xr


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
