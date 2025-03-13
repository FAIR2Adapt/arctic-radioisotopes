# Test script

# Install xarray-healpy and dggs libraries for regridding
#pip install git+https://github.com/IAOCEA/xarray-healpy.git 
#pip install git+https://github.com/xarray-contrib/xdggs.git

from pathlib import Path
import numpy as np
import xarray as xr  # N-dimensional arrays with dimension, coordinate and attribute labels
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
xr.set_options(display_expand_data=False, display_expand_attrs=False, keep_attrs=True)

# Local imports
from data_handling import load_grid_vertex, standardize_variable_names, regrid_to_dggs

def regrid():
    # ds in tripolar

    data_path = Path("./CS1-nird/data/")
    tripolar_grid_data_path = data_path / "model" / "JRAOC20TRNRPv2_hm_sst_2010-01.nc"
    ds = xr.open_dataset(tripolar_grid_data_path)

    # Get grid location information
    grid_file_path = data_path / "grid" / "grid.nc"
    plat, plon, pclat, pclon = load_grid_vertex(grid_file_path)

    ds = ds.assign_coords(lat=(["y", "x"], plat), lon=(["y", "x"], plon))

    ds = standardize_variable_names(ds)

    ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180


    # Regridded to PlateCarree dr 
    bilinear_regridded_data_path = (
        data_path / "model" / "JRAOC20TRNRPv2_hm_sst_2010-01_1x1d.nc"
    )
    dr = xr.open_dataset(bilinear_regridded_data_path)

    dr = dr.rename_dims({"lat": "latitude", "lon": "longitude"})
    dr.latitude.attrs["standard_name"] = "latitude"
    dr.longitude.attrs["standard_name"] = "longitude"
    dr[["longitude", "latitude"]].compute()
    dr = dr.rename({"lon": "longitude", "lat": "latitude"})

    ocean_mask = ~dr.sst.isel(time=0).isnull()  # Mask land as False, ocean as True

    nside = 32  # 256  # Each side of the original 12 faces in Healpix is divided into nside parts
    healpy_grid_level = int(np.log2(nside))  # Healpix level
    number_of_cells = 12 * nside**2  # The resulting total number of cells

    min_vertices = 1  # Minimum number of vertices for a valid transcription for regridding.
    # 1 is the most liberal, meaning that only one is needed

    # Perform the actual regridding
    regridded = regrid_to_dggs(dr, nside, min_vertices, method="bilinear", mask=ocean_mask)

    ds_regridded = regridded.sst.compute().squeeze()

    ds_regridded.dggs.explore()

    save_location = data_path / "SST-healpix-lvl-{healpy_grid_level}.zarr"
    ds_regridded.to_zarr(save_location, mode="w")

    return ds_regridded

def test_regrid():
    # End-to-end test that regridding does not crash
    ds_regridded = regrid()

    # Standarized names
    assert ds_regridded.latitude.attrs["standard_name"] == "latitude"
    assert ds_regridded.longitude.attrs["standard_name"] == "longitude"
