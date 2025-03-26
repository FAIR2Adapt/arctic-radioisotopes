# Test script

# Install xarray-healpy and dggs libraries for regridding
# pip install git+https://github.com/IAOCEA/xarray-healpy.git
# pip install git+https://github.com/xarray-contrib/xdggs.git

from pathlib import Path
import numpy as np
import xarray as xr  # N-dimensional arrays with dimension, coordinate and attribute labels
import warnings
import s3fs

warnings.simplefilter("ignore", category=DeprecationWarning)
xr.set_options(display_expand_data=False, display_expand_attrs=False, keep_attrs=True)

# Local imports
from data_handling import load_grid_vertex, standardize_variable_names, regrid_to_dggs, get_grid_info

def regrid():
    # Load platecarre dr 
    
    # Create the S3FileSystem with a custom endpoint
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={
            "endpoint_url": "https://server-data.fair2adapt.sigma2.no"
        }
    )

    # unregister handler to make boto3 work with CEPH
    handlers = fs.s3.meta.events._emitter._handlers
    handlers_to_unregister = handlers.prefix_search("before-parameter-build.s3")
    handler_to_unregister = handlers_to_unregister[0]
    fs.s3.meta.events._emitter.unregister(
        "before-parameter-build.s3", handler_to_unregister
    )

    fs.ls("fair2adapt")
    s3path = 's3://fair2adapt/CS1-nird/data/model/JRAOC20TRNRPv2_hm_sst_2010-01_con.nc'
    remote_files = fs.glob(s3path)
    # Iterate through remote_files to create a fileset
    fileset = [fs.open(file) for file in remote_files]

    # This works
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    dr = xr.open_mfdataset(fileset, combine='by_coords', decode_times=time_coder)

    # data_path = Path("./CS1-nird/data/")
    # tripolar_grid_data_path = data_path / "model" / "JRAOC20TRNRPv2_hm_sst_2010-01.nc"
    # ds = xr.open_dataset(tripolar_grid_data_path)

    s3gridpath = 's3://fair2adapt/CS1-nird/data/grid/grid.nc'
    remote_grid_files = fs.glob(s3gridpath)
    # Iterate through remote_files to create a fileset
    grid_fileset = [fs.open(file) for file in remote_grid_files]
    
    # # Open grid 
    grid = xr.open_mfdataset(grid_fileset, combine='by_coords', decode_times=time_coder)
    plat, plon, pclat, pclon = get_grid_info(grid)

    # ds = ds.assign_coords(lat=(["y", "x"], plat), lon=(["y", "x"], plon))

    # ds = standardize_variable_names(ds)

    # ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180

    # Regridded to PlateCarree dr
    # bilinear_regridded_data_path = (
    #     data_path / "model" / "JRAOC20TRNRPv2_hm_sst_2010-01_1x1d.nc"
    # )
    # dr = xr.open_dataset(bilinear_regridded_data_path)

    dr = dr.swap_dims({"lat": "latitude", "lon": "longitude"}) # dimensions
    dr.latitude.attrs["standard_name"] = "latitude"
    dr.longitude.attrs["standard_name"] = "longitude"
    dr[["longitude", "latitude"]].compute()
    dr = dr.rename_vars({"lon": "longitude", "lat": "latitude"}) # variables

    ocean_mask = ~dr.sst.isel(time=0).isnull()  # Mask land as False, ocean as True

    nside = 32  # 256  # Each side of the original 12 faces in Healpix is divided into nside parts
    healpy_grid_level = int(np.log2(nside))  # Healpix level
    number_of_cells = 12 * nside**2  # The resulting total number of cells

    min_vertices = (
        1  # Minimum number of vertices for a valid transcription for regridding.
    )
    # 1 is the most liberal, meaning that only one is needed

    # Perform the actual regridding
    regridded = regrid_to_dggs(
        dr, nside, min_vertices, method="bilinear", mask=ocean_mask
    )

    ds_regridded = regridded.sst.compute().squeeze()

    #ds_regridded.dggs.explore()

    ds_regridded.latitude.attrs["standard_name"] = "latitude"
    ds_regridded.longitude.attrs["standard_name"] = "longitude"

    #save_location = Path("./") / f"SST-healpix-lvl-{healpy_grid_level}.zarr"
    #ds_regridded.to_zarr(save_location, mode="w")
    
    return ds_regridded

if __name__ == "__main__":
    regrid()

def test_regrid():
    # End-to-end test that regridding does not crash
    ds_regridded = regrid()

    # Standarized names
    assert ds_regridded.latitude.attrs["standard_name"] == "latitude"
    assert ds_regridded.longitude.attrs["standard_name"] == "longitude"
