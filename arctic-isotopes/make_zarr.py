#!/usr/bin/env python3
"""Consolidate the seven NorESM2/BLOM netCDF files into one cloud-optimised Zarr store.

Three things this fixes relative to the source netCDF:

1. **Geolocation.** The netCDF carry no coordinates for the 385x360 model grid -- the
   `lat` variable in them is a 1-D zonal-diagnostic axis, not the grid. The 2-D BLOM
   tripolar `longitude`/`latitude` come from data/blom_grid.npz (fetched from the
   published store; see arctic-isotopes/fetch_blom_grid.py) and are verified bit-identical there.

2. **Chunking.** The source chunks (1, 27, 193, 180) mean rendering a single map costs
   ~16 chunk reads. Chunking as (1, 1, ny, nx) makes one rendered map exactly one chunk.

3. **Metadata.** Only temp and sst carry global attributes; the other five files have
   none. The ACDD block is promoted to dataset level and applied uniformly.

    python arctic-isotopes/make_zarr.py            # all variables
    python arctic-isotopes/make_zarr.py --vars sst # quick test on one
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec

RAW = Path("data/raw")  # override with --raw
GRID = Path("data/blom_grid.npz")
OUT = Path("data/zarr/JRAOC20TRNRPv2_2010-2018.zarr")
PREFIX = "JRAOC20TRNRPv2_hm_"
SUFFIX = "_2010-2018.nc"

VARS = ["sst", "sss", "temp", "saln", "U236GF", "U236SF", "U236LH"]

# Vestigial in every file: no data variable uses these dimensions. Dropping them avoids
# a 1-D `lat` sitting alongside the 2-D `latitude` that actually geolocates the grid.
DROP = ["depth", "depth_bnds", "lat", "region", "section", "bounds"]

# Dataset-level metadata. The per-file ACDD block (present only on temp/sst) describes a
# single variable, so title/summary/id are restated for the consolidated collection while
# the provenance, licensing and citation fields are carried over verbatim.
GLOBAL_ATTRS = {
    "Conventions": "CF-1.8, ACDD-1.3",
    "title": (
        "Monthly ocean temperature, salinity and U-236 tracers 2010-2018 from NorESM2"
    ),
    "summary": (
        "Monthly mean ocean fields from NorESM2 (BLOM-CICE5) ocean-ice simulations "
        "forced by JRA-55 atmospheric data, covering 2010-2018. Includes sea surface "
        "and 3-D temperature and salinity, and three uranium-236 tracers separating "
        "global fallout from the Sellafield and La Hague reprocessing discharges. "
        "Three-dimensional fields are on 53 isopycnal (potential density) layers, on "
        "the native BLOM tripolar curvilinear grid."
    ),
    "id": "JRAOC20TRNRPv2_2010-2018",
    "naming_authority": "no.nersc.fair2adapt",
    "keywords_vocabulary": "GCMD Science Keywords",
    "keywords": (
        "EARTH SCIENCE, OCEANS, OCEAN TEMPERATURE, OCEAN SALINITY, "
        "OCEAN CHEMISTRY, RADIONUCLIDES"
    ),
    "iso_topic_category": "oceans",
    "processing_level": "Monthly averaged model output.",
    "source": "NorESM2 (BLOM-CICE5), JRA-55 forcing",
    "cdm_data_type": "Grid",
    "geospatial_lat_units": "degrees_north",
    "geospatial_lon_units": "degrees_east",
    "time_coverage_start": "2010-01-01",
    "time_coverage_end": "2018-12-31",
    "time_coverage_resolution": "P1M",
}

# CF-compliant variable metadata. The source files carry `standard_name` values
# ("Ocean temperature", "Ocean salinity") that are NOT in the CF standard name table
# (checked against table v94, 5071 names), and none at all on sst/sss/U236*.
#
# NOTE: `temp` is mapped to sea_water_potential_temperature because BLOM is an isopycnic
# model carrying potential temperature. Worth confirming with Yanchun He (NERSC) before
# publication -- if it is conservative temperature the name must change.
#
# The U-236 tracers have no CF equivalent, which is expected for model-specific tracers:
# CF says omit standard_name rather than invent one, so they carry long_name only and are
# described semantically through I-ADOPT in the RO-Crate instead.
VAR_CF = {
    "sst": {
        "standard_name": "sea_surface_temperature",
        "long_name": "Sea surface temperature",
        "units": "degC",
    },
    "sss": {
        "standard_name": "sea_surface_salinity",
        "long_name": "Sea surface salinity",
        "units": "g kg-1",
    },
    "temp": {
        "standard_name": "sea_water_potential_temperature",
        "long_name": "Sea water potential temperature",
        "units": "degC",
    },
    "saln": {
        "standard_name": "sea_water_salinity",
        "long_name": "Sea water salinity",
        "units": "g kg-1",
    },
    "U236GF": {
        "long_name": "U-236 concentration from global fallout",
        "units": "atoms kg-1",
    },
    "U236SF": {
        "long_name": "U-236 concentration from Sellafield discharges",
        "units": "atoms kg-1",
    },
    "U236LH": {
        "long_name": "U-236 concentration from La Hague discharges",
        "units": "atoms kg-1",
    },
}

# Carried over unchanged from the source files' ACDD block.
INHERIT = [
    "data_set_language",
    "project_id",
    "project",
    "project_lead",
    "project_lead_email",
    "funding_agency",
    "principal_investigator",
    "principal_investigator_email",
    "contributor_name",
    "contributor_email",
    "contributor_role",
    "publisher_name",
    "publisher_institution",
    "publisher_email",
    "publisher_type",
    "publisher_url",
    "data_assembly_center",
    "acknowledgement",
    "license",
    "citation",
    "references",
]


def add_time_bounds(ds: xr.Dataset) -> xr.Dataset:
    """Attach CF `time_bnds` spanning each averaging month.

    CF requires bounds whenever cell_methods declares an average over a dimension: the
    timestamps sit mid-month (the 17th here), so without bounds a reader cannot tell what
    interval each mean covers.
    """
    import cftime

    times = ds.time.values
    bounds = []
    for t in times:
        start = cftime.DatetimeNoLeap(t.year, t.month, 1)
        end = (
            cftime.DatetimeNoLeap(t.year + 1, 1, 1)
            if t.month == 12
            else cftime.DatetimeNoLeap(t.year, t.month + 1, 1)
        )
        bounds.append([start, end])

    ds = ds.assign_coords(time_bnds=(("time", "bnds"), np.array(bounds)))
    ds.time.attrs["bounds"] = "time_bnds"
    ds.time.attrs.setdefault("standard_name", "time")
    ds.time.attrs.setdefault("long_name", "Time")
    return ds


def build(var_names: list[str]) -> xr.Dataset:
    grid = np.load(GRID)
    merged, inherited = None, {}

    for v in var_names:
        path = RAW / f"{PREFIX}{v}{SUFFIX}"
        # One dask block per (time, all sigma) keeps memory ~30 MB while staying an exact
        # multiple of the output chunking, so to_zarr never has to rechunk.
        ds = xr.open_dataset(
            path,
            chunks={"time": 1, "sigma": -1, "y": -1, "x": -1},
            decode_timedelta=False,
        )
        if not inherited:
            inherited = {k: ds.attrs[k] for k in INHERIT if k in ds.attrs}
        ds = ds.drop_vars([d for d in DROP if d in ds.variables], errors="ignore")
        merged = ds if merged is None else merged.merge(ds)

    merged = merged.drop_dims([d for d in DROP if d in merged.dims], errors="ignore")

    merged = merged.assign_coords(
        longitude=(("y", "x"), grid["longitude"]),
        latitude=(("y", "x"), grid["latitude"]),
    )
    merged.longitude.attrs = {
        "standard_name": "longitude",
        "units": "degrees_east",
        "long_name": "Longitude",
    }
    merged.latitude.attrs = {
        "standard_name": "latitude",
        "units": "degrees_north",
        "long_name": "Latitude",
    }

    # xarray emits the CF `coordinates` attribute itself for non-dimension coordinates,
    # so setting it here would collide at write time.
    for v in merged.data_vars:
        merged[v].attrs.pop(
            "cell_measures", None
        )  # refers to `parea`, not present here
        merged[v].attrs.pop("standard_name", None)  # source values are not CF-valid
        merged[v].attrs.update(VAR_CF.get(v, {}))
        # These are monthly averages; without cell_methods nothing in the file says so.
        merged[v].attrs["cell_methods"] = "time: mean"

    merged = add_time_bounds(merged)

    merged.attrs = {**inherited, **GLOBAL_ATTRS}
    merged.attrs["geospatial_lat_min"] = float(np.nanmin(grid["latitude"]))
    merged.attrs["geospatial_lat_max"] = float(np.nanmax(grid["latitude"]))
    merged.attrs["geospatial_lon_min"] = float(np.nanmin(grid["longitude"]))
    merged.attrs["geospatial_lon_max"] = float(np.nanmax(grid["longitude"]))
    return merged


def encoding(ds: xr.Dataset) -> dict:
    """One chunk per rendered map; zstd+shuffle instead of the source's zlib level 1."""
    codec = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        chunks = tuple(1 if d in ("time", "sigma") else ds.sizes[d] for d in dims)
        enc[v] = {"chunks": chunks, "compressors": [codec], "dtype": "float32"}
    for c in ("longitude", "latitude"):
        enc[c] = {"chunks": ds[c].shape, "compressors": [codec]}
    # time_bnds must share the reference and calendar of the axis it bounds.
    for t in ("time", "time_bnds"):
        if t in ds.variables:
            enc[t] = {
                "units": "days since 1653-01-01",
                "calendar": "noleap",
                "dtype": "float64",
            }
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", nargs="*", default=VARS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing store")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to overwrite")
        shutil.rmtree(args.out)

    ds = build(args.vars)
    print(ds)
    print("\nWriting", args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(
        args.out, mode="w", encoding=encoding(ds), zarr_format=3, consolidated=True
    )
    print("Done.")


if __name__ == "__main__":
    main()
