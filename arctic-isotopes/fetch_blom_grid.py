#!/usr/bin/env python3
"""Fetch the 2-D BLOM tripolar grid coordinates from the published Zarr store.

The source netCDF have no geolocation for their 385x360 grid, so the coordinates are
taken from the store already serving the FAIR2Adapt dashboard. That store is reachable
over HTTPS with an `X-Auth-Token` header (the proxy advertises the header name in its
CORS response, and the token is the one embedded in dashboard URLs).

Directory listing is disabled on the proxy, so arrays are opened by name rather than
discovered -- xarray's group listing comes back empty against this endpoint.

    python arctic-isotopes/fetch_grid.py --token <token>

Writes data/blom_grid.npz, consumed by make_zarr.py. Run --verify to re-check the grid
against a local field before trusting it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import fsspec
import numpy as np
import zarr

STORE = "https://fair2adapt.duckdns.org/afouilloux-noresm/JRAOC20TRNRPv2_2010-2018.zarr"
OUT = Path("data/blom_grid.npz")
LOCAL_SST = Path("data/raw/JRAOC20TRNRPv2_hm_sst_2010-2018.nc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--token",
        default=os.environ.get("F2A_TOKEN"),
        help="proxy token; defaults to $F2A_TOKEN",
    )
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="confirm the remote sst field matches the local netCDF",
    )
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("Need a token: pass --token or set F2A_TOKEN")

    fs = fsspec.filesystem(
        "http", client_kwargs={"headers": {"X-Auth-Token": args.token}}
    )

    def arr(name: str) -> np.ndarray:
        return zarr.open_array(store=fs.get_mapper(f"{args.store}/{name}"), mode="r")[
            ...
        ]

    lon, lat = arr("longitude"), arr("latitude")
    print(
        f"longitude {lon.shape} {lon.dtype}  {np.nanmin(lon):.3f} .. {np.nanmax(lon):.3f}"
    )
    print(
        f"latitude  {lat.shape} {lat.dtype}  {np.nanmin(lat):.3f} .. {np.nanmax(lat):.3f}"
    )

    if args.verify:
        # The coordinates are only meaningful if the remote grid is the same one our
        # netCDF are on, so compare an actual field rather than assuming.
        import xarray as xr

        remote = arr("sst")[0]
        local = (
            xr.open_dataset(LOCAL_SST, decode_timedelta=False).sst.isel(time=0).values
        )
        masks_match = bool((np.isnan(remote) == np.isnan(local)).all())
        both = ~np.isnan(remote) & ~np.isnan(local)
        values_match = bool(np.array_equal(remote[both], local[both]))
        print(f"verify: nan masks match={masks_match} values identical={values_match}")
        if not (masks_match and values_match):
            raise SystemExit("Remote grid does not match the local data - do not use.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, longitude=lon, latitude=lat)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
