# Data pipeline: NorESM2 netCDF → cloud-optimised Zarr

Turns the seven NorESM2/BLOM monthly-mean netCDF files into a single CF-compliant Zarr
store for the FAIR2Adapt dashboard.

| Script | Purpose |
|---|---|
| `sharepoint_login.py` | Authenticate to the NERSC SharePoint share (email one-time passcode) |
| `sharepoint_fetch.py` | Download the source netCDF over the SharePoint REST API |
| `fetch_blom_grid.py` | Retrieve the 2-D BLOM tripolar coordinates |
| `make_zarr.py` | Merge, re-chunk and write the Zarr store |

No credentials are stored in this repository. Tokens are passed on the command line or
through environment variables, and session state is written outside the working tree.

## 1. Download the source data

The share is accessible only to external guests authenticated by **email one-time
passcode**: the address has no Microsoft account behind it (Entra ID returns
`IfExistsResult: 1`), so no OAuth token can be issued and rclone, Microsoft Graph and the
anonymous `/shares` API all fail. The passcode authenticates a *browser session* only.

`sharepoint_login.py` therefore drives a headless Chromium on the machine that will hold
the data, so the transfer runs SharePoint → server rather than via a laptop.

```bash
python arctic-isotopes/sharepoint_login.py --email you@example.org
# when the message arrives, in another shell:
echo 123456 > .auth/otp.txt

python arctic-isotopes/sharepoint_fetch.py --dry-run   # list first
python arctic-isotopes/sharepoint_fetch.py             # -> data/raw/
```

Notes worth knowing before you start:

- The passcode step cannot be automated; someone must read it from the inbox.
- On a rapid re-request Microsoft **reissues the same code and sends no second email**.
  If nothing new arrives, try the previous code before assuming a failure.
- The `FedAuth` cookie lasts hours. Re-running the download is cheap: files already
  present at the correct size are skipped.
- Chrome ≥132 redacts `Cookie` headers from DevTools "Copy as cURL", so copying a session
  from a desktop browser no longer works — hence the headless login.
- Every step is screenshotted to `.auth/shots/`, which is where to look first when
  Microsoft changes its login pages and a selector stops matching.

## 2. Fetch the grid

The source files carry **no geolocation** for their 385×360 grid. Their `lat` variable is
a 1-D zonal-diagnostic axis, and `cell_measures: "area: parea"` refers to a `parea`
variable that is not present in any of them.

The 2-D `longitude`/`latitude` are read from the published store and verified against a
local field before use:

```bash
python arctic-isotopes/fetch_blom_grid.py --token "$F2A_TOKEN" --verify
```

`--verify` compares the remote `sst` field against the local netCDF and refuses to write
the grid unless they match exactly, so a mismatched grid cannot silently corrupt the
geolocation.

If the original BLOM `grid.nc` is available, prefer it: it also carries the cell vertices
(`pclat`/`pclon`) that `data_handling.py` uses, which would let the store declare CF
`bounds` on the coordinates and support conservative regridding.

## 3. Build the Zarr store

```bash
python arctic-isotopes/make_zarr.py            # all variables -> data/zarr/
python arctic-isotopes/make_zarr.py --vars sst # quick single-variable run
```

What the conversion changes relative to the source netCDF:

**Chunking.** Source chunks are `(1, 27, 193, 180)`, so rendering one map costs ~16 chunk
reads. The store uses `(time=1, sigma=1, y=385, x=360)` — one chunk per rendered map,
about 554 KB. For comparison, a single `temp` map in the earlier published store spans 16
chunks totalling roughly 55 MB.

**Compression.** Zstd level 5 with shuffle, replacing zlib level 1.

**CF compliance.** The source `standard_name` values `Ocean temperature` and
`Ocean salinity` are not in the CF standard name table (checked against v94, 5071 names),
and `sst`, `sss` and the U-236 tracers carried none at all. The store uses:

| Variable | `standard_name` |
|---|---|
| `sst` | `sea_surface_temperature` |
| `sss` | `sea_surface_salinity` |
| `temp` | `sea_water_potential_temperature` |
| `saln` | `sea_water_salinity` |
| `U236GF` / `U236SF` / `U236LH` | *(none — no CF term exists)* |

The U-236 tracers deliberately carry no `standard_name`: CF says to omit it rather than
invent a term. They are described semantically through I-ADOPT in the RO-Crate instead.

`cell_methods = "time: mean"` is set on every variable, and `time_bnds` spans each
averaging month, since CF requires bounds once `cell_methods` declares an average.

> **To confirm:** `temp` is mapped to `sea_water_potential_temperature` on the basis that
> BLOM is an isopycnic model carrying potential temperature. If it is in fact conservative
> temperature, this standard name is wrong and must be changed before a DOI is minted.

**Metadata.** Only `temp` and `sst` carried global attributes; the other five files had
none. The ACDD block is promoted to dataset level and applied uniformly, with
`geospatial_*` bounds computed from the grid and `time_coverage_*` from the time axis.

Vestigial `depth`, `lat`, `region` and `section` dimensions are dropped — no data variable
uses them, and a 1-D `lat` beside the 2-D `latitude` is a trap for anyone selecting by
position.

## Extending the metadata later

Zarr keeps attributes in JSON documents separate from chunk data, so metadata can be added
at any time without rewriting arrays — adding a DOI plus two attributes measured at 210
bytes with every chunk byte unchanged. Re-run `zarr.consolidate_metadata()` afterwards, or
readers using the consolidated index will not see the additions. Only chunking or dtype
changes force a rewrite.
