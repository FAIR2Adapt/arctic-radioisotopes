# FAIR2Adapt Case Study 1 — workshop demo notes

NorESM2/BLOM Arctic radioisotopes, 2010–2018 monthly means, in the FAIR2Adapt dashboard.

## Use the tripolar store

```
https://fair2adapt.duckdns.org/afouilloux-noresm/JRAOC20TRNRPv2_all_2010-2018.zarr
```

7 variables, native BLOM tripolar grid, and it **supports `nearside_perspective`** — which
the HEALPix store does not (see [Known limitations](#known-limitations)).

Token: `$F2A_TOKEN` (read-only, passed as
`::token=` in the URL; the proxy authenticates with an `X-Auth-Token` header).

## Demo URLs

All centred on the North Pole for Arctic stakeholders. Base:

```
https://f2a.plan4all.eu/#https://fair2adapt.duckdns.org/afouilloux-noresm/JRAOC20TRNRPv2_all_2010-2018.zarr::token=$F2A_TOKEN   # substitute the real token; do not commit it
```

Append one of:

| Demo | Parameters |
|---|---|
| Sea surface temperature — safe opener, no density slider | `::varname=sst::dimIndices_time=0::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0` |
| U-236 from **Sellafield** — headline result | `::varname=U236SF::dimIndices_time=0::dimIndices_sigma=25::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0` |
| U-236 from **La Hague** | as above with `varname=U236LH` |
| U-236 **global fallout** — baseline for contrast | as above with `varname=U236GF` |
| Ocean temperature at depth | as above with `varname=temp` |

Other variables: `saln` (3-D salinity), `sss` (surface salinity).

## Two things that will bite you on stage

### 1. Keep the density slider between 10 and 35

`sigma` is **potential density**, not depth — 53 isopycnal layers. Only levels **0–41**
contain meaningful water. Above that the layers are largely massless:

| Level | Density | Real water |
|---|---|---|
| 0 | 27.22 | 62.6% |
| 25 | 36.28 | 56.3% ← best |
| 35 | 36.98 | 56.4% ← best |
| 41 | 37.20 | ~25% |
| 45 | 37.33 | 9.1% |
| 52 | 37.80 | 0.2% |

The U-236 tracers carry values in **every** layer regardless of whether water exists there,
so dragging the slider to the right shows convincing-looking numbers that mean nothing.
This comes from the source model output, not the conversion. Worth confirming with
Yanchun He whether those layers should be masked.

### 2. Set the colour scale to logarithmic for U-236

The tracers span roughly 70× between the 1st and 99th percentile (and up to 10¹⁶ in the
massless layers). On a linear scale they look flat. Use the **Colormap & range** panel and
switch to log before showing them — that is what makes the Sellafield and La Hague plumes
visible.

## Talking points

- **Three separately-tagged U-236 sources** — global fallout, Sellafield, La Hague — so the
  contribution of each reprocessing plant to Arctic contamination can be traced
  independently. That is the scientific value of this dataset.
- **The vertical coordinate is density, not depth.** Layers follow water masses, which is
  how the transport is actually organised.
- **FAIRification, concretely**: the source netCDF had no geolocation for its 385×360 grid,
  and five of seven files had no metadata at all — no title, licence or citation. The
  published store carries CF-1.8 + ACDD-1.3 metadata, correct standard names, `cell_methods`
  and time bounds throughout.
- **Cloud-optimised**: chunked one map per chunk (~554 KB), so the dashboard fetches exactly
  what it draws instead of whole files.

## Known limitations

Both are dashboard-side, not data problems, and both are customisation scope:

1. **Chart / bounding box / polygon selection requires the HEALPix (DGGS) store.** The
   dashboard routes HEALPix data to a MapLibre renderer, which does not offer
   `nearside_perspective` — projection controls were deliberately hidden for that path
   (riomar-dashboard commit 2026-06-03). So today you can have Arctic projection **or**
   Chart, not both.
2. **A HEALPix store exists and works** —
   `JRAOC20TRNRPv2_all_2010-2018_hp.zarr`, level 8 (~25 km), verified — so if the projection
   gap is closed on the dashboard side, the data is already published and ready.

Smallest fix to ask plan4all for: enable MapLibre's globe projection for the HEALPix view.
It sits in the code path they are actively developing, unlike the WebGL HEALPix renderer,
which has not been touched since March 2026.

A further data-side improvement for any Chart work: the published store is chunked for map
rendering, so a single-cell time series touches 109 MB to return 432 bytes. A companion
store chunked `(108, 1, 4096)` would make a time series one chunk.

## What is published

| Store | Contents | Purpose |
|---|---|---|
| `JRAOC20TRNRPv2_2010-2018.zarr` | SST + temp, 2 vars | original, still live in the existing RO-Crate |
| `JRAOC20TRNRPv2_all_2010-2018.zarr` | **all 7 vars, tripolar** | **use this for the workshop** |
| `JRAOC20TRNRPv2_all_2010-2018_hp.zarr` | all 7 vars, HEALPix level 8 | for Chart, once projections are supported |

All in the `afouilloux-noresm` bucket on Pangeo@EOSC MinIO; nothing was overwritten.
