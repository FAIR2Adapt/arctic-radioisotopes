# Workshop demo links — FAIR2Adapt Case Study 1

All links resolve through the **Research Object**
(`https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9`), so the metadata panel shows the
I-ADOPT decomposition alongside the map. All are centred on the North Pole.

> Set **Colormap & range → logarithmic** before showing any U-236 variable. On a linear
> scale the tracers span ~70x and the plume structure is invisible.

---

## 1. Opener — sea surface temperature

Recognisable, 54% Arctic coverage, no density slider needed. Establishes that the grid,
projection and land mask are right before anything subtle.

<https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=sst::dimIndices_time=0::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

## 2. The headline — La Hague reaches the Arctic

U-236 discharged from La Hague, at density level 36 (37.02 kg m-3), January 2010.
Arctic mean **6.0e6 atoms kg-1** — the largest single contributor.

<https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=U236LH::dimIndices_time=0::dimIndices_sigma=36::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

## 3. Compare — Sellafield

Same layer, same month, different source. Arctic mean **3.6e6** — roughly *half* La Hague.
Two plants, two plumes, separable only because they were tagged separately in the model.

<https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=U236SF::dimIndices_time=0::dimIndices_sigma=36::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

## 4. Contrast — global fallout

Same layer and month, but weapons-testing fallout: **3.1e6**, and spatially far more
uniform. The contrast with 2 and 3 is the point — point sources versus a background.

<https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=U236GF::dimIndices_time=0::dimIndices_sigma=36::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

**The number to quote:** the two reprocessing plants together deliver **more U-236 to the
Arctic than all atmospheric nuclear testing combined, in every density layer and month
sampled** -- by a factor of **1.3 to 3.3** depending on layer, median 2.3. At the layer
shown here (36) it is 3.1x in January 2010.

Quote the range, not the single figure: the 3.1x holds only at the denser layers. At the
surface it is 1.8x, and by 2018 it has fallen to 2.5x at this layer. The robust statement
is that reprocessing exceeds fallout everywhere in the Arctic, not that it does so
threefold.

## 5. Change over time — the signal is declining

Same variable and layer, December 2018. Sellafield falls 3.6e6 -> 2.8e6 across the record
while fallout stays flat (~3.0e6), so the reprocessing share drops from 3.12x to 2.51x.
Consistent with reduced authorised discharges.

2010-01: <https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=U236SF::dimIndices_time=0::dimIndices_sigma=36::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

2018-12: <https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=U236SF::dimIndices_time=107::dimIndices_sigma=36::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

## 5b. US2 — showing it works, on RiOMAR

The chart and area selection are blocked for us only because they need the HEALPix layout,
which the dashboard cannot combine with `nearside_perspective`. The tool itself is fine —
demonstrate that on the RiOMAR case study, where both work:

<https://f2a.plan4all.eu/#https://w3id.org/ro-id/9c8ee58c-28f8-42ad-9ba8-0daf91935d31::varname=dissolved_oxygen::dimIndices_time_counter=0::dimIndices_s_rho=0>

Point to make: same dashboard, charts working. What is missing for Case Study 1 is the
*combination* of chart and Arctic projection — a customisation question, not a data one.

Pair it with the time-series figure (`figures/arctic_timeseries.png`) as the mockup of what
US2 would deliver for the Arctic.

## 6. Supporting ocean state

If asked what the tracers are riding on — temperature and salinity on the same layer.

temperature: <https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=temp::dimIndices_time=0::dimIndices_sigma=36::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

salinity: <https://f2a.plan4all.eu/#https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9::token=$F2A_TOKEN::varname=saln::dimIndices_time=0::dimIndices_sigma=36::projection=nearside_perspective::projectionCenterLat=90::projectionCenterLon=0>

---

## Numbers behind the story

Arctic (lat > 65) means at density level 36, atoms kg-1:

| | 2010-01 | 2014-06 | 2018-12 |
|---|---|---|---|
| La Hague | 6.04e6 | 5.25e6 | 4.73e6 |
| Sellafield | 3.60e6 | 3.19e6 | 2.84e6 |
| Global fallout | 3.09e6 | 3.14e6 | 3.02e6 |
| **(SF+LH)/GF** | **3.12** | **2.69** | **2.51** |

Reprocessing dominance grows with density: the Sellafield-to-fallout ratio rises from 0.61
at the surface to 1.67 at level 48, i.e. the reprocessing signal is carried in denser water
masses rather than at the surface.

## Do not use

- **Density levels above ~41.** Level 52 has essentially no water anywhere, yet the tracers
  still carry values there. It renders as noise and means nothing. Levels 25-40 are safe;
  36 is used throughout above.
- **A stale `camerastate`** in the URL. It overrides `projectionCenterLat` and will drop you
  back on the equator. Frame the view first, then copy the fresh camerastate from the URL
  bar if you want to pin a shot.
