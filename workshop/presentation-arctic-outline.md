# Arctic Radioisotopes × FAIR2Adapt — stakeholder meeting deck

Slide-by-slide content for the FAIR2Adapt PowerPoint template, mirroring the structure of
the RiOMAR stakeholder deck (`example_RIOMAR-presentation.pdf`, 21 slides).

**Assumptions to confirm** — the RiOMAR deck is in French because its stakeholders are
French; this is drafted in **English** for a Nordic/Arctic audience. Slides 12–15 are
essentially reusable from the RiOMAR deck (the FAIR2Adapt project introduction) and are
marked *[reuse]*.

---

## 1. Title

**Welcome to the Arctic Radioisotopes & FAIR2Adapt stakeholder meeting**

*Footer: replace `Presentation title | Name Surname`.*

## 2. Agenda

- Introducing the Arctic radioisotopes case study (NERSC)
- What FAIR2Adapt aims to achieve
- The visualisation tool — features and demo
- Hands-on exploration by participants
- Feedback: what is missing, what would you want next

## 3. Agenda with timings

Mirror RiOMAR slide 3 — split into introduction (5 + 10 min), FAIR2Adapt link (5 + 5 min),
tool demonstration, exploration, feedback round. Name a lead for each.

---

## 4. The question

**Where does Arctic radioactive contamination come from — and how much of it is ours?**

Three sources put uranium-236 into the Arctic Ocean:

- **Sellafield** (UK) — nuclear fuel reprocessing discharges
- **La Hague** (France) — nuclear fuel reprocessing discharges
- **Global fallout** — atmospheric nuclear weapons testing

Measurements in Arctic seawater see only the **total**. Attribution to a source is not
possible from observation alone.

*Speaker note: this is the question the dataset exists to answer.*

## 5. The approach — tag each source separately in the model

NorESM2 (BLOM–CICE5) carries **three separate passive tracers**, one per source, through
the same simulated ocean. Each is transported by the same currents, but can be read
independently.

- Model: NorESM2, CMIP6-generation Earth system model
- Ocean: BLOM (Bergen Layered Ocean Model), isopycnic coordinates
- Forcing: JRA-55 atmospheric reanalysis
- Simulated period: 1950–2018 (this dataset: 2010–2018)
- Reference: Seland et al. (2020), doi:10.5194/gmd-13-6165-2020

## 6. Why the model suits Arctic questions

The BLOM tripolar grid is **refined at high latitudes**:

| Region | Horizontal resolution |
|---|---|
| Equator | ~111 km |
| 60–70°N | ~44 km |
| **70–90°N** | **~38 km** |

Roughly three times finer where the stakeholders are. *(This is nanopublication claim 1.)*

The vertical coordinate is **potential density**, not depth — 53 isopycnal layers that
follow water masses, which is how the transport is actually organised.

## 7. What is in the dataset

| | |
|---|---|
| Variables | 7 — `sst`, `sss`, `temp`, `saln`, `U236GF`, `U236SF`, `U236LH` |
| Time | 108 monthly means, Jan 2010 – Dec 2018 |
| Vertical | 53 isopycnal layers (potential density 27.22–37.80 kg m⁻³) |
| Grid | BLOM tripolar 385 × 360, 2-D lon/lat |
| Size | 6.1 GB, cloud-optimised Zarr |

## 8. A first result

At density level 36 in the Arctic (>65°N), the two reprocessing plants together deliver
**more U-236 than all atmospheric weapons testing combined** — and the reprocessing share
is declining as discharges are reduced.

| Arctic mean (atoms kg⁻¹) | 2010-01 | 2014-06 | 2018-12 |
|---|---|---|---|
| La Hague | 6.0 × 10⁶ | 5.3 × 10⁶ | 4.7 × 10⁶ |
| Sellafield | 3.6 × 10⁶ | 3.2 × 10⁶ | 2.8 × 10⁶ |
| Global fallout | 3.1 × 10⁶ | 3.1 × 10⁶ | 3.0 × 10⁶ |
| **(SF+LH) / GF** | **3.1×** | **2.7×** | **2.5×** |

> Caution: these figures are from our own analysis of the dataset, at one density layer and
> one latitude threshold. The ratio ranges 1.3–3.3 across layers. Have Yanchun He confirm
> before presenting them as a result.

---

## 9–11. FAIR2Adapt *[reuse from RiOMAR slides 12–14]*

- FAIR to Adapt to Climate Change — https://fair2adapt-eosc.eu/
- 3-year EU project (2025–2027), INFRA-2024-EOSC-01-01, 18 partners, grant 101188256
- Four objectives: FAIRification framework; interoperability in EOSC; community
  engagement; demonstrating impact
- Six case studies — **Arctic radionuclides (NERSC)** is this one

---

## 12. The need

**From model output nobody can open, to data a stakeholder can explore in a browser.**

What the source files looked like:

- Seven separate netCDF files, 6 GB, one variable each
- **No geolocation** — no coordinates for the 385 × 360 grid
- **Five of seven files had no metadata at all** — no title, licence or citation
- Two files used `standard_name` values that do not exist in CF
- Nothing recorded that these are monthly means

## 13. What FAIRification actually changed

| Before | After |
|---|---|
| No grid coordinates | 2-D lon/lat recovered, verified bit-identical |
| 5 of 7 files with no metadata | Uniform CF-1.8 + ACDD-1.3 on all seven |
| `Ocean temperature` (not a CF name) | `sea_water_potential_temperature` |
| No record of averaging | `cell_methods` + time bounds |
| Whole-file download | One chunk per rendered map (~554 KB) |
| No semantics | I-ADOPT decomposition per variable |

## 14. Findable and citable — the Research Object

`https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9`

- Conforms to the **CCA RO-Crate profile**
- **I-ADOPT** decomposition for all seven variables — e.g. U-236 is *number density per
  unit mass* of *uranium-236 per unit mass of water* in a *marine environment*
- Two **nanopublications** carrying citable claims about the model
- Dataset, pipeline code and dashboard all linked

*Visual: the I-ADOPT decomposition diagram.*

## 15. Machine-actionable

The dashboard reads the Research Object, resolves the dataset, and renders the variable
semantics alongside the map — nothing is typed in by hand.

---

## 16. Demonstration

Links in `WORKSHOP-LINKS.md`. Suggested order:

1. Sea surface temperature — establishes the grid and Arctic projection
2. U-236 from **La Hague** — the largest contributor
3. U-236 from **Sellafield** — about half as much
4. U-236 from **global fallout** — spatially uniform by contrast
5. Sellafield in 2010 vs 2018 — the declining signal

> Set the colour scale to **logarithmic** before showing any U-236 variable, and keep the
> density slider between 10 and 35.

## 17. Try it yourself

Give participants the links and let them explore. Mirror RiOMAR slide 19 — a short URL in
the chat.

## 18–19. Feedback

RiOMAR used a **Figma** board with "Thoughts" zones and post-its. Worth reusing, with
questions adapted:

- Can you find the variable, region and period you need?
- Is potential density a usable vertical coordinate for you, or do you need depth?
- What would you want to ask that this dataset cannot answer yet?

## 20. What we could not do yet — honest limitations

- **Chart / bounding-box selection needs the HEALPix version of the data**, which the
  dashboard cannot currently show in the Arctic-centred projection. So today you can have
  the Arctic view *or* the time-series chart, not both. This is dashboard customisation
  work, and we can scope it.
- Deeper density layers (above ~41) contain no water; the tracers still carry values there.

## 21. Available data

- Dataset: `JRAOC20TRNRPv2_all_2010-2018.zarr` (7 variables, tripolar grid)
- Also available: HEALPix level 8 version, for DGGS analysis
- Research Object: `https://w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9`
- Pipeline: https://github.com/FAIR2Adapt/arctic-radioisotopes
- Licence: CC-BY 4.0
- Data: Yanchun He (NERSC), NKS-B ANTHROPIC (AFT/B(23)4)

---

## Visual assets available

| File | Use |
|---|---|
| `/tmp/check2.png` | SST + U-236 global maps, showing real vs massless layers — good for slide 8 or 12 |
| `/tmp/iadopt_last.png` | I-ADOPT decomposition diagram for U-236 — slide 14 |

Better figures can be generated on request: an Arctic-centred plume map per source, or a
time-series of the three tracers 2010–2018.
