#!/usr/bin/env python3
"""Build the presentation figures for the Arctic radioisotopes case study.

Two figures:

* `arctic_sources.png` -- the three U-236 sources side by side on a North Polar
  Stereographic projection, sharing one logarithmic colour scale so the panels are
  directly comparable. That shared scale is the whole point: it is what shows La Hague
  contributing more than Sellafield, and both exceeding the fallout background.
* `arctic_timeseries.png` -- Arctic-mean concentration of each source, 2010-2018.

Colours are the categorical palette validated with the data-viz palette checker
(CVD separation, chroma, lightness band and contrast all pass); the theme's own
blue/orange/green fails deuteranopia separation at dE 4.6, so green is replaced by purple.

    pixi run python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
import xarray as xr

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

STORE = Path("data/zarr/JRAOC20TRNRPv2_2010-2018.zarr")
SIGMA = 36  # 41.9% real water in the Arctic, reprocessing signal clearly above fallout
ARCTIC_LAT = 65.0

# Fixed order, fixed colour per source -- never reassigned when a series is dropped.
SOURCES = [
    ("U236SF", "Sellafield", "#2F5CB0"),
    ("U236LH", "La Hague", "#C1571B"),
    ("U236GF", "Global fallout", "#7A3E9D"),
]

INK = "#333333"
MUTED = "#6B7280"
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Carlito",
            "Calibri",
            "DejaVu Sans",
        ],  # Carlito is metric-compatible with Calibri
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#D1D5DB",
        "figure.facecolor": "white",
    }
)


def load() -> xr.Dataset:
    return xr.open_zarr(STORE, consolidated=True, decode_timedelta=False)


def arctic_maps(ds: xr.Dataset, out: Path, time_index: int = 0) -> None:
    lon = ds.longitude.values
    lat = ds.latitude.values
    fields = {v: ds[v].isel(time=time_index, sigma=SIGMA).values for v, _, _ in SOURCES}

    # Scale from the plotted region only, and cap the span. A raw 2nd percentile reaches
    # ~1e0 here -- a thin tail that would stretch the scale over eight decades and saturate
    # every panel to the same dark blue, destroying the comparison the figure exists to make.
    visible = lat > 58
    finite = np.concatenate(
        [
            f[visible][np.isfinite(f[visible]) & (f[visible] > 0)]
            for f in fields.values()
        ]
    )
    vmax = float(np.percentile(finite, 99.5))
    vmin = max(float(np.percentile(finite, 15)), vmax / 40)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    proj = ccrs.NorthPolarStereo(central_longitude=0)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), subplot_kw={"projection": proj})

    for ax, (var, label, colour) in zip(axes, SOURCES):
        ax.set_extent([-180, 180, 58, 90], ccrs.PlateCarree())
        mesh = ax.pcolormesh(
            lon,
            lat,
            fields[var],
            transform=ccrs.PlateCarree(),
            cmap="Blues",
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        ax.add_feature(
            cfeature.LAND,
            facecolor="#EFEFEC",
            edgecolor="#C9C9C4",
            linewidth=0.4,
            zorder=2,
        )
        ax.gridlines(color="#D1D5DB", linewidth=0.4, alpha=0.7)
        # Colour the panel title with the series colour so map and time series agree.
        ax.set_title(label, fontsize=14, pad=10, color=colour, fontweight="bold")

    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.06,
        shrink=0.55,
        extend="both",
    )
    cbar.set_label(
        "U-236 concentration (atoms kg$^{-1}$), shared log scale",
        fontsize=10,
        color=MUTED,
    )
    # Matplotlib's default log ticks collide on a narrow bar; place 1/2/5-per-decade.
    lo, hi = int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax)))
    ticks = [
        m * 10.0**e
        for e in range(lo, hi + 1)
        for m in (1, 2, 5)
        if vmin <= m * 10.0**e <= vmax
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(
        [
            f"{t / 10**np.floor(np.log10(t)):.0f}$\\times$10$^{{{int(np.floor(np.log10(t)))}}}$"
            for t in ticks
        ]
    )
    cbar.ax.tick_params(labelsize=9, colors=MUTED)
    cbar.outline.set_edgecolor("#D1D5DB")

    when = str(ds.time.values[time_index])[:7]
    fig.suptitle(
        f"Uranium-236 in the Arctic Ocean by source — {when}, "
        f"potential density {float(ds.sigma[SIGMA]):.2f} kg m$^{{-3}}$",
        fontsize=15,
        y=0.97,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (colour scale {vmin:.2e}–{vmax:.2e})")


def arctic_timeseries(ds: xr.Dataset, out: Path) -> None:
    arctic = ds.latitude.values > ARCTIC_LAT
    times = ds.time.values
    series = {}
    for var, label, _ in SOURCES:
        da = ds[var].isel(sigma=SIGMA)
        vals = []
        for t in range(ds.sizes["time"]):
            a = da.isel(time=t).values
            vals.append(np.nanmean(np.where(arctic, a, np.nan)))
        series[label] = np.array(vals)

    years = np.array([t.year + (t.month - 1) / 12 for t in times])
    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    for var, label, colour in SOURCES:
        ax.plot(years, series[label] / 1e6, color=colour, linewidth=2, label=label)
        # Direct label at the line end -- identity never rests on colour alone.
        ax.annotate(
            label,
            xy=(years[-1], series[label][-1] / 1e6),
            xytext=(8, 0),
            textcoords="offset points",
            color=colour,
            fontsize=11,
            fontweight="bold",
            va="center",
        )

    # Scale to 10^6 rather than leaving matplotlib's "1e6" offset text floating.
    ax.set_ylabel("Arctic mean concentration (10$^6$ atoms kg$^{-1}$)", fontsize=11)
    ax.set_title(
        f"Uranium-236 north of {ARCTIC_LAT:.0f}°N by source, 2010–2018",
        fontsize=15,
        pad=14,
        loc="left",
    )
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.02)
    ax.set_xlim(years[0], years[-1] + 0.85)  # room for the direct labels, no dead space
    ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    for label, v in series.items():
        print(f"  {label:16} {v[0]:.2e} -> {v[-1]:.2e}  ({100*(v[-1]/v[0]-1):+.0f}%)")
    print(f"wrote {out}")


def divider_panel(
    ds: xr.Dataset, out: Path, var: str, cmap: str, time_index: int = 0
) -> None:
    """A full-bleed portrait Arctic map for the template's divider layout.

    The divider placeholder is the right half of the slide, 6.67 x 7.50 in, and the title
    sits on the left. So this carries no title, colourbar or annotation -- it is wallpaper,
    and anything written on it would compete with the slide's own heading.
    """
    lon, lat = ds.longitude.values, ds.latitude.values
    field = ds[var].isel(time=time_index)
    field = field.isel(sigma=SIGMA).values if "sigma" in ds[var].dims else field.values

    visible = lat > 55
    good = field[visible][np.isfinite(field[visible])]
    if var.startswith("U236"):
        good = good[good > 0]
        vmax = float(np.percentile(good, 99))
        norm = LogNorm(vmin=max(float(np.percentile(good, 20)), vmax / 30), vmax=vmax)
        kw = {"norm": norm}
    else:
        kw = {
            "vmin": float(np.percentile(good, 2)),
            "vmax": float(np.percentile(good, 98)),
        }

    fig = plt.figure(figsize=(6.67, 7.50))
    ax = fig.add_axes(
        [0, 0, 1, 1], projection=ccrs.NorthPolarStereo(central_longitude=0)
    )
    ax.set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
    ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="auto",
        rasterized=True,
        **kw,
    )
    ax.add_feature(
        cfeature.LAND, facecolor="#F2F2EF", edgecolor="#D5D5D0", linewidth=0.4, zorder=2
    )
    ax.spines["geo"].set_visible(False)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def workflow_diagram(out: Path) -> None:
    """Three-band FAIRification workflow, after the RiOMAR 'de la donnee a l'outil' slide.

    Sized to the template's Image Page placeholder (11.90 x 4.06 in) so it fills without
    letterboxing. Deliberately wordless where the slide can say it aloud.
    """
    from matplotlib.patches import FancyArrow, FancyBboxPatch

    TEAL, GREEN, BLUE = "#17635F", "#35A08B", "#2E5C8A"
    BAND, INK, MUTED = "#EEF4F4", "#1F2937", "#4B5563"

    fig = plt.figure(figsize=(11.90, 4.06))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    def box(x, y, w, h, fc, ec=None, r=0.02, lw=1.2):
        # rounding_size is in data units and the axes are 2.93:1, so a radius that suits a
        # tall box deforms a short one into a lens. Clamp it to a quarter of the short side.
        radius = min(r * 100, min(w, h) * 0.25)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fc,
                edgecolor=ec or fc,
                linewidth=lw,
                zorder=2,
            )
        )

    def label(x, y, t, size=10, c=INK, w="normal", ha="left", va="center"):
        ax.text(x, y, t, fontsize=size, color=c, fontweight=w, ha=ha, va=va, zorder=4)

    # backdrop
    box(1, 14, 86, 84, BAND, r=0.01)

    bands = [
        (72, 24, TEAL, "NorESM\nDATA"),
        (46, 22, GREEN, "FAIRIFICATION"),
        (18, 24, BLUE, "TOOLS &\nDASHBOARD"),
    ]
    for y, h, colour, name in bands:
        box(3, y, 16, h, colour, r=0.02)
        label(11, y + h / 2, name, size=11, c="white", w="bold", ha="center")

    # band 1 -- the source
    label(
        22, 90, "Arctic Ocean — NorESM2/BLOM simulations, 2010–2018", size=11, w="bold"
    )
    label(
        22,
        84,
        "6 GB  ·  temperature  ·  salinity  ·  uranium-236 from three sources",
        size=9.5,
        c=MUTED,
    )
    label(
        22,
        78,
        "No coordinates. No metadata on five of seven files.",
        size=9.5,
        c="#B45309",
    )

    ax.add_patch(
        FancyArrow(
            24,
            70,
            0,
            -4,
            width=1.6,
            head_width=4,
            head_length=2.2,
            facecolor="#9CA3AF",
            edgecolor="none",
            zorder=3,
        )
    )
    box(28, 66, 12, 5, GREEN, r=0.06)
    label(34, 68.5, "prepare", size=9, c="white", w="bold", ha="center")

    # band 2 -- the three FAIRification steps
    steps = [
        (22, "Recover the grid\ncoordinates", "white", TEAL),
        (44, "Standardised metadata\nCF · ACDD · I-ADOPT", "white", TEAL),
        (66, "RO-Crate\nFAIR Digital Object", TEAL, TEAL),
    ]
    for x, text, fc, ec in steps:
        box(x, 48, 19, 16, fc, ec=ec, r=0.05)
        label(
            x + 9.5,
            56,
            text,
            size=9.5,
            c="white" if fc == TEAL else INK,
            w="bold",
            ha="center",
        )
    for x in (41.5, 63.5):
        ax.add_patch(
            FancyArrow(
                x,
                56,
                2.0,
                0,
                width=1.2,
                head_width=3.2,
                head_length=1.4,
                facecolor=TEAL,
                edgecolor="none",
                zorder=3,
            )
        )

    ax.add_patch(
        FancyArrow(
            24,
            45,
            0,
            -4,
            width=1.6,
            head_width=4,
            head_length=2.2,
            facecolor="#9CA3AF",
            edgecolor="none",
            zorder=3,
        )
    )
    box(28, 41, 18, 5, BLUE, r=0.06)
    label(37, 43.5, "publish the FDO", size=9, c="white", w="bold", ha="center")

    # band 3 -- consumption
    label(
        22,
        36,
        "The dashboard reads the FAIR Digital Object directly",
        size=11,
        w="bold",
    )
    label(
        22,
        30,
        "Everything on screen — data and metadata — comes from the FDO",
        size=9.5,
        c=MUTED,
    )
    box(22, 18, 44, 8, "white", ec=BLUE, r=0.06)
    label(
        24,
        22,
        "→  nothing is hard-coded: the FDO describes it all",
        size=10,
        c=BLUE,
        w="bold",
    )

    # reusable spine
    box(89, 14, 10, 84, "#D8EDE7", r=0.02)
    ax.add_patch(
        FancyArrow(
            94,
            30,
            0,
            52,
            width=5,
            head_width=9,
            head_length=8,
            facecolor=GREEN,
            edgecolor="none",
            zorder=3,
        )
    )
    ax.text(
        94,
        56,
        "REUSABLE",
        fontsize=12,
        color="white",
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
        zorder=4,
    )

    box(1, 2, 98, 9, GREEN, r=0.03)
    label(
        50,
        6.5,
        "A new FDO — other data, same metadata pattern — feeds the same dashboard.",
        size=10.5,
        c="white",
        w="bold",
        ha="center",
    )

    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def figma_guide(out: Path) -> None:
    """One-slide guide to commenting on the Figma board, after the RiOMAR 5-minute deck.

    Two tool boxes side by side, the warning that matters (people drag the board when they
    meant to write), and the sections they will be asked to comment in.
    """
    from matplotlib.patches import FancyBboxPatch

    TEAL, GREEN, AMBER, INK, MUTED = (
        "#17635F",
        "#35A08B",
        "#B45309",
        "#1F2937",
        "#4B5563",
    )

    fig = plt.figure(figsize=(11.90, 4.60))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    def box(x, y, w, h, fc, ec=None, r=2.0, lw=1.4):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={min(r, min(w, h) * 0.25)}",
                facecolor=fc,
                edgecolor=ec or fc,
                linewidth=lw,
                zorder=2,
            )
        )

    def txt(x, y, t, size=10, c=INK, w="normal", ha="left"):
        ax.text(
            x, y, t, fontsize=size, color=c, fontweight=w, ha=ha, va="center", zorder=4
        )

    txt(
        50,
        95,
        "Two tools — switch between them in the toolbar",
        size=12.5,
        c=TEAL,
        w="bold",
        ha="center",
    )

    box(2, 46, 46, 40, "white", ec="#D1D5DB")
    box(2, 78, 46, 8, TEAL, r=2)
    txt(25, 82, "MOVE AROUND   (Hand tool)", size=11, c="white", w="bold", ha="center")
    for i, line in enumerate(
        [
            "Click and drag to move the board",
            "Scroll or pinch to zoom",
            "Tip: hold Space = hand tool",
        ]
    ):
        txt(6, 71 - i * 7, "•  " + line, size=10.5, c=INK if i < 2 else MUTED)

    box(52, 46, 46, 40, "white", ec="#D1D5DB")
    box(52, 78, 46, 8, GREEN, r=2)
    txt(
        75,
        82,
        "WRITE / COMMENT   (Post-it / Text)",
        size=11,
        c="white",
        w="bold",
        ha="center",
    )
    for i, line in enumerate(
        [
            "Pick the Post-it or Text tool",
            "Click where you want to comment",
            "Type your comment",
        ]
    ):
        txt(56, 71 - i * 7, "•  " + line, size=10.5)

    box(2, 30, 96, 12, "#FEF6E7", ec="#F5C77E", r=2)
    txt(
        6,
        38,
        "⚠   Remember to switch tools between each action",
        size=11,
        c=AMBER,
        w="bold",
    )
    txt(
        6,
        33.5,
        "Hand = move the board  ·  Post-it / Text = write.  "
        "Otherwise you will drag the board instead of writing.",
        size=9.5,
        c=MUTED,
    )

    txt(
        2,
        24,
        "The board has two panels — add your comments in either:",
        size=10.5,
        c=INK,
        w="bold",
    )

    panels = [
        (
            "1 — User stories",
            ["US2 to US5, one zone each", "Add the questions you want answered"],
        ),
        (
            "2 — The dashboard",
            [
                "Main plot · colormaps · I-ADOPT variables",
                "nanopublications · other metadata",
            ],
        ),
    ]
    pw = 47
    for i, (name, lines) in enumerate(panels):
        x = 2 + i * (pw + 2)
        box(x, 3, pw, 17, "white", ec="#D1D5DB")
        box(x, 14, pw, 6, TEAL, r=1.5)
        txt(x + pw / 2, 17, name, size=10, c="white", w="bold", ha="center")
        for j, line in enumerate(lines):
            txt(x + 2.5, 10.5 - j * 4.5, "•  " + line, size=9.5, c=MUTED)

    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = load()
    arctic_maps(ds, args.outdir / "arctic_sources.png")
    arctic_timeseries(ds, args.outdir / "arctic_timeseries.png")
    # Divider wallpaper, one per section.
    divider_panel(ds, args.outdir / "divider_sst.png", "sst", "RdYlBu_r")
    divider_panel(ds, args.outdir / "divider_lahague.png", "U236LH", "Blues")
    divider_panel(ds, args.outdir / "divider_sellafield.png", "U236SF", "Purples")
    workflow_diagram(args.outdir / "workflow.png")
    figma_guide(args.outdir / "figma_guide.png")


if __name__ == "__main__":
    main()
