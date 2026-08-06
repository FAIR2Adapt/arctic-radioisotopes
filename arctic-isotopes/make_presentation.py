#!/usr/bin/env python3
"""Build the Arctic radioisotopes stakeholder deck from the FAIR2Adapt template.

Uses the template's own layouts and placeholders, so branding, fonts and footers come from
the template rather than being reconstructed. Layout indices in the supplied template:

    0 Cover Page        title + subtitle
    1 Text Page         title + body
    2 Text + Image Page title + body + picture
    3 Image Page 1      title + body + picture
    5 Divider Page 1    section divider
    7 BackPage         closing

Structure mirrors the RiOMAR stakeholder deck. Content is drawn from
`presentation-arctic-outline.md`; figures from `scripts/make_figures.py`.

    pixi run python scripts/make_presentation.py --author "Anne Fouilloux"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pptx import Presentation

# Override with --template; the FAIR2Adapt template is not redistributable, so it is
# not kept in the repository.
TEMPLATE = Path(os.environ.get("F2A_TEMPLATE", "FAIR2Adapt_Presentation_Template.pptx"))
FIGURES = Path("figures")
FIGMA_URL = "https://www.figma.com/board/helaXmDzUyiAgpnok82DOE/CS1"
# RiOMAR case study, where the chart and area selection already work -- useful as a
# live demonstration that the gap is in the projection combination, not the tool.
RIOMAR_DEMO = (
    "https://f2a.plan4all.eu/#https://w3id.org/ro-id/"
    "9c8ee58c-28f8-42ad-9ba8-0daf91935d31::varname=dissolved_oxygen"
    "::dimIndices_time_counter=0::dimIndices_s_rho=0"
)

COVER, TEXT, TEXT_IMAGE, IMAGE, DIVIDER, BACK = 0, 1, 2, 3, 5, 7


def drop_sample_slide(prs) -> None:
    """Remove the template's sample cover slide, relationship included.

    Removing the entry from sldIdLst alone leaves the slide part still related to the
    presentation, so python-pptx serialises it twice -- producing duplicate
    `ppt/slides/slide1.xml` entries in the zip. PowerPoint reports that as a corrupt file
    and drops content (including embedded images) when it repairs. Dropping the relationship
    is what actually detaches the part.
    """
    if len(prs.slides) != 1:
        return
    id_list = prs.slides._sldIdLst
    entry = list(id_list)[0]
    rel_id = entry.get(
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    )
    prs.part.drop_rel(rel_id)
    id_list.remove(entry)


def set_body(shape, lines: list) -> None:
    """Fill a body placeholder. Items may be str, or (text, level) for sub-bullets."""
    tf = shape.text_frame
    tf.clear()
    for i, item in enumerate(lines):
        text, level = item if isinstance(item, tuple) else (item, 0)
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.text = text
        para.level = level


def add(prs, layout_idx: int, title: str | None = None, body: list | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    if title is not None:
        for ph in slide.placeholders:
            if ph.placeholder_format.idx == 0:
                ph.text = title
                break
    if body is not None:
        for ph in slide.placeholders:
            if ph.placeholder_format.idx == 14:
                set_body(ph, body)
                break
    return slide


def add_picture_slide(prs, title: str, image: Path, body: list | None = None):
    """Use the picture placeholder if the layout has one, else place it manually."""
    slide = add(prs, IMAGE, title, body)
    for ph in list(slide.placeholders):
        if ph.placeholder_format.type == 18 or "PICTURE" in str(
            ph.placeholder_format.type
        ):
            box_l, box_t, box_w, box_h = ph.left, ph.top, ph.width, ph.height
            pic = ph.insert_picture(str(image))
            # insert_picture crops to FILL the placeholder, which would slice ~15% off the
            # top and bottom of these wide figures -- taking the title and colourbar with
            # it. Reset the crop and letterbox the image inside the box instead.
            pic.crop_left = pic.crop_right = pic.crop_top = pic.crop_bottom = 0
            from PIL import Image as PILImage

            iw, ih = PILImage.open(image).size
            scale = min(box_w / iw, box_h / ih)
            pic.width, pic.height = int(iw * scale), int(ih * scale)
            pic.left = int(box_l + (box_w - pic.width) / 2)
            pic.top = int(box_t + (box_h - pic.height) / 2)
            return slide
    # Fallback: centre it under the title.

    pic = slide.shapes.add_picture(str(image), 0, 0)
    scale = min(
        (prs.slide_width * 0.86) / pic.width, (prs.slide_height * 0.62) / pic.height
    )
    pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
    pic.left = int((prs.slide_width - pic.width) / 2)
    pic.top = int(prs.slide_height * 0.30)
    return slide


def add_text_image(
    prs, title: str, image: Path, subtitle: str = "", body: list | None = None
):
    """Layout 2: title, one-line subtitle, body text left, image right."""
    slide = prs.slides.add_slide(prs.slide_layouts[TEXT_IMAGE])
    for ph in list(slide.placeholders):
        pf = ph.placeholder_format
        if pf.idx == 0:
            ph.text = title
        elif pf.idx == 14:
            ph.text = subtitle
        elif pf.idx == 15 and body:
            set_body(ph, body)
        elif pf.idx == 10:
            ph.insert_picture(str(image))
    return slide


def add_divider(prs, title: str, subtitle: str, image: Path):
    """Section divider: text left, full-bleed image right.

    The divider layout's picture placeholder is the right half of the slide, so here
    crop-to-fill is the intended behaviour (unlike the figure slides, where it would
    slice the title off). Any placeholder left empty renders as a grey prompt box in
    PowerPoint, so the unused content placeholder is removed.
    """
    slide = add(prs, DIVIDER, title, [subtitle])
    for ph in list(slide.placeholders):
        pf = ph.placeholder_format
        if pf.idx == 1:
            ph.insert_picture(str(image))
        elif pf.idx == 15:
            ph._element.getparent().remove(ph._element)
    return slide


def build(author: str, meeting: str, out: Path) -> None:
    """Anne's two slots in the CS1 workshop: item 2 (intro, 25 min) and item 4 (demo, 30 min).

    Deliberately does NOT cover the modelling science (item 3, Yanchun) or facilitate the
    user-story review (item 5, Barbara/Lorenzo) -- it sets those up instead.
    """
    prs = Presentation(str(TEMPLATE))
    drop_sample_slide(prs)

    # ---------------------------------------------------------------- item 2 (25 min)
    cover = add(prs, COVER, "FAIR2Adapt Case Study 1 Workshop")
    for ph in cover.placeholders:
        if ph.placeholder_format.idx == 1:
            ph.text = f"{meeting}\n{author}"

    add(
        prs,
        TEXT,
        "Why we are here",
        [
            "Objective: gather your feedback on the first visualisation prototype, and "
            "validate user stories US2-US5.",
            "We want to identify the questions that turn NorESM radioactivity simulations "
            "into decision-ready information.",
            "Those questions have a name: competency questions.",
        ],
    )

    add(
        prs,
        TEXT,
        "Agenda",
        [
            "Welcome — Torill (5 min)",
            "FAIR2Adapt and this workshop — Anne (25 min)",
            "Modelling radioactivity in the Arctic — Yanchun (20 min)",
            "Visualisation prototype: demo and testing — Anne (30 min)",
            "Break (15 min)",
            "User stories and competency questions — Barbara & Lorenzo (40 min)",
            "General discussion (20 min), closing remarks",
        ],
    )

    add_text_image(
        prs,
        "FAIR2Adapt — FAIR to Adapt to Climate Change",
        FIGURES / "divider_sst.png",
        subtitle="EU project 2025-2027 · 18 partners · Grant No. 101188256",
        body=[
            "Four objectives:",
            ("A methodological framework for FAIRifying adaptation data", 1),
            ("Identify useful data and improve its interoperability in EOSC", 1),
            ("Engage communities to bring in local perspectives", 1),
            ("Demonstrate the impact of FAIR data on adaptation", 1),
            "Six case studies — Arctic radionuclides (NERSC) is Case Study 1",
        ],
    )

    add(
        prs,
        TEXT,
        "Where we are, and what comes next",
        [
            "We are a little past mid-term of the three-year project.",
            "Done for Case Study 1: the data is FAIR, published, and readable by the dashboard.",
            "Next: automate the creation of FAIR Digital Objects — a data pipeline, "
            "so this is not hand-built each time",
            "Next: develop the dashboard beyond the prototype",
            "Both depend on what you tell us today. We do not want to automate the wrong thing.",
        ],
    )

    add_divider(
        prs,
        "Case Study 1",
        "From model output to something you can use",
        FIGURES / "divider_lahague.png",
    )

    add(
        prs,
        TEXT,
        "Where we started",
        [
            "Seven netCDF files of NorESM output, 6 GB, one variable each",
            "No coordinates — nothing said where on Earth the grid points were",
            "Five of the seven files had no title, licence or citation at all",
            "Opening one needed specialist software and a full download",
            "Usable by the modellers who made it. Not by anyone else.",
        ],
    )

    add_picture_slide(
        prs, "From model output to a FAIR Digital Object", FIGURES / "workflow.png"
    )

    add_picture_slide(
        prs,
        "What that makes possible",
        FIGURES / "arctic_sources.png",
        [
            "Three sources, one ocean — separable because each was tracked "
            "separately in the model."
        ],
    )

    add(
        prs,
        TEXT,
        "Competency questions",
        [
            "A competency question is a question the data should be able to answer.",
            "They are how we turn 'make it useful' into something we can build and test.",
            "Examples, against the user stories:",
            (
                "US2 — How has the Sellafield contribution in the Barents Sea changed since 2010?",
                1,
            ),
            (
                "US3 — Where is U-236 from La Hague highest in the Arctic in a given month?",
                1,
            ),
            (
                "US4 — Which source dominates contamination at the depths that matter for fisheries?",
                1,
            ),
            ("US5 — Is the trend consistent enough to support a recommendation?", 1),
        ],
    )

    add(
        prs,
        TEXT,
        "Today we are asking you two things",
        [
            "Does the prototype let you answer the questions you actually have?",
            "What can you not answer yet — and would need to?",
            "We will capture both as competency questions against user stories US2-US5.",
            "Nothing here is fixed. This is a prototype, and the gaps are the point.",
        ],
    )

    add(
        prs,
        TEXT,
        "The Figma board",
        [
            "We will collect your comments on a Figma board as we go:",
            (FIGMA_URL, 1),
            "Two tools, switch between them in the toolbar:",
            ("Hand — click and drag to move around; + / - to zoom", 1),
            ("Post-it or Text — click into a zone and type your comment", 1),
            "Remember to switch tools between moving and writing.",
            "Link is in the chat — no account needed.",
        ],
    )

    # ---------------------------------------------------------------- item 4 (30 min)
    add_divider(
        prs,
        "The prototype",
        "Demonstration, testing, and your questions",
        FIGURES / "divider_sellafield.png",
    )

    add(
        prs,
        TEXT,
        "What you are about to see",
        [
            "The FAIR2Adapt dashboard, reading the Case Study 1 data live",
            "Seven variables: temperature, salinity, and uranium-236 from three sources",
            ("Sellafield, La Hague, and global weapons-testing fallout", 1),
            "Monthly, 2010-2018, on 53 density layers",
            "Centred on the Arctic, because that is where you work",
        ],
    )

    add(
        prs,
        TEXT,
        "Demonstration",
        [
            "Sea surface temperature — orientation: the grid, the projection, the land mask",
            "U-236 from La Hague — the largest of the three sources",
            "U-236 from Sellafield — roughly half as much",
            "U-236 from global fallout — diffuse background, for contrast",
            "Stepping through time, 2010 to 2018",
        ],
    )

    add(
        prs,
        TEXT,
        "Against the user stories",
        [
            "US3 — plot maps of radioactive tracers: works today",
            "US2 — time series with selected parameters: not yet in the Arctic view",
            (
                "The chart needs a different data layout, which the dashboard cannot yet "
                "combine with the Arctic-centred projection",
                1,
            ),
            "US4 — interpretation: partly, by stepping through months",
            "US5 — recommendation report: depends on US2 and US4",
            "This is a customisation question, not a data question — and it is scopeable.",
        ],
    )

    add_picture_slide(
        prs,
        "This is what US2 would give you",
        FIGURES / "arctic_timeseries.png",
        [
            "Produced from the data today — but not yet interactively, "
            "and not inside the Arctic view."
        ],
    )

    add(
        prs,
        TEXT,
        "US2 in the tool — the RiOMAR case study",
        [
            "The dashboard can already do this. Here it is working on RiOMAR data:",
            (RIOMAR_DEMO, 1),
            "Same tool, same dashboard, charts and area selection working.",
            "What is missing for us is the combination — chart and Arctic projection together.",
            "So this is a customisation question, not a limitation of the data.",
        ],
    )

    add(
        prs,
        TEXT,
        "Now you try",
        [
            "Links are in the chat — open them and explore",
            "Can you find the variable, region and period you need?",
            "Is potential density a usable vertical coordinate for you, or do you need depth?",
            "What did you expect to be able to do, and could not?",
            f"Put it on the Figma board as you go — {FIGMA_URL}",
        ],
    )

    add(
        prs,
        TEXT,
        "What we would like from you",
        [
            "Questions you can answer with this today",
            "Questions you would want to answer and cannot yet",
            "Questions this data could never answer — so we look elsewhere",
            "These become the competency questions that drive what gets built next.",
        ],
    )

    add(
        prs,
        TEXT,
        "Where everything lives",
        [
            "Research Object: w3id.org/ro-id/b54df87a-515d-4e50-b212-48d4e4514cc9",
            "Pipeline code: github.com/FAIR2Adapt/arctic-radioisotopes",
            "Licence: CC-BY 4.0",
            "Data: Yanchun He (NERSC), NKS-B ANTHROPIC project (AFT/B(23)4)",
            "Made FAIR under FAIR2Adapt, EU grant 101188256",
        ],
    )

    back = add(prs, BACK)
    for ph in back.placeholders:
        if ph.placeholder_format.idx == 1:
            ph.text = "Thank you"

    prs.save(str(out))
    print(f"wrote {out}  ({len(prs.slides)} slides)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="Anne Fouilloux")
    ap.add_argument("--meeting", default="7 August 2026, 13:00-15:30 CEST · online")
    ap.add_argument(
        "--out", type=Path, default=Path("Arctic_Radioisotopes_FAIR2Adapt.pptx")
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=None,
        help="FAIR2Adapt .pptx template (or set $F2A_TEMPLATE)",
    )
    args = ap.parse_args()
    if args.template:
        globals()["TEMPLATE"] = args.template
    if not TEMPLATE.exists():
        raise SystemExit(
            f"template not found: {TEMPLATE}\nPass --template or set $F2A_TEMPLATE."
        )
    build(args.author, args.meeting, args.out)


if __name__ == "__main__":
    main()
