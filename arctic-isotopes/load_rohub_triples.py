#!/usr/bin/env python3
"""Load a TTL file into a ROHUB annotation as individual triples.

`ros_add_annotations(body_specification_file=...)` accepts a TTL and reports success, but
ROHUB does not extract the triples from it -- the annotation is created with zero triples.
Adding them one at a time via `ros_add_triple` is what actually populates the annotation,
and is how the existing FAIR2Adapt ROs are built.

    export ROHUB_USERNAME=... ROHUB_PASSWORD=...
    python arctic-isotopes/load_rohub_triples.py --ro-id <id> --ttl <file.ttl>

Creates a fresh annotation on the RO and fills it. Re-running creates another annotation
rather than duplicating triples in the existing one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rdflib import Graph, Literal

PLACEHOLDER = "https://w3id.org/ro-id/NORESM-RO"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ro-id", required=True)
    ap.add_argument("--ttl", required=True, type=Path)
    ap.add_argument(
        "--annotation-id", help="fill an existing annotation instead of creating one"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ttl = args.ttl.read_text().replace(
        PLACEHOLDER, f"https://w3id.org/ro-id/{args.ro_id}"
    )
    g = Graph()
    g.parse(data=ttl, format="turtle")
    triples = list(g)
    print(f"{args.ttl.name}: {len(triples)} triples")

    if args.dry_run:
        for s, p, o in triples[:5]:
            print(f"  {str(s)[-40:]:42} {str(p).rsplit('/', 1)[-1]:24} {str(o)[:50]}")
        print("  ...")
        return

    user, pwd = os.environ.get("ROHUB_USERNAME"), os.environ.get("ROHUB_PASSWORD")
    if not (user and pwd):
        sys.exit("Set ROHUB_USERNAME and ROHUB_PASSWORD.")

    import rohub

    rohub.login(username=user, password=pwd)

    ann_id = args.annotation_id
    if not ann_id:
        resp = rohub.ros_add_annotations(identifier=args.ro_id)
        ann_id = resp["identifier"] if isinstance(resp, dict) else str(resp)
        print(f"created annotation {ann_id}")

    ok = failed = 0
    for s, p, o in triples:
        obj_class = "Literal" if isinstance(o, Literal) else "URIRef"
        try:
            rohub.ros_add_triple(
                the_subject=str(s),
                the_predicate=str(p),
                the_object=str(o),
                annotation_id=ann_id,
                object_class=obj_class,
            )
            ok += 1
        except Exception as exc:  # keep going: one bad triple should not lose the rest
            failed += 1
            print(f"  FAILED {str(s)[-30:]} {str(p).rsplit('/', 1)[-1]}: {exc}")
    print(f"loaded {ok} triples, {failed} failed, into annotation {ann_id}")


if __name__ == "__main__":
    main()
