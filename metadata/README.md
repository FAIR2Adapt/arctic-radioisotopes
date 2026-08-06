# Nanopublications — FAIR2Adapt Case Study 1

Four AIDA-sentence nanopublications asserting what the FAIRification of the NorESM2 Arctic
radioisotope dataset achieved. They exist so the dashboard and RO-Crate can display
*contextual* claims alongside the data, in the same style as the RiOMAR RO-Crate.

| File | Claim |
|---|---|
| `01-source-attribution.trig` | Separately tagged U-236 tracers allow Arctic contamination to be attributed to global fallout, Sellafield or La Hague independently |
| `02-fairification.trig` | Cloud-optimised Zarr with CF-1.8 + ACDD-1.3 makes the data findable and reusable without whole-file downloads |
| `03-grid-recovery.trig` | The 2-D BLOM coordinates missing from the source netCDF were recovered and verified bit-identical |
| `04-healpix.trig` | HEALPix level 8 preserves the ~38 km Arctic resolution of the source grid |

Each is valid TriG with the four required graphs (`Head`, `assertion`, `provenance`,
`pubinfo`) and parses cleanly with rdflib. The assertion uses an
[AIDA sentence](http://purl.org/aida/) IRI — an Atomic, Independent, Declarative,
Absolute statement — which is the convention the FAIR2Adapt/Science Live nanopublications
already follow.

## Before publishing — two things need your input

1. **`orcid:ORCID-PLACEHOLDER`** appears in the provenance and pubinfo graphs of every
   file. Replace it with the real ORCID iD of whoever is asserting these claims. I did not
   guess one, because attribution of a signed scientific claim to the wrong person is not a
   defect worth risking.
2. **The claims are mine, drafted from what the work actually established.** Read them
   before signing: a nanopublication is a citable assertion under your name, not
   documentation.

Replace the placeholder in all four:

```bash
sed -i 's|orcid:ORCID-PLACEHOLDER|orcid:0000-0002-1234-5678|g' metadata/*.trig
```

## Publishing

The `TEMP` in `this:`/`sub:` prefixes is deliberate — trusty URIs are assigned at signing
time, when the content hash is computed.

```bash
pip install nanopub
np setup                       # one-time: creates an RSA keypair, links your ORCID
np sign metadata/01-source-attribution.trig
np publish metadata/01-source-attribution.signed.trig
```

Use the test server first (`np publish --test`) to confirm they render before writing to
the production network — published nanopublications are immutable.

## Linking back

Once published, each nanopublication gets a trusty URI
(`https://w3id.org/np/RA...`). Add them to the RO-Crate as resources so ROHUB lists them
alongside the dataset, matching how the RiOMAR RO exposes its AIDA claims:

```json
{ "@id": "https://w3id.org/np/RA...", "@type": "CreativeWork",
  "name": "AIDA Claim 1: ...", "encodingFormat": "application/trig" }
```

and reference them from the root dataset via `mentions` or `subjectOf`.
