#!/usr/bin/env python3
"""Recursively download a SharePoint / OneDrive folder using a browser session cookie.

Needed because the FAIR2Adapt share is reachable only as an email-OTP guest: there is
no Microsoft account behind the address, so no OAuth token can be issued (rclone, Graph
and the anonymous /shares API are all dead ends). The passcode authenticates a *browser
session*, so we borrow that session's cookie and talk to the SharePoint REST API.

Usage:
    python arctic-isotopes/sharepoint_login.py --email you@example.org   # writes .auth/cookie.txt
    python arctic-isotopes/fetch_sharepoint.py --dry-run                 # list without downloading
    python arctic-isotopes/fetch_sharepoint.py                           # download to data/raw

--curl also accepts a DevTools 'Copy as cURL' dump, though Chrome now redacts the Cookie
header from those, so the login script is the reliable source.

Reruns are safe: files already present at the right size are skipped, so an expired
session mid-transfer just means re-running the login and this script again.

Depends only on the standard library, so it runs before the project environment is set up.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SITE = "https://nerscno-my.sharepoint.com/personal/torill_hamre_nersc_no"
ROOT = "/personal/torill_hamre_nersc_no/Documents/FAIR2Adapt/Shared"


def cookie_from_curl(path: Path) -> str:
    """Pull the Cookie header out of a DevTools 'Copy as cURL' dump.

    Handles bash (-H 'Cookie: ...' / -b '...') and cmd/PowerShell (double-quoted) forms.
    """
    text = path.read_text(errors="replace")
    patterns = [
        r"-H\s+'[Cc]ookie:\s*(.*?)'",
        r'-H\s+"[Cc]ookie:\s*(.*?)"',
        r"-b\s+'(.*?)'",
        r'-b\s+"(.*?)"',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.S)
        if m:
            return m.group(1).strip()

    # Chrome omits cookies when copying service-worker navigations, so also accept a
    # bare 'FedAuth=...; rtFa=...' line pasted straight from the Application tab.
    bare = " ".join(text.split())
    if re.search(r"\b(FedAuth|rtFa)=", bare):
        return (
            bare[len("cookie:") :].strip()
            if bare.lower().startswith("cookie:")
            else bare
        )

    sys.exit(
        f"No cookie found in {path}.\n"
        "Either paste a 'Copy as cURL' of an XHR/Fetch request to "
        "nerscno-my.sharepoint.com, or paste the FedAuth and rtFa values directly as:\n"
        "  FedAuth=<value>; rtFa=<value>"
    )


def api(cookie: str, url: str, raw: bool = False):
    """Call the SharePoint REST API. Returns parsed JSON, or the raw response if raw."""
    req = urllib.request.Request(url)
    req.add_header("Cookie", cookie)
    req.add_header("User-Agent", "Mozilla/5.0")
    if not raw:
        req.add_header("Accept", "application/json;odata=nometadata")
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            sys.exit(
                f"HTTP {e.code} — the session cookie is expired or not authorised.\n"
                "Re-open the folder in your browser, redo 'Copy as cURL', and rerun."
            )
        body = e.read()[:400].decode(errors="replace")
        sys.exit(f"HTTP {e.code} for {url}\n{body}")
    return resp if raw else json.load(resp)


def quote_sp(server_relative: str) -> str:
    """Quote a server-relative path for embedding in a REST method argument."""
    return urllib.parse.quote(server_relative.replace("'", "''"), safe="/")


def walk(cookie: str, folder: str):
    """Yield (server_relative_path, size) for every file under folder, recursively."""
    enc = quote_sp(folder)
    files = api(cookie, f"{SITE}/_api/web/GetFolderByServerRelativeUrl('{enc}')/Files")
    for f in files.get("value", []):
        yield f["ServerRelativeUrl"], int(f.get("Length", 0))

    folders = api(
        cookie, f"{SITE}/_api/web/GetFolderByServerRelativeUrl('{enc}')/Folders"
    )
    for sub in folders.get("value", []):
        # SharePoint exposes internal 'Forms' folders on every library; skip them.
        if sub["Name"] == "Forms":
            continue
        yield from walk(cookie, sub["ServerRelativeUrl"])


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def download(cookie: str, remote: str, dest: Path, size: int) -> str:
    """Download one file. Returns 'skip' if a complete copy is already on disk."""
    rel = remote[len(ROOT) :].lstrip("/")
    out = dest / rel
    if out.exists() and size and out.stat().st_size == size:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)

    enc = quote_sp(remote)
    resp = api(
        cookie, f"{SITE}/_api/web/GetFileByServerRelativeUrl('{enc}')/$value", raw=True
    )
    tmp = out.with_suffix(out.suffix + ".part")
    got = 0
    with open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            got += len(chunk)
            if size:
                pct = 100 * got / size
                print(
                    f"\r  {rel}  {human(got)}/{human(size)} ({pct:.0f}%)",
                    end="",
                    flush=True,
                )
    print()
    if size and got != size:
        tmp.unlink(missing_ok=True)
        sys.exit(f"Short read for {rel}: got {got} of {size} bytes. Rerun to retry.")
    tmp.rename(out)
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--curl",
        type=Path,
        default=Path(".auth/cookie.txt"),
        help="file holding the session cookie (default: .auth/cookie.txt from the login script)",
    )
    p.add_argument("--dest", default=Path("data/raw"), type=Path)
    p.add_argument("--folder", default=ROOT, help="server-relative folder to fetch")
    p.add_argument(
        "--dry-run", action="store_true", help="list files without downloading"
    )
    args = p.parse_args()

    if not args.curl.exists():
        sys.exit(
            f"No session at {args.curl}. Run first:\n"
            "  python arctic-isotopes/sharepoint_login.py --email <your address>"
        )
    cookie = cookie_from_curl(args.curl)
    print(f"Listing {args.folder} ...")
    items = list(walk(cookie, args.folder))
    if not items:
        sys.exit("No files found. Check --folder matches the shared path.")

    total = sum(s for _, s in items)
    print(f"{len(items)} file(s), {human(total)} total\n")
    for remote, size in items:
        print(f"  {remote[len(ROOT):].lstrip('/')}  ({human(size)})")

    if args.dry_run:
        return

    print()
    counts = {"ok": 0, "skip": 0}
    for remote, size in items:
        counts[download(cookie, remote, args.dest, size)] += 1
    print(
        f"\nDone: {counts['ok']} downloaded, {counts['skip']} already present -> {args.dest}"
    )


if __name__ == "__main__":
    main()
