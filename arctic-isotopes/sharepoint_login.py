#!/usr/bin/env python3
"""Authenticate to the FAIR2Adapt SharePoint share on this VM, via email one-time passcode.

The share is only reachable as a B2B guest whose address has no Microsoft account, so the
passcode flow is browser-only. Rather than borrowing a cookie from a laptop browser, this
drives a headless Chromium *here* — the resulting session lives on the VM, so the netCDF
data transfers SharePoint -> VM directly.

The passcode arrives in the user's inbox mid-flow, so the script parks at the code prompt
and polls .auth/otp.txt. Write the six digits there and it resumes:

    python arctic-isotopes/sharepoint_login.py --email you@example.org
    # in another shell, once the email arrives:
    echo 123456 > .auth/otp.txt

On success it writes the session cookie to .auth/cookie.txt, which fetch_sharepoint.py
reads by default. The cookie is valid for hours only; a new passcode is needed after that.
Screenshots of every step land in .auth/shots/ for when Microsoft changes its login pages.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SHARE_URL = (
    "https://nerscno-my.sharepoint.com/personal/torill_hamre_nersc_no/"
    "Documents/FAIR2Adapt/Shared"
)
# Everything auth-related stays under .auth/, which .gitignore excludes — the cookie is a
# live credential and the browser profile holds the signed-in session.
AUTH = Path(".auth")
OTP_FILE = AUTH / "otp.txt"
COOKIE_FILE = AUTH / "cookie.txt"
SHOTS = AUTH / "shots"
PROFILE = AUTH / "chromium-profile"

# Microsoft's login pages are a state machine; identify each state by a selector and
# react, rather than assuming a fixed page order (it varies for guest/OTP accounts).
# The guest flow federates to login.live.com, which renders <button>Send code</button>
# rather than the input[value=...] that login.microsoftonline.com uses, so match on text.
EMAIL_BOX = "input[name='loginfmt'], input[type='email']"
NEXT_BTN = "input[type='submit'], button[type='submit']"
SEND_CODE = (
    "button:has-text('Send code'), input[value='Send code'], "
    "button:has-text('Send a code'), #idSIButton9"
)
# The current MSA code page renders six single-character boxes rather than one field.
OTP_BOX = (
    "input[name='otc'], #iOttText, #idTxtBx_OTC_Password, "
    "input[autocomplete='one-time-code'], input[maxlength='1']"
)
STAY_SIGNED_IN = "#idSIButton9, button:has-text('Yes'), input[value='Yes']"


def shot(page, name: str) -> Path:
    SHOTS.mkdir(parents=True, exist_ok=True)
    p = SHOTS / f"{name}.png"
    page.screenshot(path=str(p), full_page=False)
    print(f"  [screenshot] {p}")
    return p


def visible(page, selector: str) -> bool:
    try:
        return page.locator(selector).first.is_visible(timeout=1500)
    except Exception:
        return False


def wait_for_otp(timeout_s: int = 900, accept_existing: bool = False) -> str:
    """Block until the passcode is written to OTP_FILE.

    A code already on disk is only trusted when this run did not click 'Send code' —
    requesting a new one invalidates the previous, so a stale file would fail the login.
    """
    if not accept_existing:
        OTP_FILE.unlink(missing_ok=True)
    elif OTP_FILE.exists():
        print(f">>> Using the passcode already staged in {OTP_FILE}", flush=True)
    print(f"\n>>> Waiting for the emailed code. Write it to: {OTP_FILE}\n", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if OTP_FILE.exists():
            code = "".join(c for c in OTP_FILE.read_text() if c.isdigit())
            if len(code) >= 6:
                print(f">>> Got a {len(code)}-digit code, continuing.", flush=True)
                return code
        time.sleep(3)
    sys.exit("Timed out waiting for the passcode.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument(
        "--headed", action="store_true", help="run with a visible browser (needs X)"
    )
    ap.add_argument("--url", default=SHARE_URL)
    args = ap.parse_args()

    AUTH.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(PROFILE),
            headless=not args.headed,
            args=["--disable-blink-features=AutomationControlled"],
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1400, "height": 950},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.set_default_timeout(45_000)

        print(f"Opening {args.url}")
        page.goto(args.url, wait_until="domcontentloaded")
        shot(page, "01-landing")

        submitted_code = False
        requested_code = False
        for step in range(2, 25):
            time.sleep(2)
            host = page.url.split("/")[2] if "//" in page.url else page.url

            # Reached SharePoint without a login prompt -> already authenticated.
            if "sharepoint.com" in host and not visible(page, EMAIL_BOX):
                if not visible(page, OTP_BOX):
                    break

            if visible(page, EMAIL_BOX):
                print(f"[{step}] entering email")
                page.fill(EMAIL_BOX, args.email)
                shot(page, f"{step:02d}-email")
                page.click(NEXT_BTN)
                continue

            if visible(page, OTP_BOX):
                if submitted_code:
                    shot(page, f"{step:02d}-after-code")
                    print(
                        "[!] still on the code page - it may have been wrong or expired"
                    )
                    sys.exit(
                        "Passcode rejected. Rerun and use a freshly requested code."
                    )
                shot(page, f"{step:02d}-code-prompt")
                code = wait_for_otp(accept_existing=not requested_code)
                digit_boxes = page.locator("input[maxlength='1']")
                if digit_boxes.count() >= 6:
                    # Six separate boxes that auto-advance: focus the first and type.
                    digit_boxes.first.click()
                    page.keyboard.type(code, delay=90)
                else:
                    page.fill(OTP_BOX, code)
                time.sleep(2)
                if visible(page, NEXT_BTN):
                    page.click(NEXT_BTN)  # some variants auto-submit on the sixth digit
                submitted_code = True
                continue

            if visible(page, SEND_CODE):
                print(f"[{step}] requesting a passcode")
                requested_code = True
                shot(page, f"{step:02d}-send-code")
                page.click(SEND_CODE)
                continue

            shot(page, f"{step:02d}-unknown")
            # Print the clickable labels so a selector mismatch diagnoses itself rather
            # than needing a screenshot round-trip.
            try:
                labels = page.eval_on_selector_all(
                    "button, input[type='submit'], a[role='button'], h1, h2",
                    "els => els.filter(e => e.offsetParent)"
                    ".map(e => (e.innerText || e.value || '').trim()).filter(Boolean)",
                )
            except Exception:
                labels = []
            print(f"[{step}] unrecognised page, host={page.url.split('/')[2]}")
            print(f"      visible controls: {labels[:12]}")
            time.sleep(3)
        else:
            shot(page, "99-gave-up")
            sys.exit("Could not complete login; inspect the screenshots.")

        # 'Stay signed in?' keeps FedAuth alive far longer, which matters for big transfers.
        if visible(page, STAY_SIGNED_IN):
            page.click(STAY_SIGNED_IN)
            time.sleep(3)

        page.goto(args.url, wait_until="domcontentloaded")
        time.sleep(3)
        shot(page, "90-final")
        print(f"Final URL: {page.url}")

        cookies = ctx.cookies()
        names = {c["name"] for c in cookies}
        if not ({"FedAuth", "rtFa"} & names):
            sys.exit(
                f"Logged in but no FedAuth/rtFa cookie found. Saw: {sorted(names)}"
            )

        header = "; ".join(
            f"{c['name']}={c['value']}"
            for c in cookies
            if "sharepoint.com" in c["domain"]
        )
        COOKIE_FILE.write_text(header)
        COOKIE_FILE.chmod(0o600)
        print(f"\nSession saved to {COOKIE_FILE} ({len(header)} bytes)")
        print("Next: python arctic-isotopes/fetch_sharepoint.py --dry-run")
        ctx.close()


if __name__ == "__main__":
    main()
