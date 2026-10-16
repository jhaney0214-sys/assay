"""Is there a newer edition of an index Assay covers than the one it used?

    python tools/editions.py

Re-running the rank-change test is cheap once you know a new edition is out;
finding out was the expensive part. This asks each publisher, with HEAD
requests only, whether the file a later edition would live at exists, and
whether the file Assay used has changed size since it was fetched. It never
downloads a data file. When it reports one, the commands to fetch it and
re-run the tests are printed with it, for you to run.

Three indices, with the edition each script uses read from that script, so
this cannot fall out of step with what was fetched:

  HDI   tools/fetch_undp.py, tools/build_hdi_panel.py   annual report
  EPI   tools/fetch_epi.py, tools/build_epi_panel.py    every two years
  SVI   tools/fetch_svi.py                              every two years

**What it cannot see.** A later edition is found by trying the URL it would
have if the publisher kept its naming pattern. A renamed file or a moved
directory is missed, so a "nothing newer" answer means "nothing at the
addresses tried", and the report lists them. A file revised in place at the
same URL (EPI published a corrigendum to its weights) is caught only by a
change in size against the local cache, and only for a file in the cache.

A probe that answers 200 with an HTML page is not counted as a find: some
servers answer a missing file with a page that says so.

Requests go one at a time, a second apart per host, with a User-Agent naming
the tool. robots.txt on hdr.undp.org and epi.yale.edu (read 2026-10-15)
disallows neither path; svi.cdc.gov serves no robots.txt.
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_epi_panel  # noqa: E402
import build_hdi_panel  # noqa: E402
import fetch_epi  # noqa: E402
import fetch_svi  # noqa: E402
import fetch_undp  # noqa: E402

USER_AGENT = "Assay edition check (research; urllib; HEAD requests only)"
CACHE = ROOT / "data" / "cache"
INTERVAL = 1.0
LOOK_AHEAD = 4  # editions this many years past the one used are tried

NEW, REVISED, CURRENT, GONE, UNKNOWN = (
    "NEW EDITION", "REVISED IN PLACE", "current", "USED FILE GONE", "could not tell")


# --------------------------------------------------------------- the indices

def hdi_edition(url):
    """(first year, label) from an HDR file URL: 2023-24_HDR -> (2023, '2023-24')."""
    m = re.search(r"/(\d{4})(?:-(\d{2}))?_HDR/", url)
    if not m:
        raise ValueError("not an HDR file URL: %s" % url)
    return int(m.group(1)), m.group(0)[1:-5]


def hdi_candidates(url, ahead=LOOK_AHEAD):
    """Where a later report's time-series file would be, under both of UNDP's
    naming patterns (a single year, 2025_HDR/HDR25_; or two, 2023-24_HDR/HDR23-24_)."""
    first, _label = hdi_edition(url)
    stem = "Composite_indices_complete_time_series.csv"
    base = url.split("/sites/default/files/")[0] + "/sites/default/files/"
    out = []
    for y in range(first + 1, first + 1 + ahead):
        yy, nn = y % 100, (y + 1) % 100
        out.append((str(y), "%s%d_HDR/HDR%02d_%s" % (base, y, yy, stem)))
        out.append(("%d-%02d" % (y, nn), "%s%d-%02d_HDR/HDR%02d-%02d_%s" % (base, y, nn, yy, nn, stem)))
    return out


def year_in(url, pattern):
    m = re.search(pattern, url)
    if not m:
        raise ValueError("no edition year in %s" % url)
    return int(m.group(1))


def epi_candidates(url, ahead=LOOK_AHEAD):
    used = year_in(url, r"epi(\d{4})results\.csv")
    return [(str(y), url.replace("epi%d" % used, "epi%d" % y))
            for y in range(used + 1, used + 1 + ahead)]


def epi_pages(url, ahead=LOOK_AHEAD):
    """Each EPI edition since 2026 has a downloads page, /<year>/downloads, and
    a year with none answers a real 404 (checked 2026-10-15: /2026 200, /2024
    and /2028 404). The page is the find; the file names inside it carry dates
    no probe can guess (epi2026results2026-07-07.xlsx)."""
    used = year_in(url, r"epi(\d{4})results")
    host = "/".join(url.split("/")[:3])
    return [(str(y), "%s/%d/downloads" % (host, y)) for y in range(used + 1, used + 1 + ahead)]


def svi_candidates(url, ahead=LOOK_AHEAD):
    used = year_in(url, r"/Data/(\d{4})/")
    return [(str(y), url.replace(str(used), str(y)))
            for y in range(used + 1, used + 1 + ahead)]


def indices():
    """Each index: the URL used, its edition, where a later one would be, and how to re-run."""
    return [
        {"key": "hdi", "name": "UNDP Human Development Index", "cache": "undp",
         "used": fetch_undp.URL, "edition": hdi_edition(fetch_undp.URL)[1],
         "also": {"tools/build_hdi_panel.py": build_hdi_panel.URL},
         "candidates": hdi_candidates(fetch_undp.URL),
         "rerun": ["set URL in tools/fetch_undp.py and tools/build_hdi_panel.py to the new file",
                   "py tools/build_hdi_panel.py",
                   "py ax.py drift data/hdi-panel.json --draws 600",
                   "py tools/fetch_undp.py --year <latest year in the file>"]},
        {"key": "epi", "name": "Yale Environmental Performance Index", "cache": "epi",
         "used": fetch_epi.RESULTS,
         "edition": str(year_in(fetch_epi.RESULTS, r"epi(\d{4})results")),
         "also": {"tools/build_epi_panel.py": build_epi_panel.RESULTS,
                  "weights (tools/fetch_epi.py)": fetch_epi.WEIGHTS},
         "candidates": epi_candidates(fetch_epi.RESULTS),
         "edition_pages": epi_pages(fetch_epi.RESULTS),
         "rerun": ["open the new edition's downloads page and pick the results and weights files",
                   "check their format: fetch_epi.py reads CSV, and the 2026 edition's results are .xlsx",
                   "set RESULTS and WEIGHTS in tools/fetch_epi.py and tools/build_epi_panel.py",
                   "read the new edition's errata first (2024 had a weights corrigendum)",
                   "py tools/build_epi_panel.py",
                   "py ax.py drift data/epi-panel.json --draws 800 --no-structure",
                   "py tools/fetch_epi.py"]},
        {"key": "svi", "name": "CDC/ATSDR Social Vulnerability Index", "cache": "svi",
         "used": fetch_svi.URL, "edition": str(fetch_svi.YEAR),
         "also": {},
         "candidates": svi_candidates(fetch_svi.URL),
         "rerun": ["set YEAR in tools/fetch_svi.py",
                   "py tools/fetch_svi.py",
                   "py ax.py audit data/svi-<year>.json --draws 2000",
                   "two SVI editions would allow a county rank-change panel; none is built yet"]},
    ]


# --------------------------------------------------------------- asking

class _KeepHead(urllib.request.HTTPRedirectHandler):
    """urllib turns a redirected HEAD into a GET, which would download the
    file the redirect points at. This keeps it a HEAD."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.method = req.get_method()
        return new


def head(url, timeout=30):
    """(status, headers) for a HEAD request. Never reads a body."""
    opener = urllib.request.build_opener(_KeepHead)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {})


class Prober:
    """Calls `ask(url) -> (status, headers)` one at a time, `interval` apart per host."""

    def __init__(self, ask=head, interval=INTERVAL, sleep=time.sleep, clock=time.time):
        self.ask, self.interval, self.sleep, self.clock = ask, interval, sleep, clock
        self.last = {}
        self.asked = []

    def __call__(self, url):
        host = urllib.parse.urlsplit(url).netloc
        wait = self.interval - (self.clock() - self.last.get(host, -1e9))
        if wait > 0:
            self.sleep(wait)
        self.last[host] = self.clock()
        self.asked.append(url)
        try:
            status, headers = self.ask(url)
        except (urllib.error.URLError, OSError) as exc:
            return {"status": None, "error": "%s: %s" % (type(exc).__name__, exc)}
        headers = {k.lower(): v for k, v in headers.items()}
        length = headers.get("content-length")
        return {"status": status,
                "type": headers.get("content-type", ""),
                "bytes": int(length) if length and length.isdigit() else None,
                "modified": headers.get("last-modified")}


def is_file(answer):
    """A data file answered: 200, and not an HTML page standing in for a 404."""
    return answer.get("status") == 200 and "html" not in answer.get("type", "").lower()


def cached_bytes(cache_dir, source, url):
    # netcache's layout: <cache>/<source>/<sha1 of the URL>.meta.json
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    meta = pathlib.Path(cache_dir) / source / (key + ".meta.json")
    try:
        return json.loads(meta.read_text(encoding="utf-8"))["bytes"]
    except (OSError, ValueError, KeyError):
        return None


def assess(index, probe, cache_dir=CACHE):
    """What the publisher says about one index. Returns a dict the report prints."""
    used = probe(index["used"])
    found, pages = [], []
    for label, url in index["candidates"]:
        answer = probe(url)
        if is_file(answer):
            found.append((label, url, answer))
        elif answer.get("status") == 200:
            pages.append((label, url, answer))
    for label, url in index.get("edition_pages", []):
        answer = probe(url)
        if answer.get("status") == 200:
            found.append((label + " page", url, answer))
    cached = cached_bytes(cache_dir, index["cache"], index["used"])
    if found:
        verdict = NEW
    elif used.get("status") in (404, 410) or (used.get("status") == 200 and not is_file(used)):
        # An HTML page at the file's address is the file gone: EPI's 2024
        # results URL answers with the site's downloads page since 2026.
        verdict = GONE
    elif not is_file(used):
        verdict = UNKNOWN
    elif cached is not None and used.get("bytes") is not None and used["bytes"] != cached:
        verdict = REVISED
    else:
        verdict = CURRENT
    return {"index": index, "verdict": verdict, "used": used, "cached": cached,
            "found": found, "pages": pages, "mismatch": {k: v for k, v in index["also"].items()
                                         if k.startswith("tools/") and v != index["used"]}}


# --------------------------------------------------------------- report

def _answer(a, page=False):
    if a.get("status") is None:
        return a.get("error", "no answer")
    parts = ["HTTP %s" % a["status"]]
    if a.get("bytes") is not None:
        parts.append("{:,} bytes".format(a["bytes"]))
    if a.get("modified"):
        parts.append("modified " + a["modified"])
    if a.get("status") == 200 and not is_file(a):
        parts.append("the edition's page" if page else "an HTML page, not a file")
    return ", ".join(parts)


def report(results, today):
    out = ["Edition check, %s. HEAD requests only; nothing was downloaded." % today, ""]
    for r in results:
        ix = r["index"]
        out.append("%s (%s): %s" % (ix["name"], ix["key"].upper(), r["verdict"]))
        out.append("  used     %-8s %s" % (ix["edition"], ix["used"]))
        cached = "" if r["cached"] is None else "; cached copy {:,} bytes".format(r["cached"])
        out.append("           %s%s" % (_answer(r["used"]), cached))
        for script, url in sorted(r["mismatch"].items()):
            out.append("  WARNING  %s uses a different file: %s" % (script, url))
        for label, url, answer in r["found"]:
            out.append("  newer    %-8s %s" % (label, url))
            out.append("           %s" % _answer(answer, page=label.endswith(" page")))
        for label, url, answer in r["pages"]:
            out.append("  ignored  %-8s %s" % (label, url))
            out.append("           %s" % _answer(answer))
        if r["verdict"] in (NEW, REVISED):
            out.append("  to re-run (fetching the file is your call):")
            out += ["    " + step for step in ix["rerun"]]
        else:
            tried = ix["candidates"] + ix.get("edition_pages", [])
            out.append("  tried %d later addresses: %s" % (
                len(tried), ", ".join(label for label, _ in tried)))
        out.append("")
    news = [r["index"]["key"].upper() for r in results if r["verdict"] == NEW]
    out.append("New editions: %s." % (", ".join(news) if news else "none found"))
    return "\n".join(out)


def main(argv=None, probe=None, cache_dir=CACHE, today=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["hdi", "epi", "svi"], action="append",
                    help="check only this index (repeatable)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    probe = probe or Prober()
    chosen = [ix for ix in indices() if not args.only or ix["key"] in args.only]
    results = [assess(ix, probe, cache_dir) for ix in chosen]
    print(report(results, today or time.strftime("%Y-%m-%d")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
