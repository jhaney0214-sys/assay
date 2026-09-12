"""Build the HDI panel as HDI actually computes it: three normalised sub-indices.

    python tools/build_hdi_panel.py

A first version of this file fed the four raw indicators to a linear model and
called it HDI. It was not: the model tracked the published score at r = 0.967
with rank disagreements up to 47 places, so every statement made from it would
have been about the model rather than about HDI. That is the exact error Assay
exists to catch, committed by Assay's own tooling.

HDI is a geometric mean of three sub-indices, each normalised against fixed
goalposts:

    health    = (LE - 20) / (85 - 20)
    education = mean( EYS / 18 , MYS / 15 )
    income    = (ln GNI - ln 100) / (ln 75000 - ln 100)
    HDI       = (health * education * income) ^ (1/3)

So this writes the LOGS of those three sub-indices. Ranking by a weighted sum of
logs is exactly ranking by a weighted geometric mean, which means the linear
machinery in `aggregation.py` reproduces HDI's own arithmetic rather than
approximating it - and reweighting the columns is reweighting the exponents,
which is the real methodological choice UNDP made.
"""

import argparse
import csv
import io
import json
import math
import os
import re
import urllib.request

URL = ("https://hdr.undp.org/sites/default/files/2023-24_HDR/"
       "HDR23-24_Composite_indices_complete_time_series.csv")

COLS = ["log Health index", "log Education index", "log Income index"]
WEIGHTS = {c: 1 / 3 for c in COLS}
EPS = 1e-6


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit("could not decode the UNDP file")


def to_float(v):
    v = (v or "").strip()
    if v in ("", "..", "n/a", "NA"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def subindices(le, eys, mys, gni):
    """HDI's own normalisation, with UNDP's published goalposts.

    Every goalpost is a CAP as well as a scale. Omitting the GNI cap put
    Liechtenstein 0.031 above its published HDI and Qatar, Singapore and
    Ireland above theirs - the five richest countries, and only those, because
    they are the only ones past 75,000. A missing cap does not look like a bug;
    it looks like a handful of countries being slightly wrong.
    """
    health = (min(le, 85.0) - 20.0) / (85.0 - 20.0)
    education = (min(eys, 18.0) / 18.0 + min(mys, 15.0) / 15.0) / 2.0
    gni = min(gni, 75000.0)
    income = (math.log(gni) - math.log(100.0)) / (math.log(75000.0) - math.log(100.0))
    return health, education, income


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/raw/undp_hdr.csv")
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache = os.path.join(here, args.cache)

    if os.path.exists(cache):
        raw = open(cache, "rb").read()
    else:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        raw = urllib.request.urlopen(urllib.request.Request(
            URL, headers={"User-Agent": "Assay (research; urllib)"}), timeout=120).read()
        open(cache, "wb").write(raw)
    rows = list(csv.DictReader(io.StringIO(decode(raw))))

    years = sorted(int(m.group(1)) for c in rows[0]
                   for m in [re.match(r"^hdi_((?:19|20)\d\d)$", c)] if m)

    out, per_year, degenerate = [], {}, 0
    for r in rows:
        iso = (r.get("iso3") or "").strip()
        if len(iso) != 3 or not iso.isalpha():
            continue
        for y in years:
            le = to_float(r.get("le_%d" % y))
            eys = to_float(r.get("eys_%d" % y))
            mys = to_float(r.get("mys_%d" % y))
            gni = to_float(r.get("gnipc_%d" % y))
            hdi = to_float(r.get("hdi_%d" % y))
            if None in (le, eys, mys, gni, hdi) or gni <= 0:
                continue
            h, e, i = subindices(le, eys, mys, gni)
            if min(h, e, i) <= EPS:
                # A zero sub-index sends the geometric mean to zero and its log
                # to -inf. Rare, and dropped rather than clipped, because
                # clipping would invent a value the index does not have.
                degenerate += 1
                continue
            out.append([r["country"].strip(), y,
                        math.log(h), math.log(e), math.log(i), hdi])
            per_year[y] = per_year.get(y, 0) + 1

    path = os.path.join(here, "data", "hdi-panel.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "year"] + COLS + ["HDI"])
        w.writerows(sorted(out, key=lambda r: (r[1], r[0])))
    print("years %d-%d | %d country-years | %d..%d countries per year"
          % (years[0], years[-1], len(out),
             min(per_year.values()), max(per_year.values())))
    if degenerate:
        print("%d country-years dropped for a zero sub-index" % degenerate)
    print("wrote %s" % path)

    spec = {
        "name": "UNDP Human Development Index panel",
        "data": os.path.basename(path),
        "entity_col": "country",
        "year_col": "year",
        "score_cols": COLS,
        "published_col": "HDI",
        "weights": WEIGHTS,
        "claimed_dimensions": 3,
        "orient": False,
        "aggregation_form":
            "geometric mean; columns are LOGS of HDI's own normalised "
            "sub-indices, so a weighted sum of them is HDI's own arithmetic",
        "source": "UNDP Human Development Report 2023-24, " + URL,
    }
    with open(path.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print("wrote %s" % path.replace(".csv", ".json"))


if __name__ == "__main__":
    main()
