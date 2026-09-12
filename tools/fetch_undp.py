"""Build an HDI indicator table from UNDP's published composite-indices file.

    python tools/fetch_undp.py --year 2022

UNDP publishes the whole Human Development Report time series as one open CSV -
no key, no registration, no form. This pulls the four indicators HDI is built
from, for one year, plus HDI itself.

**Why HDI is a fair test where V-Dem was not.** Its four indicators are measured
by four different institutions from four unrelated instruments: life expectancy
from civil registration and UN population estimates, expected and mean years of
schooling from education ministries and household surveys, GNI per capita from
national accounts. None of them contains any of the others. If they turn out to
be one dimension, that is a fact about countries rather than about a formula.

**What is still not clean.** The two schooling indicators are combined into one
education index inside HDI, so they are two measurements of one declared
dimension rather than two dimensions - the audit should find them close, and
that is HDI working as documented, not a defect. And HDI is a geometric mean, so
the linear reproduction test is disabled by the spec's `aggregation_form`.
"""

import argparse
import csv
import io
import math
import os
import urllib.request

URL = ("https://hdr.undp.org/sites/default/files/2023-24_HDR/"
       "HDR23-24_Composite_indices_complete_time_series.csv")

# The four indicators HDI aggregates, and what they measure.
INDICATORS = {
    "le":    "Life expectancy at birth",
    "eys":   "Expected years of schooling",
    "mys":   "Mean years of schooling",
    "gnipc": "GNI per capita (log)",
}


def decode(raw):
    """Decode strictly, trying each candidate, never with errors='replace'.

    An earlier version decoded with errors='replace' and cached the result.
    That silently destroyed exactly two country names - Cote d'Ivoire and
    Turkiye, the only two carrying non-ASCII - replacing their accented
    characters with U+FFFD. Nothing failed; the file merely became wrong, and
    it surfaced two steps later as an unrelated-looking print error.

    'replace' turns a decoding question into corrupted data that travels. Try
    each encoding strictly and say which one worked.
    """
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit("could not decode the UNDP file in any expected encoding")


def to_float(v):
    v = (v or "").strip()
    if v in ("", "..", "n/a", "NA"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--cache", default="data/raw/undp_hdr.csv")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache = os.path.join(here, args.cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if os.path.exists(cache):
        raw = open(cache, "rb").read()
        print("using cached %s" % cache)
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "Assay (research; urllib)"})
        raw = urllib.request.urlopen(req, timeout=120).read()
        open(cache, "wb").write(raw)          # bytes, undecoded - see below
        print("downloaded %s" % cache)
    body = decode(raw)

    rows = list(csv.DictReader(io.StringIO(body)))
    cols = {k: "%s_%d" % (k, args.year) for k in INDICATORS}
    hdi_col = "hdi_%d" % args.year
    have = rows[0]
    for c in list(cols.values()) + [hdi_col]:
        if c not in have:
            raise SystemExit("column %s not in file - try another --year" % c)

    out_rows, dropped = [], 0
    for r in rows:
        # UNDP appends eleven aggregate rows to the country list, and they do
        # NOT carry a blank iso3 - they carry pseudo-codes beginning "ZZ":
        # regions (ZZE.AS, ZZK.WORLD) and, worse, HDI groups (ZZA.VHHD, "Very
        # high human development"). The group rows are defined BY the index
        # under audit, so leaving them in would inflate every correlation with
        # rows that are guaranteed to be extreme on it. A first version of this
        # file filtered on blank iso3 and let all eleven through.
        iso = (r.get("iso3") or "").strip()
        if len(iso) != 3 or not iso.isalpha():
            dropped += 1
            continue
        vals = [to_float(r[cols[k]]) for k in INDICATORS]
        # HDI's income index is built from ln(GNI), not GNI. Auditing the index
        # on its own terms means applying the transform the index applies:
        # raw GNI is heavily right-skewed, and correlating it untransformed
        # understates its relationship to the other three - which would have
        # flattered the audit's own conclusion.
        gi = list(INDICATORS).index("gnipc")
        if vals[gi] is not None and vals[gi] > 0:
            vals[gi] = math.log(vals[gi])
        elif vals[gi] is not None:
            vals[gi] = None
        hdi = to_float(r[hdi_col])
        if any(v is None for v in vals) or hdi is None:
            dropped += 1
            continue
        out_rows.append([r["country"].strip()] + vals + [hdi])

    print("%d countries complete in %d (%d rows dropped)"
          % (len(out_rows), args.year, dropped))
    if len(out_rows) < 50:
        raise SystemExit("too few complete countries; try another --year")

    out_csv = os.path.join(here, "data", "hdi-%d.csv" % args.year)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country"] + list(INDICATORS.values()) + ["HDI"])
        w.writerows(sorted(out_rows))
    print("wrote %s" % out_csv)

    spec = {
        "name": "UNDP Human Development Index %d" % args.year,
        "data": os.path.basename(out_csv),
        "entity_col": "country",
        "score_cols": list(INDICATORS.values()),
        "published_col": "HDI",
        "claimed_dimensions": 3,
        "orient": True,
        "aggregation_form":
            "geometric mean of three sub-indices, education being the mean of "
            "the two schooling indicators - not a weighted sum",
        "source": "UNDP Human Development Report 2023-24 composite indices, " + URL,
        "note": (
            "HDI declares three dimensions - a long and healthy life, knowledge, "
            "and a decent standard of living - built from four indicators. The "
            "two schooling indicators belong to one declared dimension, so four "
            "columns against three claimed dimensions is the documented design "
            "and not itself a finding."
        ),
    }
    import json
    with open(out_csv.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print("wrote %s" % out_csv.replace(".csv", ".json"))


if __name__ == "__main__":
    main()
