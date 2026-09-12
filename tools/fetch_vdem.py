"""Build a V-Dem index table from Our World in Data's per-index CSVs.

    python tools/fetch_vdem.py --year 2024

Five separate CSVs, joined on country and year. No key, no registration, no
third-party client - urllib against a public endpoint, the same shape as
Farewatch's and Greenlight's fetchers.

**What this table can and cannot be used to claim.** V-Dem's liberal,
participatory, deliberative and egalitarian indices each incorporate its
electoral democracy index by construction. They therefore correlate for an
arithmetic reason, and any audit of this table is auditing a formula rather than
the world. The spec written here records that, and `assay/report.py` refuses to
report the resulting collapse as a finding.

It is kept as the worked example precisely because it is the trap: it is exactly
what a careless audit of a famous index looks like, and the tool has to survive
being pointed at it.
"""

import argparse
import csv
import json
import os
import urllib.request

BASE = "https://ourworldindata.org/grapher/%s.csv"

SLUGS = {
    "electoral-democracy-index": "Electoral democracy index",
    "liberal-democracy-index": "Liberal democracy index",
    "participatory-democracy-index": "Participatory democracy index",
    "deliberative-democracy-index-vdem": "Deliberative democracy index",
    "egalitarian-democracy-index-vdem": "Egalitarian democracy index",
}

# OWID mixes real countries with continent and income aggregates in the same
# column. An aggregate has no ISO3 code, or one of the OWID_ pseudo-codes.
def is_country(row):
    code = (row.get("Code") or "").strip()
    return bool(code) and not code.startswith("OWID_")


def fetch(slug):
    req = urllib.request.Request(BASE % slug,
                                 headers={"User-Agent": "Assay (research; urllib)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8", "replace")
    return list(csv.DictReader(body.splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_csv = args.out or os.path.join(here, "data", "vdem-%d.csv" % args.year)
    out_spec = out_csv.replace(".csv", ".json")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    tables = {}
    for slug, col in SLUGS.items():
        rows = fetch(slug)
        got = {}
        for r in rows:
            if not is_country(r):
                continue
            try:
                if int(r["Year"]) != args.year:
                    continue
            except (KeyError, ValueError):
                continue
            val = (r.get(col) or "").strip()
            if val:
                got[r["Entity"].strip()] = val
        tables[col] = got
        print("  %-34s %4d countries in %d" % (slug, len(got), args.year))

    cols = list(SLUGS.values())
    common = set.intersection(*(set(t) for t in tables.values()))
    print("\n%d countries with all five indices in %d" % (len(common), args.year))
    if len(common) < 30:
        raise SystemExit(
            "only %d countries complete - try an earlier --year; OWID lags "
            "V-Dem's release by a cycle." % len(common))

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["country"] + cols)
        for country in sorted(common):
            wr.writerow([country] + [tables[c][country] for c in cols])
    print("wrote %s" % out_csv)

    spec = {
        "name": "V-Dem democracy indices %d" % args.year,
        "data": os.path.basename(out_csv),
        "entity_col": "country",
        "score_cols": cols,
        "claimed_dimensions": 5,
        "orient": True,
        "source": "V-Dem via Our World in Data, %s" % (BASE % "<index>"),
        "shared_construction": {
            "columns": cols[1:],
            "note": (
                "V-Dem's liberal, participatory, deliberative and egalitarian\n"
                "indices each INCORPORATE the electoral democracy index by\n"
                "construction. Four of these five columns contain the fifth.\n"
                "Their correlation is therefore a property of the formula, not\n"
                "a measurement, and this table cannot support a claim that the\n"
                "five varieties of democracy are really one. Testing that needs\n"
                "the component indices with the shared electoral term removed."
            ),
        },
    }
    with open(out_spec, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print("wrote %s" % out_spec)


if __name__ == "__main__":
    main()
