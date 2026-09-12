"""Build EPI's two-period panel: Yale's own backcast baseline against current.

    python tools/build_epi_panel.py

EPI publishes every score twice in one file - `.old` is a BACKCAST, the score a
country would have had on data from about ten years earlier run through the
*current* methodology, and `.new` is the current score. Yale computes both, and
the difference is what its own reports call the ten-year change in performance.
That makes the two comparable in a way that two separate EPI releases are not:
comparing the 2016 and 2024 reports would confound real change with a changed
method, which is precisely why the backcast exists.

**Three policy objectives, not eleven issue categories.** The single-year audit
uses the eleven, because that is where the dimensional question lives. The drift
analysis uses the three, for two reasons found by checking rather than assumed:

  * Every country has all three, so nothing is dropped. At issue-category level
    75 of 180 countries fall out, and not at random - Forests is missing for 49
    and Fisheries for 39, so the survivors are the ones with coastlines and
    forests.
  * The three reproduce the published EPI far more closely (a country lands
    within 1.2 points rather than 5.6), though still not exactly.

The 45/25/30 split across Ecosystem Vitality, Environmental Health and Climate
Change is Yale's own published top-level weighting, and it is the number a
reweighting test should perturb: it is the choice a committee actually made.
"""

import argparse
import csv
import io
import json
import os
import urllib.request

RESULTS = "https://epi.yale.edu/downloads/epi2024results.csv"
WEIGHTS = "https://epi.yale.edu/downloads/epi2024weights.csv"
UA = {"User-Agent": "Mozilla/5.0 (compatible; Assay research; urllib)"}

# The backcast is "approximately ten years prior" (Yale's wording), so these
# labels are nominal periods, not exact years. They exist to give the panel two
# ordered points; nothing downstream reads them as dates.
BASELINE_YEAR, CURRENT_YEAR = 2014, 2024


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit("could not decode an EPI file")


def grab(url, cache):
    if os.path.exists(cache):
        return decode(open(cache, "rb").read())
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    open(cache, "wb").write(raw)
    return decode(raw)


def to_float(v):
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    res = list(csv.DictReader(io.StringIO(
        grab(RESULTS, os.path.join(here, "data/raw/epi2024results.csv")))))
    wts = list(csv.DictReader(io.StringIO(
        grab(WEIGHTS, os.path.join(here, "data/raw/epi2024weights.csv")))))

    po = [r for r in wts if (r.get("Type") or "").strip() == "PolicyObjective"]
    cols, weights = [], {}
    for r in po:
        ab = r["Abbreviation"].strip()
        label = "%s (%s)" % (r["Variable"].strip(), ab)
        cols.append((ab, label))
        weights[label] = float(r["Weight"])
    print("policy objectives: " + ", ".join("%s=%.2f" % (l, weights[l]) for _, l in cols))

    rows, dropped = [], 0
    for r in res:
        name = (r.get("country") or "").strip()
        if not name:
            continue
        rec = {}
        for suffix, year in ((".old", BASELINE_YEAR), (".new", CURRENT_YEAR)):
            vals = [to_float(r.get(ab + suffix)) for ab, _ in cols]
            epi = to_float(r.get("EPI" + suffix))
            if epi is None or any(v is None for v in vals):
                rec = None
                break
            rec[year] = vals + [epi]
        if rec is None:
            dropped += 1
            continue
        for year, vals in rec.items():
            rows.append([name, year] + vals)

    print("%d countries with both periods complete (%d dropped)"
          % (len(rows) // 2, dropped))

    path = os.path.join(here, "data", "epi-panel.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "year"] + [l for _, l in cols] + ["EPI"])
        w.writerows(sorted(rows, key=lambda r: (r[1], r[0])))
    print("wrote %s" % path)

    spec = {
        "name": "Yale EPI 2024 - backcast baseline vs current",
        "data": os.path.basename(path),
        "entity_col": "country",
        "year_col": "year",
        "score_cols": [l for _, l in cols],
        "published_col": "EPI",
        "weights": weights,
        "claimed_dimensions": 3,
        "aggregation_form": "linear",
        "orient": False,
        "source": "Yale EPI 2024, %s and %s" % (RESULTS, WEIGHTS),
        "note": (
            "Periods are labelled %d and %d but the baseline is Yale's backcast "
            "on data 'approximately ten years prior', not a calendar year. The "
            "gap is ONE ten-year change, not a series of annual ones, so it is "
            "not comparable to HDI's year-over-year pairs without saying so."
            % (BASELINE_YEAR, CURRENT_YEAR)
        ),
    }
    with open(path.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print("wrote %s" % path.replace(".csv", ".json"))


if __name__ == "__main__":
    main()
