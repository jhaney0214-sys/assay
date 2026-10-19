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
import json
import os

import epi_editions
from epi_editions import to_float

# The default edition's files; tools/editions.py reads these. 2026 publishes
# the change (`.chg`) instead of the backcast, and epi_editions rebuilds `.old`
# from it, so the panel below is the same construction for both editions.
RESULTS = epi_editions.EDITIONS[2024]["results"]
WEIGHTS = epi_editions.EDITIONS[2024]["weights"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", type=int, default=2024)
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    e = epi_editions.EDITIONS.get(args.edition)

    # The backcast is "approximately ten years prior" (Yale's wording), so these
    # labels are nominal periods, not exact years. They exist to give the panel
    # two ordered points; nothing downstream reads them as dates.
    BASELINE_YEAR, CURRENT_YEAR = e["baseline"] if e else None, args.edition

    res, _, objs = epi_editions.load(args.edition, here)
    cols = [(ab, label) for ab, label, _ in objs]
    weights = {label: w for _, label, w in objs}
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

    path = os.path.join(here, "data", "epi-panel.csv" if args.edition == 2024
                        else "epi-panel-%d.csv" % args.edition)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "year"] + [l for _, l in cols] + ["EPI"])
        w.writerows(sorted(rows, key=lambda r: (r[1], r[0])))
    print("wrote %s" % path)

    spec = {
        "name": "Yale EPI %d - backcast baseline vs current" % args.edition,
        "data": os.path.basename(path),
        "entity_col": "country",
        "year_col": "year",
        "score_cols": [l for _, l in cols],
        "published_col": "EPI",
        "weights": weights,
        "claimed_dimensions": 3,
        "aggregation_form": "linear",
        "orient": False,
        "source": "Yale EPI %d, %s and %s" % (args.edition, e["results"], e["weights"]),
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
