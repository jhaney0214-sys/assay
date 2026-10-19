"""Build an EPI issue-category table from Yale's own published files.

    python tools/fetch_epi.py                  # the 2024 edition
    python tools/fetch_epi.py --edition 2026   # the 2026 edition

Each edition publishes country scores at every level of the hierarchy and the
weight of each level; tools/epi_editions.py knows where each keeps them and
how to read them (2024: two CSVs; 2026: two .xlsx workbooks, with a weights
column in mixed units that is not read).

This is the first target where all four of Assay's tests apply at once. EPI's
issue categories are measured from unrelated instruments (satellite aerosol
retrievals, fisheries landings, protected-area registries, greenhouse gas
inventories), they aggregate LINEARLY, and Yale publishes the weights
machine-readably. So the reproduction test has something to check, and the rank
envelope can perturb the authors' own numbers rather than inventing a range.

Yale issued a corrigendum after the 2024 publication: the Wastewater Collected
and Wastewater Treated indicator weights should read 1.5% each rather than 2%.
That sits below the issue-category level audited here, so it should not move
these totals - but it is exactly the class of error the reproduction test
exists to surface, and worth knowing about when reading the result.
"""

import argparse
import csv
import json
import os

import epi_editions
from epi_editions import to_float

# The default edition's files. tools/editions.py reads these to say which
# edition the bundled data is from and whether a later one is out.
RESULTS = epi_editions.EDITIONS[2024]["results"]
WEIGHTS = epi_editions.EDITIONS[2024]["weights"]

NOTES = {
    2024: ("Eleven issue categories weighted into three policy objectives "
           "(Climate Change Mitigation, Environmental Health, Ecosystem "
           "Vitality) and then into the EPI. Weights are Yale's own published "
           "'EPI Percent' shares. A corrigendum revised two INDICATOR weights "
           "below this level (Wastewater Collected and Treated, 1.5% not 2%)."),
    2026: ("Twelve issue categories weighted into three policy objectives "
           "(Ecosystem Vitality, Environmental Health, Climate Change) and then "
           "into the EPI. Each category's weight is its Weight_level times its "
           "objective's, from the methods workbook's Weights sheet; that sheet's "
           "Weight_EPI column mixes percentages and fractions and is not read. "
           "Forests is FOR (ECS in 2024) and Grasslands (GRS) is new."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", type=int, default=2024,
                    help="EPI edition (%s)" % ", ".join(map(str, sorted(epi_editions.EDITIONS))))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    e = epi_editions.EDITIONS.get(args.edition)

    res, cats, _ = epi_editions.load(args.edition, here)
    weights = {label: pct for _, label, pct in cats}
    cols = [(ab, ab + ".new", label) for ab, label, _ in cats]

    total = sum(weights.values())
    print("%d issue categories, published weights sum to %.2f%%" % (len(cols), total))
    if abs(total - 100.0) > 0.51:
        raise SystemExit("weights do not sum to 100% - the hierarchy changed")

    # Which categories are causing the drops? EPI's missingness is structural,
    # not random: a landlocked country has no Fisheries score and a desert
    # state has no Forests score. Listwise deletion therefore removes a
    # particular KIND of country, and the survivors are not a random sample of
    # the world. Report it rather than let it disappear into a row count.
    miss = {}
    for ab, c, label in cols:
        miss[label] = sum(1 for r in res if to_float(r.get(c)) is None)
    worst = sorted(miss.items(), key=lambda kv: -kv[1])[:3]
    print("\nmissingness by category (top 3 of %d countries):" % len(res))
    for label, n in worst:
        print("  %-42s missing in %3d" % (label[:42], n))

    # Incomplete rows are written too, with their gaps left blank. The audit
    # drops them itself and then reports who it dropped and how they score on
    # the published EPI; a file holding only the survivors would hide that.
    rows, dropped = [], 0
    for r in res:
        vals = [to_float(r[c]) for _, c, _ in cols]
        epi = to_float(r.get("EPI.new"))
        name = (r.get("country") or "").strip()
        if not name:
            continue
        if epi is None or any(v is None for v in vals):
            dropped += 1
        rows.append([name] + ["" if v is None else v for v in vals]
                    + ["" if epi is None else epi])
    print("\n%d countries complete (%d to be dropped by the audit)"
          % (len(rows) - dropped, dropped))
    if dropped > 0.25 * len(res):
        # Named from the data: 2024's gaps were Forests and Fisheries, 2026's
        # Fisheries, Grasslands and Forests. The audit's "who was left out"
        # section measures whether the dropped countries score differently.
        gaps = [label for label, n in worst if n]
        print("  WARNING: %.0f%% of countries dropped, almost all for %s.\n"
              "  The audited sample is the countries scored on all of them."
              % (100.0 * dropped / len(res), " or ".join(gaps)))

    out_csv = args.out or os.path.join(here, "data", "epi-%d.csv" % args.edition)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country"] + [l for _, _, l in cols] + ["EPI"])
        w.writerows(sorted(rows, key=lambda r: r[0]))
    print("wrote %s" % out_csv)

    spec = {
        "name": "Yale Environmental Performance Index %d" % args.edition,
        "data": os.path.basename(out_csv),
        "entity_col": "country",
        "score_cols": [l for _, _, l in cols],
        "published_col": "EPI",
        "weights": weights,
        "claimed_dimensions": 3,
        "aggregation_form": "linear",
        "orient": True,
        "source": "Yale EPI %d, %s and %s" % (args.edition, e["results"], e["weights"]),
        "note": NOTES[args.edition],
    }
    with open(out_csv.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print("wrote %s" % out_csv.replace(".csv", ".json"))


if __name__ == "__main__":
    main()
