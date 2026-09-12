"""Build an EPI issue-category table from Yale's own published files.

    python tools/fetch_epi.py

Two files, both plain CSV over HTTPS with no key or form:

    epi2024results.csv   country scores at every level of the hierarchy
    epi2024weights.csv   the weight of each level, including "EPI Percent" -
                         each issue category's effective share of the total

This is the first target where all four of Assay's tests apply at once. EPI's
eleven issue categories are measured from unrelated instruments (satellite
aerosol retrievals, fisheries landings, protected-area registries, greenhouse
gas inventories), they aggregate LINEARLY, and Yale publishes the weights
machine-readably. So the reproduction test has something to check, and the rank
envelope can perturb the authors' own numbers rather than inventing a range.

Yale also issued a corrigendum after publication: the Wastewater Collected and
Wastewater Treated indicator weights should read 1.5% each rather than 2%. That
sits below the issue-category level audited here, so it should not move these
totals - but it is exactly the class of error the reproduction test exists to
surface, and worth knowing about when reading the result.
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
    v = (v or "").strip().replace("%", "")
    if v in ("", "NA", "n/a", ".."):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    res = list(csv.DictReader(io.StringIO(
        grab(RESULTS, os.path.join(here, "data/raw/epi2024results.csv")))))
    wts = list(csv.DictReader(io.StringIO(
        grab(WEIGHTS, os.path.join(here, "data/raw/epi2024weights.csv")))))

    cats = [r for r in wts if (r.get("Type") or "").strip() == "IssueCategory"]
    weights, labels, cols = {}, {}, []
    for r in cats:
        ab = r["Abbreviation"].strip()
        pct = to_float(r["EPI Percent"])
        col = ab + ".new"
        if pct is None or col not in res[0]:
            raise SystemExit("issue category %s has no weight or no score column" % ab)
        label = "%s (%s)" % (r["Variable"].strip(), ab)
        labels[ab] = label
        weights[label] = pct
        cols.append((ab, col, label))

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

    rows, dropped = [], 0
    for r in res:
        vals = [to_float(r[c]) for _, c, _ in cols]
        epi = to_float(r.get("EPI.new"))
        name = (r.get("country") or "").strip()
        if not name or epi is None or any(v is None for v in vals):
            dropped += 1
            continue
        rows.append([name] + vals + [epi])
    print("\n%d countries complete (%d dropped)" % (len(rows), dropped))
    if dropped > 0.25 * len(res):
        print("  WARNING: %.0f%% of countries dropped, and the two categories "
              "above\n  explain most of it. The audited sample is biased toward "
              "countries\n  that have both a coastline and forests."
              % (100.0 * dropped / len(res)))

    out_csv = args.out or os.path.join(here, "data", "epi-2024.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country"] + [l for _, _, l in cols] + ["EPI"])
        w.writerows(sorted(rows))
    print("wrote %s" % out_csv)

    spec = {
        "name": "Yale Environmental Performance Index 2024",
        "data": os.path.basename(out_csv),
        "entity_col": "country",
        "score_cols": [l for _, _, l in cols],
        "published_col": "EPI",
        "weights": weights,
        "claimed_dimensions": 3,
        "aggregation_form": "linear",
        "orient": True,
        "source": "Yale EPI 2024, %s and %s" % (RESULTS, WEIGHTS),
        "note": (
            "Eleven issue categories weighted into three policy objectives "
            "(Climate Change Mitigation, Environmental Health, Ecosystem "
            "Vitality) and then into the EPI. Weights are Yale's own published "
            "'EPI Percent' shares. A corrigendum revised two INDICATOR weights "
            "below this level (Wastewater Collected and Treated, 1.5% not 2%)."
        ),
    }
    with open(out_csv.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print("wrote %s" % out_csv.replace(".csv", ".json"))


if __name__ == "__main__":
    main()
