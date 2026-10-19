"""Every figure on Against the Source's EPI page, printed from the data.

    python tools/findings.py

That site's rule is that no figure is typed: each finding names the script
in the owning project that prints its numbers. This is that script for the
page on where EPI's ranking leans. It reads the bundled 2024 and 2026 tables
(tools/fetch_epi.py) and prints, per edition, what the page states and nothing
it does not. Imports nothing from Sextant, so it runs on a clone of Assay alone.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from assay import aggregation, load  # noqa: E402

EDITIONS = (2024, 2026)


def short(label):
    return label.rsplit(" (", 1)[0]


def edition(year):
    idx = load.from_spec(os.path.join(HERE, "data", "epi-%d.json" % year))
    rep = aggregation.reproduces_published(idx)
    loo = aggregation.leave_one_out(idx)
    left = load.left_out(idx)
    rows = {short(r["column"]): r for r in loo["rows"]}
    heaviest = max(loo["rows"], key=lambda r: r["weight"])
    leans = loo["rows"][0]
    print("EPI %d" % year)
    print("  countries in the index          %d" % left["rows_in"])
    print("  scored on every category        %d (dropped %d)" % (idx.n, left["dropped"]))
    print("  issue categories                %d" % idx.k)
    print("  recomputed from Yale's weights  median %.2f points from the published EPI"
          % rep["median_abs_score_error"])
    print("  heaviest category               %s, %.1f%% of the weight"
          % (short(heaviest["column"]), 100 * heaviest["weight"]))
    print("  leaned on most                  %s, %.1f%% of the weight"
          % (short(leans["column"]), 100 * leans["weight"]))
    print("  left out in turn (median country moves, of %d):" % idx.n)
    for r in loo["rows"]:
        print("    %-34s %5.1f%% weight  %2.0f places  (90th percentile %2.0f)"
              % (short(r["column"]), 100 * r["weight"], r["median_shift"], r["p90_shift"]))
    print("  dropped for an empty score in   %s"
          % ", ".join("%s (%d)" % (short(c), blank) for c, blank, _bad, _ex in left["by_column"]))
    print("  dropped countries vs kept       %.2f SD %s on the published EPI"
          % (abs(left["gap_in_sd"]), "lower" if left["gap_in_sd"] < 0 else "higher"))
    print()
    return rows, heaviest, leans


def main():
    for year in EDITIONS:
        edition(year)
    return 0


if __name__ == "__main__":
    sys.exit(main())
