"""Build a theme table from CDC/ATSDR's Social Vulnerability Index, by county.

    python tools/fetch_svi.py

The third index, and chosen to be unlike the first two. HDI and EPI both rank
countries and both come from institutions drawing on the same international
statistics, so agreeing on them is not two independent tests. The SVI ranks
3,144 US counties, is built by a public-health agency for emergency planning,
and has nothing to do with development or the environment.

It declares four themes: socioeconomic status, household characteristics,
racial and ethnic minority status, and housing type and transportation. Each
county gets a percentile rank on each theme and one overall. The audit asks
what it asks of any index: are there four things here.

**Two ways this is a weaker test than HDI, said before the result.** All four
themes are computed from one survey, the American Community Survey, where
HDI's four indicators come from four institutions. And the overall rank is the
percentile of a sum of sums of percentiles, not a weighted sum of the theme
ranks, so the weights cannot be reproduced linearly; the spec says so and that
test is switched off, as it is for HDI.

A county CDC could not rank carries -999 in a column. Since 2026-10-18 that
cell is written blank and the county kept, so the audit can drop it and report
who it dropped; before, this script dropped it and only counted.
"""

import argparse
import csv
import io
import json
import os

import rawcache

YEAR = 2022
URL = ("https://svi.cdc.gov/Documents/Data/%d/csv/states_counties/SVI_%d_US_county.csv"
       % (YEAR, YEAR))
THEMES = {
    "RPL_THEME1": "Socioeconomic status",
    "RPL_THEME2": "Household characteristics",
    "RPL_THEME3": "Racial and ethnic minority status",
    "RPL_THEME4": "Housing type and transportation",
}
OVERALL = "RPL_THEMES"
MISSING = -999.0


def rows_from(text):
    """([entity, theme ranks..., overall] rows, number incomplete).

    An unranked cell (-999) is written blank; its county stays in the file.
    """
    out, incomplete = [], 0
    for r in csv.DictReader(io.StringIO(text)):
        values = [float(r[c]) for c in list(THEMES) + [OVERALL]]
        if any(v <= MISSING for v in values):
            incomplete += 1
        # LOCATION is "Autauga County, Alabama": unique, where COUNTY alone is not.
        out.append([r["LOCATION"]] + ["" if v <= MISSING else v for v in values])
    return out, incomplete


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = rawcache.get("svi", URL, "Assay (research; urllib)", timeout=180)
    rows, dropped = rows_from(raw.decode("utf-8-sig"))

    stem = "svi-%d" % YEAR
    with open(os.path.join(here, "data", stem + ".csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["county"] + list(THEMES.values()) + ["SVI"])
        w.writerows(rows)
    spec = {
        "name": "CDC/ATSDR Social Vulnerability Index %d, US counties" % YEAR,
        "data": stem + ".csv",
        "entity_col": "county",
        "score_cols": list(THEMES.values()),
        "published_col": "SVI",
        "claimed_dimensions": len(THEMES),
        "aggregation_form": ("percentile rank of the sum of the four themes' sums of "
                             "variable percentile ranks - not a weighted sum of the theme ranks"),
        "source": "CDC/ATSDR Social Vulnerability Index %d, county file, %s" % (YEAR, URL),
        "note": ("The four theme columns are each theme's percentile rank among counties, "
                 "as published. All four are computed from one survey, the American "
                 "Community Survey, so unlike HDI's indicators they share an instrument."),
    }
    with open(os.path.join(here, "data", stem + ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    print("%d counties written to data/%s.csv, %d of them with a missing rank "
          "for the audit to drop" % (len(rows), stem, dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
