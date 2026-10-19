"""HDI 2004 as the Human Development Report 2006 printed it, for one replication.

    python tools/fetch_hdr2006.py

Foster, McGillivray and Seth (2009, OPHI Working Paper 26) report how many HDI
country comparisons survive reweighting: 10,875 of 15,576 (69.8%) under every
weighting at all, and 91.8% within a quarter of the way to the simplex's
corners. They used HDI 2004 from this report: 177 countries, three dimension
indices averaged with equal weights. That is the one published figure the
weight-sensitivity side of Assay can be held to, and tests/test_pairwise.py
holds `aggregation.pairwise_robustness` to it.

**The source is the report's own PDF**, Table 1, since UNDP no longer serves
the 2006 tables as data. HDR materials are CC BY 3.0 IGO (hdr.undp.org/terms-use,
read 2026-10-19). Those terms also ask anyone keeping HDI data to keep it
current; this file is a historical edition kept to replicate a 2009 paper, and
is labelled so. For current HDI, see fetch_undp.py.

**What the table cannot give.** It prints the dimension indices to two
decimals, so many pairs tie on a dimension, and a tie counts as "not reversed".
On these values the paper's procedure finds more robust pairs than the paper
did. Rebuilding unrounded indices from the raw columns is not possible either:
about 35 countries have a raw indicator UNDP estimated and did not print. The
test therefore asks whether the paper's figure lies inside the range the
rounding allows, not whether the rounded table hits it.

Needs pypdf (pip install pypdf) to read the PDF; nothing else in Assay does.
"""

import csv
import json
import logging
import os
import re
import sys

import rawcache

URL = ("https://hdr.undp.org/system/files/documents/"
       "human-development-report-2006-english.human-development-report-2006-english")
# Table 1 sits on these pages of the 2006 PDF (zero-based), found by Norway's
# 0.965 on 2026-10-19. Every one of the 177 ranks must be found or the run stops.
PAGES = range(298, 304)
ROW = re.compile(r"^(\d{1,3}) (.+?) (\d\.\d{3}) .* (\d\.\d\d) (\d\.\d\d) (\d\.\d\d) (\S+)\s*$")
COLUMNS = ["Life expectancy index", "Education index", "GDP index"]


def rows_from(page_texts):
    """{rank: (country, hdi, three indices)} from Table 1's page text."""
    got = {}
    for text in page_texts:
        for line in text.splitlines():
            m = ROW.match(line.strip())
            if m:
                got[int(m.group(1))] = (m.group(2).strip(), m.group(3),
                                        m.group(4), m.group(5), m.group(6))
    return got


def main():
    try:
        import pypdf
    except ImportError:
        raise SystemExit("needs pypdf to read the report: pip install pypdf")
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = rawcache.get("undp", URL, "Assay (research; urllib)", timeout=180)
    if raw[:5] != b"%PDF-":
        raise SystemExit("%s did not return a PDF" % URL)
    import io
    reader = pypdf.PdfReader(io.BytesIO(raw))
    got = rows_from(reader.pages[i].extract_text() or "" for i in PAGES)
    missing = [k for k in range(1, 178) if k not in got]
    if missing:
        raise SystemExit("Table 1 ranks not found: %s" % missing)

    stem = "hdr2006-hdi-2004"
    with open(os.path.join(here, "data", stem + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "country"] + COLUMNS + ["HDI"])
        for k in range(1, 178):
            country, hdi, le, ed, gdp = got[k]
            w.writerow([k, country, le, ed, gdp, hdi])
    spec = {
        "name": "UNDP Human Development Index 2004, as printed in HDR 2006",
        "data": stem + ".csv",
        "entity_col": "country",
        "score_cols": COLUMNS,
        "published_col": "HDI",
        "weights": {c: 1 for c in COLUMNS},
        "claimed_dimensions": 3,
        "aggregation_form": "linear",
        "orient": True,
        "source": "UNDP, Human Development Report 2006, Table 1, " + URL,
        "note": ("A historical edition, kept to replicate Foster, McGillivray and "
                 "Seth (2009). Before 2010 the HDI was the arithmetic mean of these "
                 "three indices. Indices are printed to two decimals, HDI to three."),
    }
    with open(os.path.join(here, "data", stem + ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    print("177 countries written to data/%s.csv" % stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
