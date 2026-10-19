"""Where each EPI edition keeps its scores and weights, and how to read them.

fetch_epi.py and build_epi_panel.py both read one edition at a time through
here, so a new edition is one entry below and both scripts agree on it.

The two editions differ in more than the address:

  2024  two CSVs. Weights.csv gives each issue category's share of the total
        as "EPI Percent", and each policy objective's weight as "Weight". The
        results carry each score twice: `.new`, and `.old`, Yale's backcast on
        data about ten years earlier through the current method.

  2026  two .xlsx workbooks (read 2026-10-19 from epi.yale.edu/2026/downloads).
        The weights are the methods workbook's "Weights" sheet, and its
        Weight_EPI column mixes units: Forests reads 4.499999 (a percentage)
        and Biodiversity & Habitat 0.2475 (a fraction). It is not read. Each
        category's share is its Weight_level times its policy objective's,
        which is unambiguous and sums to one within every objective. The
        results publish `.chg`, the change from data up to ten years earlier,
        in place of `.old`; old = new - chg recovers the same backcast.
        Forests is FOR (ECS in 2024), and Grasslands (GRS) is new: twelve
        categories, not eleven.
"""

import rawcache
import table

UA = "Mozilla/5.0 (compatible; Assay research; urllib)"
BASE_2026 = "https://epi.yale.edu/sites/default/files/2026-09/"

EDITIONS = {
    2024: {"results": "https://epi.yale.edu/downloads/epi2024results.csv",
           "weights": "https://epi.yale.edu/downloads/epi2024weights.csv",
           "legacy_results": "data/raw/epi2024results.csv",
           "legacy_weights": "data/raw/epi2024weights.csv",
           "baseline": 2014},
    2026: {"results": BASE_2026 + "epi2026results2026-07-07.xlsx",
           "weights": BASE_2026 + "epi2026methods.xlsx",
           "results_sheet": "data", "weights_sheet": "Weights",
           "baseline": 2016},
}


def to_float(v):
    v = (v or "").strip().replace("%", "")
    if v in ("", "NA", "n/a", ".."):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _grab(url, here, legacy=None, sheet=None):
    import os
    try:
        raw = rawcache.get("epi", url, UA, timeout=120,
                           legacy=os.path.join(here, legacy) if legacy else None)
        return table.rows(raw, sheet)
    except table.TableError as exc:
        raise SystemExit("%s: %s" % (url, exc))


def shares_2026(wts):
    """(categories, objectives) from the 2026 Weights sheet's rows.

    A category's share of the EPI is its Weight_level times its objective's.
    Weight_EPI is never read: it holds percentages for some rows and
    fractions for others (Forests 4.499999, Biodiversity & Habitat 0.2475).
    """
    kind = lambda r: (r.get("Type") or "").strip()
    level = {r["TLA"].strip(): to_float(r["Weight_level"]) for r in wts}
    parent = {r["TLA"].strip(): (r.get("NextLevel") or "").strip() for r in wts}
    cats, objs = [], []
    for r in wts:
        t = r["TLA"].strip()
        label = "%s (%s)" % (r["Component"].strip(), t)
        if kind(r) == "IssueCategory":
            mine, up = level.get(t), level.get(parent.get(t))
            cats.append((t, label, None if mine is None or up is None else 100.0 * mine * up))
        elif kind(r) == "PolicyObjective":
            objs.append((t, label, level.get(t)))
    return cats, objs


def load(edition, here):
    """(results rows, categories, objectives) for one edition.

    categories: [(tla, label, share of the EPI in percent)]
    objectives: [(tla, label, weight as a fraction)]
    Every row gets `TLA.old` where the edition publishes `.chg` instead.
    """
    if edition not in EDITIONS:
        raise SystemExit("no EPI edition %s here; known: %s"
                         % (edition, ", ".join(map(str, sorted(EDITIONS)))))
    e = EDITIONS[edition]
    res = _grab(e["results"], here, e.get("legacy_results"), e.get("results_sheet"))
    wts = _grab(e["weights"], here, e.get("legacy_weights"), e.get("weights_sheet"))
    kind = lambda r: (r.get("Type") or "").strip()

    if edition == 2024:
        cats = [(r["Abbreviation"].strip(), "%s (%s)" % (r["Variable"].strip(), r["Abbreviation"].strip()),
                 to_float(r["EPI Percent"])) for r in wts if kind(r) == "IssueCategory"]
        objs = [(r["Abbreviation"].strip(), "%s (%s)" % (r["Variable"].strip(), r["Abbreviation"].strip()),
                 to_float(r["Weight"])) for r in wts if kind(r) == "PolicyObjective"]
    else:
        cats, objs = shares_2026(wts)
        # The backcast, from the change Yale publishes in its place.
        tlas = ["EPI"] + [t for t, _, _ in cats] + [t for t, _, _ in objs]
        for r in res:
            for t in tlas:
                new, chg = to_float(r.get(t + ".new")), to_float(r.get(t + ".chg"))
                r[t + ".old"] = "" if new is None or chg is None else repr(round(new - chg, 6))

    missing = [t for t, _, w in cats + objs if w is None]
    if missing:
        raise SystemExit("EPI %s: no weight for %s" % (edition, ", ".join(missing)))
    absent = [t for t, _, _ in cats + objs if t + ".new" not in res[0]]
    if absent:
        raise SystemExit("EPI %s: no score column for %s" % (edition, ", ".join(absent)))
    return res, cats, objs
