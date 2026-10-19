"""EPI 2026 beside 2024: the edition's files read right, and what replicated.

The 2026 tables (data/epi-2026.*, data/epi-panel-2026.*) were built on
2026-10-19 by tools/fetch_epi.py and tools/build_epi_panel.py with
--edition 2026. These tests need no download: they read the committed tables
and exercise the weight rule on rows shaped like Yale's sheet.

Imports nothing from Sextant, so it runs on a runner without the sibling.

Run:  python -m unittest tests.test_epi_editions
"""

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import epi_editions  # noqa: E402
from assay import aggregation, load  # noqa: E402


def row(kind, tla, name, nxt, level, epi):
    return {"Type": kind, "TLA": tla, "Component": name, "NextLevel": nxt,
            "Weight_level": level, "Weight_EPI": epi}


class TheWeightRule(unittest.TestCase):

    def test_mixed_units_in_weight_epi_are_never_read(self):
        # Shaped like Yale's 2026 sheet: Weight_EPI holds a percentage for one
        # category (4.499999) and a fraction for another (0.2475).
        rows = [row("EPI", "EPI", "EPI", "", "", ""),
                row("PolicyObjective", "ECO", "Ecosystem Vitality", "EPI", "0.45", "0.45"),
                row("IssueCategory", "FOR", "Forests", "ECO", "0.45", "4.499999"),
                row("IssueCategory", "BDH", "Biodiversity & Habitat", "ECO", "0.55", "0.2475"),
                row("PolicyObjective", "PCC", "Climate Change", "EPI", "0.55", "0.55"),
                row("IssueCategory", "CCH", "Climate Change Mitigation", "PCC", "1", "0.55")]
        cats, objs = epi_editions.shares_2026(rows)
        shares = {t: w for t, _, w in cats}
        self.assertAlmostEqual(shares["FOR"], 20.25)
        self.assertAlmostEqual(shares["BDH"], 24.75)
        self.assertAlmostEqual(shares["CCH"], 55.0)
        self.assertAlmostEqual(sum(shares.values()), 100.0)
        self.assertEqual([t for t, _, _ in objs], ["ECO", "PCC"])

    def test_a_category_without_a_level_weight_has_no_share(self):
        rows = [row("PolicyObjective", "ECO", "Ecosystem Vitality", "EPI", "0.5", ""),
                row("IssueCategory", "FOR", "Forests", "ECO", "", "4.5")]
        cats, _ = epi_editions.shares_2026(rows)
        self.assertIsNone(cats[0][2])


class TheTables(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT / "data" / "epi-2026.json").read_text("utf-8"))
        cls.idx = load.from_spec(str(ROOT / "data" / "epi-2026.json"))

    def test_twelve_categories_whose_weights_make_the_whole(self):
        self.assertEqual(len(self.spec["score_cols"]), 12)
        self.assertIn("Forests (FOR)", self.spec["score_cols"])
        self.assertIn("Grasslands (GRS)", self.spec["score_cols"])
        self.assertAlmostEqual(sum(self.spec["weights"].values()), 100.0, places=6)
        self.assertAlmostEqual(self.spec["weights"]["Biodiversity & Habitat (BDH)"], 24.75)
        self.assertAlmostEqual(self.spec["weights"]["Sanitation & Drinking Water (H2O)"], 5.0)

    def test_the_weights_reproduce_the_published_epi(self):
        # 2024 lands 0.22 points out at the median; 2026 to rounding.
        rep = aggregation.reproduces_published(self.idx)
        self.assertTrue(rep["exact"])
        self.assertLess(rep["max_abs_score_error"], 0.05)

    def test_the_panel_rebuilt_from_the_change_has_every_country(self):
        panel, _ = load.panel_from_spec(str(ROOT / "data" / "epi-panel-2026.json"))
        self.assertEqual(sorted(panel), [2016, 2026])
        self.assertEqual(panel[2016].n, 177)
        self.assertEqual(panel[2026].n, 177)


class WhatReplicated(unittest.TestCase):
    """The 2024 findings, asked again of 2026. Recorded 2026-10-19."""

    @classmethod
    def setUpClass(cls):
        cls.a = load.from_spec(str(ROOT / "data" / "epi-2024.json"))
        cls.b = load.from_spec(str(ROOT / "data" / "epi-2026.json"))

    def test_the_ranking_still_leans_on_biodiversity_over_climate(self):
        for idx in (self.a, self.b):
            res = aggregation.leave_one_out(idx)
            self.assertEqual(res["rows"][0]["column"], "Biodiversity & Habitat (BDH)")
            climate = next(r for r in res["rows"] if r["column"].endswith("(CCH)"))
            self.assertGreater(climate["weight"], res["rows"][0]["weight"])

    def test_fisheries_still_runs_against_the_index(self):
        self.assertIn("Fisheries (FSH)", self.a.flipped)
        self.assertIn("Fisheries (FSH)", self.b.flipped)

    def test_the_dropping_went_from_selective_to_the_line(self):
        a, b = load.left_out(self.a), load.left_out(self.b)
        self.assertTrue(a["selective"])
        self.assertFalse(a["near_line"])
        self.assertTrue(b["near_line"])
        self.assertLess(b["gap_in_sd"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
