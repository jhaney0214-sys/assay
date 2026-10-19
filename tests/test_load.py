"""assay/load.py: specs that would mislead are refused, cells are read as the
browser page reads them, and the rows listwise deletion drops are accounted for.

Imports nothing from Sextant, so it runs on a runner without the sibling.

Run:  python -m unittest tests.test_load
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assay import load  # noqa: E402

SPEC = {"name": "t", "data": "t.csv", "entity_col": "country",
        "score_cols": ["a", "b", "c"], "published_col": "overall",
        "weights": {"a": 50, "b": 30, "c": 20}, "claimed_dimensions": 3}


def write(tmp, rows, spec=None, header=("country", "a", "b", "c", "overall")):
    """A CSV of `rows` and a spec beside it; returns the spec's path."""
    with open(os.path.join(tmp, "t.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")
    path = os.path.join(tmp, "t.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec or SPEC, fh)
    return path


def complete(n, start=0):
    """n complete rows whose overall score rises with the row number."""
    return [("e%d" % i, i % 7 + 1, (i * 3) % 11 + 1, i % 5 + 1, i)
            for i in range(start, start + n)]


class TestCells(unittest.TestCase):

    def test_an_infinity_is_not_a_number(self):
        # float() takes all of these; one in a column made every result nan.
        for s in ("inf", "-inf", "Infinity", "-Infinity", "INF"):
            self.assertIsNone(load._to_float(s), s)

    def test_ordinary_numbers_still_read(self):
        self.assertEqual(load._to_float(" 1,234.5 "), 1234.5)
        self.assertEqual(load._to_float("-0.25"), -0.25)

    def test_only_a_plain_decimal_reads(self):
        # float() takes all of these. docs/audit.js does not, and the two
        # must drop the same rows (tools/conform_browser.py checks it).
        for s in ("1_000", "١٢", "0x10", "1e", "--1", "3 (est.)"):
            self.assertIsNone(load._to_float(s), s)
        for s, v in ((".5", 0.5), ("5.", 5.0), ("-.5e-3", -0.0005), ("+2", 2.0)):
            self.assertEqual(load._to_float(s), v, s)

    def test_blank_and_unreadable_are_told_apart(self):
        for s in ("", " ", "NA", "n/a", "nan", "-", "null", None):
            self.assertTrue(load._is_blank(s), repr(s))
        for s in ("12%", "n.a.", "..", "inf"):
            self.assertFalse(load._is_blank(s), s)
            self.assertIsNone(load._to_float(s), s)


class TestSpecs(unittest.TestCase):

    def check(self, **change):
        spec = dict(SPEC, **change)
        load.check_spec(spec, "t.json")

    def refused(self, *words, panel=False, **change):
        spec = {k: v for k, v in dict(SPEC, **change).items() if v is not None}
        with self.assertRaises(ValueError) as cm:
            load.check_spec(spec, "t.json", panel=panel)
        for word in words:
            self.assertIn(word, str(cm.exception))

    def test_a_good_spec_passes(self):
        self.check()
        self.check(weights=None, claimed_dimensions=None)

    def test_a_misspelt_weight_is_refused_by_name(self):
        # This used to load, giving pillar 'a' a weight of 0.
        self.refused("weights name A", "not in score_cols",
                     weights={"A": 50, "b": 30, "c": 20})

    def test_a_pillar_with_no_weight_is_refused(self):
        self.refused("no weight given for c", "Write 0", weights={"a": 50, "b": 30})

    def test_an_explicit_zero_weight_is_allowed(self):
        self.check(weights={"a": 50, "b": 50, "c": 0})

    def test_impossible_weights_are_refused(self):
        self.refused("zero or more", weights={"a": -1, "b": 30, "c": 20})
        self.refused("zero or more", weights={"a": "50", "b": 30, "c": 20})
        self.refused("sum to zero", weights={"a": 0, "b": 0, "c": 0})

    def test_a_missing_key_is_named(self):
        self.refused("missing entity_col", entity_col=None)

    def test_a_panel_needs_its_year_column(self):
        self.refused("year_col", panel=True)

    def test_score_columns_are_checked(self):
        self.refused("non-empty list", score_cols=[])
        self.refused("more than once", score_cols=["a", "b", "a"], weights=None)

    def test_claimed_dimensions_must_fit_the_columns(self):
        self.refused("claimed_dimensions", claimed_dimensions=4)
        self.refused("claimed_dimensions", claimed_dimensions=0)
        self.refused("claimed_dimensions", claimed_dimensions=2.5)

    def test_every_bundled_spec_passes(self):
        for path in sorted((ROOT / "data").glob("*.json")):
            spec = json.loads(path.read_text("utf-8"))
            load.check_spec(spec, path.name, panel="year_col" in spec)

    def test_from_spec_applies_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, complete(20),
                         spec=dict(SPEC, weights={"a": 1, "b": 1, "cc": 1}))
            with self.assertRaises(ValueError):
                load.from_spec(path)


class TestDraftSpec(unittest.TestCase):
    """ax.py init: what it guesses, and that it never guesses a claim."""

    def draft(self, name):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name + ".csv")
            shutil.copy(ROOT / "data" / (name + ".csv"), path)
            return load.draft_spec(path)

    def test_hdi_total_is_found_and_flagged_for_checking(self):
        spec, checks = self.draft("hdi-2022")
        self.assertEqual(spec["entity_col"], "country")
        self.assertEqual(spec["published_col"], "HDI")
        self.assertEqual(len(spec["score_cols"]), 4)
        self.assertTrue(any("correlates 0.983" in c for c in checks))

    def test_a_loose_total_is_named_as_the_closest_not_chosen(self):
        spec, checks = self.draft("epi-2024")
        self.assertIsNone(spec["published_col"])
        self.assertIn("EPI", spec["score_cols"])
        self.assertTrue(any("most like a total is 'EPI'" in c for c in checks))

    def test_shared_construction_is_named_as_the_other_reading(self):
        # V-Dem has no total; its liberal index contains its electoral index.
        spec, checks = self.draft("vdem-2024")
        self.assertIsNotNone(spec["published_col"])
        self.assertTrue(any("shared_construction" in c for c in checks))

    def test_it_never_writes_a_claim(self):
        spec, checks = self.draft("svi-2022")
        self.assertNotIn("weights", spec)
        self.assertIsNone(spec["claimed_dimensions"])
        self.assertTrue(any(c.startswith("weights:") for c in checks))
        load.check_spec(spec, "draft")

    def test_without_an_entity_column_it_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "n.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("a,b\n" + "".join("%d,%d\n" % (i, i * 2) for i in range(12)))
            with self.assertRaises(ValueError) as cm:
                load.draft_spec(path)
        self.assertIn("entity column", str(cm.exception))

    def test_the_cli_will_not_overwrite_a_spec(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hdi.csv")
            import shutil
            shutil.copy(ROOT / "data" / "hdi-2022.csv", path)
            run = lambda: subprocess.run([sys.executable, str(ROOT / "ax.py"), "init", path],
                                         capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            with open(os.path.join(tmp, "hdi.json"), encoding="utf-8") as fh:
                first = fh.read()
            second = run()
            self.assertEqual(second.returncode, 1)
            self.assertIn("not overwritten", second.stdout)
            with open(os.path.join(tmp, "hdi.json"), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), first)


class TestLeftOut(unittest.TestCase):

    def test_nothing_dropped_is_nothing_to_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = load.from_spec(write(tmp, complete(20)))
        self.assertEqual(idx.dropped, 0)
        self.assertIsNone(load.left_out(idx))

    def test_an_index_built_in_memory_has_nothing_to_report(self):
        idx = load.Index("m", ["x"] * 3, ["a"], [[1.0]] * 3)
        idx.dropped = 2
        self.assertIsNone(load.left_out(idx))

    def test_the_column_that_cost_each_row_is_counted(self):
        rows = complete(20) + [("g1", "", 2, 3, 1), ("g2", "", "", 3, 2),
                               ("g3", 1, "12%", 3, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            idx = load.from_spec(write(tmp, rows))
        left = load.left_out(idx)
        self.assertEqual((left["rows_in"], left["dropped"]), (23, 3))
        cols = {c: (blank, bad, ex) for c, blank, bad, ex in left["by_column"]}
        self.assertEqual(cols["a"], (2, 0, []))
        self.assertEqual(cols["b"], (1, 1, ["12%"]))
        self.assertNotIn("c", cols)
        self.assertEqual(left["names"], ["g1", "g2", "g3"])

    def test_hdi_drops_two_countries_the_index_did_not_score_either(self):
        left = load.left_out(load.from_spec(str(ROOT / "data" / "hdi-2022.json")))
        self.assertEqual((left["rows_in"], left["dropped"], left["compared"]), (195, 2, 0))
        self.assertEqual(sorted(left["names"]),
                         ["Korea (Democratic People's Rep. of)", "Monaco"])

    def test_dropping_the_bottom_of_the_table_is_called_selective(self):
        # The overall score rises with the row number, so dropping the first
        # rows drops the lowest scorers.
        rows = [(n, "", b, c, o) for n, _, b, c, o in complete(8)] + complete(30, start=8)
        with tempfile.TemporaryDirectory() as tmp:
            left = load.left_out(load.from_spec(write(tmp, rows)))
        self.assertEqual(left["compared"], 8)
        self.assertLess(left["gap_in_sd"], -load.SELECTIVE_AT)
        self.assertTrue(left["selective"])
        self.assertLess(left["dropped_mean_pct"], left["kept_mean_pct"])

    def test_dropping_across_the_table_is_not(self):
        rows = complete(40)
        rows = [(n, "" if i % 5 == 2 else a, b, c, o)
                for i, (n, a, b, c, o) in enumerate(rows)]
        with tempfile.TemporaryDirectory() as tmp:
            left = load.left_out(load.from_spec(write(tmp, rows)))
        self.assertEqual(left["dropped"], 8)
        self.assertFalse(left["selective"])
        self.assertLess(abs(left["gap_in_sd"]), load.SELECTIVE_AT)

    def test_no_published_score_means_no_comparison(self):
        rows = complete(20) + [("g%d" % i, "", 2, 3, "") for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            left = load.left_out(load.from_spec(write(tmp, rows)))
        self.assertEqual(left["dropped"], 4)
        self.assertEqual(left["compared"], 0)
        self.assertNotIn("gap_in_sd", left)

    def test_epi_drops_its_lower_scorers(self):
        # The bundled EPI table keeps its incomplete rows since 2026-10-18 so
        # that this can be asked of it. 75 of 180 countries are dropped, all
        # for an empty Forests or Fisheries score, and they sit lower on the
        # published EPI than the 105 the audit runs on.
        left = load.left_out(load.from_spec(str(ROOT / "data" / "epi-2024.json")))
        self.assertEqual((left["rows_in"], left["dropped"], left["compared"]),
                         (180, 75, 75))
        self.assertEqual([c for c, *_ in left["by_column"]],
                         ["Forests (ECS)", "Fisheries (FSH)"])
        self.assertTrue(left["selective"])
        self.assertLess(left["gap_in_sd"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
