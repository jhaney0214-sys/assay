"""`claims.json` against the engine, the prose, and the hand-written test.

`tests/test_claims.py` already pins the headline share and correlation, and it
cannot run on a runner: it imports Sextant's factor machinery through
`assay.structure`, and putting a token for a private repository on a public one
was declined on 2026-09-19. So the guard it provides is real and runs only
where the two projects are siblings.

**No figure is written into this file as a literal.** Every number it checks
comes from `claims.json`, because a test that hardcodes the value it is
guarding is a second copy of the claim — the defect, one level up. Running the
coverage check against an earlier draft of this file, which quoted both
numbers in this docstring, reported exactly that.

This file splits that chain at the ledger:

    engine --(TheLedgerAgreesWithTheEngine)--> claims.json --(everything
    else here)--> README.md, docs/index.html, tests/test_browser_page.py

Only the first link needs Sextant. **The other two run on the runner**, because
comparing a JSON file against prose imports nothing at all. That is the whole
reason the ledger is a file rather than a dict inside a test: a number that
lives in Python can only be checked by running Python.

What this adds over `tests/test_claims.py`, which is kept rather than replaced:

  * `docs/index.html` is checked in CI without a browser. The seven browser
    tests do assert the page, but they assert the value is PRESENT; they
    cannot see a second, different value three paragraphs down.
  * a wrong number sitting beside a right one is caught at all.
  * two numbers the README leans on that nothing checked before: the row-count
    denominator and the rank envelope's median move.
  * the ledger and `tests/test_claims.py` are pinned to each other, so the
    two cannot become two sources of truth.

    python -m unittest tests.test_claims_ledger
"""

import io
import json
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import claims  # noqa: E402  (vendored; see tools/claims.py)

LEDGER = ROOT / "claims.json"

#: Swept for files that quote a claim and are not pinned to it. Deliberately
#: wider than `appears_in`: the point is to find the file nobody remembered.
#:
#: Recursive on purpose. The first draft listed `*.md` and `docs/*.html`, and
#: staging the regression it exists to catch — a new `docs/NEWPAGE.md`
#: quoting the headline share — left the suite green, because `*.md` does not
#: descend. A sweep that cannot see a new subdirectory is the blind spot it
#: was written to remove. 26 files at the current commit.
SCAN = ["**/*.md", "**/*.html", "**/*.js", "**/*.py"]


def read(*parts):
    with io.open(str(ROOT.joinpath(*parts)), encoding="utf-8") as handle:
        return handle.read()


class TheLedgerItself(unittest.TestCase):

    def test_it_exists_and_parses(self):
        self.assertTrue(LEDGER.exists(), "claims.json is gone")
        self.assertTrue(claims.load(LEDGER))

    def test_the_schema_holds(self):
        findings = claims.check_schema(claims.load(LEDGER))
        fatal = [f.line() for f in findings if f.fatal]
        self.assertEqual(fatal, [], "\n".join(fatal))

    def test_every_claim_says_how_it_is_known(self):
        for claim in claims.load(LEDGER):
            self.assertIn(claim["status"], claims.STATUSES)
            self.assertGreater(
                len(claim["anchor"]), 40,
                "%s's anchor is too short to be somewhere a reader could go "
                "and disagree" % claim["id"])

    def test_the_rank_envelope_is_not_called_measured(self):
        """The grade is the point, and this is the one that is not measured.

        `median_spread` is a median over simulated reweightings. Calling it
        measured would be the failure this whole mechanism exists to stop:
        a modeled number wearing the authority of an observed one.
        """
        by_id = {c["id"]: c for c in claims.load(LEDGER)}
        self.assertEqual(by_id["hdi_median_rank_move"]["status"], "modeled")
        self.assertEqual(by_id["hdi_pc1_share"]["status"], "measured")


class TheLedgerAgreesWithEveryDocument(unittest.TestCase):
    """Runs on a runner. Imports nothing from this project."""

    def test_the_published_strings_round_from_the_recorded_values(self):
        findings = claims.check_rounding(claims.load(LEDGER))
        fatal = [f.line() for f in findings if f.fatal]
        self.assertEqual(fatal, [], "\n".join(fatal))

    def test_every_document_still_carries_its_claims(self):
        findings = claims.check_prose(claims.load(LEDGER), ROOT)
        fatal = [f.line() for f in findings if f.fatal]
        self.assertEqual(fatal, [], "\n".join(fatal))

    def test_no_document_carries_a_contradicting_number(self):
        """The half a bare `assertIn` cannot do.

        `test_browser_page.py` asserts the published share is on the page. It
        says nothing about a second, different share also being on the page,
        which is what editing one sentence and not the next leaves behind.
        """
        findings = [f for f in claims.check_prose(claims.load(LEDGER), ROOT)
                    if f.kind == "contradiction" and f.fatal]
        self.assertEqual([f.line() for f in findings], [])

    def test_the_contradiction_scan_was_actually_armed(self):
        """A NOT CHECKED note here means the scan found no window to look in.

        A `near` phrase deleted from the prose silently disarms the guard, and
        an instrument that could not look must not report that it looked.
        """
        notes = [f.line() for f in claims.check_prose(claims.load(LEDGER), ROOT)
                 if "NOT CHECKED" in f.message]
        self.assertEqual(notes, [], "\n".join(notes))

    def test_no_unpinned_file_quotes_a_claim(self):
        """The direction that can see an absence.

        A file that quotes a published value and is not in `appears_in` is
        guarded by nothing, and checking the files that ARE listed can never
        find it.
        """
        findings = claims.check_coverage(claims.load(LEDGER), ROOT, SCAN)
        fatal = [f.line() for f in findings if f.fatal]
        self.assertEqual(fatal, [], "\n".join(fatal))

    def test_the_whole_verify_is_clean(self):
        _, findings = claims.verify(ROOT, scan=SCAN)
        fatal = [f.line() for f in findings if f.fatal]
        self.assertEqual(fatal, [], "\n".join(fatal))


class TheLedgerAndTheHandWrittenTestCannotSeparate(unittest.TestCase):
    """Two records of the same numbers would be the defect, one level up.

    `tests/test_claims.py` keeps its own `HDI` and `AS_WRITTEN` dicts, which is
    how it worked before this ledger existed. Both are kept - that file runs
    the engine and this one cannot - but they must not be free to drift, or
    the fix for "a number nobody re-runs" would be a second number nobody
    re-runs. This reads that file as text, so it needs no import and runs on
    the runner where that file itself cannot.
    """

    def setUp(self):
        self.text = read("tests", "test_claims.py")
        self.by_id = {c["id"]: c for c in claims.load(LEDGER)}

    def constant(self, key):
        found = re.search(r'"%s":\s*([0-9.]+)' % re.escape(key), self.text)
        self.assertIsNotNone(
            found, "tests/test_claims.py no longer records %r" % key)
        return float(found.group(1))

    def test_the_pc1_share_constants_agree(self):
        self.assertAlmostEqual(self.constant("pc1_share"),
                               self.by_id["hdi_pc1_share"]["raw"], places=15)

    def test_the_correlation_constants_agree(self):
        self.assertAlmostEqual(self.constant("r_with_pc1"),
                               self.by_id["hdi_r_with_pc1"]["raw"], places=15)

    def test_the_row_count_constants_agree(self):
        self.assertEqual(self.constant("n"),
                         self.by_id["hdi_countries"]["raw"])

    def test_the_published_strings_agree(self):
        for claim_id, key in (("hdi_pc1_share", "pc1_share"),
                              ("hdi_r_with_pc1", "r_with_pc1")):
            found = re.search(r'"%s":\s*"([^"]+)"' % key, self.text)
            self.assertIsNotNone(found)
            self.assertEqual(found.group(1), self.by_id[claim_id]["value"])


class TheLedgerAgreesWithTheEngine(unittest.TestCase):
    """The one link that needs Sextant as a sibling, and says so when it has none.

    Skipping rather than failing is right here - a runner genuinely cannot do
    this - but a silent skip is not, because a green suite that never ran its
    only engine check is the exact shape of failure this project has already
    been bitten by. The workflow prints what did not run.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from assay import aggregation, load, structure
        except ImportError as exc:
            raise unittest.SkipTest(
                "the engine needs Sextant as a sibling checkout: %s" % exc)
        index = load.from_spec(str(ROOT / "data" / "hdi-2022.json"))
        struct = structure.analyse(index)
        pc1 = structure.pc1_scores(index)
        decorative = aggregation.weights_are_decorative(index, pc1)
        envelope = aggregation.rank_envelope(index)
        cls.computed = {
            "hdi_pc1_share": struct["pc1_share"],
            "hdi_r_with_pc1": decorative["r_with_pc1"],
            "hdi_countries": struct["n"],
            "hdi_median_rank_move": envelope["median_spread"],
        }

    def test_every_recorded_value_is_what_the_engine_returns_today(self):
        findings = claims.check_computed(claims.load(LEDGER), self.computed)
        fatal = [f.line() for f in findings if f.fatal]
        self.assertEqual(fatal, [], "\n".join(fatal))

    def test_nothing_in_the_ledger_went_unrecomputed(self):
        """A claim with a `raw` that nothing checks is the original defect."""
        notes = [f.line() for f in
                 claims.check_computed(claims.load(LEDGER), self.computed)
                 if "nothing recomputed it" in f.message]
        self.assertEqual(notes, [], "\n".join(notes))

    def test_the_findings_themselves_still_hold(self):
        """Not the digits - the claims they support.

        If either of these stopped being true the README would need
        rewriting, not renumbering.
        """
        self.assertGreater(self.computed["hdi_pc1_share"], 0.80)
        self.assertGreater(self.computed["hdi_r_with_pc1"], 0.99)


if __name__ == "__main__":
    unittest.main()
