"""Assay's published numbers, recomputed from the engine that produced them.

`84.1%` and `r = 0.9976` are Assay's headline result and its only external
validation anchor - the README leans on them four times, and
`tests/test_browser_page.py` asserts them against the rendered page. Until
2026-09-19 they were pinned in exactly one direction:

* **the browser drifting away from them** was caught, because those seven
  tests run in CI against a real chromium;
* **the engine drifting away from them** was caught by nothing. The Python
  value lived in a trailing comment (`# python: 0.8409571`) and in prose.
  Had the engine moved - or Sextant's factor machinery underneath it - both
  repositories' CI would have stayed green while the README went quietly
  wrong.

`tools/conform_browser.py` compares both live implementations and is the real
drift guard, but it runs by hand and cannot run on Assay's runner: it needs
Sextant as a sibling, and putting a token for a private repository on a public
one was declined on 2026-09-19. This file closes the half of the gap that
needs no cross-repo access, by deriving the numbers here and failing when any
place that writes them down disagrees.

The pattern is Journeyman's `CountedClaimsInProse`, which exists for the same
reason: a number written into a document is a claim nobody re-runs.

**This does not run in CI.** It imports Sextant's factor machinery through
`assay.structure`, exactly as `test_structure`, `test_aggregation` and
`test_report` do, so it runs in the local 73 - the run that gates a publish -
and the workflow names it among what a runner cannot do.

    python -m unittest tests.test_claims
"""

import io
import pathlib
import re
import unittest

from assay import aggregation, load, structure

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Recomputed 2026-09-19 and equal to what the engine returns today. These are
# not free-standing constants: every test below either derives them again or
# requires some other file to agree with them, so changing one here without
# the engine agreeing fails immediately.
HDI = {
    "n": 193,
    "k": 4,
    "pc1_share": 0.8409571221535493,
    "r_with_pc1": 0.9975743676413303,
}

# How each number is written where a person reads it. A formatting change is a
# real change to a published claim, so it belongs in the test rather than in a
# tolerance.
AS_WRITTEN = {
    "pc1_share": "84.1%",
    "r_with_pc1": "0.9976",
}


def hdi():
    index = load.from_spec(str(ROOT / "data" / "hdi-2022.json"))
    struct = structure.analyse(index)
    pc1 = structure.pc1_scores(index)
    decorative = aggregation.weights_are_decorative(index, pc1)
    return {
        "n": struct["n"],
        "k": struct["k"],
        "pc1_share": struct["pc1_share"],
        "r_with_pc1": decorative["r_with_pc1"],
    }


def read(*parts):
    with io.open(ROOT.joinpath(*parts), encoding="utf-8") as handle:
        return handle.read()


class TheEngineStillProducesThem(unittest.TestCase):
    """The direction that had no check at all."""

    @classmethod
    def setUpClass(cls):
        cls.got = hdi()

    def test_the_shape_of_the_data_is_unchanged(self):
        self.assertEqual(self.got["n"], HDI["n"])
        self.assertEqual(self.got["k"], HDI["k"],
                         "four pillars, with HDI's own total held out")

    def test_pc1_share_is_what_the_readme_claims(self):
        self.assertAlmostEqual(self.got["pc1_share"], HDI["pc1_share"],
                               places=12)

    def test_weights_barely_beat_one_component(self):
        self.assertAlmostEqual(self.got["r_with_pc1"], HDI["r_with_pc1"],
                               places=12)

    def test_the_finding_itself_still_holds(self):
        """Not just the digits - the claim they support.

        84.1% on one component against three claimed dimensions, and a
        published score that is very nearly PC1. If either stopped being true
        the README would need rewriting, not renumbering.
        """
        self.assertGreater(self.got["pc1_share"], 0.80)
        self.assertGreater(self.got["r_with_pc1"], 0.99)

    def test_rounding_to_the_published_form_is_stable(self):
        self.assertEqual("%.1f%%" % (self.got["pc1_share"] * 100),
                         AS_WRITTEN["pc1_share"])
        self.assertEqual("%.4f" % self.got["r_with_pc1"],
                         AS_WRITTEN["r_with_pc1"])


class EveryPlaceThatWritesThemDown(unittest.TestCase):
    """A number in prose is a claim, and this is the list of claims."""

    def test_the_readme_quotes_the_computed_share(self):
        text = read("README.md")
        self.assertIn(AS_WRITTEN["pc1_share"], text,
                      "README no longer states the variance share it claims")

    def test_the_readme_quotes_the_computed_correlation(self):
        self.assertIn(AS_WRITTEN["r_with_pc1"], read("README.md"))

    def test_the_readme_says_how_many_countries(self):
        self.assertIn("%d countries" % HDI["n"], read("README.md"))

    def test_the_browser_test_asserts_the_same_numbers(self):
        """So the two halves of the drift guard cannot separate.

        `test_browser_page.py` runs in CI and this file does not, so if the
        engine moves and that file is left alone, the failure has to surface
        here - naming the file that still carries the old number.
        """
        text = read("tests", "test_browser_page.py")
        for key, written in AS_WRITTEN.items():
            self.assertIn(written, text,
                          "tests/test_browser_page.py no longer asserts %s "
                          "(%s); the browser and the engine are now pinned "
                          "to different numbers" % (key, written))

    def test_the_browser_tests_python_comment_is_current(self):
        """The comment that was the only record of the Python value."""
        text = read("tests", "test_browser_page.py")
        found = re.search(r"#\s*python:\s*([0-9.]+)", text)
        self.assertIsNotNone(found, "the `# python:` note was removed")
        self.assertAlmostEqual(float(found.group(1)), HDI["pc1_share"],
                               places=6,
                               msg="the comment records a value the engine no "
                                   "longer produces")


if __name__ == "__main__":
    unittest.main()
