"""aggregation.pairwise_robustness, and the external figure it is held to.

Foster, McGillivray and Seth (2009, OPHI Working Paper 26) report how many
HDI 2004 country comparisons no reweighting reverses. Their four figures, and
what Assay gets on the HDI 2004 table as HDR 2006 printed it:

    figure                      paper    Assay, within the table's rounding
    all 177, every weighting    10,875   10,820 - 10,889 (95% of draws)
    all 177, r = 0.25           91.8%    91.9% - 92.1%
    top ten, every weighting    4 of 45  1 - 5
    top ten, r = 0.25           21 of 45 16 - 24

Three land inside the range the printed table's rounding allows. The fourth
misses by about 0.2 points, some 30 pairs of 15,576, and is not explained.

Also here: the regression test for applying an index's weights to a sub-score
the loader had turned round (fixed 2026-10-19).

Imports nothing from Sextant, so it runs on a runner without the sibling.

Run:  python -m unittest tests.test_pairwise
"""

import itertools
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assay import aggregation, load  # noqa: E402

HDR2006 = ROOT / "data" / "hdr2006-hdi-2004.json"


def index(X, weights=None, name="t"):
    X = np.asarray(X, float)
    cols = ["c%d" % j for j in range(X.shape[1])]
    w = dict(zip(cols, weights)) if weights is not None else None
    return load.Index(name, ["e%d" % i for i in range(len(X))], cols, X.tolist(), weights=w)


def brute_force(X, w0, r, grid=40):
    """Robust pairs by sampling S_r densely, to check the corner rule against."""
    X = np.asarray(X, float)
    k = X.shape[1]
    corners = [(1 - r) * w0 + r * np.eye(k)[d] for d in range(k)]
    mixes = [m for m in itertools.product(range(grid + 1), repeat=k) if sum(m) == grid]
    W = np.array([sum(a / grid * c for a, c in zip(m, corners)) for m in mixes])
    c0 = X @ w0
    S = X @ W.T
    robust = 0
    for i in range(len(X)):
        for j in range(len(X)):
            if c0[i] > c0[j] and np.all(S[i] >= S[j] - 1e-12):
                robust += 1
    return robust


class TheMethod(unittest.TestCase):

    def test_the_corner_rule_agrees_with_sampling_the_whole_set(self):
        rng = np.random.default_rng(3)
        X = rng.random((25, 3))
        w0 = np.array([0.5, 0.3, 0.2])
        res = aggregation.pairwise_robustness(index(X, w0), orders=(1.0, 0.5, 0.25))
        for o in res["orders"]:
            self.assertEqual(o["robust"], brute_force(X, w0, o["r"]), o["r"])

    def test_every_weighting_means_better_on_every_sub_score(self):
        X = [[3, 3, 3], [2, 2, 2], [4, 1, 1], [1, 1, 0.5]]
        res = aggregation.pairwise_robustness(index(X, [1, 1, 1]), orders=(1.0,))
        # [3,3,3] beats [2,2,2] and [1,1,.5]; [2,2,2] beats [1,1,.5];
        # [4,1,1] beats [1,1,.5]. [3,3,3] over [4,1,1] is reversed by weight on c0.
        self.assertEqual(res["orders"][0]["robust"], 4)
        self.assertEqual(res["pairs"], 6)

    def test_robustness_only_falls_as_the_set_widens(self):
        rng = np.random.default_rng(4)
        res = aggregation.pairwise_robustness(index(rng.random((60, 4)), [1, 2, 3, 4]),
                                              orders=(0.1, 0.25, 0.5, 1.0))
        counts = [o["robust"] for o in res["orders"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_without_published_weights_it_says_it_used_z_scores(self):
        rng = np.random.default_rng(5)
        res = aggregation.pairwise_robustness(index(rng.random((30, 3))))
        self.assertEqual(res["scale"], "z-scores, equal weights")


class TheWeightsMeetTheSubScoresAsPublished(unittest.TestCase):
    """The orientation flip is for reading structure, never for weighting."""

    def test_a_turned_round_sub_score_still_reproduces_the_index(self):
        rng = np.random.default_rng(6)
        X = rng.random((80, 3))
        X[:, 2] = 1 - X[:, 0] + 0.05 * rng.random(80)      # runs against the rest
        w = [0.45, 0.45, 0.10]
        idx = index(X, w)
        idx.published = (X @ np.array(w)).tolist()
        load.orient(idx)
        self.assertEqual(idx.flipped, ["c2"])
        rep = aggregation.reproduces_published(idx)
        self.assertTrue(rep["exact"])
        self.assertLess(rep["max_abs_score_error"], 1e-9)

    def test_as_published_undoes_exactly_the_flip(self):
        idx = index([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]] * 3, [1, 1])
        idx.published = [r[0] for r in idx.values]
        load.orient(idx)
        self.assertEqual(idx.flipped, ["c1"])
        self.assertEqual(idx.as_published()[:5, 1].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(idx.matrix()[:5, 1].tolist(), [-5, -4, -3, -2, -1])

    def test_epi_reproduces_from_its_published_weights(self):
        # With Fisheries turned round, this read a typical error of 2.2
        # points, and the README published it as a finding about EPI.
        idx = load.from_spec(str(ROOT / "data" / "epi-2024.json"))
        self.assertIn("Fisheries (FSH)", idx.flipped)
        rep = aggregation.reproduces_published(idx)
        self.assertGreater(rep["r"], 0.998)
        self.assertLess(rep["median_abs_score_error"], 0.3)


@unittest.skipUnless(HDR2006.exists(), "data/hdr2006-hdi-2004.json not built")
class FosterMcGillivraySeth(unittest.TestCase):
    """HDI 2004, as HDR 2006 printed it, against the paper's four figures."""

    @classmethod
    def setUpClass(cls):
        cls.idx = load.from_spec(str(HDR2006))
        cls.X = cls.idx.matrix()

    def _rounding_range(self, X, draws, seed=0):
        """Counts over draws of the indices within their printed rounding."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(draws):
            Xj = X + rng.uniform(-0.005, 0.005, X.shape)
            res = aggregation.pairwise_robustness(index(Xj, [1, 1, 1]))
            out.append([o["robust"] for o in res["orders"]])
        return np.array(out)

    def test_the_table_is_the_papers(self):
        self.assertEqual(self.idx.n, 177)
        self.assertEqual(aggregation.pairwise_robustness(self.idx)["pairs"], 15576)
        # Before 2010 the HDI was the plain mean of the three indices.
        self.assertTrue(aggregation.reproduces_published(self.idx)["exact"])

    def test_the_printed_table_finds_more_robust_pairs_than_the_paper(self):
        # Two-decimal indices tie often, and a tie is never a reversal.
        res = aggregation.pairwise_robustness(self.idx)
        self.assertEqual([o["robust"] for o in res["orders"]], [11156, 14373])

    def test_full_robustness_lands_inside_the_rounding(self):
        counts = self._rounding_range(self.X, 300)[:, 0]
        lo, hi = np.percentile(counts, [2.5, 97.5])
        self.assertLessEqual(lo, 10875)
        self.assertGreaterEqual(hi, 10875)

    def test_order_quarter_misses_by_about_a_fifth_of_a_point(self):
        # The one figure that does not land: 91.8% in the paper, 92.0% here at
        # the median. Pinned as a near miss so a change in either direction
        # is seen, not as agreement.
        shares = self._rounding_range(self.X, 300)[:, 1] / 15576
        self.assertGreater(np.percentile(shares, 2.5), 0.918)
        self.assertLess(abs(np.median(shares) - 0.918), 0.005)

    def test_the_top_ten_land_inside_the_rounding(self):
        counts = self._rounding_range(self.X[:10], 1000)
        for col, paper in ((0, 4), (1, 21)):
            lo, hi = np.percentile(counts[:, col], [2.5, 97.5])
            self.assertLessEqual(lo, paper)
            self.assertGreaterEqual(hi, paper)


if __name__ == "__main__":
    unittest.main(verbosity=2)
