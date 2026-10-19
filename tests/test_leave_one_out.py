"""aggregation.leave_one_out: which sub-score the ranking leans on.

Imports nothing from Sextant, so it runs on a runner without the sibling.

Run:  python -m unittest tests.test_leave_one_out
"""

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assay import aggregation, load  # noqa: E402


def index(X, weights=None):
    X = np.asarray(X, float)
    cols = ["c%d" % j for j in range(X.shape[1])]
    w = dict(zip(cols, weights)) if weights is not None else None
    return load.Index("t", ["e%d" % i for i in range(len(X))], cols, X.tolist(), weights=w)


def row(res, col):
    return next(r for r in res["rows"] if r["column"] == col)


class TheMethod(unittest.TestCase):

    def test_a_pillar_that_repeats_another_moves_less_than_an_independent_one(self):
        rng = np.random.default_rng(1)
        a = rng.random(60)
        X = np.c_[a, a, rng.random(60)]
        res = aggregation.leave_one_out(index(X, [1, 1, 1]))
        # Dropping one of two identical pillars halves its share and
        # rescales; dropping the independent one removes all disagreement.
        self.assertGreater(row(res, "c2")["median_shift"], row(res, "c0")["median_shift"])

    def test_a_light_pillar_can_matter_more_than_a_heavy_one(self):
        # c0 and c1 agree; c2 disagrees with both. c2 carries less weight than
        # c0 and still moves the ranking further when it is left out.
        rng = np.random.default_rng(2)
        a = rng.random(100)
        X = np.c_[a + 0.05 * rng.random(100), a + 0.05 * rng.random(100), rng.random(100)]
        res = aggregation.leave_one_out(index(X, [0.4, 0.35, 0.25]))
        self.assertEqual(res["rows"][0]["column"], "c2")
        self.assertGreater(row(res, "c2")["median_shift"], row(res, "c0")["median_shift"])

    def test_a_zero_weight_pillar_moves_nothing(self):
        rng = np.random.default_rng(3)
        res = aggregation.leave_one_out(index(rng.random((40, 3)), [1, 1, 0]))
        r = row(res, "c2")
        self.assertEqual((r["median_shift"], r["max_shift"], r["top_changed"]), (0, 0, 0))
        self.assertAlmostEqual(r["rank_r"], 1.0)

    def test_it_reads_the_same_composite_as_the_pairwise_test(self):
        rng = np.random.default_rng(4)
        idx = index(rng.random((30, 3)))
        self.assertEqual(aggregation.leave_one_out(idx)["scale"],
                         aggregation.pairwise_robustness(idx)["scale"])


class EPI(unittest.TestCase):

    def test_epi_leans_on_biodiversity_more_than_on_its_heaviest_pillar(self):
        # Climate Change Mitigation carries 30% of EPI's weight and
        # Biodiversity & Habitat 25%, but leaving out Biodiversity moves the
        # median country 7 places and leaving out Climate 4.
        res = aggregation.leave_one_out(load.from_spec(str(ROOT / "data" / "epi-2024.json")))
        self.assertEqual(res["scale"], "published sub-scores, published weights")
        first = res["rows"][0]
        self.assertEqual(first["column"], "Biodiversity & Habitat (BDH)")
        climate = row(res, "Climate Change Mitigation (CCH)")
        self.assertGreater(climate["weight"], first["weight"])
        self.assertEqual((first["median_shift"], climate["median_shift"]), (7.0, 4.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
