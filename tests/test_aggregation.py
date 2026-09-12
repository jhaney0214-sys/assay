"""The weighting tests, against indices built to have a known answer.

The case worth reading twice is `test_stability_is_not_robustness`: a
one-dimensional index barely moves under reweighting, and that stability is a
symptom of the redundancy rather than evidence of a sound ranking. If Assay ever
reports the first without the second, this is the test that should fail.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assay import aggregation, structure
from assay.load import Index
from tests.test_structure import genuinely_k, one_dimensional


class TestRanks(unittest.TestCase):

    def test_rank_one_is_the_highest_score(self):
        r = aggregation._ranks([10.0, 30.0, 20.0])
        self.assertEqual(list(r), [3, 1, 2])

    def test_ranks_are_a_permutation(self):
        rng = np.random.default_rng(0)
        r = aggregation._ranks(rng.standard_normal(50))
        self.assertEqual(sorted(r), list(range(1, 51)))


class TestReproduction(unittest.TestCase):

    def test_composite_built_from_weights_is_reproducible(self):
        """A published index adds RAW sub-scores, so the check must too."""
        idx = genuinely_k(k=4)
        weights = {"pillar_0": 40, "pillar_1": 30, "pillar_2": 20, "pillar_3": 10}
        idx.weights = weights
        idx.published = aggregation.weighted_score_raw(idx, [40, 30, 20, 10]).tolist()
        rep = aggregation.reproduces_published(idx)
        self.assertTrue(rep["reproducible"])
        self.assertAlmostEqual(rep["r"], 1.0, places=9)
        self.assertEqual(rep["max_rank_gap"], 0)
        self.assertLess(rep["max_abs_score_error"], 1e-9)

    def test_z_scored_composite_is_NOT_treated_as_reproducing_a_raw_one(self):
        """The bug EPI exposed: weighting z-scores is different arithmetic.

        Columns with different spreads make the two composites disagree, and
        checking the wrong one would call a faithful index irreproducible.
        """
        idx = genuinely_k(k=4, seed=11)
        X = idx.matrix()
        X[:, 0] *= 40.0                      # one column on a much wider scale
        X[:, 1] *= 0.05
        idx.values = X.tolist()
        w = [40, 30, 20, 10]
        idx.weights = dict(zip(idx.columns, w))
        idx.published = aggregation.weighted_score_raw(idx, w).tolist()

        rep = aggregation.reproduces_published(idx)
        self.assertTrue(rep["reproducible"])          # raw: reproduces exactly
        z = aggregation.weighted_score(idx, w)        # standardised: does not
        import numpy as np
        self.assertLess(abs(np.corrcoef(z, idx.published)[0, 1]), 0.99)

    def test_undocumented_processing_shows_up_as_irreproducible(self):
        idx = genuinely_k(k=4, seed=3)
        idx.weights = {"pillar_0": 25, "pillar_1": 25, "pillar_2": 25, "pillar_3": 25}
        # Published score secretly uses a different weighting entirely.
        idx.published = aggregation.weighted_score(idx, [90, 5, 3, 2]).tolist()
        rep = aggregation.reproduces_published(idx)
        self.assertFalse(rep["reproducible"])
        self.assertGreater(rep["max_rank_gap"], 5)

    def test_absent_weights_report_nothing_rather_than_guessing(self):
        idx = genuinely_k()
        idx.published = [0.0] * idx.n
        self.assertIsNone(aggregation.reproduces_published(idx))


class TestDecorative(unittest.TestCase):

    def test_one_dimensional_index_has_decorative_weights(self):
        idx = one_dimensional(k=6)
        idx.published = aggregation.weighted_score(
            idx, [30, 25, 20, 10, 10, 5]).tolist()
        pc1 = structure.pc1_scores(idx)
        dec = aggregation.weights_are_decorative(idx, pc1)
        self.assertTrue(dec["decorative"])

    def test_weights_that_do_work_are_not_called_decorative(self):
        idx = genuinely_k(k=4, seed=1)
        # Almost all weight on one independent pillar: PC1 cannot recover this.
        idx.published = aggregation.weighted_score(idx, [97, 1, 1, 1]).tolist()
        pc1 = structure.pc1_scores(idx)
        dec = aggregation.weights_are_decorative(idx, pc1)
        self.assertFalse(dec["decorative"])


class TestRankEnvelope(unittest.TestCase):

    def test_independent_pillars_make_ranks_move_a_lot(self):
        idx = genuinely_k(n=120, k=5, seed=2)
        env = aggregation.rank_envelope(idx, draws=300, mode="uniform")
        self.assertGreater(env["median_spread"], 10)

    def test_stability_is_not_robustness(self):
        """A one-dimensional index moves less - BECAUSE it is redundant.

        Stated as a comparison at equal field size, not an absolute threshold.
        Raw rank spread scales with how many entities are packed into the field,
        so "under twelve places" means nothing without an n beside it.
        """
        flat = one_dimensional(n=150, k=6, loading=0.95)
        real = genuinely_k(n=150, k=6, seed=5)
        e_flat = aggregation.rank_envelope(flat, draws=300, mode="uniform")
        e_real = aggregation.rank_envelope(real, draws=300, mode="uniform")

        self.assertLess(e_flat["median_spread_share"],
                        e_real["median_spread_share"] / 2)
        s = structure.analyse(flat)
        self.assertGreater(s["pc1_share"], 0.7)            # because it is one thing
        self.assertLess(s["dimensions"]["high"], flat.claimed_dimensions)

    def test_spread_share_is_scale_free(self):
        """The same structure at two field sizes gives a comparable share."""
        small = genuinely_k(n=60, k=4, seed=8)
        large = genuinely_k(n=240, k=4, seed=9)
        a = aggregation.rank_envelope(small, draws=250, mode="uniform")
        b = aggregation.rank_envelope(large, draws=250, mode="uniform")
        self.assertLess(abs(a["median_spread_share"] - b["median_spread_share"]), 0.15)
        # ...while the raw places differ by a lot, which is the whole point.
        self.assertGreater(b["median_spread"], a["median_spread"] * 2)

    def test_best_rank_is_never_worse_than_worst(self):
        idx = genuinely_k(n=60, k=4)
        env = aggregation.rank_envelope(idx, draws=200, mode="uniform")
        self.assertTrue(np.all(env["best"] <= env["worst"]))
        self.assertTrue(np.all(env["spread"] == env["worst"] - env["best"]))

    def test_near_mode_moves_less_than_uniform(self):
        idx = genuinely_k(n=100, k=5, seed=4)
        idx.weights = {c: 20 for c in idx.columns}
        near = aggregation.rank_envelope(idx, draws=300, mode="near",
                                         concentration=200.0)
        wide = aggregation.rank_envelope(idx, draws=300, mode="uniform")
        self.assertLess(near["median_spread"], wide["median_spread"])

    def test_mode_defaults_to_near_only_when_weights_exist(self):
        idx = genuinely_k(n=60, k=4)
        self.assertEqual(aggregation.rank_envelope(idx, draws=50)["mode"], "uniform")
        idx.weights = {c: 25 for c in idx.columns}
        self.assertEqual(aggregation.rank_envelope(idx, draws=50)["mode"], "near")

    def test_envelope_is_reproducible_from_its_seed(self):
        idx = genuinely_k(n=80, k=4)
        a = aggregation.rank_envelope(idx, draws=100, mode="uniform", seed=7)
        b = aggregation.rank_envelope(idx, draws=100, mode="uniform", seed=7)
        self.assertTrue(np.array_equal(a["spread"], b["spread"]))


class TestMovable(unittest.TestCase):

    def test_most_movable_is_sorted_by_spread(self):
        idx = genuinely_k(n=60, k=4)
        env = aggregation.rank_envelope(idx, draws=200, mode="uniform")
        rows = aggregation.most_movable(idx, env, limit=5)
        spreads = [r[4] for r in rows]
        self.assertEqual(spreads, sorted(spreads, reverse=True))
        self.assertEqual(len(rows), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
