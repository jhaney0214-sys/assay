"""Does Assay give the right answer on data whose answer is known?

Every test here builds an index whose true structure is decided in advance,
because an audit tool that cannot be audited is worth nothing. The two cases
that matter are the two conclusions the tool is allowed to reach: an index that
really is one thing wearing six names, and an index whose pillars really are
distinct.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assay import aggregation, structure
from assay.load import Index, orient


def one_dimensional(n=150, k=6, loading=0.85, seed=0):
    """k sub-scores that are all the same underlying thing plus noise."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    X = np.column_stack([loading * f + np.sqrt(1 - loading ** 2) * rng.standard_normal(n)
                         for _ in range(k)])
    cols = ["pillar_%d" % i for i in range(k)]
    return Index("one-dimensional", ["e%d" % i for i in range(n)], cols, X.tolist(),
                 claimed_dimensions=k)


def genuinely_k(n=250, k=4, seed=0):
    """k sub-scores that really are k independent things."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    cols = ["pillar_%d" % i for i in range(k)]
    return Index("orthogonal", ["e%d" % i for i in range(n)], cols, X.tolist(),
                 claimed_dimensions=k)


class TestDimensionality(unittest.TestCase):

    def test_one_factor_is_found_as_one(self):
        idx = one_dimensional()
        res = structure.analyse(idx)
        self.assertEqual(res["dimensions"]["parallel"], 1)
        self.assertLess(res["dimensions"]["high"], idx.claimed_dimensions)

    def test_one_factor_concentrates_variance(self):
        res = structure.analyse(one_dimensional())
        self.assertGreater(res["pc1_share"], 0.6)

    def test_orthogonal_columns_are_not_collapsed(self):
        idx = genuinely_k()
        res = structure.analyse(idx)
        # Independent columns must not be reported as one thing.
        self.assertLess(res["pc1_share"], 0.45)

    def test_map_is_withheld_when_there_are_too_few_columns(self):
        idx = one_dimensional(k=3)
        res = structure.analyse(idx)
        self.assertFalse(res["dimensions"]["map_usable"])
        self.assertIsNone(res["dimensions"]["map"])

    def test_map_runs_when_there_are_enough_columns(self):
        res = structure.analyse(one_dimensional(k=8))
        self.assertTrue(res["dimensions"]["map_usable"])

    def test_variance_shares_sum_to_one(self):
        res = structure.analyse(one_dimensional())
        self.assertAlmostEqual(sum(res["variance_explained"]), 1.0, places=6)


class TestRedundancy(unittest.TestCase):

    def test_duplicated_column_is_caught(self):
        idx = one_dimensional(k=4)
        X = idx.matrix()
        X = np.column_stack([X, X[:, 0]])            # exact duplicate
        idx2 = Index("dup", idx.entities, idx.columns + ["copy_of_0"], X.tolist())
        res = structure.analyse(idx2)
        pairs = {(a, b) for a, b, _ in res["redundant"]}
        self.assertIn(("pillar_0", "copy_of_0"), pairs)

    def test_independent_columns_are_not_flagged(self):
        res = structure.analyse(genuinely_k())
        self.assertEqual(res["redundant"], [])


class TestOrientation(unittest.TestCase):

    def test_reversed_column_is_flipped_back(self):
        idx = one_dimensional(k=5)
        X = idx.matrix()
        X[:, 2] = -X[:, 2]
        idx.values = X.tolist()
        idx.published = (X[:, 0] + X[:, 1]).tolist()
        orient(idx)
        self.assertIn("pillar_2", idx.flipped)
        R = structure.correlations(idx)
        self.assertGreater(R[2, 0], 0)

    def test_already_aligned_columns_are_left_alone(self):
        idx = one_dimensional(k=5)
        idx.published = idx.matrix().sum(axis=1).tolist()
        orient(idx)
        self.assertEqual(idx.flipped, [])


class TestPC1(unittest.TestCase):

    def test_pc1_is_oriented_positive(self):
        idx = one_dimensional()
        pc1 = structure.pc1_scores(idx)
        col0 = idx.matrix()[:, 0]
        self.assertGreater(np.corrcoef(pc1, col0)[0, 1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
