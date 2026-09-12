"""The report has to render, in every branch, without being run by hand.

This file exists because a formatting bug shipped: the uniform-weights branch
had a `%d` on one line and its argument on the next, and nothing executed that
path until a real index was audited. Every branch below is one that a real
index will take.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assay import aggregation, report, structure
from tests.test_structure import genuinely_k, one_dimensional


def _render(idx, draws=120, mode=None):
    struct = structure.analyse(idx)
    pc1 = structure.pc1_scores(idx)
    agg = {"reproduces": aggregation.reproduces_published(idx),
           "decorative": aggregation.weights_are_decorative(idx, pc1)}
    env = aggregation.rank_envelope(idx, draws=draws, mode=mode)
    movable = aggregation.most_movable(idx, env, limit=5)
    return report.render(idx, struct, agg, env, movable)


class TestRenders(unittest.TestCase):

    def test_uniform_mode_renders(self):
        """The branch that shipped broken."""
        txt = _render(genuinely_k(n=60, k=4), mode="uniform")
        self.assertIn("upper bound", txt)

    def test_near_mode_renders(self):
        idx = genuinely_k(n=60, k=4)
        idx.weights = {c: 25 for c in idx.columns}
        idx.published = aggregation.weighted_score(idx, [25] * 4).tolist()
        txt = _render(idx, mode="near")
        self.assertIn("OWN published weights", txt)

    def test_one_dimensional_index_renders_and_says_so(self):
        idx = one_dimensional(n=120, k=6)
        idx.published = aggregation.weighted_score(idx, [1] * 6).tolist()
        txt = _render(idx)
        self.assertIn("HOW MANY DIMENSIONS", txt)
        self.assertIn("data supports", txt)

    def test_index_with_no_published_score_renders(self):
        txt = _render(genuinely_k(n=50, k=5))
        self.assertNotIn("r(published score", txt)

    def test_every_branch_produces_the_closing_caveat(self):
        for idx in (genuinely_k(n=50, k=4), one_dimensional(n=80, k=5)):
            self.assertIn("does NOT show", _render(idx))


class TestJointReading(unittest.TestCase):
    """One dominant dimension and volatile ranks can both be true."""

    def test_one_dimension_with_volatile_ranks_says_both(self):
        # Densely packed field, one strong factor, meaningful residual - the
        # shape HDI actually has.
        idx = one_dimensional(n=193, k=4, loading=0.88)
        idx.claimed_dimensions = 3
        txt = _render(idx, draws=200, mode="uniform")
        self.assertIn("not in conflict", txt)
        self.assertIn("band, not a position", txt)

    def test_tight_field_with_one_dimension_says_stability_is_redundancy(self):
        idx = one_dimensional(n=20, k=6, loading=0.99)
        txt = _render(idx, draws=200, mode="uniform")
        self.assertIn("same fact", txt)


class TestSharedConstruction(unittest.TestCase):

    def test_warning_is_rendered_before_the_numbers(self):
        idx = one_dimensional(n=100, k=5)
        idx.shared_construction = {
            "columns": idx.columns[1:],
            "note": "these share a term by construction",
        }
        txt = _render(idx)
        self.assertIn("SHARED CONSTRUCTION", txt)
        self.assertLess(txt.index("SHARED CONSTRUCTION"),
                        txt.index("HOW MANY DIMENSIONS"))

    def test_collapse_is_not_called_a_finding(self):
        idx = one_dimensional(n=100, k=5)
        idx.shared_construction = {"note": "shared term"}
        txt = _render(idx)
        self.assertIn("not a finding", txt)

    def test_without_the_flag_the_collapse_is_reported_plainly(self):
        idx = one_dimensional(n=100, k=5)
        txt = _render(idx)
        self.assertNotIn("SHARED CONSTRUCTION", txt)
        self.assertIn("single underlying thing", txt)



class TestWeakSingleFactor(unittest.TestCase):
    """One factor retained is not the same as one thing measured."""

    def test_weak_pc1_is_not_called_a_single_underlying_thing(self):
        import numpy as np
        from assay.load import Index
        rng = np.random.default_rng(3)
        n, k = 105, 11
        f = rng.standard_normal(n)
        # Loosely related columns: a weak common factor, mostly noise.
        X = np.column_stack([0.45 * f + rng.standard_normal(n) for _ in range(k)])
        idx = Index("loose", ["e%d" % i for i in range(n)],
                    ["c%d" % i for i in range(k)], X.tolist(),
                    claimed_dimensions=3)
        txt = _render(idx, draws=150, mode="uniform")
        self.assertNotIn("single underlying thing", txt)
        self.assertIn("do not resolve into named dimensions", txt)


class TestFlippedSubScores(unittest.TestCase):

    def test_a_flipped_sub_score_is_reported_as_substantive(self):
        import numpy as np
        from assay.load import Index, orient
        idx = one_dimensional(n=90, k=5)
        X = idx.matrix()
        idx.published = X[:, :4].sum(axis=1).tolist()
        X[:, 4] = -X[:, 4]
        idx.values = X.tolist()
        orient(idx)
        txt = _render(idx, draws=120, mode="uniform")
        self.assertIn("RUN AGAINST THEIR OWN INDEX", txt)

class TestReproductionNuance(unittest.TestCase):
    """Ordering can follow from the weights while the score does not."""

    def test_approximate_reproduction_is_not_called_exact(self):
        import numpy as np
        idx = genuinely_k(n=100, k=5, seed=21)
        w = [30, 25, 20, 15, 10]
        idx.weights = dict(zip(idx.columns, w))
        base = aggregation.weighted_score_raw(idx, w)
        rng = np.random.default_rng(5)
        idx.published = (base + rng.standard_normal(idx.n) * 0.10 * base.std()).tolist()
        txt = _render(idx, draws=150, mode="near")
        self.assertIn("ORDERING follows", txt)
        self.assertNotIn("The stated method is the actual method", txt)

    def test_exact_reproduction_is_called_exact(self):
        idx = genuinely_k(n=100, k=5, seed=22)
        w = [30, 25, 20, 15, 10]
        idx.weights = dict(zip(idx.columns, w))
        idx.published = aggregation.weighted_score_raw(idx, w).tolist()
        txt = _render(idx, draws=150, mode="near")
        self.assertIn("The stated method is the actual method", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
