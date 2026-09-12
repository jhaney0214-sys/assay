"""The drift analysis, against movements whose robustness is known in advance.

The test that matters most is `test_pairing_cancels_shared_uncertainty`. A
naive version of this analysis compares a rank change against the static rank
envelope, and that is wrong: a country whose rank is uncertain by twenty places
can still have moved up robustly, because the same weighting uncertainty
applies to both years and cancels. If the pairing is ever lost, that test fails.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assay import drift
from assay.load import Index


def make(n, k, seed=0, shift=None):
    """An index of n entities on k sub-scores; `shift` adds to one column."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    if shift is not None:
        col, who, amount = shift
        X[who, col] += amount
    cols = ["c%d" % i for i in range(k)]
    return Index("t", ["e%d" % i for i in range(n)], cols, X.tolist(),
                 weights={c: 1.0 for c in cols})


class TestRobustness(unittest.TestCase):

    def test_a_move_driven_by_one_sub_score_is_not_robust(self):
        """Improve only on c0; downweighting c0 reverses the move."""
        before = make(80, 4, seed=1)
        X = before.matrix().copy()
        X[5, 0] += 3.0                      # entity 5 improves on one column only
        after = Index("t2", before.entities, before.columns, X.tolist(),
                      weights=before.weights)
        w = {c: 1.0 for c in before.columns}
        res = drift.rank_change_robustness(before, after, w, draws=400,
                                           concentration=6.0)
        i = res["countries"].index("e5")
        self.assertGreater(res["base_delta"][i], 0)      # it did rise
        self.assertLess(res["agreement"][i], 1.0)        # but not under all weights

    def test_a_move_across_every_sub_score_is_robust(self):
        before = make(80, 4, seed=2)
        X = before.matrix().copy()
        X[7, :] += 3.0                      # improves on all columns
        after = Index("t2", before.entities, before.columns, X.tolist(),
                      weights=before.weights)
        res = drift.rank_change_robustness(before, after,
                                           {c: 1.0 for c in before.columns},
                                           draws=400, concentration=6.0)
        i = res["countries"].index("e7")
        self.assertGreater(res["base_delta"][i], 0)
        self.assertEqual(res["agreement"][i], 1.0)

    def test_pairing_cancels_shared_uncertainty(self):
        """An unusual entity can have a wide static envelope and a robust move.

        Its profile makes its RANK weight-sensitive, but the same sensitivity
        applies in both years, so the CHANGE survives. Comparing the change to
        the static envelope instead would wrongly call this unreliable.
        """
        from assay.aggregation import rank_envelope
        before = make(90, 5, seed=3)
        X = before.matrix().copy()
        # Lopsided LOW, not high: a top-ranked entity cannot rise, so delta
        # pins at 0 under some weights and the strict same-sign test counts
        # that as disagreement. The ceiling is real behaviour, not a bug -
        # it just is not what this test is about.
        X[11, 0] -= 4.0                      # lopsided profile -> wide envelope
        before = Index("b", before.entities, before.columns, X.tolist(),
                       weights=before.weights)
        Y = X.copy()
        Y[11, :] += 2.0                      # then improves on everything
        after = Index("a", before.entities, before.columns, Y.tolist(),
                      weights=before.weights)

        env = rank_envelope(before, draws=400, mode="uniform")
        i_env = before.entities.index("e11")
        self.assertGreater(env["spread"][i_env], 8)      # wide static envelope

        res = drift.rank_change_robustness(before, after,
                                           {c: 1.0 for c in before.columns},
                                           draws=400, concentration=6.0)
        i = res["countries"].index("e11")
        # Robust, not necessarily unanimous - an occasional draw ties.
        self.assertTrue(res["stable"][i])
        self.assertGreater(res["agreement"][i], 0.99)

    def test_only_countries_in_both_years_are_compared(self):
        before = make(60, 3, seed=4)
        after = Index("a", before.entities[:50], before.columns,
                      before.matrix()[:50].tolist(), weights=before.weights)
        res = drift.rank_change_robustness(before, after,
                                           {c: 1.0 for c in before.columns},
                                           draws=100)
        self.assertEqual(len(res["countries"]), 50)
        self.assertEqual(res["excluded"], 10)

    def test_too_few_common_entities_raises(self):
        before = make(60, 3, seed=5)
        after = Index("a", before.entities[:5], before.columns,
                      before.matrix()[:5].tolist(), weights=before.weights)
        with self.assertRaises(ValueError):
            drift.rank_change_robustness(before, after,
                                         {c: 1.0 for c in before.columns})


class TestThreshold(unittest.TestCase):

    def test_threshold_needs_every_larger_bin_to_hold(self):
        bins = [{"move": 1, "n": 10, "stable_share": 0.2},
                {"move": 2, "n": 10, "stable_share": 0.95},
                {"move": 3, "n": 10, "stable_share": 0.5},   # dips again
                {"move": 4, "n": 10, "stable_share": 0.99}]
        # 4 holds, 3 does not, so the answer is 4 - not 2.
        self.assertEqual(drift.reliable_threshold(bins), 4)

    def test_threshold_is_none_when_nothing_is_reliable(self):
        bins = [{"move": 1, "n": 10, "stable_share": 0.1},
                {"move": 2, "n": 10, "stable_share": 0.2}]
        self.assertIsNone(drift.reliable_threshold(bins))


class TestFidelity(unittest.TestCase):

    def test_a_faithful_model_reports_near_perfect_rank_agreement(self):
        idx = make(70, 4, seed=6)
        w = {c: 1.0 for c in idx.columns}
        idx.published = (idx.matrix() @ np.full(4, 0.25)).tolist()
        f = drift.model_fidelity(idx, w)
        self.assertGreater(f["rank_r"], 0.999)
        self.assertEqual(f["max_rank_gap"], 0)

class TestDeviationWeights(unittest.TestCase):
    """The interpretable parameterisation must actually mean what it says."""

    def test_no_weight_moves_further_than_delta(self):
        rng = np.random.default_rng(0)
        base = np.array([1/3, 1/3, 1/3])
        for delta in (0.01, 0.05, 0.10):
            W = drift._draw_weights_within(base, 500, delta, rng)
            self.assertLessEqual(np.abs(W - base).max(), delta + 1e-9)

    def test_weights_still_sum_to_one(self):
        rng = np.random.default_rng(1)
        W = drift._draw_weights_within(np.array([0.5, 0.3, 0.2]), 300, 0.08, rng)
        self.assertTrue(np.allclose(W.sum(axis=1), 1.0))

    def test_weights_are_never_negative(self):
        rng = np.random.default_rng(2)
        W = drift._draw_weights_within(np.array([0.05, 0.475, 0.475]), 400, 0.10, rng)
        self.assertGreaterEqual(W.min(), 0.0)


class TestSweep(unittest.TestCase):

    def test_allowing_more_disagreement_never_increases_robustness(self):
        before = make(70, 4, seed=7)
        X = before.matrix().copy()
        X[:, 0] += np.linspace(0, 2, 70)
        after = Index("a", before.entities, before.columns, X.tolist(),
                      weights=before.weights)
        panel = {2000: before, 2001: after}
        w = {c: 1.0 for c in before.columns}
        rows = drift.sweep(panel, w, [(2000, 2001)], [0.01, 0.05, 0.15], draws=200)
        shares = [r["robust_share"] for r in rows]
        self.assertGreaterEqual(shares[0], shares[-1])

    def test_bound_is_the_smallest_threshold(self):
        rows = [{"delta": 0.01, "threshold": 2, "robust_share": .8, "moved": 10},
                {"delta": 0.05, "threshold": 4, "robust_share": .5, "moved": 10}]
        self.assertEqual(drift.parameter_free_bound(rows), 2)

    def test_bound_is_none_if_any_setting_has_no_threshold(self):
        rows = [{"delta": 0.01, "threshold": 2, "robust_share": .8, "moved": 10},
                {"delta": 0.05, "threshold": None, "robust_share": .5, "moved": 10}]
        self.assertIsNone(drift.parameter_free_bound(rows))

class TestChangeFidelity(unittest.TestCase):
    """A model can miss the level and still reproduce the change."""

    def test_a_constant_offset_cancels_in_the_change(self):
        before = make(80, 3, seed=30)
        X = before.matrix().copy()
        # A real, directional movement - not noise. Sign agreement is
        # meaningless when most changes are zero places, which is what a
        # noise-only fixture produces.
        Y = X + np.linspace(-1.5, 1.5, before.n)[:, None] + rng_shift(X.shape, 40)
        after = Index("a", before.entities, before.columns, Y.tolist(),
                      weights=before.weights)
        w = {c: 1.0 for c in before.columns}
        # Published = model score plus a fixed per-country offset, identical in
        # both periods. Levels disagree; the change does not.
        # Small relative to the score's own spread (~0.58 here). An offset of
        # several standard deviations would not model "undocumented processing",
        # it would model a different index.
        off = np.linspace(-0.12, 0.12, before.n)
        before.published = (X @ np.full(3, 1/3) + off).tolist()
        after.published = (Y @ np.full(3, 1/3) + off).tolist()

        level = drift.model_fidelity(before, w)
        chg = drift.change_fidelity(before, after, w)
        # Ranks are ordinal, so a per-country offset perturbs them in both
        # periods and the cancellation is partial, not exact. The claim that
        # matters - and the one EPI exhibits - is that the CHANGE survives far
        # better than the level does.
        self.assertGreater(level["max_rank_gap"], 0)      # levels disagree
        self.assertGreater(chg["r"], 0.95)                # the change tracks
        self.assertGreater(chg["same_direction"], 0.90)

    def test_change_fidelity_needs_both_published(self):
        before = make(40, 3, seed=31)
        after = Index("a", before.entities, before.columns,
                      before.matrix().tolist(), weights=before.weights)
        self.assertIsNone(drift.change_fidelity(before, after,
                                                {c: 1.0 for c in before.columns}))


def rng_shift(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape) * 0.4


class TestTimescale(unittest.TestCase):

    def test_a_wider_gap_is_at_least_as_robust(self):
        """More real movement accumulates, so a longer gap survives better."""
        rng = np.random.default_rng(40)
        cols = ["c0", "c1", "c2"]
        n = 60
        base = rng.standard_normal((n, 3))
        panel = {}
        for t in range(11):
            drifted = base + rng.standard_normal((n, 3)) * 0.15 + t * 0.10 * np.linspace(-1, 1, n)[:, None]
            panel[2000 + t] = Index("y%d" % t, ["e%d" % i for i in range(n)],
                                    cols, drifted.tolist(),
                                    weights={c: 1.0 for c in cols})
        rows = drift.timescale(panel, {c: 1.0 for c in cols}, [1, 10],
                               delta=0.05, draws=200)
        self.assertEqual([r["gap"] for r in rows], [1, 10])
        self.assertGreaterEqual(rows[-1]["robust_share"], rows[0]["robust_share"])

    def test_a_gap_with_no_pairs_is_skipped(self):
        panel = {2000: make(40, 3, seed=41), 2001: make(40, 3, seed=42)}
        rows = drift.timescale(panel, {c: 1.0 for c in panel[2000].columns},
                               [1, 50], delta=0.05, draws=100)
        self.assertEqual([r["gap"] for r in rows], [1])

def _drifting_panel(n=60, years=11, seed=50):
    """A panel with steady real movement, so gaps accumulate signal."""
    rng = np.random.default_rng(seed)
    cols = ["c0", "c1", "c2"]
    base = rng.standard_normal((n, 3))
    trend = np.linspace(-1, 1, n)[:, None]
    panel = {}
    for t in range(years):
        X = base + t * 0.10 * trend + rng.standard_normal((n, 3)) * 0.15
        panel[2000 + t] = Index("y", ["e%d" % i for i in range(n)], cols,
                                X.tolist(), weights={c: 1.0 for c in cols})
    return panel, {c: 1.0 for c in cols}


class TestSurface(unittest.TestCase):

    def test_robustness_falls_in_both_parameters(self):
        panel, w = _drifting_panel()
        years = sorted(panel)
        pairs = list(zip(years, years[1:]))
        cells = drift.surface(panel, w, pairs, [0.01, 0.10], [0.75, 0.99],
                              draws=200)
        inv = drift.invariants(cells)
        self.assertTrue(inv["monotone_in_tolerance"])
        self.assertTrue(inv["monotone_in_bar"])
        self.assertLess(inv["share_min"], inv["share_max"])

    def test_every_cell_of_the_grid_is_present(self):
        panel, w = _drifting_panel()
        years = sorted(panel)
        cells = drift.surface(panel, w, list(zip(years, years[1:])),
                              [0.01, 0.05, 0.10], [0.75, 0.90, 0.99], draws=150)
        self.assertEqual(len(cells), 9)

    def test_the_bar_is_applied_after_the_draws_not_during(self):
        """Two bars on one tolerance must come from the same simulation.

        If the bar leaked into the draws, the same tolerance would give
        different underlying agreement for different bars and the surface
        would not be comparable across its own columns.
        """
        panel, w = _drifting_panel()
        years = sorted(panel)
        pairs = list(zip(years, years[1:]))
        mag, agree = drift._pooled(panel, w, pairs, 0.05, 200, 0)
        cells = drift.surface(panel, w, pairs, [0.05], [0.80, 0.95], draws=200)
        moved = mag > 0
        for c in cells:
            expected = float((agree >= c["bar"])[moved].mean())
            self.assertAlmostEqual(c["robust_share"], expected, places=9)


class TestGradient(unittest.TestCase):

    def test_a_rising_gradient_is_reported_as_holding(self):
        rows = [{"gap": 1, "cells": [{"delta": .01, "bar": .9, "robust_share": .3}]},
                {"gap": 5, "cells": [{"delta": .01, "bar": .9, "robust_share": .6}]}]
        self.assertTrue(drift.gradient_holds(rows))

    def test_one_bad_corner_breaks_the_claim(self):
        rows = [{"gap": 1, "cells": [{"delta": .01, "bar": .9, "robust_share": .3},
                                     {"delta": .10, "bar": .9, "robust_share": .5}]},
                {"gap": 5, "cells": [{"delta": .01, "bar": .9, "robust_share": .6},
                                     {"delta": .10, "bar": .9, "robust_share": .4}]}]
        self.assertFalse(drift.gradient_holds(rows))

    def test_a_single_gap_cannot_establish_a_gradient(self):
        self.assertIsNone(drift.gradient_holds(
            [{"gap": 1, "cells": [{"delta": .01, "bar": .9, "robust_share": .3}]}]))


class TestInvariants(unittest.TestCase):

    def test_incomplete_thresholds_are_flagged(self):
        cells = [{"delta": .01, "bar": .9, "robust_share": .8, "threshold": 2, "n": 10},
                 {"delta": .10, "bar": .9, "robust_share": .3, "threshold": None, "n": 10}]
        inv = drift.invariants(cells)
        self.assertFalse(inv["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
