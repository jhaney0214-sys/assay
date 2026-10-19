"""Are the published weights doing any work, and would other weights change the answer?

An index that names six pillars and weights them 20/20/15/15/15/15 is making two
claims at once: that the pillars are distinct, and that those numbers are the
right way to trade them off. `structure.py` tests the first. This tests the
second, and it asks three questions in increasing order of how uncomfortable the
answer tends to be.

1. **Does the published composite follow from the published weights?** If an
   index ships both, the composite should be reproducible. When it is not, the
   difference is undocumented processing - a cap, a rescale, a hidden variable -
   and every downstream comparison is against a number nobody outside can build.

2. **Does the weighting beat one unweighted component?** If the published score
   correlates above about 0.95 with the first principal component of its own
   sub-scores, the weights are decorative: an equal-weighted first component
   would rank the same entities the same way, and the deliberation that produced
   20/20/15/15/15/15 changed nothing.

3. **How far can an entity move under weights the author could also have
   defended?** This is the statistic worth publishing. If a country ranks 4th
   under the published weights and anywhere from 2nd to 29th under small
   perturbations of those same weights, its rank is not a measurement of the
   country. It is a measurement of a committee's choice.

**The trap this module exists to avoid.** A ranking that barely moves under
reweighting looks robust, and gets reported that way. It has a second
explanation that looks identical in the output: the sub-scores are so correlated
that no weighting could separate them. Stability is only evidence of robustness
when the pillars are actually distinct, so nothing here is reported without the
dimensionality result beside it.
"""

import numpy as np

DECORATIVE_AT = 0.95        # r(published, PC1) above this and the weights add nothing
DEFAULT_DRAWS = 2000


def weighted_score(index, weights):
    """Composite from STANDARDISED sub-scores and a weight vector.

    Standardising is right when the question is about structure or about which
    ordering a weighting produces, because it stops the widest-scaled column
    dominating for a reason unrelated to its weight. The sub-scores point the
    way the index publishes them, since these are the index's weights.
    """
    Z = index.standardised(as_published=True)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return Z @ w


def weighted_score_raw(index, weights):
    """Composite from RAW sub-scores - what a published index actually computes.

    Standardising is wrong here and the difference is not cosmetic. An index
    adds its sub-scores on their own scale; weighting z-scores instead is a
    different arithmetic whenever the columns have different spreads, and it
    would make a perfectly reproducible index look irreproducible. EPI, whose
    eleven issue categories vary widely in spread, is the case that exposed it.
    The values are as published, never orientation-flipped (load.as_published).
    """
    X = index.as_published()
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return X @ w


def reproduces_published(index):
    """Does the index's own composite follow from its own stated weights?

    Returns None when the index ships no weights or no published score - an
    absence to report, not a failure to paper over.

    **Only meaningful for a LINEAR composite.** HDI is a geometric mean, GII a
    harmonic mean of gendered geometric means. Running a weighted-sum check
    against either would report "not reproducible" and the fault would be this
    tool's, not the index's - the arithmetic being tested would be arithmetic
    nobody claimed. A spec therefore declares `aggregation_form`, and anything
    but "linear" skips this test rather than failing it.
    """
    if index.published is None or not index.weights:
        return None
    if getattr(index, "aggregation_form", "linear") != "linear":
        return None
    w = [index.weights.get(c, 0.0) for c in index.columns]
    if sum(w) == 0:
        return None
    ours = weighted_score_raw(index, w)
    theirs = np.asarray(index.published, dtype=float)
    r = float(np.corrcoef(ours, theirs)[0, 1])
    rank_ours = _ranks(ours)
    rank_theirs = _ranks(theirs)
    # How far off is the score itself, not just the ordering? A published index
    # on a 0-100 scale should come back within rounding.
    resid = float(np.abs(ours - theirs).max())
    # Scored relative to the published score's own spread, never in absolute
    # points. Half a point is nothing on a 0-100 index and catastrophic on a
    # 0-1 one; an absolute threshold silently means something different for
    # every index audited.
    med = float(np.median(np.abs(ours - theirs)))
    spread = float(np.std(theirs)) or 1.0
    return {
        "r": r,
        "max_rank_gap": int(np.abs(rank_ours - rank_theirs).max()),
        "median_rank_gap": float(np.median(np.abs(rank_ours - rank_theirs))),
        "max_abs_score_error": resid,
        "median_abs_score_error": med,
        "median_error_in_sd": med / spread,
        "reproducible": r >= 0.99,
        "exact": r >= 0.99 and (med / spread) < 0.02,
    }


def weights_are_decorative(index, pc1):
    """Compare the published ranking against one unweighted component."""
    if index.published is None:
        return None
    theirs = np.asarray(index.published, dtype=float)
    r = float(np.corrcoef(theirs, pc1)[0, 1])
    return {
        "r_with_pc1": abs(r),
        "decorative": abs(r) >= DECORATIVE_AT,
    }


def _ranks(scores):
    """Rank 1 = highest score; equal scores share the best rank (1, 2, 2, 4).

    Competition ranking, the way an index publishes ranks. Until 2026-10-19
    ties took whatever order numpy's default sort left them in, which is
    not even defined: two countries tied at HDI 0.947 got ranks 9 and 10 in
    an order nobody chose, and a rank change could come from the sort. The
    browser engine (docs/audit.js `ranks`) does the same, and
    tools/conform_browser.py holds the two together.
    """
    s = -np.asarray(scores, dtype=float)
    # Each score's rank is one past the number of scores strictly above it.
    return np.searchsorted(np.sort(s), s, side="left") + 1


def rank_envelope(index, draws=DEFAULT_DRAWS, mode=None, concentration=50.0, seed=0):
    """Where could each entity rank, under weights the author could have chosen?

    `mode` is "near" - perturbations around the index's own published weights -
    or "uniform", any weighting at all. "near" is the default whenever published
    weights exist, because it is the version that cannot be dismissed: it only
    asks what happens if the committee had argued slightly differently.

    `concentration` controls how near. Higher is tighter; 50 keeps draws close
    to the published vector, 5 is a loose reading of "defensible".
    """
    rng = np.random.default_rng(seed)
    k = index.k
    if mode is None:
        mode = "near" if index.weights else "uniform"

    if mode == "near" and index.weights:
        base = np.array([index.weights.get(c, 0.0) for c in index.columns], float)
        if base.sum() == 0:
            mode, base = "uniform", None
        else:
            base = base / base.sum()
            alpha = np.clip(base * concentration, 0.05, None)
    if mode != "near" or not index.weights:
        alpha = np.ones(k)
        mode = "uniform"

    # Perturbing the PUBLISHED weights only means something on the sub-scores
    # as published; any weighting at all ("uniform") is read over the oriented
    # ones, where every pillar points the way the index calls good.
    Z = index.standardised(as_published=(mode == "near"))
    all_ranks = np.empty((draws, index.n), dtype=np.int32)
    for d in range(draws):
        w = rng.dirichlet(alpha)
        all_ranks[d] = _ranks(Z @ w)

    best = all_ranks.min(axis=0)
    worst = all_ranks.max(axis=0)
    spread = worst - best

    if index.published is not None:
        published_rank = _ranks(index.published)
    else:
        published_rank = _ranks(Z @ (alpha / alpha.sum()))

    top = 10 if index.n >= 20 else max(3, index.n // 4)
    # Top by rank, not by position in a sort, so a tie at the boundary is in.
    published_top = np.flatnonzero(published_rank <= top)
    churn = [int((all_ranks[d][published_top] > top).sum()) for d in range(draws)]

    return {
        "mode": mode,
        "draws": draws,
        "concentration": concentration if mode == "near" else None,
        "best": best,
        "worst": worst,
        "spread": spread,
        "published_rank": published_rank,
        "median_spread": float(np.median(spread)),
        "p90_spread": float(np.percentile(spread, 90)),
        "max_spread": int(spread.max()),
        # Raw places are not comparable between indices. Twenty places means
        # something different in a field of 25 than in a field of 190, and the
        # first version of this module reported only the raw number - which made
        # a tightly-packed 150-entity index look more volatile than a genuinely
        # arbitrary 30-entity one. Share of the field is the comparable figure.
        "median_spread_share": float(np.median(spread)) / index.n,
        "p90_spread_share": float(np.percentile(spread, 90)) / index.n,
        "max_spread_share": float(spread.max()) / index.n,
        "top_n": top,
        "mean_top_churn": float(np.mean(churn)),
        "max_top_churn": int(np.max(churn)),
    }


def composite_basis(index):
    """(values, weights summing to 1, description) for recomputing a composite.

    The published sub-scores with the published weights when the index is a
    weighted sum of them, which is what the index itself computes; otherwise
    z-scores with equal weights, and the description says so, because a sum
    of raw sub-scores on unrelated scales is not a composite of anything.
    Shared by `pairwise_robustness` and `leave_one_out` so the two always
    describe the same composite.
    """
    linear = getattr(index, "aggregation_form", "linear") == "linear"
    if index.weights and linear:
        X = index.as_published()
        w0 = np.array([index.weights.get(c, 0.0) for c in index.columns], float)
        scale = "published sub-scores, published weights"
    else:
        X = index.standardised()
        w0 = np.ones(index.k)
        scale = "z-scores, equal weights"
    return X, w0 / w0.sum(), scale


def leave_one_out(index, top=10):
    """How far the ranking moves when each sub-score is left out in turn.

    The other weights are rescaled to sum to one, so the question is the one
    an index's critics ask: what if this pillar were not in it? The answer
    names the pillar the ranking leans on, which the weights alone do not.
    A pillar can carry a small weight and still move the ranking a long way
    if it is the one the others disagree with, and a heavy one can move it
    little if it repeats them.

    Compared against the composite recomputed with every sub-score, not the
    published score, so that what is measured is the pillar and not whatever
    separates the recomputation from the published number (that is
    `reproduces_published`'s question).
    """
    X, w0, scale = composite_basis(index)
    base = _ranks(X @ w0)
    n = index.n
    top = min(top, n)
    base_top = np.flatnonzero(base <= top)
    rows = []
    for j, col in enumerate(index.columns):
        w = w0.copy()
        w[j] = 0.0
        if w.sum() <= 0:
            continue
        ranks = _ranks(X @ (w / w.sum()))
        moved = np.abs(ranks - base)
        rows.append({
            "column": col,
            "weight": float(w0[j]),
            "median_shift": float(np.median(moved)),
            "p90_shift": float(np.percentile(moved, 90)),
            "max_shift": int(moved.max()),
            "rank_r": float(np.corrcoef(ranks, base)[0, 1]),
            "top_changed": int((ranks[base_top] > top).sum()),
        })
    rows.sort(key=lambda r: (-r["median_shift"], -r["max_shift"]))
    return {"scale": scale, "n": n, "top": top, "rows": rows}


def pairwise_robustness(index, orders=(1.0, 0.25)):
    """Which pairwise comparisons no defensible weighting reverses.

    Foster, McGillivray and Seth (2009, OPHI Working Paper 26) call x above y
    robust over a set of weightings W when the composite puts x above y at
    the initial weights w0 and no w in W reverses it. For W the simplex S_r,
    the weightings within a fraction r of the way from w0 to each corner, it
    is exact and needs no sampling (their Theorem 2): x stays at or above y
    everywhere in S_r if and only if it does at each of S_r's corners, where
    entity x scores (1-r)C0(x) + r x_d. r = 1 is every weighting at all,
    which reduces to x beating y on every sub-score.

    This is the deterministic counterpart of `rank_envelope`, and the only
    weight-sensitivity figure here with a published value to land on: on
    HDI 2004 the paper reports 69.8% of 15,576 comparisons robust at r = 1
    and 91.8% at r = 0.25 (tools/fetch_hdr2006.py, tests/test_pairwise.py).

    Scale: see `composite_basis`.
    """
    X, w0, scale = composite_basis(index)
    c0 = X @ w0
    n = index.n
    total = n * (n - 1) // 2
    out = {"scale": scale, "pairs": total, "orders": []}
    for r in orders:
        V = (1.0 - r) * c0[:, None] + r * X
        robust = 0
        # One entity against every lower-scoring one at a time: n rows of k
        # comparisons, so 3,144 counties is 4.9 million pairs without ever
        # holding them all at once.
        for i in range(n):
            below = c0 < c0[i]
            if below.any():
                robust += int(np.all(V[below] <= V[i], axis=1).sum())
        out["orders"].append({"r": r, "robust": robust, "share": robust / total})
    return out


def most_movable(index, env, limit=10):
    """The entities whose rank is least a property of themselves."""
    rows = []
    for i, name in enumerate(index.entities):
        rows.append((name, int(env["published_rank"][i]), int(env["best"][i]),
                     int(env["worst"][i]), int(env["spread"][i])))
    rows.sort(key=lambda r: -r[4])
    return rows[:limit]
