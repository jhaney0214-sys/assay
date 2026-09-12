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
    dominating for a reason unrelated to its weight.
    """
    Z = index.standardised()
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
    """
    X = index.matrix()
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
    """Rank 1 = highest score."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


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

    Z = index.standardised()
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
    published_top = set(np.argsort(published_rank)[:top])
    churn = [len(published_top - set(np.argsort(all_ranks[d])[:top])) for d in range(draws)]

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


def most_movable(index, env, limit=10):
    """The entities whose rank is least a property of themselves."""
    rows = []
    for i, name in enumerate(index.entities):
        rows.append((name, int(env["published_rank"][i]), int(env["best"][i]),
                     int(env["worst"][i]), int(env["spread"][i])))
    rows.sort(key=lambda r: -r[4])
    return rows[:limit]
