"""Is a reported year-over-year rank change bigger than the method's own uncertainty?

Rank movements get reported as events. "Country X climbed four places" is a
headline, a ministry press release, and sometimes a policy argument. The move is
computed under one particular weighting, chosen by a committee, and nobody
checks what the same two years of data would have said under a weighting the
same committee could equally have defended.

That is what this module measures, and it is the question Assay was adapted to
answer rather than the one it was built for.

The test is paired, which matters. It would be weaker to compare a rank change
against the static rank envelope: a country whose rank is uncertain by twenty
places can still have moved up robustly, because the same weighting uncertainty
applies to both years and cancels. So each weight draw is applied to BOTH years
and the change is recomputed:

    for each candidate weighting w:
        delta(w) = rank_before(w) - rank_after(w)

If delta keeps its sign across the draws, the movement survives the arbitrary
part of the method. If it flips, the reported change is a property of the
weighting rather than of the country - and the direction of the headline was
decided by a choice, not by the data.

**What this is not.** It says nothing about whether the underlying indicators
moved, or whether the move was caused by policy. A country can genuinely improve
and still show an unstable rank change, because rank is positional and depends
on everyone else. The claim is narrow and about reporting: a rank change smaller
than the method's own uncertainty should not be read as news.
"""

import collections

import numpy as np

from .aggregation import _ranks

SIGN_STABLE_AT = 0.90       # share of draws agreeing on direction


def panel_years(index_by_year):
    return sorted(index_by_year)


def model_fidelity(index, weights):
    """How well does the linear model track the index's own published score?

    Reported before anything else. HDI is a geometric mean, so the linear
    composite used here is a MODEL of it; if the model does not track the
    published score, every downstream statement is about the model instead.
    """
    if index.published is None:
        return None
    w = np.array([weights[c] for c in index.columns], float)
    ours = index.matrix() @ (w / w.sum())
    theirs = np.asarray(index.published, float)
    return {
        "r": float(np.corrcoef(ours, theirs)[0, 1]),
        "rank_r": float(np.corrcoef(_ranks(ours), _ranks(theirs))[0, 1]),
        "max_rank_gap": int(np.abs(_ranks(ours) - _ranks(theirs)).max()),
    }


def change_fidelity(before, after, weights):
    """Does the model reproduce the published CHANGE, not just the level?

    This is the fidelity test that matters for drift, and it is not the same as
    `model_fidelity`. A model can sit a few places off the published ranking in
    every year - because of undocumented processing between an index's
    components and its headline - and still track the year-to-year MOVEMENT
    exactly, because a constant offset cancels in a difference.

    EPI forced this. Its published score is not reproducible from its own
    published policy objectives (a country lands up to 1.2 points out), so the
    level fidelity is mediocre; whether the change survives is a separate
    question and has to be asked separately.
    """
    if before.published is None or after.published is None:
        return None
    common = [c for c in before.entities if c in set(after.entities)]
    bi = {c: i for i, c in enumerate(before.entities)}
    ai = {c: i for i, c in enumerate(after.entities)}
    w = np.array([weights[c] for c in before.columns], float)
    w = w / w.sum()

    mb = before.matrix()[[bi[c] for c in common]] @ w
    ma = after.matrix()[[ai[c] for c in common]] @ w
    pb = np.asarray(before.published, float)[[bi[c] for c in common]]
    pa = np.asarray(after.published, float)[[ai[c] for c in common]]

    model_delta = _ranks(mb) - _ranks(ma)
    pub_delta = _ranks(pb) - _ranks(pa)
    same_sign = float((np.sign(model_delta) == np.sign(pub_delta)).mean())
    return {
        "n": len(common),
        "r": float(np.corrcoef(model_delta, pub_delta)[0, 1]),
        "same_direction": same_sign,
        "max_gap": int(np.abs(model_delta - pub_delta).max()),
        "median_gap": float(np.median(np.abs(model_delta - pub_delta))),
    }


def structural_drift(index_by_year, seed=0):
    """Does the index's dimensional structure change over the panel?"""
    from .structure import analyse
    out = []
    for y in panel_years(index_by_year):
        idx = index_by_year[y]
        if idx.n < 30:
            continue
        s = analyse(idx, seed=seed)
        out.append({
            "year": y,
            "n": idx.n,
            "pc1_share": s["pc1_share"],
            "dimensions": s["dimensions"]["high"],
        })
    return out


def _draw_weights(base, draws, concentration, rng):
    alpha = np.clip(np.asarray(base, float) / np.sum(base) * concentration, 0.05, None)
    return rng.dirichlet(alpha, size=draws)


def _draw_weights_within(base, draws, delta, rng):
    """Weightings differing from the published ones by at most `delta` each.

    Dirichlet concentration was the first parameterisation here and it is a bad
    one: nobody can say whether a concentration of 50 is a reasonable amount of
    disagreement, so the headline became a function of an opaque knob. The
    threshold moved from 4 places to 9 across plausible settings, which is
    exactly the failure Assay exists to catch, committed by Assay.

    `delta` is in the same units as the weights. For HDI's three equal thirds,
    delta = 0.05 means each dimension may be weighted anywhere from 0.283 to
    0.383 - a claim a reader can judge without knowing what a Dirichlet is.

    Directions are drawn in the simplex tangent plane (components summing to
    zero, so the weights keep summing to one) and scaled to a random fraction
    of delta, giving the whole neighbourhood rather than only its surface.
    """
    base = np.asarray(base, float)
    base = base / base.sum()
    k = base.size
    d = rng.standard_normal((draws, k))
    d -= d.mean(axis=1, keepdims=True)                    # stay on the simplex
    scale = np.abs(d).max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    d /= scale                                            # max component = 1
    t = rng.random((draws, 1)) * delta
    W = np.clip(base + t * d, 0.0, None)
    return W / W.sum(axis=1, keepdims=True)


def rank_change_robustness(before, after, weights, draws=1000,
                           concentration=50.0, seed=0, delta=None):
    """Does each country's rank change keep its direction under other weights?

    Both years are ranked over the SAME country set - the intersection - so a
    change cannot come from the field growing. Countries entering or leaving the
    index between the two years are excluded and counted.
    """
    common = [c for c in before.entities if c in set(after.entities)]
    if len(common) < 20:
        raise ValueError("only %d countries in both years" % len(common))
    bi = {c: i for i, c in enumerate(before.entities)}
    ai = {c: i for i, c in enumerate(after.entities)}
    Xb = before.matrix()[[bi[c] for c in common]]
    Xa = after.matrix()[[ai[c] for c in common]]

    base = np.array([weights[c] for c in before.columns], float)
    base = base / base.sum()
    base_delta = _ranks(Xb @ base) - _ranks(Xa @ base)     # positive = improved

    rng = np.random.default_rng(seed)
    W = (_draw_weights_within(base, draws, delta, rng) if delta is not None
         else _draw_weights(base, draws, concentration, rng))
    deltas = np.empty((draws, len(common)), dtype=np.int32)
    for d in range(draws):
        deltas[d] = _ranks(Xb @ W[d]) - _ranks(Xa @ W[d])

    sign = np.sign(base_delta)
    agree = np.where(
        sign == 0,
        (deltas == 0).mean(axis=0),
        (np.sign(deltas) == sign).mean(axis=0),
    )
    return {
        "countries": common,
        "excluded": len(before.entities) - len(common),
        "base_delta": base_delta,
        "agreement": agree,
        "delta_p10": np.percentile(deltas, 10, axis=0),
        "delta_p90": np.percentile(deltas, 90, axis=0),
        "stable": agree >= SIGN_STABLE_AT,
        "draws": draws,
        "concentration": None if delta is not None else concentration,
        "delta": delta,
    }


def stability_by_size(res, max_move=12):
    """At what size does a reported move become trustworthy?

    Bins countries by how far they moved under the published weighting and
    reports what share of each bin keeps its direction. This is the number a
    reader needs: the smallest move worth reporting.
    """
    out = []
    mag = np.abs(res["base_delta"])
    for m in range(0, max_move + 1):
        sel = mag == m
        if sel.sum() == 0:
            continue
        out.append({
            "move": m,
            "n": int(sel.sum()),
            "stable_share": float(res["stable"][sel].mean()),
        })
    big = mag > max_move
    if big.sum():
        out.append({
            "move": max_move + 1,       # rendered as "13+"
            "n": int(big.sum()),
            "stable_share": float(res["stable"][big].mean()),
        })
    return out


def reliable_threshold(bins, need=0.90):
    """Smallest move size at which that size and every larger one are reliable."""
    ok = None
    for b in reversed(bins):
        if b["stable_share"] >= need:
            ok = b["move"]
        else:
            break
    return ok


def summarise_pairs(index_by_year, weights, pairs, draws=1000,
                    concentration=50.0, seed=0, delta=None):
    """Run the robustness test over many consecutive-year pairs at once."""
    rows, all_bins = [], collections.defaultdict(lambda: [0, 0.0])
    for y0, y1 in pairs:
        res = rank_change_robustness(index_by_year[y0], index_by_year[y1],
                                     weights, draws=draws,
                                     concentration=concentration, seed=seed,
                                     delta=delta)
        moved = np.abs(res["base_delta"]) > 0
        rows.append({
            "from": y0,
            "to": y1,
            "n": len(res["countries"]),
            "moved": int(moved.sum()),
            "stable_share_of_moved": float(res["stable"][moved].mean()) if moved.sum() else 1.0,
            "median_move": float(np.median(np.abs(res["base_delta"]))),
        })
        for b in stability_by_size(res):
            slot = all_bins[b["move"]]
            slot[0] += b["n"]
            slot[1] += b["stable_share"] * b["n"]
    bins = [{"move": m, "n": c, "stable_share": s / c}
            for m, (c, s) in sorted(all_bins.items()) if c]
    return rows, bins


def sweep(index_by_year, weights, pairs, deltas, draws=250, seed=0):
    """The threshold as a function of how much disagreement is allowed.

    Reporting one row of this table is what made the first version of the
    finding attackable. The curve IS the result: a reader who thinks half a
    percentage point of disagreement is generous and a reader who thinks ten
    points is stingy can both read their own answer off it, and the claim that
    survives every row is the one worth publishing.
    """
    out = []
    for d in deltas:
        rows, bins = summarise_pairs(index_by_year, weights, pairs,
                                     draws=draws, seed=seed, delta=d)
        moved = sum(r["moved"] for r in rows)
        share = (sum(r["stable_share_of_moved"] * r["moved"] for r in rows) / moved
                 if moved else 1.0)
        out.append({
            "delta": d,
            "robust_share": share,
            "threshold": reliable_threshold(bins),
            "moved": moved,
        })
    return out


def parameter_free_bound(sweep_rows):
    """The largest move size that is unreliable at EVERY setting tested.

    A threshold of N means moves below N are unreliable there. The claim that
    holds regardless of the parameter is therefore driven by the SMALLEST
    threshold in the sweep - the most permissive setting for the index.
    """
    thresholds = [r["threshold"] for r in sweep_rows if r["threshold"] is not None]
    if not thresholds or len(thresholds) != len(sweep_rows):
        return None
    return min(thresholds)


def timescale(index_by_year, weights, gaps, delta=0.05, draws=300, seed=0):
    """Robustness as a function of the gap between the two years compared.

    The controlled version of the question. Comparing HDI's annual changes
    against EPI's ten-year change confounds two things - the index and the
    timescale - and the confound flatters whichever conclusion you wanted.
    Holding the index, the countries and the weight tolerance fixed and moving
    only the gap isolates the timescale, and it turns out to be what matters.
    """
    out = []
    years = sorted(index_by_year)
    for gap in gaps:
        pairs = [(y, y + gap) for y in years if y + gap in index_by_year]
        if not pairs:
            continue
        rows, bins = summarise_pairs(index_by_year, weights, pairs,
                                     draws=draws, seed=seed, delta=delta)
        moved = sum(r["moved"] for r in rows)
        out.append({
            "gap": gap,
            "pairs": len(pairs),
            "changes": moved,
            "robust_share": (sum(r["stable_share_of_moved"] * r["moved"]
                                 for r in rows) / moved) if moved else 1.0,
            "threshold": reliable_threshold(bins),
        })
    return out


def _pooled(index_by_year, weights, pairs, delta, draws, seed):
    """Run the draws once for a given tolerance and pool every pair.

    Agreement does not depend on the sign bar - the bar is applied to it
    afterwards. Computing agreement once and evaluating many bars against it is
    what makes a two-parameter surface affordable: the cost is one run per
    tolerance, not one per cell.
    """
    mags, agrees = [], []
    for y0, y1 in pairs:
        res = rank_change_robustness(index_by_year[y0], index_by_year[y1],
                                     weights, draws=draws, seed=seed, delta=delta)
        mags.append(np.abs(res["base_delta"]))
        agrees.append(res["agreement"])
    return np.concatenate(mags), np.concatenate(agrees)


def _bins_from(mag, stable, max_move=12):
    out = []
    for m in range(0, max_move + 1):
        sel = mag == m
        if sel.sum():
            out.append({"move": m, "n": int(sel.sum()),
                        "stable_share": float(stable[sel].mean())})
    big = mag > max_move
    if big.sum():
        out.append({"move": max_move + 1, "n": int(big.sum()),
                    "stable_share": float(stable[big].mean())})
    return out


def surface(index_by_year, weights, pairs, deltas, bars, draws=300, seed=0):
    """Robustness over BOTH free parameters at once.

    The weight tolerance was swept first and the sign bar was left fixed, which
    meant the headline was still a function of a number chosen by hand - the
    same failure, one level down. Neither parameter has a principled value, so
    neither gets to be hidden: the result is a surface, and what is worth
    reporting is what holds across all of it.
    """
    cells = []
    for delta in deltas:
        mag, agree = _pooled(index_by_year, weights, pairs, delta, draws, seed)
        moved = mag > 0
        for bar in bars:
            stable = agree >= bar
            bins = _bins_from(mag, stable)
            cells.append({
                "delta": delta,
                "bar": bar,
                "robust_share": float(stable[moved].mean()) if moved.sum() else 1.0,
                "threshold": reliable_threshold(bins),
                "n": int(moved.sum()),
            })
    return cells


def invariants(cells):
    """What survives every cell of the surface - the part worth stating.

    A range rather than a point, plus the two monotonicities. If either
    monotonicity failed, the surface would not be describing one underlying
    phenomenon and the summary would be misleading.
    """
    shares = [c["robust_share"] for c in cells]
    thr = [c["threshold"] for c in cells if c["threshold"] is not None]
    deltas = sorted({c["delta"] for c in cells})
    bars = sorted({c["bar"] for c in cells})

    def at(d, b):
        return next(c for c in cells if c["delta"] == d and c["bar"] == b)

    mono_delta = all(at(deltas[i], b)["robust_share"] >= at(deltas[i + 1], b)["robust_share"] - 1e-9
                     for b in bars for i in range(len(deltas) - 1))
    mono_bar = all(at(d, bars[i])["robust_share"] >= at(d, bars[i + 1])["robust_share"] - 1e-9
                   for d in deltas for i in range(len(bars) - 1))
    return {
        "share_min": min(shares),
        "share_max": max(shares),
        "threshold_min": min(thr) if thr else None,
        "threshold_max": max(thr) if thr else None,
        "complete": len(thr) == len(cells),
        "monotone_in_tolerance": mono_delta,
        "monotone_in_bar": mono_bar,
    }


def timescale_surface(index_by_year, weights, gaps, deltas, bars,
                      draws=250, seed=0):
    """The timescale gradient at every corner of the parameter surface.

    One gradient at one setting would be another single-cell claim. What is
    being asserted is that robustness rises with the gap REGARDLESS of the two
    parameters, so it has to be shown regardless of them.
    """
    years = sorted(index_by_year)
    out = []
    for gap in gaps:
        pairs = [(y, y + gap) for y in years if y + gap in index_by_year]
        if not pairs:
            continue
        row = {"gap": gap, "pairs": len(pairs), "cells": []}
        for delta in deltas:
            mag, agree = _pooled(index_by_year, weights, pairs, delta, draws, seed)
            moved = mag > 0
            for bar in bars:
                row["cells"].append({
                    "delta": delta, "bar": bar,
                    "robust_share": float((agree >= bar)[moved].mean()),
                })
        row["min_share"] = min(c["robust_share"] for c in row["cells"])
        row["max_share"] = max(c["robust_share"] for c in row["cells"])
        out.append(row)
    return out


def gradient_holds(rows):
    """Does robustness rise with the gap at EVERY corner of the surface?"""
    if len(rows) < 2:
        return None
    keys = [(c["delta"], c["bar"]) for c in rows[0]["cells"]]
    for k in keys:
        series = [next(c["robust_share"] for c in r["cells"]
                       if (c["delta"], c["bar"]) == k) for r in rows]
        if any(series[i] > series[i + 1] + 1e-9 for i in range(len(series) - 1)):
            return False
    return True
