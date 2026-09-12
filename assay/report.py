"""Say what was measured, what it means, and what it does not mean.

The register is deliberate. An audit of somebody else's index is easy to write
as an accusation and hard to write as a finding, and only the second one is
worth anything. So: no adjectives about the authors, no claim that a weighting
is wrong because it is a choice, and every number carries the n it came from.

The one verdict this tool will state plainly is the arithmetic one - that an
index naming six pillars has one dimension is not an opinion about the authors,
it is a property of the numbers they published.
"""

import numpy as np

from .aggregation import DECORATIVE_AT
from .structure import REDUNDANT_AT


def _bar(x, width=28):
    return "#" * int(round(x * width))


def render(index, struct, agg, env, movable):
    L = []
    w = L.append

    w("=" * 74)
    w("ASSAY  %s" % index.name)
    w("=" * 74)
    w(index.describe())
    if index.source:
        w("source: %s" % index.source)
    w("")

    if index.has_shared_construction:
        sc = index.shared_construction
        w("!" * 74)
        w("SHARED CONSTRUCTION - READ BEFORE ANY NUMBER BELOW")
        w("")
        for line in str(sc.get("note", "")).splitlines():
            w("  " + line)
        if sc.get("columns"):
            w("")
            w("  affected: %s" % ", ".join(sc["columns"]))
        w("")
        w("  These sub-scores share an input by definition, so they would")
        w("  correlate even if the world contained no pattern at all. Nothing")
        w("  below about their correlation, dimensionality or redundancy is")
        w("  evidence ABOUT THE WORLD. It is a restatement of the formula.")
        w("")
        w("  A real finding needs sub-scores built from independent inputs, or")
        w("  the component parts with the shared term divided out.")
        w("!" * 74)
        w("")

    # ---- dimensionality -------------------------------------------------
    dim = struct["dimensions"]
    w("HOW MANY DIMENSIONS ARE ACTUALLY THERE")
    w("")
    claimed = index.claimed_dimensions or index.k
    lo, hi = dim["low"], dim["high"]
    span = str(lo) if lo == hi else "%d-%d" % (lo, hi)
    w("  sub-scores published      %d" % index.k)
    w("  dimensions claimed        %d" % claimed)
    w("  dimensions found          %s   (parallel analysis %s%s)"
      % (span, dim["parallel"],
         ", MAP %s" % dim["map"] if dim["map_usable"] else ", MAP not usable"))
    if not dim["map_usable"]:
        w("      MAP needs about five columns to partial anything out; with %d"
          % index.k)
        w("      it is not reported rather than reported unreliably.")
    w("")
    w("  variance by component:")
    for i, v in enumerate(struct["variance_explained"][:6], 1):
        w("    PC%-2d  %5.1f%%  %s" % (i, 100 * v, _bar(v)))
    w("")

    pc1 = struct["pc1_share"]
    if hi < claimed and index.has_shared_construction:
        w("  >> The index names %d dimensions and the data supports %s, with"
          % (claimed, span))
        w("     %.0f%% of variance on one component - but see the construction"
          % (100 * pc1))
        w("     warning above. This collapse is what the formula guarantees.")
        w("     It is not a finding, and must not be published as one.")
    elif hi < claimed and pc1 >= 0.6:
        w("  >> The index names %d dimensions. The data supports %s." % (claimed, span))
        w("     %.0f%% of all sub-score variance sits on one component, so the"
          % (100 * pc1))
        w("     pillars are largely re-descriptions of a single underlying thing.")
    elif hi < claimed:
        # EPI produced this case: one factor retained, but explaining under half
        # the variance. Saying "the index is one thing" there would be as wrong
        # as accepting its three dimensions. Retention counts how many factors
        # beat noise; it does not say the retained one explains much.
        w("  >> The index names %d dimensions and the retention tests support %s"
          % (claimed, span))
        w("     - but that factor explains only %.0f%% of the variance."
          % (100 * pc1))
        w("")
        w("     Read this as: the claimed structure is not there, and NEITHER")
        w("     is a single clean alternative. The remaining %.0f%% is spread"
          % (100 * (1 - pc1)))
        w("     across components too weak to retain individually. An index")
        w("     with genuinely distinct pillars and one with one pillar would")
        w("     both look unlike this; what this looks like is a set of loosely")
        w("     related measurements that do not resolve into named dimensions.")
    else:
        w("  >> The claimed dimension count is supported: %s found against %d claimed."
          % (span, claimed))
    w("")

    # ---- sub-scores pointing the wrong way ------------------------------
    if index.flipped:
        w("SUB-SCORES THAT RUN AGAINST THEIR OWN INDEX")
        w("")
        for c in index.flipped:
            w("  %s" % c)
        w("")
        w("  These correlate NEGATIVELY with the published composite, and were")
        w("  turned round so the structure below is readable. That is a")
        w("  substantive fact rather than a formatting step: a country scoring")
        w("  well here tends to score worse overall. Either the sub-score")
        w("  measures something in tension with the rest of the index, or its")
        w("  direction is not what its name suggests. Worth checking against")
        w("  the index's own documentation before reading anything else.")
        w("")

    # ---- redundancy -----------------------------------------------------
    if struct["redundant"]:
        w("SUB-SCORES THAT ARE THE SAME SUB-SCORE   (|r| >= %.2f)" % REDUNDANT_AT)
        w("")
        for a, b, r in struct["redundant"]:
            w("  r = %+.3f   %s  <->  %s" % (r, a, b))
        w("")
        w("  Two pillars that move together across every entity are one pillar")
        w("  counted twice, and counting it twice doubles its weight silently.")
        w("")

    # ---- do the weights follow? -----------------------------------------
    if agg["reproduces"] is not None:
        rep = agg["reproduces"]
        w("DOES THE PUBLISHED SCORE FOLLOW FROM THE PUBLISHED WEIGHTS?")
        w("")
        w("  r(recomputed, published)  %.4f" % rep["r"])
        w("  largest rank disagreement %d places" % rep["max_rank_gap"])
        w("  median score error        %.2f  (%.1f%% of one SD of the"
          % (rep["median_abs_score_error"], 100 * rep["median_error_in_sd"]))
        w("                                  published score)")
        w("  largest score error       %.2f" % rep["max_abs_score_error"])
        if rep["exact"]:
            w("  >> Reproducible. The stated method is the actual method.")
        elif rep["reproducible"]:
            # EPI's case: the ordering follows from the published weights, but
            # the level does not. Reporting only the correlation would have
            # called this exact, and reporting only the score error would have
            # called it broken. It is neither.
            w("  >> The ORDERING follows from the published weights (r %.4f),"
              % rep["r"])
            w("     but the SCORE does not: a typical country lands %.1f points"
              % rep["median_abs_score_error"])
            w("     from its published value and the worst is %.1f, with rank"
              % rep["max_abs_score_error"])
            w("     disagreements up to %d places." % rep["max_rank_gap"])
            w("")
            w("     A weighted sum of the published sub-scores is therefore a")
            w("     good model of the index but not the index itself - there is")
            w("     processing between the two. Usually that is aggregation")
            w("     happening at a lower level than the one audited here, which")
            w("     is benign; it is worth confirming rather than assuming.")
        else:
            w("  >> NOT reproducible from the stated weights alone. Something")
            w("     undocumented sits between the sub-scores and the headline")
            w("     number - a cap, a rescale, or an input that is not published.")
        w("")

    # ---- are the weights decorative? ------------------------------------
    if agg["decorative"] is not None:
        dec = agg["decorative"]
        w("DO THE WEIGHTS ADD ANYTHING OVER ONE UNWEIGHTED COMPONENT?")
        w("")
        w("  r(published score, PC1)   %.4f" % dec["r_with_pc1"])
        if dec["decorative"]:
            w("  >> Above %.2f: the weighting is decorative. An equal-weighted"
              % DECORATIVE_AT)
            w("     first component ranks these entities the same way, so the")
            w("     specific trade-off the weights encode is not what produced")
            w("     the ranking.")
        else:
            w("  >> The weights do real work; the published ranking is not")
            w("     recoverable from an unweighted component.")
        w("")

    # ---- rank envelope --------------------------------------------------
    w("HOW MUCH OF A RANK IS THE ENTITY, AND HOW MUCH IS THE WEIGHTING?")
    w("")
    if env["mode"] == "near":
        w("  %d draws, perturbing the index's OWN published weights"
          % env["draws"])
        w("  (Dirichlet, concentration %.0f - small disagreements, not new indices)"
          % env["concentration"])
    else:
        w("  %d draws over all weightings. There are no published weights to"
          % env["draws"])
        w("  perturb, so this is the permissive reading and should be read as")
        w("  an upper bound on how far a rank could move.")
    w("")
    w("  median rank spread        %.1f places  (%.0f%% of the field)"
      % (env["median_spread"], 100 * env["median_spread_share"]))
    w("  90th percentile spread    %.1f places  (%.0f%% of the field)"
      % (env["p90_spread"], 100 * env["p90_spread_share"]))
    w("  largest spread            %d places  (%.0f%% of the field)"
      % (env["max_spread"], 100 * env["max_spread_share"]))
    w("      Places are not comparable between indices - twenty places means")
    w("      something different in a field of 25 than in a field of 190 - so")
    w("      the share of the field is the figure to quote.")
    w("  top-%d churn              %.1f of %d change places on an average draw"
      % (env["top_n"], env["mean_top_churn"], env["top_n"]))
    w("")
    w("  most weight-dependent entities:")
    w("    %-34s published   best  worst  spread" % "")
    for name, pub, best, worst, spread in movable:
        w("    %-34s %6d   %5d  %5d  %6d" % (name[:34], pub, best, worst, spread))
    w("")

    # ---- the joint reading ----------------------------------------------
    w("=" * 74)
    w("READING THESE TOGETHER")
    w("")
    stable = env["median_spread_share"] <= 0.05
    one_dim = hi < claimed and pc1 >= 0.5
    if stable and one_dim:
        w("  The ranking barely moves under reweighting, and that is NOT")
        w("  evidence that it is robust. With %.0f%% of variance on one" % (100 * pc1))
        w("  component, no weighting could have separated these entities.")
        w("  The stability and the redundancy are the same fact.")
    elif one_dim:
        # Both at once, which looked contradictory until HDI produced it: one
        # dominant dimension AND volatile ranks. There is no contradiction. The
        # dimension decides the broad ordering; the leftover variance decides
        # positions inside a densely packed field, where a small score change
        # crosses many neighbours.
        w("  Two things are true at once here, and they are not in conflict.")
        w("")
        w("  %.0f%% of the variance is one dimension, so the broad ordering is"
          % (100 * pc1))
        w("  settled by a single underlying factor and the claimed %d dimensions"
          % claimed)
        w("  are not separable in the data. But individual ranks still move a")
        w("  lot, because %d entities are packed closely enough that the" % index.n)
        w("  remaining %.0f%% decides who sits above whom." % (100 * (1 - pc1)))
        w("")
        w("  So: the index is measuring one thing, and the precise rank it")
        w("  reports is not a reliable property of the entity. A place in the")
        w("  table is worth reading as a band, not a position.")
    elif stable:
        w("  The ranking is stable under reweighting and the pillars are")
        w("  distinct, which is the good case: the ordering survives the")
        w("  arbitrary part of the method.")
    else:
        w("  The ranking moves substantially under weights the authors could")
        w("  have chosen instead. Rank differences smaller than the spread")
        w("  above should not be reported as differences between entities.")
    w("")
    w("  What this audit does NOT show: whether the sub-scores measure what")
    w("  they are named after, whether the entities are comparable, or whether")
    w("  the index is useful. It tests internal structure against the index's")
    w("  own claims, on %d entities with complete data." % index.n)
    w("=" * 74)
    return "\n".join(L)
