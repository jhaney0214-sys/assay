"""Say what was measured, what it means, and what it does not mean.

The register is deliberate. An audit of somebody else's index is easy to write
as an accusation and hard to write as a finding, and only the second one is
worth anything. So: no adjectives about the authors, no claim that a weighting
is wrong because it is a choice, and every number carries the n it came from.

The one verdict this tool will state plainly is the arithmetic one - that an
index naming six pillars has one dimension is not an opinion about the authors,
it is a property of the numbers they published.
"""

import textwrap

import numpy as np

from .aggregation import DECORATIVE_AT
from .structure import REDUNDANT_AT
from . import structure


def _bar(x, width=28):
    return "#" * int(round(x * width))


# Dropped entities are named when there are this few; past it, a list is noise.
NAME_DROPPED_UP_TO = 10


def _wrap(text, width):
    return textwrap.wrap(text, width, subsequent_indent="  ")


def _left_out(w, left, index):
    w("WHO WAS LEFT OUT")
    w("")
    w("  %d of %d rows dropped for a missing or unreadable value; everything"
      % (left["dropped"], left["rows_in"]))
    w("  below is about the %d that stayed." % index.n)
    names = left.get("names") or []
    if names and len(names) <= NAME_DROPPED_UP_TO:
        w("")
        for line in _wrap("dropped: " + "; ".join(names), 70):
            w("  " + line)
    w("")
    for col, blank, bad, examples in left["by_column"]:
        tail = ""
        if bad:
            tail = "   %d unreadable, e.g. %s" % (bad, ", ".join(repr(e) for e in examples))
        w("    %-40s %4d empty%s" % (col[:40], blank, tail))
    if len(left["by_column"]) > 1:
        w("    (a row can be missing more than one)")
    if any(bad for _, _, bad, _ in left["by_column"]):
        w("")
        w("  An unreadable cell is text that is not a number. If it is a")
        w("  formatting habit of the file (a footnote mark, a unit) and not a")
        w("  gap, those rows were dropped by mistake.")
    w("")
    pub = left["published_col"] or "published score"
    if "gap_in_sd" not in left:
        if index.published is None:
            w("  The index publishes no overall score here, so whether the")
            w("  dropped rows differ from the kept ones cannot be tested.")
        elif left["compared"] == 0:
            # HDI 2022: North Korea and Monaco. The index did not score them
            # either, so this audit and the index cover the same countries.
            w("  None of the dropped rows has a published %s either: the index" % pub)
            w("  left them out too, so the audit covers the same field as the")
            w("  index. How they would have scored is not known to either.")
        else:
            w("  Only %d dropped row(s) carry a published %s, too few to"
              % (left["compared"], pub))
            w("  compare. Whether the dropping is selective is not known.")
        w("")
        return
    side = "lower" if left["gap_in_sd"] < 0 else "higher"
    w("  On the published %s, %d of the dropped rows can be compared:"
      % (pub, left["compared"]))
    w("    kept     average percentile %5.1f" % left["kept_mean_pct"])
    w("    dropped  average percentile %5.1f   (%.2f SD %s)"
      % (left["dropped_mean_pct"], abs(left["gap_in_sd"]), side))
    if left.get("near_line"):
        w("  >> Too close to call: the gap is %.2f SD, and the report calls"
          % abs(left["gap_in_sd"]))
        w("     dropping selective from 0.25. The entities removed score %s"
          % side)
        w("     than those kept, by about as much as the line itself; read")
        w("     the audit as describing the complete part of the field and")
        w("     possibly not the whole of it.")
    elif left["selective"]:
        w("  >> The dropping is selective: the entities removed score %s on the"
          % side)
        w("     index itself than those kept. The audit describes the index")
        w("     where it is complete, which is not the whole field.")
    else:
        w("  >> The dropped rows score close to the kept ones on the published")
        w("     index, which is evidence against the dropping biasing what")
        w("     follows. It is not proof: they may differ on what is missing.")
    unscored = left["dropped"] - left["compared"]
    if unscored:
        w("     %d dropped row(s) have no published score and are not compared."
          % unscored)
    w("")


def _leave_one_out(w, loo):
    w("WHICH SUB-SCORE DOES THE RANKING LEAN ON?")
    w("")
    w("  Each sub-score left out in turn, the other weights rescaled, against")
    w("  the ranking with all of them (%s)." % loo["scale"])
    w("")
    w("    %-34s weight  median move  90th pct  top-%d out"
      % ("", loo["top"]))
    for r in loo["rows"]:
        w("    %-34s %5.1f%%  %5.0f (%2.0f%%)   %6.0f   %5d"
          % (r["column"][:34], 100 * r["weight"], r["median_shift"],
             100 * r["median_shift"] / loo["n"], r["p90_shift"], r["top_changed"]))
    w("")
    rows = loo["rows"]
    most = rows[0]
    heaviest = max(rows, key=lambda r: r["weight"])
    meds = [r["median_shift"] for r in rows]
    if (most["column"] != heaviest["column"]
            and most["median_shift"] > heaviest["median_shift"]):
        # EPI 2024: Biodiversity & Habitat at 25% moves the median country 7
        # places, Climate Change Mitigation at 30% only 4. A pillar matters by
        # how much it disagrees with the rest, not by its weight alone.
        text = ("The ranking leans hardest on %s (%.0f%% of the weight), not on "
                "%s, the heaviest at %.0f%%. Leaving out the first moves the "
                "median entity %.0f places, the second %.0f. A sub-score counts "
                "by how far it disagrees with the others as well as by its weight."
                % (most["column"], 100 * most["weight"], heaviest["column"],
                   100 * heaviest["weight"], most["median_shift"],
                   heaviest["median_shift"]))
    elif len(meds) > 1 and meds[0] - meds[1] <= max(1.0, 0.1 * meds[0]):
        # HDI 2022: three of four at 5 places. Naming one would be a tie-break.
        text = ("No one sub-score carries the ranking: leaving out any one moves "
                "the median entity %.0f to %.0f places." % (min(meds), max(meds)))
    else:
        text = "The sub-score that moves the ranking most is %s." % most["column"]
    for i, line in enumerate(_wrap(text, 66)):
        w(("  >> " if i == 0 else "     ") + line.strip())
    w("")


def render(index, struct, agg, env, movable, left=None, pairs=None, loo=None):
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

    # ---- who was dropped ------------------------------------------------
    # Before any number, because every number below is about the rows that
    # stayed. EPI is the case: 75 of 180 countries dropped, and they score
    # lower on the published EPI than the 105 the audit runs on.
    if left is not None:
        _left_out(w, left, index)

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
    alike = struct["pairwise"]
    kind = structure.grade(alike["mean"])
    if hi < claimed and index.has_shared_construction:
        w("  >> The index names %d dimensions and the data supports %s, with"
          % (claimed, span))
        w("     %.0f%% of variance on one component - but see the construction"
          % (100 * pc1))
        w("     warning above. This collapse is what the formula guarantees.")
        w("     It is not a finding, and must not be published as one.")
    elif hi < claimed and kind == "one":
        w("  >> The index names %d dimensions. The data supports %s." % (claimed, span))
        w("     %.0f%% of all sub-score variance sits on one component, so the"
          % (100 * pc1))
        w("     pillars are largely re-descriptions of a single underlying thing.")
        w("     They correlate %.2f with each other on average, and %.2f at the"
          % (alike["mean"], alike["min"]))
        w("     least alike pair.")
    elif hi < claimed and kind == "overlap":
        # CDC's Social Vulnerability Index produced this case: one factor
        # retained and 61% on it, from four themes correlating 0.27 to 0.58.
        # That is four things that rise together, not one thing named four
        # times, and the report said the second until 2026-10-13.
        w("  >> The index names %d dimensions and the retention tests support %s."
          % (claimed, span))
        w("     %.0f%% of sub-score variance sits on one component." % (100 * pc1))
        w("")
        w("     Read this as: the pillars OVERLAP HEAVILY and are still not one")
        w("     thing. They correlate %.2f with each other on average and only"
          % alike["mean"])
        w("     %.2f at the least alike pair, so each carries something the"
          % alike["min"])
        w("     others do not. What the data does not support is %d separate"
          % claimed)
        w("     dimensions: the differences between pillars are real and too")
        w("     small to stand apart from what they share.")
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

    # ---- leave one out --------------------------------------------------
    if loo is not None and loo["rows"]:
        _leave_one_out(w, loo)

    # ---- pairwise robustness, exact -------------------------------------
    if pairs is not None:
        w("WHICH COMPARISONS DOES NO WEIGHTING REVERSE?")
        w("")
        w("  Exact, not sampled (Foster, McGillivray & Seth 2009): 'A above B'")
        w("  holds everywhere in a set of weightings if it holds at the set's")
        w("  corners. On %s, of %d pairs:" % (pairs["scale"], pairs["pairs"]))
        w("")
        for o in pairs["orders"]:
            if o["r"] == 1.0:
                label = "under every weighting at all"
            else:
                label = ("under weightings up to %d%% of the way from the starting"
                         % round(100 * o["r"]))
            w("    %5.1f%%  %s" % (100 * o["share"], label))
            if o["r"] != 1.0:
                w("            weights toward all the weight on any one sub-score")
        w("")
        w("  'Every weighting' needs A to beat B on every sub-score, so it falls")
        w("  as sub-scores are added; compare indices at the narrower set. On")
        w("  HDI 2004 three of the paper's four figures land inside the printed")
        w("  table's rounding and one misses by 0.2 points (tests/test_pairwise.py).")
        w("")

    # ---- the joint reading ----------------------------------------------
    w("=" * 74)
    w("READING THESE TOGETHER")
    w("")
    stable = env["median_spread_share"] <= 0.05
    one_dim = hi < claimed and kind == "one"
    overlapping = hi < claimed and kind == "overlap"
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
    elif overlapping and not stable:
        w("  The pillars overlap enough that one component sets the broad")
        w("  ordering, and differ enough that the weighting matters: the")
        w("  remaining %.0f%% of variance is where one pillar says something"
          % (100 * (1 - pc1)))
        w("  another does not. Other weightings would move ranks a long way")
        w("  (the envelope above), whatever the published one happens to do,")
        w("  so a rank belongs to the weights that produced it as much as to")
        w("  the entity.")
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
