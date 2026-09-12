"""Render the drift analysis.

The claim being made is narrow and the report keeps it narrow: a rank change
smaller than the method's own uncertainty is not distinguishable from the
method. It is not a claim that nothing changed, nor that the index is wrong,
nor that the country did not improve.

The model-fidelity block comes first and is not optional. If the composite used
here does not reproduce the published index, everything after it describes the
model instead, and the reader has to be told that before reading a number.
"""


def render(name, fidelity, structure_rows, pair_rows, bins, threshold,
           draws, concentration, source=None, sweep_rows=None, bound=None,
           change_fidelity=None, timescale_rows=None, delta=None,
           surface_rows=None, inv=None, ts_surface=None,
           gradient_everywhere=None):
    L = []
    w = L.append
    w("=" * 74)
    w("ASSAY DRIFT  %s" % name)
    w("=" * 74)
    if source:
        w("source: %s" % source)
    w("")

    # ---- fidelity, before anything is claimed ---------------------------
    w("DOES THE MODEL REPRODUCE THE PUBLISHED INDEX?")
    w("")
    w("  year    rank r     max rank gap")
    for y, f in fidelity:
        w("  %-6d  %.5f   %d places" % (y, f["rank_r"], f["max_rank_gap"]))
    worst = max(f["max_rank_gap"] for _, f in fidelity)
    best_r = min(f["rank_r"] for _, f in fidelity)
    if best_r >= 0.999 and worst <= 3:
        w("")
        w("  >> Reproduced. Everything below is about the published index.")
    else:
        w("")
        w("  >> NOT reproduced closely (rank r as low as %.4f, gaps to %d"
          % (best_r, worst))
        w("     places). Everything below describes the MODEL, not the index.")
    w("")

    # ---- fidelity of the CHANGE, which is what drift actually uses -------
    if change_fidelity:
        c = change_fidelity
        w("DOES THE MODEL REPRODUCE THE PUBLISHED *CHANGE*?")
        w("")
        w("  r(model change, published change)  %.4f" % c["r"])
        w("  same direction                     %.1f%% of %d countries"
          % (100 * c["same_direction"], c["n"]))
        w("  median disagreement                %.1f places" % c["median_gap"])
        w("  largest disagreement               %d places" % c["max_gap"])
        w("")
        if c["r"] >= 0.99 and c["same_direction"] >= 0.95:
            w("  >> The change is reproduced even where the level is not. A")
            w("     constant offset between model and published score cancels")
            w("     in a difference, so drift results stand.")
        else:
            w("  >> The change is NOT reproduced closely. Drift results below")
            w("     describe the model's movement, not the published index's.")
        w("")

    # ---- structural drift ------------------------------------------------
    if structure_rows:
        w("HAS THE STRUCTURE CHANGED OVER THE PANEL?")
        w("")
        w("  year     n    PC1 share   dimensions retained")
        step = max(1, len(structure_rows) // 8)
        for r in structure_rows[::step]:
            w("  %-6d %4d     %5.1f%%      %d"
              % (r["year"], r["n"], 100 * r["pc1_share"], r["dimensions"]))
        first, last = structure_rows[0], structure_rows[-1]
        delta = last["pc1_share"] - first["pc1_share"]
        w("")
        w("  PC1 share moved %+.1f points between %d and %d."
          % (100 * delta, first["year"], last["year"]))
        if abs(delta) < 0.03:
            w("  That is flat: the index's internal structure is stable, so a")
            w("  finding about its dimensionality in one year holds in others.")
        elif delta > 0:
            w("  The sub-scores have grown MORE redundant over the panel - the")
            w("  index is measuring closer to one thing than it used to.")
        else:
            w("  The sub-scores have grown LESS redundant over the panel.")
        w("")

    # ---- the headline ----------------------------------------------------
    w("=" * 74)
    w("ARE REPORTED YEAR-OVER-YEAR RANK CHANGES ROBUST TO THE WEIGHTING?")
    w("")
    w("  Each candidate weighting is applied to BOTH years and the change")
    w("  recomputed, so the uncertainty that is common to the two years")
    w("  cancels. A change 'keeps direction' when it has the same sign under")
    w("  at least 90%% of %d draws around the index's own published weights."
      % draws)
    w("")

    total_moved = sum(r["moved"] for r in pair_rows)
    if total_moved:
        w("  %d pairs, %d rank changes that actually happened."
          % (len(pair_rows), total_moved))
    w("")

    if surface_rows and inv:
        w("  TWO CHOICES HAVE TO BE MADE, AND NEITHER HAS A RIGHT ANSWER:")
        w("  how much weight disagreement counts as defensible, and how often a")
        w("  change must keep its sign to be called robust. Both are swept,")
        w("  because quoting one cell of this table as 'the' answer is the")
        w("  failure this tool exists to catch.")
        w("")
        deltas = sorted({c["delta"] for c in surface_rows})
        bars = sorted({c["bar"] for c in surface_rows})
        w("  share of changes keeping direction:")
        w("    tolerance " + "".join("   bar %.2f" % b for b in bars))
        for d in deltas:
            row = [c for c in surface_rows if c["delta"] == d]
            w("    +/-%4.1f pts" % (100 * d)
              + "".join("    %5.1f%%" % (100 * c["robust_share"]) for c in row))
        w("")
        w("  smallest reliable move (places):")
        w("    tolerance " + "".join("   bar %.2f" % b for b in bars))
        for d in deltas:
            row = [c for c in surface_rows if c["delta"] == d]
            w("    +/-%4.1f pts" % (100 * d)
              + "".join("    %6s" % (c["threshold"] if c["threshold"] is not None
                                     else "none") for c in row))
        w("")
        w("  >> The share ranges from %.0f%% to %.0f%% across this surface."
          % (100 * inv["share_min"], 100 * inv["share_max"]))
        w("     There is therefore NO single percentage to quote, and any")
        w("     report of one is a report about the two settings behind it.")
        w("")
        if inv["monotone_in_tolerance"] and inv["monotone_in_bar"]:
            w("     Robustness falls monotonically in both parameters, so the")
            w("     surface describes one phenomenon rather than noise.")
        if inv["threshold_min"] is not None and inv["complete"]:
            w("     WHAT HOLDS EVERYWHERE: a move of fewer than %d place(s) is"
              % inv["threshold_min"])
            w("     never reliable, at any setting on this surface.")
        w("")
    w("  size of move    changes    keeps direction")
    for b in bins:
        lab = "%d+" % b["move"] if b is bins[-1] and b["move"] > 12 else str(b["move"])
        bar = "#" * int(round(b["stable_share"] * 22))
        w("    %-4s %11d      %5.1f%%  %s" % (lab, b["n"], 100 * b["stable_share"], bar))
    w("")
    if sweep_rows:
        w("  HOW MUCH DOES THIS DEPEND ON HOW MUCH DISAGREEMENT IS ALLOWED?")
        w("")
        w("  Each published weight may vary by up to +/- delta. The first")
        w("  version of this analysis reported a single row of this table and")
        w("  was therefore a statement about a parameter, not about the index.")
        w("")
        w("    weights vary by     changes robust    smallest reliable move")
        for r in sweep_rows:
            w("      +/- %4.1f points        %5.1f%%              %s"
              % (100 * r["delta"], 100 * r["robust_share"],
                 r["threshold"] if r["threshold"] is not None else "none"))
        w("")
        if bound is not None:
            w("  >> HOLDS AT EVERY SETTING TESTED: a move of fewer than %d"
              % bound)
            w("     place(s) is unreliable however little disagreement you")
            w("     allow - including the tightest setting here, where each")
            w("     weight moves by at most %.1f of a percentage point."
              % (100 * sweep_rows[0]["delta"]))
            w("")
            w("     Above that the answer depends on what you count as a")
            w("     defensible weighting, and the table says how. It is not a")
            w("     number this tool can settle for the reader.")
            w("")

    if threshold is not None:
        w("  At the setting used for the table above, the smallest reliable")
        w("  move is %d places." % threshold)
        w("     A reported rise or fall of fewer than %d places cannot be"
          % threshold)
        w("     separated from the choice of weights, and should not be read")
        w("     as the country having moved.")
    else:
        w("  >> No move size reaches 90%% reliability in this panel.")
    w("")
    if ts_surface:
        w("")
        w("  DOES THE GAP BETWEEN THE COMPARED YEARS MATTER?")
        w("")
        w("  Same index, same countries. Each row is shown at its worst and")
        w("  best corner of the parameter surface above, so the gradient is")
        w("  not itself a single-setting claim.")
        w("")
        w("    gap      pairs    worst corner    best corner")
        for r in ts_surface:
            w("    %2d yr %8d       %6.1f%%        %6.1f%%"
              % (r["gap"], r["pairs"], 100 * r["min_share"], 100 * r["max_share"]))
        w("")
        if gradient_everywhere:
            lo, hi = ts_surface[0], ts_surface[-1]
            w("  >> Robustness rises with the gap at EVERY corner tested.")
            w("     Even at the harshest setting it goes from %.0f%% at %d year"
              % (100 * lo["min_share"], lo["gap"]))
            w("     to %.0f%% at %d years; at the mildest, %.0f%% to %.0f%%."
              % (100 * hi["min_share"], hi["gap"],
                 100 * lo["max_share"], 100 * hi["max_share"]))
            w("")
            w("     This is the finding, and it is parameter-free: an ANNUAL")
            w("     rank change is largely a property of the method, and a")
            w("     DECADAL one is largely a property of the country. Annual")
            w("     releases are the ones that get reported.")
        else:
            w("  >> The gradient does NOT hold at every corner, so it cannot be")
            w("     stated without naming the settings it depends on.")
        w("")

    w("  What this does NOT say: that the underlying indicators did not move,")
    w("  or that no progress occurred. Rank is positional - a country can")
    w("  improve and still not move reliably, because everyone else moved too.")
    w("  The claim is about reporting a rank change as news.")
    w("=" * 74)
    return "\n".join(L)
