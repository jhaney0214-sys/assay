"""Assay - test whether a published index measures what it says it measures.

    python ax.py audit data/whr2024.json
    python ax.py audit data/whr2024.json --draws 5000 --mode uniform
    python ax.py columns data/whr2024.csv

An index spec is a small JSON file naming the data, the entity column, the
sub-scores, the published composite, and - if the index publishes them - the
weights and the number of dimensions it claims. The spec is what makes the
audit an audit rather than an opinion: every comparison is against the index's
own account of itself.
"""

import argparse
import csv
import io
import json
import os
import sys

# Windows consoles default to cp1252, which cannot print a country called
# Cote d'Ivoire with its accent. Without this the audit computes correctly and
# then dies on the way to the screen.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from assay import aggregation, drift, driftreport, load, report, structure


def cmd_columns(args):
    """Print a file's columns, so a spec can be written without guessing."""
    with open(args.path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("no rows")
        return 1
    print("%d rows, %d columns\n" % (len(rows), len(rows[0])))
    for c in rows[0]:
        sample = [r[c] for r in rows[:3]]
        print("  %-42s %s" % (c, " | ".join(str(s)[:18] for s in sample)))
    return 0


def cmd_audit(args):
    idx = load.from_spec(args.spec)
    struct = structure.analyse(idx, seed=args.seed)
    pc1 = structure.pc1_scores(idx)

    agg = {
        "reproduces": aggregation.reproduces_published(idx),
        "decorative": aggregation.weights_are_decorative(idx, pc1),
    }
    env = aggregation.rank_envelope(idx, draws=args.draws, mode=args.mode,
                                    concentration=args.concentration, seed=args.seed)
    movable = aggregation.most_movable(idx, env, limit=args.show)

    text = report.render(idx, struct, agg, env, movable)
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print("\nwritten to %s" % args.out)

    if args.json:
        payload = {
            "index": idx.name,
            "n": idx.n,
            "k": idx.k,
            "claimed_dimensions": idx.claimed_dimensions,
            "dimensions_found": [struct["dimensions"]["low"],
                                 struct["dimensions"]["high"]],
            "pc1_share": struct["pc1_share"],
            "variance_explained": struct["variance_explained"],
            "redundant_pairs": struct["redundant"],
            "reproduces_published": agg["reproduces"],
            "weights_decorative": agg["decorative"],
            "rank_envelope": {k: (v.tolist() if hasattr(v, "tolist") else v)
                              for k, v in env.items()},
            "flipped": idx.flipped,
            "dropped_rows": idx.dropped,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print("json written to %s" % args.json)
    return 0


def cmd_drift(args):
    panel, spec = load.panel_from_spec(args.spec)
    weights = spec.get("weights")
    if not weights:
        raise SystemExit("a drift panel needs published weights in its spec")
    years = sorted(panel)
    # Consecutive in the sorted list, not literally y+1: EPI's panel is a
    # baseline and a current period ten years apart, and requiring adjacent
    # calendar years silently produced no pairs at all.
    pairs = list(zip(years, years[1:]))
    if not pairs:
        raise SystemExit("no consecutive-year pairs in the panel")

    probe = sorted({years[0], years[len(years) // 2], years[-1]})
    fidelity = [(y, drift.model_fidelity(panel[y], weights)) for y in probe]
    chg_fid = drift.change_fidelity(panel[pairs[0][0]], panel[pairs[0][1]], weights)
    struct = drift.structural_drift(panel, seed=args.seed) if not args.no_structure else []

    deltas = [float(x) for x in args.deltas.split(",")] if args.deltas else None
    rows, bins = drift.summarise_pairs(panel, weights, pairs, draws=args.draws,
                                       seed=args.seed,
                                       delta=(deltas[len(deltas) // 2] if deltas else None),
                                       concentration=args.concentration)
    thr = drift.reliable_threshold(bins)
    # The one-parameter sweep is a single column of the surface below, so it
    # is no longer computed separately - showing both would invite reading the
    # column as the answer.
    sweep_rows, bound = None, None
    gaps = [int(g) for g in args.gaps.split(",")] if args.gaps else None
    bars = [float(b) for b in args.bars.split(",")] if args.bars else [0.90]
    surface_rows = (drift.surface(panel, weights, pairs, deltas, bars,
                                  draws=max(200, args.draws // 2), seed=args.seed)
                    if deltas else None)
    inv = drift.invariants(surface_rows) if surface_rows else None
    ts_surface = (drift.timescale_surface(
        panel, weights, gaps, deltas or [0.05], bars,
        draws=max(150, args.draws // 4), seed=args.seed) if gaps else None)
    grad = drift.gradient_holds(ts_surface) if ts_surface else None
    ts = None
    text = driftreport.render(spec.get("name", args.spec), fidelity, struct,
                              rows, bins, thr, args.draws, args.concentration,
                              source=spec.get("source"), sweep_rows=sweep_rows,
                              bound=bound, change_fidelity=chg_fid,
                              timescale_rows=ts, surface_rows=surface_rows,
                              inv=inv, ts_surface=ts_surface,
                              gradient_everywhere=grad)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text + "\n")
        print("\nwritten to %s" % args.out)
    if args.json:
        payload = {"index": spec.get("name"), "pairs": rows, "bins": bins,
                   "reliable_threshold": thr, "structure": struct,
                   "sweep": sweep_rows, "parameter_free_bound": bound,
                   "change_fidelity": chg_fid, "surface": surface_rows,
                   "invariants": inv, "timescale_surface": ts_surface,
                   "gradient_everywhere": grad,
                   "fidelity": [{"year": y, **f} for y, f in fidelity],
                   "draws": args.draws, "concentration": args.concentration}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(payload, open(args.json, "w", encoding="utf-8"), indent=2, default=str)
        print("json written to %s" % args.json)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="ax", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="audit an index from its spec")
    a.add_argument("spec")
    a.add_argument("--draws", type=int, default=aggregation.DEFAULT_DRAWS)
    a.add_argument("--mode", choices=["near", "uniform"], default=None,
                   help="near: perturb the index's own weights (default when it "
                        "publishes them). uniform: any weighting at all.")
    a.add_argument("--concentration", type=float, default=50.0,
                   help="how near 'near' is; higher is tighter (default 50)")
    a.add_argument("--show", type=int, default=10)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", help="write the text report here")
    a.add_argument("--json", help="write machine-readable results here")
    a.set_defaults(func=cmd_audit)

    c = sub.add_parser("columns", help="list a CSV's columns")
    c.add_argument("path")
    c.set_defaults(func=cmd_columns)

    d = sub.add_parser("drift", help="are year-over-year rank changes robust?")
    d.add_argument("spec")
    d.add_argument("--draws", type=int, default=600)
    d.add_argument("--concentration", type=float, default=50.0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-structure", action="store_true")
    d.add_argument("--bars", default="0.75,0.90,0.99",
                   help="comma-separated sign-agreement bars to sweep")
    d.add_argument("--gaps", default="",
                   help="comma-separated year gaps to compare (e.g. 1,2,5,10). "
                        "Isolates timescale from index.")
    d.add_argument("--deltas", default="0.01,0.02,0.033,0.05,0.10",
                   help="comma-separated weight deviations to sweep, in weight "
                        "units (0.01 = one percentage point). Empty to skip.")
    d.add_argument("--out")
    d.add_argument("--json")
    d.set_defaults(func=cmd_drift)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
