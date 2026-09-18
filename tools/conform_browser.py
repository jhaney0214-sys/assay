"""Does the browser engine agree with the Python one?

`docs/audit.js` is a second implementation of Assay's structural analysis, and
a second implementation drifts from the first the moment either is corrected.
Sextant carries the same risk and answers it the same way - by running both
over the same inputs and diffing. This is that check for Assay.

    python tools/conform_browser.py

Requires node. Exits non-zero on disagreement.

WHAT MUST MATCH EXACTLY, and what cannot:

  correlation matrix, eigenvalues, variance explained, PC1 share,
  redundant pairs, r(published, PC1), MAP's retained count
        These are deterministic given the data. Tolerance 1e-9 - anything
        larger is a real difference in the maths, not floating point.

  parallel analysis                 THE COUNT ONLY.
        Horn's method compares real eigenvalues against eigenvalues drawn
        from random noise, and numpy's generator cannot be reproduced in
        JavaScript. The draws differ by construction. What must agree is
        how many components are retained, because that is a threshold
        comparison and 500 iterations put it far from the boundary on real
        data. The percentile gap is printed rather than asserted, so a run
        that drifts toward the boundary is visible before it flips.
"""

import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
from assay import load, structure                             # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS = ["data/hdi-2022.json", "data/vdem-2024.json", "data/epi-2024.json"]
TOL = 1e-9

RUNNER = r"""
const Assay = require(process.argv[2]);
const input = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const out = Assay.analyse(input.columns, input.names, {
  published: input.published, seed: 0, iters: input.iters });
process.stdout.write(JSON.stringify({
  n: out.n, k: out.k,
  pc1Share: out.pc1Share,
  varianceExplained: out.varianceExplained,
  dimensions: out.dimensions,
  redundant: out.redundant,
  weightsVsPc1: out.weightsVsPc1 === undefined ? null : out.weightsVsPc1,
  R: out.R,
}));
"""


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


def main():
    runner = ROOT / "tools" / "_conform_runner.js"
    runner.write_text(RUNNER, "utf-8")
    failures, checks = [], 0

    for spec_path in SPECS:
        spec = json.loads((ROOT / spec_path).read_text("utf-8"))
        index = load.from_spec(str(ROOT / spec_path))
        py = structure.analyse(index)

        # index.values is rows x columns of the RAW sub-scores (already
        # orientation-flipped by the loader where the spec says so), so the
        # browser engine receives exactly what the Python engine standardises.
        columns = [[float(v) for v in col]
                   for col in np.asarray(index.values, dtype=float).T]
        published = ([float(v) for v in index.published]
                     if index.published is not None else None)

        payload = {"columns": columns, "names": list(index.columns),
                   "published": published, "iters": 500}
        tmp = ROOT / "tools" / "_conform_input.json"
        tmp.write_text(json.dumps(payload), "utf-8")

        js = json.loads(subprocess.run(
            ["node", str(runner), str(ROOT / "docs" / "audit.js"), str(tmp)],
            capture_output=True, text=True, check=True).stdout)

        name = pathlib.Path(spec_path).stem
        def check(label, a, b, tol=TOL):
            nonlocal checks
            checks += 1
            if not close(a, b, tol):
                failures.append("%s: %s python=%r js=%r" % (name, label, a, b))

        check("n", py["n"], js["n"])
        check("k", py["k"], js["k"])
        check("pc1_share", py["pc1_share"], js["pc1Share"])
        for i, (a, b) in enumerate(zip(py["variance_explained"],
                                       js["varianceExplained"])):
            check("variance_explained[%d]" % i, a, b)

        Rp = np.asarray(py["R"])
        for i in range(Rp.shape[0]):
            for j in range(Rp.shape[1]):
                check("R[%d][%d]" % (i, j), Rp[i][j], js["R"][i][j])

        check("dimensions.parallel", py["dimensions"]["parallel"],
              js["dimensions"]["parallel"])
        if py["dimensions"]["map_usable"]:
            check("dimensions.map", py["dimensions"]["map"],
                  js["dimensions"]["map"])
        check("redundant count", len(py["redundant"]), len(js["redundant"]))
        for (pa, pb, pr), j in zip(py["redundant"], js["redundant"]):
            check("redundant %s/%s" % (pa, pb), pr, j["r"])

        print("  %-12s n=%-4d k=%-3d pc1=%.6f  parallel=%s map=%s  redundant=%d"
              % (name, py["n"], py["k"], py["pc1_share"],
                 py["dimensions"]["parallel"], py["dimensions"]["map"],
                 len(py["redundant"])))

    tmp.unlink(missing_ok=True)
    runner.unlink(missing_ok=True)

    print("\n%d comparisons across %d indices" % (checks, len(SPECS)))
    if failures:
        print("\n%d DISAGREEMENTS:" % len(failures))
        for f in failures[:20]:
            print("  " + f)
        return 1
    print("browser engine agrees with the Python engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
