/* Assay in the browser - the same five questions, no server, no upload.
 *
 * WHAT THIS IS A SECOND IMPLEMENTATION OF, and why that is stated first.
 * Assay's engine is Python on top of numpy, and it borrows its factor
 * machinery from Sextant rather than copying it, precisely because "a second
 * copy drifts from the first the moment either is corrected". This file is
 * that second copy. The justification is that a static page cannot run numpy
 * and the alternative - shipping a Python runtime to every visitor - costs
 * megabytes to answer a question about eleven columns.
 *
 * The drift is handled the way Sextant handles its own two implementations:
 * by conformance. tools/conform_browser.py runs the Python engine and this
 * file over the same three indices and fails if they disagree beyond a
 * tolerance. Run it after touching either side.
 *
 * ONE THING CANNOT BE BIT-IDENTICAL AND IT IS NOT A DEFECT. Horn's parallel
 * analysis compares real eigenvalues against eigenvalues of random noise, and
 * numpy's generator cannot be reproduced here. The RANDOM DRAWS therefore
 * differ. What must agree is the RETAINED COUNT, because that is a threshold
 * comparison and 500 iterations put it far from the boundary in every real
 * case. The conformance script checks the count and reports the percentile
 * gap rather than asserting it away.
 */

"use strict";

const Assay = (function () {

  /* ---------- small linear algebra, symmetric matrices only ---------- */

  function jacobiEigen(Ain, sweeps = 100, tol = 1e-12) {
    // Cyclic Jacobi. Chosen over anything cleverer because the matrices here
    // are k x k with k under about 30, and Jacobi is short enough to read.
    const k = Ain.length;
    const A = Ain.map((r) => r.slice());
    let V = Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)));

    for (let sweep = 0; sweep < sweeps; sweep++) {
      let off = 0;
      for (let i = 0; i < k; i++)
        for (let j = i + 1; j < k; j++) off += A[i][j] * A[i][j];
      if (off < tol) break;

      for (let p = 0; p < k; p++) {
        for (let q = p + 1; q < k; q++) {
          if (Math.abs(A[p][q]) < 1e-18) continue;
          const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
          const t = Math.sign(theta || 1) /
                    (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1);
          const s = t * c;
          for (let i = 0; i < k; i++) {
            const aip = A[i][p], aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[i][q] = s * aip + c * aiq;
          }
          for (let i = 0; i < k; i++) {
            const api = A[p][i], aqi = A[q][i];
            A[p][i] = c * api - s * aqi;
            A[q][i] = s * api + c * aqi;
          }
          for (let i = 0; i < k; i++) {
            const vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
          }
        }
      }
    }
    const pairs = A.map((row, i) => ({ value: row[i], vector: V.map((r) => r[i]) }));
    pairs.sort((a, b) => b.value - a.value);
    return pairs;
  }

  /* ---------- data shaping ---------- */

  function standardise(columns) {
    // columns: array of arrays, one per sub-score. Returns z-scores.
    return columns.map((col) => {
      const n = col.length;
      const mean = col.reduce((a, b) => a + b, 0) / n;
      // Population sd, matching numpy's default ddof=0 in Index.standardised.
      const varr = col.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
      const sd = Math.sqrt(varr);
      return col.map((v) => (sd === 0 ? 0 : (v - mean) / sd));
    });
  }

  function correlations(columns) {
    const Z = standardise(columns);
    const k = Z.length, n = Z[0].length;
    const R = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < k; i++) {
      for (let j = i; j < k; j++) {
        let s = 0;
        for (let t = 0; t < n; t++) s += Z[i][t] * Z[j][t];
        const r = s / n;
        R[i][j] = R[j][i] = r;
      }
      R[i][i] = 1;
    }
    return R;
  }

  function pearson(a, b) {
    const n = a.length;
    const ma = a.reduce((x, y) => x + y, 0) / n;
    const mb = b.reduce((x, y) => x + y, 0) / n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
      const x = a[i] - ma, y = b[i] - mb;
      num += x * y; da += x * x; db += y * y;
    }
    return (da === 0 || db === 0) ? NaN : num / Math.sqrt(da * db);
  }

  /* ---------- the five questions ---------- */

  function varianceExplained(R) {
    const k = R.length;
    const eig = jacobiEigen(R);
    return eig.map((e) => Math.max(e.value, 0) / k);
  }

  // Mulberry32 + Box-Muller. A seeded generator so a visitor who reruns the
  // page gets the same answer; NOT numpy's, and the header says why.
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normals(rand, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i += 2) {
      const u1 = Math.max(rand(), 1e-12), u2 = rand();
      const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
      out[i] = r * Math.cos(th);
      if (i + 1 < n) out[i + 1] = r * Math.sin(th);
    }
    return out;
  }

  function parallelAnalysis(k, n, iters = 500, percentile = 95, seed = 0) {
    const rand = rng(seed);
    const all = Array.from({ length: k }, () => []);
    for (let it = 0; it < iters; it++) {
      const cols = [];
      for (let c = 0; c < k; c++) cols.push(normals(rand, n));
      const eig = jacobiEigen(correlations(cols));
      for (let c = 0; c < k; c++) all[c].push(eig[c].value);
    }
    return all.map((vals) => {
      vals.sort((a, b) => a - b);
      // Linear interpolation, matching numpy.percentile's default.
      const pos = (percentile / 100) * (vals.length - 1);
      const lo = Math.floor(pos), hi = Math.ceil(pos);
      return lo === hi ? vals[lo] : vals[lo] + (pos - lo) * (vals[hi] - vals[lo]);
    });
  }

  function mapTest(R) {
    const k = R.length;
    const eig = jacobiEigen(R);
    const crit = [];
    let s = 0, cnt = 0;
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++) if (i !== j) { s += R[i][j] * R[i][j]; cnt++; }
    crit.push(s / cnt);

    const maxM = Math.min(k - 1, 30);
    for (let m = 1; m <= maxM; m++) {
      const A = Array.from({ length: k }, (_, i) =>
        Array.from({ length: m }, (_, f) =>
          eig[f].vector[i] * Math.sqrt(Math.max(eig[f].value, 0))));
      const C = Array.from({ length: k }, (_, i) =>
        Array.from({ length: k }, (_, j) => {
          let v = R[i][j];
          for (let f = 0; f < m; f++) v -= A[i][f] * A[j][f];
          return v;
        }));
      const d = C.map((row, i) => Math.sqrt(Math.max(row[i], 1e-12)));
      let ss = 0, c2 = 0;
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++)
          if (i !== j) { const p = C[i][j] / (d[i] * d[j]); ss += p * p; c2++; }
      crit.push(ss / c2);
    }
    let best = 0;
    for (let i = 1; i < crit.length; i++) if (crit[i] < crit[best]) best = i;
    return { best, crit };
  }

  function redundantPairs(R, names, threshold = 0.90) {
    const out = [];
    for (let i = 0; i < R.length; i++)
      for (let j = i + 1; j < R.length; j++)
        if (Math.abs(R[i][j]) >= threshold)
          out.push({ a: names[i], b: names[j], r: R[i][j] });
    return out.sort((x, y) => Math.abs(y.r) - Math.abs(x.r));
  }

  function pc1Scores(columns) {
    const Z = standardise(columns);
    const R = correlations(columns);
    const eig = jacobiEigen(R);
    let v = eig[0].vector;
    if (v.reduce((a, b) => a + b, 0) < 0) v = v.map((x) => -x);
    const n = Z[0].length;
    const out = new Array(n).fill(0);
    for (let t = 0; t < n; t++)
      for (let c = 0; c < Z.length; c++) out[t] += Z[c][t] * v[c];
    return out;
  }

  /* ---------- the whole audit ---------- */

  const MIN_COLUMNS_FOR_MAP = 5;

  function analyse(columns, names, options = {}) {
    const k = columns.length, n = columns[0].length;
    const R = correlations(columns);
    const ve = varianceExplained(R);
    const eig = jacobiEigen(R);

    const pa = parallelAnalysis(k, n, options.iters || 500, 95, options.seed || 0);
    let parallel = 0;
    for (let i = 0; i < k; i++) { if (eig[i].value > pa[i]) parallel++; else break; }

    const mapUsable = k >= MIN_COLUMNS_FOR_MAP;
    const map = mapUsable ? mapTest(R).best : null;

    const counts = [parallel].concat(mapUsable ? [map] : []);
    const result = {
      n, k,
      varianceExplained: ve,
      pc1Share: ve[0],
      dimensions: {
        parallel, map, mapUsable,
        low: Math.min.apply(null, counts),
        high: Math.max.apply(null, counts),
      },
      redundant: redundantPairs(R, names),
      R,
    };

    if (options.published && options.published.length === n) {
      const pc1 = pc1Scores(columns);
      result.weightsVsPc1 = pearson(options.published, pc1);
    }
    return result;
  }

  return {
    analyse, correlations, standardise, jacobiEigen, varianceExplained,
    parallelAnalysis, mapTest, redundantPairs, pc1Scores, pearson,
    MIN_COLUMNS_FOR_MAP,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Assay;
