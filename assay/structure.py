"""How many things is this index actually measuring?

A published index asserts a number of dimensions by naming them: six pillars,
twelve components, four sub-indices. That is a claim about the world, and it is
testable, because if two pillars move together across every entity then they
are not two pillars.

The dimensionality machinery is imported from Sextant rather than copied. It is
the same Horn parallel analysis and Velicer MAP, already tested there against
synthetic structures with known answers, and a second copy would drift from the
first the moment either was fixed.

**Pearson here, not polychoric.** Sextant needs polychoric because GSS answers
are ordered categories with few levels, where treating codes as numbers
attenuates correlations badly. Published sub-scores are continuous quantities on
an interval scale, so that correction does not apply and would only add
assumptions.

**Both retention tests are weak when there are few columns.** Sextant's problem
was over-retention at n above 10,000; the problem here is the opposite shape -
an index has 3 to 12 pillars and 50 to 200 entities, and MAP in particular is
unreliable below about five columns because there is almost nothing left to
partial out. Where the tests disagree, the honest answer is the range.
"""

import os
import sys

import numpy as np

_SEXTANT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Sextant")
if _SEXTANT not in sys.path:
    sys.path.insert(0, _SEXTANT)

try:
    from sextant.factor import parallel_analysis, map_test, principal_axis, promax
except ImportError as exc:                                      # pragma: no cover
    raise ImportError(
        "Assay reuses Sextant's factor machinery and could not import it from\n"
        "  %s\n"
        "Keep the two projects as siblings in the workstation, or fix the path "
        "here. The code is deliberately not duplicated: a second copy drifts "
        "from the first the moment either is corrected." % _SEXTANT) from exc

MIN_COLUMNS_FOR_MAP = 5
REDUNDANT_AT = 0.90


def correlations(index):
    """Correlation matrix of the standardised sub-scores."""
    return np.corrcoef(index.standardised(), rowvar=False)


def dimensionality(R, n_obs, seed=0):
    """Bracket the number of real dimensions with two disagreeing methods.

    Returns a dict. `parallel` and `map` are counts; where they differ the
    answer is a range, and the caller is expected to report it as one.
    """
    k = R.shape[0]
    eig = np.linalg.eigvalsh(R)[::-1]
    noise = parallel_analysis(k, n_obs, seed=seed)
    retained_pa = int((eig > noise).sum())

    if k >= MIN_COLUMNS_FOR_MAP:
        retained_map, crit = map_test(R)
        map_usable = True
    else:
        retained_map, crit, map_usable = None, None, False

    return {
        "k": k,
        "n": n_obs,
        "eigenvalues": eig.tolist(),
        "noise_threshold": noise.tolist(),
        "parallel": retained_pa,
        "map": retained_map,
        "map_usable": map_usable,
        "low": retained_map if map_usable and retained_map is not None
               else retained_pa,
        "high": retained_pa,
    }


def variance_explained(R):
    """Share of sub-score variance taken by each component, largest first."""
    eig = np.linalg.eigvalsh(R)[::-1]
    eig = np.clip(eig, 0, None)
    return (eig / eig.sum()).tolist()


def loadings(R, m):
    """Rotated loadings, so a reader can see WHICH pillars are the same pillar.

    Rotation is attempted, not assumed. An index with an exactly duplicated
    column - which happens, and is itself a finding - gives a singular
    correlation matrix, and promax inverts a matrix built from it. Falling back
    to the unrotated solution keeps the audit running and reports which one the
    reader is looking at; crashing here would suppress the duplicate finding
    that caused it.
    """
    if m < 1:
        return None, None, "none"
    L, _ = principal_axis(R, m)
    if m == 1:
        return L, None, "unrotated (single factor)"
    try:
        Lr, Phi = promax(L)
        return Lr, Phi, "promax"
    except np.linalg.LinAlgError:
        return L, None, "unrotated (rotation failed - near-singular correlations)"


def redundant_pairs(R, columns, threshold=REDUNDANT_AT):
    """Sub-score pairs correlating so highly they cannot be separate pillars."""
    out = []
    k = len(columns)
    for i in range(k):
        for j in range(i + 1, k):
            r = float(R[i, j])
            if abs(r) >= threshold:
                out.append((columns[i], columns[j], r))
    return sorted(out, key=lambda t: -abs(t[2]))


def pc1_scores(index):
    """Entity scores on the first principal component, oriented positive."""
    Z = index.standardised()
    R = np.corrcoef(Z, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    v = vecs[:, int(np.argmax(vals))]
    if v.sum() < 0:
        v = -v
    return Z @ v


def analyse(index, seed=0):
    """The whole structural picture for one index."""
    R = correlations(index)
    dim = dimensionality(R, index.n, seed=seed)
    var = variance_explained(R)
    m = max(1, dim["low"])
    L, Phi, rotation = loadings(R, m)
    return {
        "R": R,
        "dimensions": dim,
        "variance_explained": var,
        "pc1_share": var[0],
        "loadings": L,
        "factor_correlations": Phi,
        "rotation": rotation,
        "redundant": redundant_pairs(R, index.columns),
        "n": index.n,
        "k": index.k,
    }
