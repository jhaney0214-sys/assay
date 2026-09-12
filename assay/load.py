"""Reading a published index into something testable.

An index arrives as a table: one row per entity (country, university, city),
one column per sub-score, and usually a published overall score or rank. That
last column is the claim. Everything here exists to test it.

Two things are done on the way in, and both are corrections for mistakes that
would otherwise be invisible in the output.

**Orientation.** Sub-scores do not all point the same way. An index can mix
"life expectancy" (higher is better) with "homicide rate" (lower is better),
and a correlation matrix built across that mixture reports a negative
correlation between two things that are really the same direction of the same
idea. The first principal component then splits into two halves that cancel.
Columns are therefore oriented against the published composite where one
exists, and the flips are reported rather than silently applied - if a flip
surprises you, the index's own documentation is worth rereading.

**Standardisation.** Pillars are published on whatever scale their author
liked: 0-100 here, 0-1 there, raw dollars somewhere else. Combining those
without standardising lets the widest-scaled column dominate any recomputed
composite for a reason that has nothing to do with weight. z-scores are taken
per column, over the entities actually present.

Missing data is dropped listwise and the count is reported. Indices are small -
50 to 200 rows - so a pairwise estimator buys little and costs interpretability,
and unlike a survey the missingness here is usually "this country was not
scored", which is not ignorable.
"""

import csv
import json
import os


class Index:
    """A published index, ready to be tested against its own claims."""

    def __init__(self, name, entities, columns, values,
                 published=None, weights=None, source=None, claimed_dimensions=None,
                 shared_construction=None, aggregation_form='linear'):
        self.name = name
        self.entities = list(entities)
        self.columns = list(columns)
        self.values = values                      # list of rows, floats
        self.published = published                # published overall score, or None
        self.weights = weights                    # {column: weight}, or None
        self.source = source
        self.claimed_dimensions = claimed_dimensions
        # Sub-scores that share an input BY CONSTRUCTION. V-Dem's liberal,
        # participatory, deliberative and egalitarian indices each incorporate
        # its electoral index, so those four correlate for an arithmetic reason
        # and no empirical one. Without this field the audit would report a
        # definition as a discovery - the same error as reading questionnaire
        # structure as opinion structure.
        self.shared_construction = shared_construction
        # 'linear', or a description of the real formula. Anything but
        # linear disables the weighted-sum reproduction test.
        self.aggregation_form = aggregation_form
        self.flipped = []
        self.dropped = 0

    @property
    def n(self):
        return len(self.entities)

    @property
    def k(self):
        return len(self.columns)

    def matrix(self):
        """Values as a numpy array, entities x columns."""
        import numpy as np
        return np.asarray(self.values, dtype=float)

    def standardised(self):
        """z-scored columns. A column with zero variance is left at zero."""
        import numpy as np
        X = self.matrix()
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        return (X - mu) / sd

    def describe(self):
        bits = ["%s - %d entities, %d sub-scores" % (self.name, self.n, self.k)]
        if self.claimed_dimensions:
            bits.append("claims %d dimensions" % self.claimed_dimensions)
        if self.published is not None:
            bits.append("published composite present")
        if self.weights:
            bits.append("published weights present")
        if self.dropped:
            bits.append("%d row(s) dropped for missing values" % self.dropped)
        if self.flipped:
            bits.append("oriented: flipped %s" % ", ".join(self.flipped))
        return "; ".join(bits)

    @property
    def has_shared_construction(self):
        return bool(self.shared_construction)


def _to_float(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if s == "" or s.lower() in ("na", "n/a", "nan", "-", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def from_csv(path, entity_col, score_cols, published_col=None,
             name=None, weights=None, claimed_dimensions=None, source=None,
             shared_construction=None, aggregation_form='linear'):
    """Load an index from a CSV.

    `score_cols` is the list of sub-score column names - the pillars the index
    says it is made of. `published_col` is its own overall score, if it ships
    one; without it the aggregation tests are skipped rather than guessed at.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("no rows in %s" % path)

    missing = [c for c in [entity_col] + list(score_cols) if c not in rows[0]]
    if published_col and published_col not in rows[0]:
        missing.append(published_col)
    if missing:
        raise ValueError("columns not in file: %s\navailable: %s"
                         % (", ".join(missing), ", ".join(sorted(rows[0]))))

    entities, values, published, dropped = [], [], [], 0
    for r in rows:
        vals = [_to_float(r[c]) for c in score_cols]
        pub = _to_float(r[published_col]) if published_col else None
        if any(v is None for v in vals) or (published_col and pub is None):
            dropped += 1
            continue
        entities.append(str(r[entity_col]).strip())
        values.append(vals)
        published.append(pub)

    if len(entities) < 10:
        raise ValueError(
            "only %d complete rows - too few to say anything about structure. "
            "Check the column names and the missing-value markers." % len(entities))

    idx = Index(name or os.path.basename(path), entities, score_cols, values,
                published=(published if published_col else None),
                weights=weights, source=source,
                claimed_dimensions=claimed_dimensions,
                shared_construction=shared_construction,
                aggregation_form=aggregation_form)
    idx.dropped = dropped
    return idx


def orient(index):
    """Point every sub-score the same way, and say which were turned round.

    Against the published composite when there is one, because that defines
    which direction the index itself calls "good". Without it, against the
    column that correlates most strongly with the rest - the majority vote,
    the same device `Sextant/tools/norms.py` uses for factor orientation.
    """
    import numpy as np
    X = index.matrix()
    if index.published is not None:
        ref = np.asarray(index.published, dtype=float)
    else:
        Z = index.standardised()
        R = np.corrcoef(Z, rowvar=False)
        ref = Z[:, int(np.argmax(np.abs(R).sum(axis=0)))]

    flipped = []
    for j, col in enumerate(index.columns):
        if np.corrcoef(X[:, j], ref)[0, 1] < 0:
            X[:, j] = -X[:, j]
            flipped.append(col)
    index.values = X.tolist()
    index.flipped = flipped
    return index


def from_spec(path):
    """Load an index from a JSON spec beside its data file.

    A spec records what the index CLAIMS - its pillars, its published weights,
    how many dimensions it says it has - so the audit compares against the
    author's own account rather than against an assumption of ours.
    """
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    data = spec["data"]
    if not os.path.isabs(data):
        data = os.path.join(os.path.dirname(os.path.abspath(path)), data)
    idx = from_csv(
        data,
        entity_col=spec["entity_col"],
        score_cols=spec["score_cols"],
        published_col=spec.get("published_col"),
        name=spec.get("name"),
        weights=spec.get("weights"),
        claimed_dimensions=spec.get("claimed_dimensions"),
        source=spec.get("source"),
        shared_construction=spec.get("shared_construction"),
        aggregation_form=spec.get("aggregation_form", "linear"),
    )
    return orient(idx) if spec.get("orient", True) else idx


def panel_from_spec(path):
    """Load a panel spec into {year: Index}, one Index per year.

    A panel spec is an ordinary spec plus `year_col`. Each year becomes a
    standalone Index so every existing test applies to it unchanged, and the
    drift analysis compares them.
    """
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    data = spec["data"]
    if not os.path.isabs(data):
        data = os.path.join(os.path.dirname(os.path.abspath(path)), data)
    year_col = spec["year_col"]
    score_cols = spec["score_cols"]
    pub_col = spec.get("published_col")

    with open(data, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    by_year = {}
    for r in rows:
        try:
            y = int(str(r[year_col]).strip())
        except (KeyError, ValueError):
            continue
        vals = [_to_float(r[c]) for c in score_cols]
        pub = _to_float(r[pub_col]) if pub_col else None
        if any(v is None for v in vals) or (pub_col and pub is None):
            continue
        by_year.setdefault(y, {"e": [], "v": [], "p": []})
        by_year[y]["e"].append(str(r[spec["entity_col"]]).strip())
        by_year[y]["v"].append(vals)
        by_year[y]["p"].append(pub)

    out = {}
    for y, d in sorted(by_year.items()):
        if len(d["e"]) < 20:
            continue
        idx = Index("%s %d" % (spec.get("name", "panel"), y), d["e"], score_cols,
                    d["v"], published=(d["p"] if pub_col else None),
                    weights=spec.get("weights"), source=spec.get("source"),
                    claimed_dimensions=spec.get("claimed_dimensions"),
                    shared_construction=spec.get("shared_construction"),
                    aggregation_form=spec.get("aggregation_form", "linear"))
        out[y] = orient(idx) if spec.get("orient", True) else idx
    return out, spec
