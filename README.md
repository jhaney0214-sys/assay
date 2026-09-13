# Assay

**Status: production.** 66 tests; the dimensionality path is validated against a published HDI result, the rank-envelope path has no external anchor. See [PRODUCTION.md](../PRODUCTION.md).

Tests whether a published index measures what it says it measures. Named for
the metallurgical test that tells you what an ore is actually made of, rather
than what the seller called it.

Status: **engine built, tested, and validated against a published result.**
Three indices audited: one was a trap the tool correctly refused, one
reproduced a figure from the literature on data it had not seen, and one
exercised every test at once. Adapted since to answer a question the
literature has not: whether a reported year-over-year rank change survives
the index's own weighting uncertainty.

## Why

An index that names six pillars is making two claims at once: that the pillars
are distinct things, and that the weights trading them off are the right ones.
Both are testable, and neither is usually tested.

There are over a hundred ESG rating systems, each with its own weights. The
academic response to university rankings has moved to arguing that single-score
aggregation implies a precision the underlying data cannot support. The critique
exists in journals. It does not exist as something you can run.

This is the same position [Sextant](../Sextant) takes about political compasses,
pointed at other people's numbers instead of our own: **nothing may be asserted
about structure that the data has not been asked about.**

## What it tests

| Question | Method | The failure it catches |
|-|-|-|
| How many dimensions are really there? | Horn's parallel analysis + Velicer's MAP on the sub-score correlations | Six pillars that are one pillar with six names |
| Which pillars are the same pillar? | Pairwise correlation above 0.90 | A thing counted twice, and so weighted twice, silently |
| Does the published score follow from the published weights? | Recompute and compare | Undocumented processing between inputs and headline |
| Do the weights beat one unweighted component? | r(published, PC1) | Weights that are decorative — a committee's deliberation that changed no ranking |
| How much of a rank is the entity? | Rank envelope over thousands of weightings the author could also have defended | A rank reported as a property of a country when it is a property of a choice |

## The trap this is built around

**A ranking that barely moves under reweighting looks robust, and has a second
explanation that looks identical in the output: the sub-scores are so correlated
that no weighting could have separated them.** Stability is evidence of
robustness only when the pillars are actually distinct. Nothing here reports
one without the other, and `tests/test_aggregation.py::test_stability_is_not_robustness`
fails if that ever stops being true.

## The worked example, and why it is not a finding

`python tools/fetch_vdem.py --year 2024` builds a table of V-Dem's five
democracy indices for 174 countries. Auditing it returns 97.1% of variance on
one component and ten sub-score pairs correlating above 0.94, which looks like a
devastating result about the "five varieties of democracy".

It is not a result at all. V-Dem's liberal, participatory, deliberative and
egalitarian indices each **incorporate the electoral democracy index by
construction** — four of the five columns contain the fifth. They would correlate
if the world contained no pattern whatsoever.

So a spec can declare `shared_construction`, and the report then refuses to call
the collapse a finding, in a warning printed above every number. This is the
same error as reading questionnaire structure as opinion structure, which is why
[Sextant](../Sextant) parcels its items — **the shape of the instrument
masquerading as the shape of the world.**

One thing in that run does survive: even among indices correlating at 0.95+,
which one you weight moves Belarus from 117th to 169th and Singapore from 81st
to 127th. Rank differences smaller than that spread are not differences between
countries.

**A real finding needs an index whose sub-scores are built from independent
inputs.** That is what HDI provides.

## The validation: HDI

```bash
python tools/fetch_undp.py --year 2022
python ax.py audit data/hdi-2022.json --draws 2000
```

HDI's four indicators are measured by four different institutions from four
unrelated instruments — life expectancy from civil registration, the two
schooling measures from education ministries and household surveys, GNI from
national accounts. None contains any of the others, so there is no shared
construction to explain away a result.

On 193 countries in 2022, Assay reports **84.1% of variance on one component**
against three claimed dimensions, and r(HDI, PC1) = 0.9976.

**This is not a new finding, and that is the point.** McGillivray argued in
1991 that HDI is redundant with its own components, and the PCA literature
reports the first component accounting for **78–90%** of HDI variance across
1975–2005. Assay's 84.1% for 2022 falls inside that published range, computed
from raw UNDP indicators by code that had never seen those papers.

That is the only kind of check that means anything: an external anchor, not
another internal consistency test. It is the thing
[Greenlight](../Greenlight)'s revenue model still lacks.

Two honest caveats on the number itself. `r(HDI, PC1) = 0.9976` is not
independent evidence — once four indicators are collinear, *any* sensible
weighting tracks the first component, so "the weights are decorative" is a
restatement of the collinearity rather than a second finding. And the ranks are
volatile despite the single dimension: the median country moves 41 places of
193 under reweighting, because one factor sets the broad ordering while the
remaining 16% decides who sits above whom in a densely packed field. The report
now says both.

## The first index where all four tests fire: EPI

```bash
python tools/fetch_epi.py
python ax.py audit data/epi-2024.json --draws 2000
```

Yale's Environmental Performance Index publishes eleven issue categories, their
scores, and a machine-readable `Weights.csv` giving each category's share of the
total. The categories are measured from unrelated instruments — satellite
aerosol retrievals, fisheries landings, protected-area registries, greenhouse
gas inventories — and aggregate linearly. So every test applies at once, and the
answers differ from HDI's in ways that matter.

**The claimed structure is not there, and neither is a clean alternative.** EPI
names three policy objectives; both retention tests return one factor — but that
factor explains only **45%** of the variance. This is neither a three-dimensional
index nor a one-dimensional one. The remaining 55% is spread across components
too weak to retain individually, which is what a set of loosely related
measurements looks like when it does not resolve into named dimensions.

**The weights do real work**, unlike HDI's. r(EPI, PC1) = 0.921, below the 0.95
line, so the published ranking is not recoverable from an unweighted component.
Yale's weighting is load-bearing.

**The ordering reproduces; the score does not.** r(recomputed, published) =
0.9962, but a typical country lands 2.2 points from its published value — 19% of
one standard deviation — with rank disagreements up to 10 places. A weighted sum
of the published categories is a good model of EPI, not EPI itself. Reporting
only the correlation would have called it exact; reporting only the score error
would have called it broken.

**Fisheries runs against its own index.** It correlates negatively with EPI, so
a country scoring well on fisheries tends to score worse overall.

Two limits on all of that. **75 of 180 countries are dropped**, and not at
random: Forests is missing for 49 and Fisheries for 39, so the audited sample is
biased toward countries with both a coastline and forests. And the rank envelope
here perturbs Yale's *own* published weights rather than inventing a range —
the median country still moves 28 places of 105.

## The new question: is a reported rank change real?

```bash
python tools/build_hdi_panel.py
python ax.py drift data/hdi-panel.json --draws 600
```

Rank movements are reported as events — "Country X climbed four places" is a
headline and sometimes a policy argument. The move is computed under one
weighting, chosen by a committee. Nobody checks what the same two years would
have said under a weighting the same committee could equally have defended.

The test is **paired**, which is what makes it work. Comparing a change against
the static rank envelope would be wrong: a country whose rank is uncertain by
twenty places can still have moved up robustly, because the same uncertainty
applies to both years and cancels. So each candidate weighting is applied to
*both* years and the change recomputed.

**There is no single percentage to report, and that is the first result.** Two
choices have to be made and neither has a right answer: how much weight
disagreement counts as defensible, and how often a change must keep its sign to
be called robust. Both are swept.

HDI, annual changes, share keeping direction:

| weights vary by | bar 0.75 | bar 0.90 | bar 0.99 |
|-|-|-|-|
| ±1.0 points | 88.1% | 77.1% | 65.1% |
| ±2.0 | 79.4% | 63.2% | 49.1% |
| ±5.0 | 64.9% | 45.0% | 26.9% |
| ±10.0 | 54.8% | 33.4% | 15.7% |

**16% to 88%.** Anyone quoting one cell is reporting their own settings. An
earlier version of this file quoted 27.4%, then 53.4%; both were true of one
corner and neither was a fact about HDI.

Robustness falls monotonically in both parameters, so the surface describes one
phenomenon rather than noise. Two things survive all of it:

**1. A move of fewer than two places is never reliable**, at any setting on the
surface.

**2. Robustness rises with the gap between the compared years, at every corner
tested.** Same index, same countries, only the distance changes:

| gap | pairs | worst corner | best corner |
|-|-|-|-|
| 1 year | 32 | 15.6% | 88.0% |
| 2 | 31 | 30.6% | 93.0% |
| 5 | 28 | 52.6% | 96.2% |
| 10 | 23 | 67.3% | 97.8% |
| 20 | 13 | 78.3% | 98.8% |

**That is the finding, and it is parameter-free: an annual rank change is
largely a property of the method, a decadal one largely a property of the
country.** Annual releases are the ones that get reported.

### The second index, and the thing that actually decides it

```bash
python tools/build_epi_panel.py
python ax.py drift data/epi-panel.json --draws 800 --no-structure
```

EPI publishes every score twice in one file: `.new` is current, `.old` is a
**backcast** — the score a country would have had on data from about ten years
earlier, run through the *current* methodology. Yale computes both, and the
difference is its own reported ten-year change. That makes them comparable in a
way two separate EPI releases are not.

EPI's ten-year change is robust across **77%–99%** of its own surface, against
**16%–88%** for HDI's annual changes. That looks like a difference between the
indices and is not — it confounds index with timescale.

Matched on timescale, they agree. HDI at a ten-year gap spans 67%–98%, which
overlaps EPI's 77%–99% heavily; the two ranges only separate when HDI is read
annually and EPI decadally. Two indices built by different institutions from
unrelated measurements behave alike once the gap is the same.

Two caveats. EPI's panel is a single ten-year gap, not a series, so its 175
changes give thin bins. And EPI's published score is **not exactly reproducible
from its own published components** at either level of its hierarchy — a country
lands up to 1.2 points out from the three policy objectives, and 5.6 from the
eleven issue categories. That is why `change_fidelity` exists: the level is not
reproduced, but the change is (r = 0.998, 97.8% same direction), because a
per-country offset largely cancels in a difference.

### Why this is not Foster, McGillivray & Seth (2009)

[OPHI Working Paper 26](https://ophi.org.uk/sites/default/files/OPHI-wp26_vs5.pdf)
asks the closest existing question: is a comparison between two countries
reversed under an alternative weighting? They apply it to HDI 1998 and 2004 and
report that about 70% of within-year pairwise comparisons are fully robust.

That is **cross-sectional**. Each year is analysed on its own, and the object is
"does A rank above B". The object here is "did A move relative to the field
between two years", which their method does not address — the paper contains no
treatment of change over time.

The two results fit together: comparisons between distant countries are mostly
robust, while the small movements that get reported annually mostly are not.

## Running it

```bash
cd "C:/Users/Jhane/AI Workstation/Assay"
python tools/fetch_vdem.py --year 2024
python ax.py audit data/vdem-2024.json --draws 2000 --out out/vdem-2024.txt
python ax.py columns data/vdem-2024.csv        # to write a spec for a new index
```

A spec is a small JSON file recording what the index claims about itself:

```json
{
  "name": "Example Index 2026",
  "data": "example.csv",
  "entity_col": "country",
  "score_cols": ["pillar_a", "pillar_b", "pillar_c"],
  "published_col": "overall",
  "weights": {"pillar_a": 50, "pillar_b": 30, "pillar_c": 20},
  "claimed_dimensions": 3,
  "shared_construction": null
}
```

Everything is compared against that, so the audit tests the index against its
own account of itself rather than against an assumption of ours.

Tests are `unittest`; there is no pytest here and `unittest discover` fails
because `tests/` has no `__init__.py`:

```bash
for t in tests/test_*.py; do PYTHONPATH=. python "$t"; done
```

## What is not established

- **One validation is one validation.** HDI's 84.1% matching the published
  78–90% range shows the dimensionality path is right on one index whose answer
  was already known. The rank-envelope and weight-reproduction paths have no
  external anchor at all yet.
- Two indices now, not one - but both are country-level and both are
  built by Western institutions from overlapping data infrastructure. That is
  not the same as two independent replications.
- Both free parameters are now swept rather than chosen, but the RANGES swept
  (±1 to ±10 points; bars 0.75 to 0.99) are themselves judgements. They are
  reported rather than hidden, which is the most that can honestly be done.
- The drift finding has NO external anchor. The dimensionality result could be
  checked against McGillivray and the PCA literature and landed inside it;
  nobody has published a comparable number for rank-change robustness, which
  is what makes it novel and also what leaves it unverified.
- The analysis is GENEROUS to the indices: it compares years within one
  methodology, while real reported changes compare releases and carry
  methodology drift too. These are lower bounds on unreliability.
- The 0.90 redundancy threshold and the 0.95 "decorative weights" threshold are
  conventions, not measurements. Sextant's replication thresholds were moved once
  by measuring them; these have not been.
- `rank_envelope`'s "near" mode perturbs published weights with a Dirichlet
  concentration of 50 by default. That number encodes a judgement about what
  counts as a disagreement the authors could have had, and it is not derived
  from anything.
- Listwise deletion drops countries that are not scored on every pillar, and
  that missingness is unlikely to be random — an index tends not to score the
  places it finds hardest to score.

## Layout

```
assay/      load.py  structure.py  aggregation.py  report.py
            drift.py  driftreport.py
tools/      fetch_vdem.py  fetch_undp.py  fetch_epi.py
            build_hdi_panel.py  build_epi_panel.py
tests/      66 tests
ax.py       the CLI
```

`structure.py` **imports Sextant's factor machinery rather than copying it** —
the same parallel analysis, MAP, principal axis and promax, already tested there
against synthetic structures with known answers. A second copy would drift from
the first the moment either was fixed. Keep the two projects as siblings.

## Dependencies

numpy, scipy, pandas (via Sextant). All free. No API keys, no registration, no
paid data.

## Data

Three sources, all public, all fetched over plain HTTP with no key and no
registration: V-Dem via Our World in Data, the Human Development Index from the
United Nations Development Programme, and the Environmental Performance Index
from Yale.

`data/` ships the derived tables those fetches produce - national-level
indicator and sub-score columns, a few hundred rows each - so a clone can
reproduce every result in this file without re-fetching. They are published
summary statistics rather than microdata, and each is rebuildable from scratch
with the matching `tools/fetch_*.py`. Attribution belongs to the three
publishers named above; nothing here is offered under a licence Assay is in a
position to grant.
