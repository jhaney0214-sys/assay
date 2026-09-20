"""A ledger of the numbers a project publishes, and the checks that keep them true.

Six places in this workstation had independently invented the same mechanism —
a graded claim, an anchor, a date, and something that re-checks it — before any
of them were the same code:

  METHOD.md / method_health.py   a dated lesson with a re-test horizon
  Sextant                        statute / secondary / unconfirmed on state law
  Assay                          "measured vs. estimated", two analysis paths
  Farewatch                      "measured vs. estimated" on all-in trip cost
  Greenlight                     "trust the shape, not the dollars"
  Ferrule, then Assay            CountedClaimsInProse, then tests/test_claims.py

`NEXT.md` then proposed three more of it under three more names — a confidence
ledger, a source registry, a freshness monitor. This is that mechanism, written
once. It is `netcache.py`'s situation exactly: four projects had a fetch cache
before one module had it.

The failure it exists to catch is the one Ferrule's class is named after: **a
number written into a document is a claim nobody re-runs.** In the project this
was first built for, a headline share and a correlation sat in the README four
times, in the public page twice, and — until a day before this was written — in
a trailing `# python:` comment as the only record of what the engine actually
produced. Had the engine moved, both repositories' CI would have stayed green
while the prose went quietly wrong.

## The chain this builds, and the one link it cannot close

    engine  --(project's own test)-->  raw  --(here)-->  value  --(here)-->  prose

Only the first link needs the project's code, so only the first link is a test a
project has to write. Everything right of `raw` is text against text, needs no
imports, and therefore **runs on a runner that cannot import the project at
all** — which is the specific reason this is worth extracting. Assay's
hand-written version of this check cannot run in its own CI: it imports Sextant's
factor machinery through a sibling checkout, and putting a token for a private
repository on a public one was declined on 2026-09-19. Splitting the chain at
`raw` moves two of its three links into CI without touching that decision.

## What `verify` checks

  schema          required fields, a status from the vocabulary, dates that
                  parse, ids that are unique, a horizon after its own check date
  rounding        `format % raw` still renders `value` — a rounding change is a
                  change to a published claim, not a tolerance
  presence        every path in `appears_in` exists and contains `value`
  contradiction   a DIFFERENT number of the same shape sitting next to the
                  claim, which is what a half-finished edit leaves behind.
                  Needs `near`; without it the report says NOT CHECKED rather
                  than passing silently
  coverage        (with --scan) a file that quotes `value` and is not listed in
                  `appears_in` — the occurrence nobody remembered to pin

The contradiction scan is the half that a bare `assertIn(value, text)` cannot
do. `assertIn` asks whether the right number is present; it says nothing about a
wrong one being present too, and "12.5% here, 12.4% three paragraphs down" is
exactly what a partial edit produces.

## What it does not do, on purpose

It does not discover claims. Handing it a README and asking which figures are
claims produces a flood — years, version numbers, test counts, dollar amounts,
HTTP statuses — and a checker that cries wolf gets muted, which is worse than
not having it. Claims are declared, the way `publish_audit.py`'s leak terms are
declared. Discovery is a second slice, if ever.

It does not decide whether a claim is TRUE. `status` is the project's own
assertion about its own number, and `anchor` is where a reader goes to disagree.
This checks that the assertion is stated, dated, and consistent everywhere it
appears — not that it is right.

**Meant to be VENDORED, not imported across a repo boundary**, the same as
`netcache.py`: every project here is an independently cloneable repository with
no runtime dependency on another. This copy, in `AI Workstation/tools/`, is the
one with tests; treat it as the source of truth and diff a vendored copy against
it rather than hand-patching both.

    python claims.py verify ../Assay
    python claims.py verify ../Assay --scan "**/*.md" "docs/*.html"
    python claims.py stale ../Assay ../Outcrop --asof 2027-01-01
    python claims.py render ../Assay --format md
"""

import argparse
import datetime
import io
import json
import pathlib
import re
import sys

#: What a project is allowed to say about how it knows a number. Deliberately
#: four, deliberately ordered weakest-last, and deliberately not extensible by
#: a caller: five projects each invented their own vocabulary for this, which is
#: how the workstation ended up with "measured vs. estimated" in two places and
#: "statute / secondary / unconfirmed" in a third. One vocabulary or none.
STATUSES = ("measured", "estimated", "modeled", "unconfirmed")

REQUIRED = ("id", "claim", "value", "status", "anchor", "checked_on")
OPTIONAL = ("recheck_by", "appears_in", "raw", "format", "scale", "near",
            "window", "allow", "durable", "notes", "tolerance")

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: How far either side of a `near` phrase the contradiction scan looks. A
#: paragraph of HTML is frequently one very long line, so a line-scoped window
#: would miss a headline figure sitting alone in a <b> two lines above the
#: sentence that explains it.
DEFAULT_WINDOW = 300


class LedgerError(Exception):
    """The ledger file itself is unreadable or malformed."""


class Finding(object):
    """One thing `verify` has to say. Not all findings are failures."""

    def __init__(self, kind, claim_id, message, path=None, fatal=True):
        self.kind = kind
        self.claim_id = claim_id
        self.message = message
        self.path = path
        self.fatal = fatal

    def __repr__(self):
        return "<Finding %s %s%s>" % (
            self.kind, self.claim_id, "" if self.fatal else " (not fatal)")

    def as_dict(self):
        return {"kind": self.kind, "claim": self.claim_id,
                "message": self.message, "path": self.path,
                "fatal": self.fatal}

    def line(self):
        where = " [%s]" % self.path if self.path else ""
        mark = "FAIL" if self.fatal else "note"
        return "%s  %-14s %s%s: %s" % (
            mark, self.kind, self.claim_id, where, self.message)


def load(path):
    """Read a ledger file. Raises LedgerError rather than returning nothing.

    A ledger that cannot be read is not a ledger with no claims in it. That
    distinction has been recorded five separate times in this workstation as
    the thing instruments get wrong — an instrument that could not look
    reporting that it found nothing — so it is not going to be re-introduced
    here by returning an empty list on a missing file.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise LedgerError("no ledger at %s" % path)
    try:
        with io.open(str(path), encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise LedgerError("%s is not valid JSON: %s" % (path, exc))
    if isinstance(data, dict):
        data = data.get("claims", data)
    if not isinstance(data, list):
        raise LedgerError(
            "%s must hold a list of claims, or an object with a 'claims' list"
            % path)
    return data


def shape_of(value):
    """A regex matching any number written the same way `value` is.

    "12.5%"    -> \\d+\\.\\d%       so 12.4% and 9.7% both match
    "0.4412"   -> \\d+\\.\\d{4}
    "88 rows"  -> \\d+ rows

    The decimal place count is held exactly and the integer part is not,
    because a claim changing from 12.5% to 12.54% is a formatting change the
    `rounding` check already owns, while 12.5% -> 9.7% is the drift this is
    looking for. Non-numeric text is matched literally, which is what keeps
    "88 rows" from matching every two-digit number in a file.
    """
    out = []
    last = 0
    for match in NUMBER.finditer(value):
        out.append(re.escape(value[last:match.start()]))
        whole = match.group(0)
        # Both guards are load-bearing, and the trailing one was found by
        # running this against a real project rather than by review. A
        # four-decimal shape happily matches the leading characters of the
        # SAME claim's full-precision value sitting in a test file, so every
        # claim reported its own `raw` as a contradiction of itself.
        out.append(r"(?<![\d.])")
        if "." in whole:
            out.append(r"\d+\.\d{%d}" % len(whole.split(".", 1)[1]))
        else:
            out.append(r"\d+")
        out.append(r"(?!\d)")
        last = match.end()
    out.append(re.escape(value[last:]))
    joined = "".join(out)
    if not NUMBER.search(value):
        # A claim whose value holds no digits at all ("refuses rather than
        # guesses"). There is no shape to drift, so match only itself.
        joined = re.escape(value)
    return re.compile(joined)


WHITESPACE = re.compile(r"\s+")


def normalise(text):
    """Collapse runs of whitespace, so a claim survives a line wrap.

    Found by pointing this at Assay's real README: it is hard-wrapped at 79
    columns, and a claim written as several words happens to break across two
    lines. A literal search reported the README as not carrying a number the
    README plainly carries. Where an author chose to wrap is not a fact about
    the claim, so it must not be able to fail the check — or pass it, which is
    the worse direction: a wrap falling between two halves of a contradiction
    would hide it.
    """
    return WHITESPACE.sub(" ", text)


def excerpt(text, start, end, margin=60):
    """The matched number with enough around it to recognise the sentence."""
    left = max(0, start - margin)
    right = min(len(text), end + margin)
    return "%s%s%s" % ("..." if left else "", text[left:right],
                       "..." if right < len(text) else "")


def windows(text, phrases, width):
    """Character ranges around each occurrence of any phrase. Merged."""
    spans = []
    lowered = text.lower()
    for phrase in phrases:
        needle = phrase.lower()
        start = lowered.find(needle)
        while start != -1:
            spans.append((max(0, start - width),
                          min(len(text), start + len(needle) + width)))
            start = lowered.find(needle, start + 1)
    spans.sort()
    merged = []
    for span in spans:
        if merged and span[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    return merged


def _date(value, field, claim_id, findings):
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError):
        findings.append(Finding(
            "schema", claim_id,
            "%s is %r, which is not an ISO date (YYYY-MM-DD)"
            % (field, value)))
        return None


def check_schema(claims):
    """Everything checkable without reading a single prose file."""
    findings = []
    seen = {}
    for position, claim in enumerate(claims):
        if not isinstance(claim, dict):
            findings.append(Finding(
                "schema", "#%d" % position, "claim is not an object"))
            continue
        claim_id = claim.get("id", "#%d" % position)

        for field in REQUIRED:
            if not claim.get(field):
                findings.append(Finding(
                    "schema", claim_id, "missing required field %r" % field))

        unknown = set(claim) - set(REQUIRED) - set(OPTIONAL)
        for field in sorted(unknown):
            findings.append(Finding(
                "schema", claim_id,
                "unknown field %r; a typo here fails open, so it is a failure"
                % field))

        if "id" in claim:
            if not ID_PATTERN.match(str(claim["id"])):
                findings.append(Finding(
                    "schema", claim_id,
                    "id must be lowercase letters, digits and underscores"))
            if claim["id"] in seen:
                findings.append(Finding(
                    "schema", claim_id,
                    "duplicate id, also used at position %d" % seen[claim["id"]]))
            seen[claim["id"]] = position

        if claim.get("status") and claim["status"] not in STATUSES:
            findings.append(Finding(
                "schema", claim_id,
                "status %r is not one of %s"
                % (claim["status"], ", ".join(STATUSES))))

        checked = None
        if claim.get("checked_on"):
            checked = _date(claim["checked_on"], "checked_on", claim_id,
                            findings)
        if claim.get("recheck_by"):
            horizon = _date(claim["recheck_by"], "recheck_by", claim_id,
                            findings)
            if horizon and checked and horizon <= checked:
                findings.append(Finding(
                    "schema", claim_id,
                    "recheck_by %s is not after checked_on %s"
                    % (claim["recheck_by"], claim["checked_on"])))
        elif not claim.get("durable"):
            findings.append(Finding(
                "schema", claim_id,
                "no recheck_by and no `durable` reason; a claim with neither "
                "is a claim nobody has decided how long to trust"))

        if ("raw" in claim) != ("format" in claim):
            findings.append(Finding(
                "schema", claim_id,
                "`raw` and `format` only mean anything together; one without "
                "the other cannot check the rounding"))

        if "scale" in claim and "raw" not in claim:
            findings.append(Finding(
                "schema", claim_id,
                "`scale` converts `raw` into published units and there is no "
                "`raw` to convert"))

        allow = claim.get("allow")
        if allow is not None:
            if not isinstance(allow, dict):
                findings.append(Finding(
                    "schema", claim_id,
                    "`allow` maps each excused number to WHY it is legitimate; "
                    "a bare list is an exemption nobody can disagree with"))
            else:
                for excused, reason in sorted(allow.items()):
                    if not str(reason).strip():
                        findings.append(Finding(
                            "schema", claim_id,
                            "`allow` entry %r has no reason" % excused))
            if not claim.get("near"):
                findings.append(Finding(
                    "schema", claim_id,
                    "`allow` excuses matches found by the `near` scan, and "
                    "there is no `near` to scan", fatal=False))

        if claim.get("window") is not None and not claim.get("near"):
            findings.append(Finding(
                "schema", claim_id,
                "`window` sets the width of the `near` scan, and there is no "
                "`near` to scan", fatal=False))
    return findings


def rendered_value(claim):
    """`format % (raw * scale)` — what the ledger's own number looks like.

    `scale` exists because `raw` has to stay exactly what the engine returns,
    so that `check_computed` can compare the two without either side knowing
    how the other is written. An engine commonly returns a proportion while
    the README publishes a percentage. One of those has to move, and it
    must not be `raw`: the moment the ledger stores a display-scaled number,
    the comparison against the engine needs the scale applied in the project's
    own test, which is the code this is trying to stop every project writing.
    """
    raw = claim["raw"]
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = raw * claim.get("scale", 1)
    return claim["format"] % raw


def check_rounding(claims):
    """`format % (raw * scale)` still renders `value`.

    A number rounding differently is a different published claim, so this is a
    failure and not a tolerance. Assay's test says the same thing in its own
    words: "a formatting change is a real change to a published claim."
    """
    findings = []
    for claim in claims:
        if "raw" not in claim or "format" not in claim:
            continue
        claim_id = claim.get("id", "?")
        try:
            rendered = rendered_value(claim)
        except (TypeError, ValueError) as exc:
            findings.append(Finding(
                "rounding", claim_id,
                "format %r cannot render raw %r: %s"
                % (claim["format"], claim["raw"], exc)))
            continue
        if rendered != claim["value"]:
            findings.append(Finding(
                "rounding", claim_id,
                "raw %r renders as %r through %r, but the ledger publishes %r"
                % (claim["raw"], rendered, claim["format"], claim["value"])))
    return findings


def check_prose(claims, root):
    """Presence and contradiction, over the files each claim names."""
    root = pathlib.Path(root)
    findings = []
    cache = {}

    def text_of(relative):
        if relative not in cache:
            target = root / relative
            if not target.exists():
                cache[relative] = None
            else:
                with io.open(str(target), encoding="utf-8",
                             errors="replace") as handle:
                    cache[relative] = normalise(handle.read())
        return cache[relative]

    for claim in claims:
        claim_id = claim.get("id", "?")
        value = claim.get("value")
        paths = claim.get("appears_in") or []
        if not value:
            continue
        if not paths:
            findings.append(Finding(
                "unpublished", claim_id,
                "no `appears_in`; the ledger holds this number but nothing "
                "says it is written down anywhere", fatal=False))
            continue

        shape = shape_of(value)
        near = claim.get("near") or []
        width = claim.get("window") or DEFAULT_WINDOW
        fired = set()

        for relative in paths:
            text = text_of(relative)
            if text is None:
                findings.append(Finding(
                    "presence", claim_id, "file does not exist",
                    path=relative))
                continue
            if value not in text:
                findings.append(Finding(
                    "presence", claim_id,
                    "does not contain %r" % value, path=relative))
                continue

            if not near:
                findings.append(Finding(
                    "contradiction", claim_id,
                    "NOT CHECKED: no `near` phrases, so a different number of "
                    "the same shape elsewhere in this file would not be seen",
                    path=relative, fatal=False))
                continue

            spans = windows(text, near, width)
            if not spans:
                findings.append(Finding(
                    "contradiction", claim_id,
                    "NOT CHECKED: none of the `near` phrases (%s) appear here, "
                    "so there was no window to scan"
                    % ", ".join(repr(p) for p in near),
                    path=relative, fatal=False))
                continue

            allow = claim.get("allow") or {}
            wrong = []
            for start, end in spans:
                for match in shape.finditer(text, start, end):
                    found = match.group(0)
                    if found == value:
                        continue
                    if found in allow:
                        fired.add(found)
                        continue
                    wrong.append((found, excerpt(text, match.start(),
                                                 match.end())))
            for found, context in sorted(set(wrong)):
                findings.append(Finding(
                    "contradiction", claim_id,
                    "the ledger publishes %r and %r sits beside it: %s"
                    % (value, found, context), path=relative))

        for excused in sorted(set(claim.get("allow") or {}) - fired):
            findings.append(Finding(
                "allow", claim_id,
                "excuses %r and nothing in %s matches it any more; an "
                "exemption whose reason has gone stale gets copied forward"
                % (excused, " or ".join(paths)), fatal=False))
    return findings


def check_coverage(claims, root, patterns):
    """Files that quote a claim and are not pinned to it.

    This is the direction `project_rows.py` had to add for the same reason: a
    check that starts from what is written down cannot see what was never
    written down. A README listed in `appears_in` is guarded; the public page
    that quotes the same number and was never listed is not, and no amount of
    checking the README finds it.
    """
    root = pathlib.Path(root)
    findings = []
    candidates = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))

    for claim in claims:
        value = claim.get("value")
        if not value:
            continue
        claim_id = claim.get("id", "?")
        pinned = set(claim.get("appears_in") or [])
        for target in sorted(set(candidates)):
            if not target.is_file():
                continue
            relative = target.relative_to(root).as_posix()
            if relative in pinned:
                continue
            try:
                with io.open(str(target), encoding="utf-8",
                             errors="replace") as handle:
                    text = normalise(handle.read())
            except OSError:
                continue
            if value in text:
                findings.append(Finding(
                    "coverage", claim_id,
                    "quotes %r and is not in `appears_in`, so nothing checks "
                    "it" % value, path=relative))
    return findings


def verify(root, ledger_path=None, scan=None):
    """Every check, against one project directory. Returns a list of Findings."""
    root = pathlib.Path(root)
    claims = load(ledger_path or root / "claims.json")
    findings = check_schema(claims)
    findings.extend(check_rounding(claims))
    findings.extend(check_prose(claims, root))
    if scan:
        findings.extend(check_coverage(claims, root, scan))
    return claims, findings


def check_computed(claims, computed, default_places=12):
    """The one link a project has to close with its own code.

    `computed` maps a claim id to what the engine returns today. A claim with
    no `raw` is not checkable this way and is reported as such rather than
    passed over, because a silent skip here would leave the whole chain
    looking green while its first link was never tested — which is how the
    a trailing `# python:` comment came to be the only record of a real
    project's computed value.
    """
    findings = []
    by_id = {}
    for claim in claims:
        if claim.get("id"):
            by_id[claim["id"]] = claim

    for claim_id in sorted(computed):
        if claim_id not in by_id:
            findings.append(Finding(
                "computed", claim_id,
                "the engine produced this and the ledger has no such claim"))
    for claim_id, claim in sorted(by_id.items()):
        if "raw" not in claim:
            continue
        if claim_id not in computed:
            findings.append(Finding(
                "computed", claim_id,
                "has a `raw` value and nothing recomputed it", fatal=False))
            continue
        got = computed[claim_id]
        want = claim["raw"]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            tolerance = claim.get("tolerance")
            if tolerance is None:
                tolerance = 10.0 ** -default_places
            if abs(got - want) > tolerance:
                findings.append(Finding(
                    "computed", claim_id,
                    "engine returns %r; the ledger records %r (tolerance %g)"
                    % (got, want, tolerance)))
        elif got != want:
            findings.append(Finding(
                "computed", claim_id,
                "engine returns %r; the ledger records %r" % (got, want)))
    return findings


def stale(claims, asof=None, source=None):
    """Claims past their own re-check horizon. `source` labels the project."""
    asof = asof or datetime.date.today()
    out = []
    for claim in claims:
        horizon = claim.get("recheck_by")
        if not horizon:
            continue
        try:
            when = datetime.date.fromisoformat(horizon)
        except (TypeError, ValueError):
            continue
        if when < asof:
            out.append((source, claim, (asof - when).days))
    return out


def render(claims, fmt="md"):
    """The grade block a project puts next to its own numbers."""
    rows = []
    for claim in claims:
        rows.append((claim.get("value", ""), claim.get("claim", ""),
                     claim.get("status", ""), claim.get("anchor", ""),
                     claim.get("checked_on", "")))
    if fmt == "md":
        lines = ["| Value | Claim | Status | Anchor | Checked |",
                 "| --- | --- | --- | --- | --- |"]
        for row in rows:
            lines.append("| %s | %s | %s | %s | %s |"
                         % tuple(str(cell).replace("|", "\\|") for cell in row))
        return "\n".join(lines)
    if fmt == "html":
        def esc(cell):
            return (str(cell).replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))
        lines = ['<table class="claims">',
                 "<thead><tr><th>Value</th><th>Claim</th><th>Status</th>"
                 "<th>Anchor</th><th>Checked</th></tr></thead>", "<tbody>"]
        for row in rows:
            lines.append("<tr>%s</tr>" % "".join(
                '<td class="claim-%s">%s</td>' % (
                    "status" if index == 2 else "cell", esc(cell))
                for index, cell in enumerate(row)))
        lines.extend(["</tbody>", "</table>"])
        return "\n".join(lines)
    raise ValueError("unknown format %r" % fmt)


def _report(findings, quiet=False):
    fatal = [f for f in findings if f.fatal]
    notes = [f for f in findings if not f.fatal]
    for finding in fatal:
        print(finding.line())
    if not quiet:
        for finding in notes:
            print(finding.line())
    return fatal, notes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check a project's published numbers against its ledger.")
    sub = parser.add_subparsers(dest="command")

    verify_cmd = sub.add_parser("verify", help="every check, one project")
    verify_cmd.add_argument("project")
    verify_cmd.add_argument("--ledger", default=None)
    verify_cmd.add_argument("--scan", nargs="+", default=None,
                            help="globs to sweep for unpinned occurrences")
    verify_cmd.add_argument("--json", default=None)
    verify_cmd.add_argument("--quiet", action="store_true",
                            help="failures only; notes are suppressed")

    stale_cmd = sub.add_parser("stale", help="claims past their horizon")
    stale_cmd.add_argument("projects", nargs="+")
    stale_cmd.add_argument("--asof", default=None)
    stale_cmd.add_argument("--ledger", default="claims.json")

    render_cmd = sub.add_parser("render", help="the grade block")
    render_cmd.add_argument("project")
    render_cmd.add_argument("--ledger", default=None)
    render_cmd.add_argument("--format", default="md", choices=("md", "html"))

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2

    if args.command == "verify":
        root = pathlib.Path(args.project)
        try:
            claims, findings = verify(root, args.ledger, args.scan)
        except LedgerError as exc:
            print("FAIL  ledger         %s" % exc)
            return 1
        fatal, notes = _report(findings, args.quiet)
        if args.json:
            with io.open(args.json, "w", encoding="utf-8") as handle:
                json.dump({"project": str(root), "claims": len(claims),
                           "findings": [f.as_dict() for f in findings]},
                          handle, indent=2)
        if not args.scan:
            print("note  coverage       NOT CHECKED: pass --scan to sweep for "
                  "files that quote a claim and are not pinned to it")
        print("%d claims, %d failures, %d notes"
              % (len(claims), len(fatal), len(notes)))
        return 1 if fatal else 0

    if args.command == "stale":
        asof = (datetime.date.fromisoformat(args.asof) if args.asof
                else datetime.date.today())
        rows = []
        failed = False
        for project in args.projects:
            path = pathlib.Path(project)
            try:
                claims = load(path / args.ledger)
            except LedgerError as exc:
                print("FAIL  ledger         %s" % exc)
                failed = True
                continue
            rows.extend(stale(claims, asof, source=path.name))
        for source, claim, days in sorted(rows, key=lambda r: -r[2]):
            print("STALE %-12s %-24s %s days past %s  (%s)"
                  % (source, claim.get("id", "?"), days,
                     claim.get("recheck_by"), claim.get("claim", "")))
        print("%d claims past their horizon as of %s" % (len(rows), asof))
        return 1 if (rows or failed) else 0

    if args.command == "render":
        root = pathlib.Path(args.project)
        try:
            claims = load(args.ledger or root / "claims.json")
        except LedgerError as exc:
            print("FAIL  ledger         %s" % exc)
            return 1
        print(render(claims, args.format))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
