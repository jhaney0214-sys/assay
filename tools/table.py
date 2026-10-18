"""A downloaded table as a list of dicts, whether it came as CSV or .xlsx.

Yale's EPI published CSV through 2024 and .xlsx for 2026 (tools/editions.py,
checked 2026-10-18), so the fetch scripts read the bytes through here and do
not care which arrived. The .xlsx reader is standard library only (zipfile and
ElementTree), reads one sheet, and returns every cell as text, the way
csv.DictReader does, so the scripts' own to_float() parsing is unchanged.

An HTML page is refused rather than parsed: the 2024 EPI addresses now answer
HTTP 200 with the site's page instead of the file, and a cache refresh that
stored that page would otherwise read as a CSV with no score columns.
"""

import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "p": "http://schemas.openxmlformats.org/package/2006/relationships"}


class TableError(Exception):
    pass


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise TableError("could not decode the file as text")


def rows(raw, sheet=None):
    """Header-keyed rows from CSV or .xlsx bytes. `sheet` names an .xlsx sheet (default: the first)."""
    if raw[:4] == b"PK\x03\x04":
        return xlsx_rows(raw, sheet)
    text = decode(raw)
    if re.match(r"\s*<(!doctype html|html)", text[:200], re.I):
        raise TableError("an HTML page, not a data file: the address no longer serves the file")
    return list(csv.DictReader(io.StringIO(text)))


def _col(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def xlsx_rows(raw, sheet=None):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        book = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = [(s.get("name"), s.get("{%s}id" % NS["r"])) for s in book.find("m:sheets", NS)]
        if not sheets:
            raise TableError("the workbook has no sheets")
        names = [n or "" for n, _ in sheets]
        if sheet is not None and sheet not in names:
            raise TableError("no sheet %r; the workbook has %s" % (sheet, ", ".join(names)))
        rid = dict(sheets)[sheet] if sheet is not None else sheets[0][1]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") or "" for r in rels.findall("p:Relationship", NS) if r.get("Id") == rid)
        path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
        grid = []
        for row in ET.fromstring(z.read(path)).iter("{%s}row" % NS["m"]):
            cells = {}
            for c in row.findall("m:c", NS):
                kind, v = c.get("t"), c.find("m:v", NS)
                if kind == "inlineStr":
                    text = "".join(t.text or "" for t in c.iter("{%s}t" % NS["m"]))
                elif v is None:
                    text = ""
                elif kind == "s":
                    text = shared[int(v.text or 0)]
                else:
                    text = v.text or ""
                cells[_col(c.get("r") or "A")] = text
            if cells:
                grid.append([cells.get(i, "") for i in range(max(cells) + 1)])
    if not grid:
        return []
    head = [h.strip() for h in grid[0]]
    return [{h: (r[i] if i < len(r) else "") for i, h in enumerate(head) if h} for r in grid[1:]]
