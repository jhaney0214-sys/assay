"""tools/table.py: CSV and .xlsx read the same, and an HTML page is refused.

Run:  python -m unittest tests.test_table
"""

import io
import pathlib
import sys
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import table  # noqa: E402

M = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def workbook(sheets):
    """A minimal .xlsx: {name: sheet xml body}, shared strings 'country' and 'EPI.new'."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", '<workbook %s %s><sheets>%s</sheets></workbook>' % (M, R, "".join(
            '<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (n, i + 1, i + 1) for i, n in enumerate(sheets))))
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">%s'
                   '</Relationships>' % "".join('<Relationship Id="rId%d" Target="worksheets/sheet%d.xml"/>'
                                                % (i + 1, i + 1) for i in range(len(sheets))))
        z.writestr("xl/sharedStrings.xml", '<sst %s><si><t>country</t></si><si><t>EPI.new</t></si>'
                                           '<si><r><t>Den</t></r><r><t>mark</t></r></si></sst>' % M)
        for i, body in enumerate(sheets.values()):
            z.writestr("xl/worksheets/sheet%d.xml" % (i + 1), '<worksheet %s><sheetData>%s</sheetData></worksheet>'
                       % (M, body))
    return buf.getvalue()


SHEET = ('<row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="s"><v>1</v></c></row>'
         '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>75.3</v></c></row>'
         '<row r="3"><c r="A3" t="inlineStr"><is><t>Chile</t></is></c></row>')


class Xlsx(unittest.TestCase):

    def test_reads_like_the_csv_would(self):
        rows = table.rows(workbook({"Results": SHEET}))
        self.assertEqual(rows, [{"country": "Denmark", "EPI.new": "75.3"}, {"country": "Chile", "EPI.new": ""}])
        self.assertEqual(rows, table.rows(b"country,EPI.new\nDenmark,75.3\nChile,\n"))

    def test_a_named_sheet_and_a_missing_one(self):
        raw = workbook({"Notes": '<row r="1"><c r="A1" t="inlineStr"><is><t>x</t></is></c></row>', "Results": SHEET})
        self.assertEqual(table.rows(raw, sheet="Results")[0]["country"], "Denmark")
        self.assertEqual(table.rows(raw), [])
        with self.assertRaises(table.TableError):
            table.rows(raw, sheet="Weights")


class Refusals(unittest.TestCase):

    def test_an_html_page_in_place_of_the_file_is_refused(self):
        with self.assertRaises(table.TableError):
            table.rows(b"\n<!DOCTYPE html><html><head><title>EPI</title></head></html>")

    def test_a_bom_csv_still_reads(self):
        self.assertEqual(table.rows("﻿code,v\nDNK,1\n".encode("utf-8")), [{"code": "DNK", "v": "1"}])


if __name__ == "__main__":
    unittest.main()
