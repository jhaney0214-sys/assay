"""tools/fetch_svi.py's reshaping, without the network.

Run:  python -m unittest tests.test_fetch_svi
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_svi  # noqa: E402
from assay import load  # noqa: E402

HEAD = "ST,COUNTY,LOCATION,RPL_THEME1,RPL_THEME2,RPL_THEME3,RPL_THEME4,RPL_THEMES\n"


class Reshape(unittest.TestCase):

    def test_a_county_is_named_by_its_location_since_county_names_repeat(self):
        text = HEAD + ('1,Washington County,"Washington County, Alabama",0.5,0.4,0.3,0.2,0.35\n'
                       '5,Washington County,"Washington County, Arkansas",0.1,0.2,0.3,0.4,0.25\n')
        rows, dropped = fetch_svi.rows_from(text)
        self.assertEqual(dropped, 0)
        self.assertEqual([r[0] for r in rows],
                         ["Washington County, Alabama", "Washington County, Arkansas"])
        self.assertEqual(rows[0][1:], [0.5, 0.4, 0.3, 0.2, 0.35])

    def test_a_county_cdc_could_not_rank_is_kept_blank_and_counted(self):
        # Kept so the audit can drop it and say who it dropped.
        text = HEAD + ('1,A,"A, Alabama",0.5,0.4,0.3,0.2,0.35\n'
                       '2,B,"B, Alaska",-999,0.2,0.3,0.4,-999\n')
        rows, incomplete = fetch_svi.rows_from(text)
        self.assertEqual((len(rows), incomplete), (2, 1))
        self.assertEqual(rows[1], ["B, Alaska", "", 0.2, 0.3, 0.4, ""])


class TheBundledFile(unittest.TestCase):

    def test_the_spec_loads_with_every_county_and_four_themes(self):
        index = load.from_spec(str(ROOT / "data" / "svi-2022.json"))
        self.assertEqual(index.n, 3144)
        self.assertEqual(len(fetch_svi.THEMES), 4)


if __name__ == "__main__":
    unittest.main()
