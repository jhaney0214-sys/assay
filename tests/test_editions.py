"""tools/editions.py against staged publisher answers, with the network shut.

Run:  python -m unittest tests.test_editions

Every answer comes from tests/fixtures/editions_answers.json. urllib's opener
is patched to fail for the whole module, so a test that reached the network
would error rather than pass.
"""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import editions  # noqa: E402

ANSWERS = json.loads((ROOT / "tests" / "fixtures" / "editions_answers.json")
                     .read_text(encoding="utf-8"))
NEW_HDI = "a new HDI report, EPI unchanged, SVI's missing file answered with an HTML page"


def no_network(*_a, **_k):
    raise AssertionError("a test reached for the network")


def setUpModule():
    global _patches
    _patches = [mock.patch("urllib.request.urlopen", no_network),
                mock.patch.object(urllib.request.OpenerDirector, "open", no_network)]
    for p in _patches:
        p.start()


def tearDownModule():
    for p in _patches:
        p.stop()


def staged(*scenarios):
    """An `ask` that answers from the fixture's scenarios, later ones winning; 404 otherwise."""
    table = {}
    for name in scenarios:
        table.update({k: v for k, v in ANSWERS[name].items()})

    def ask(url):
        status, headers = table.get(url, [404, {"Content-Type": "text/html"}])
        return status, headers
    return ask


def prober(ask):
    return editions.Prober(ask=ask, interval=0, sleep=lambda s: None)


def run(ask, cache_dir, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), mock.patch.object(sys.stdout, "reconfigure",
                                                            create=True):
        code = editions.main(list(argv), probe=prober(ask), cache_dir=cache_dir,
                             today="2026-10-15")
    return code, out.getvalue()


def write_cache(cache_dir, source, url, size):
    key = editions.hashlib.sha1(url.encode("utf-8")).hexdigest()
    d = pathlib.Path(cache_dir) / source
    d.mkdir(parents=True, exist_ok=True)
    (d / (key + ".meta.json")).write_text(json.dumps({"url": url, "bytes": size}),
                                          encoding="utf-8")


class WhatIsUsed(unittest.TestCase):

    def test_the_edition_is_read_from_the_fetch_scripts(self):
        got = {ix["key"]: ix["edition"] for ix in editions.indices()}
        self.assertEqual(got, {"hdi": "2023-24", "epi": "2024", "svi": "2022"})

    def test_the_fetch_and_panel_scripts_use_the_same_file(self):
        # A new edition set in one script and not the other would audit one
        # edition and measure rank change on another.
        for ix in editions.indices():
            self.assertEqual(editions.assess(ix, prober(staged()), "nowhere")["mismatch"], {},
                             ix["key"])

    def test_later_hdi_reports_are_tried_under_both_naming_patterns(self):
        urls = [u for _l, u in editions.hdi_candidates(editions.fetch_undp.URL, ahead=2)]
        self.assertEqual([u.split("/files/")[1].split("_Composite")[0] for u in urls],
                         ["2024_HDR/HDR24", "2024-25_HDR/HDR24-25",
                          "2025_HDR/HDR25", "2025-26_HDR/HDR25-26"])

    def test_later_epi_and_svi_editions_follow_the_used_address(self):
        self.assertEqual(editions.epi_candidates("https://x/downloads/epi2024results.csv", 2),
                         [("2025", "https://x/downloads/epi2025results.csv"),
                          ("2026", "https://x/downloads/epi2026results.csv")])
        self.assertEqual(editions.svi_candidates("https://x/Data/2022/csv/SVI_2022_US.csv", 1),
                         [("2023", "https://x/Data/2023/csv/SVI_2023_US.csv")])


class Verdicts(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = self.tmp.name
        write_cache(self.cache, "epi", editions.fetch_epi.RESULTS, 123308)

    def tearDown(self):
        self.tmp.cleanup()

    def verdicts(self, *scenarios):
        p = prober(staged(*scenarios))
        return {ix["key"]: editions.assess(ix, p, self.cache) for ix in editions.indices()}

    def test_a_new_report_is_found_and_an_html_page_is_not(self):
        v = self.verdicts(NEW_HDI)
        self.assertEqual(v["hdi"]["verdict"], editions.NEW)
        self.assertEqual([label for label, _u, _a in v["hdi"]["found"]], ["2025"])
        self.assertEqual(v["epi"]["verdict"], editions.CURRENT)
        self.assertEqual(v["svi"]["verdict"], editions.CURRENT)
        self.assertEqual(v["svi"]["found"], [])

    def test_a_file_that_changed_size_in_place_is_flagged(self):
        v = self.verdicts(NEW_HDI, "EPI's results file changed size at the same address")
        self.assertEqual(v["epi"]["verdict"], editions.REVISED)

    def test_a_used_file_that_is_gone_says_so(self):
        v = self.verdicts(NEW_HDI, "the SVI file Assay used is gone")
        self.assertEqual(v["svi"]["verdict"], editions.GONE)

    def test_an_edition_page_counts_as_a_new_edition(self):
        v = self.verdicts("EPI's 2024 file address answers with an HTML page and a 2026 "
                          "downloads page exists (as on 2026-10-15)")
        self.assertEqual(v["epi"]["verdict"], editions.NEW)
        self.assertEqual([label for label, _u, _a in v["epi"]["found"]], ["2026 page"])

    def test_an_html_page_at_the_used_address_is_the_file_gone(self):
        v = self.verdicts("the EPI file address answers with an HTML page and no later "
                          "edition is found")
        self.assertEqual(v["epi"]["verdict"], editions.GONE)

    def test_no_answer_is_not_called_current(self):
        def down(url):
            raise urllib.error.URLError("timed out")
        v = {ix["key"]: editions.assess(ix, prober(down), self.cache)
             for ix in editions.indices()}
        self.assertEqual({r["verdict"] for r in v.values()}, {editions.UNKNOWN})


class Asking(unittest.TestCase):

    def test_only_heads_are_sent_one_per_address(self):
        asked = []

        def ask(url):
            asked.append(url)
            return staged(NEW_HDI)(url)
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(ask, tmp)
        self.assertEqual(code, 0)
        expected = sum(1 + len(ix["candidates"]) + len(ix.get("edition_pages", []))
                       for ix in editions.indices())
        self.assertEqual(len(asked), expected)
        self.assertEqual(len(set(asked)), expected)
        self.assertIn("nothing was downloaded", out)
        self.assertIn("New editions: HDI.", out)
        self.assertIn("py tools/build_hdi_panel.py", out)
        self.assertIn("an HTML page, not a file", out)

    def test_only_narrows_the_check(self):
        asked = []

        def ask(url):
            asked.append(url)
            return 404, {}
        with tempfile.TemporaryDirectory() as tmp:
            run(ask, tmp, "--only", "epi")
        self.assertTrue(all("epi.yale.edu" in u for u in asked), asked)

    def test_requests_to_one_host_are_spaced(self):
        now = [0.0]
        slept = []

        def sleep(s):
            slept.append(round(s, 3))
            now[0] += s
        p = editions.Prober(ask=lambda u: (404, {}), interval=1.0, sleep=sleep,
                            clock=lambda: now[0])
        p("https://a.org/1")
        p("https://b.org/1")
        p("https://a.org/2")
        self.assertEqual(slept, [1.0])

    def test_the_real_head_sends_head_and_keeps_it_head_across_a_redirect(self):
        req = urllib.request.Request("https://a.org/x", method="HEAD")
        new = editions._KeepHead().redirect_request(req, None, 302, "Found", {},
                                                    "https://b.org/y")
        self.assertEqual(new.get_method(), "HEAD")

        seen = {}

        class Resp:
            status, headers = 200, {"Content-Type": "text/csv"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                raise AssertionError("a HEAD check must not read a body")

        def fake_open(self, request, timeout=None):
            seen["method"] = request.get_method()
            seen["ua"] = request.get_header("User-agent")
            return Resp()
        with mock.patch.object(urllib.request.OpenerDirector, "open", fake_open):
            status, headers = editions.head("https://a.org/x")
        self.assertEqual((status, seen["method"]), (200, "HEAD"))
        self.assertIn("Assay", seen["ua"])


if __name__ == "__main__":
    unittest.main()
