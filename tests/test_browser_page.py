"""Does the page actually do what the engine does?

`tools/conform_browser.py` proves `docs/audit.js` agrees with the Python
engine. That is necessary and not sufficient: the page can hold a correct
engine and still mis-wire it - read the wrong column, skip a step, or fail to
suppress a result it is supposed to suppress. These tests drive the rendered
page and read the numbers a visitor would see.

    python tests/test_browser_page.py

Requires playwright and a chromium; both are skipped cleanly when absent, so
this does not break a checkout that only wants the Python suite.
"""

import contextlib
import functools
import http.server
import pathlib
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

CHROMIUM_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/opt/pw-browsers/chromium/chrome-linux/chrome",
]


@functools.cache
def _chromium():
    for path in CHROMIUM_CANDIDATES:
        if pathlib.Path(path).exists():
            return path
    # Otherwise the one `playwright install chromium` put in its own cache,
    # which is where it lives on any machine that is not the container.
    if HAVE_PLAYWRIGHT:
        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        if path and pathlib.Path(path).exists():
            return path
    return None


try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False


@contextlib.contextmanager
def serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/audit.html" % server.server_address[1]
    finally:
        server.shutdown()


@unittest.skipUnless(HAVE_PLAYWRIGHT and _chromium(),
                     "playwright or chromium not available")
class TestAuditPage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._ctx = serve(DOCS)
        cls.url = cls._ctx.__enter__()
        cls._pw = sync_playwright().start()
        cls.browser = cls._pw.chromium.launch(executable_path=_chromium())

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls._pw.stop()
        cls._ctx.__exit__(None, None, None)

    def _run(self, example, mark_published=False):
        page = self.browser.new_page()
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.goto(self.url, wait_until="networkidle")
        page.click("[data-example='%s']" % example)
        page.wait_for_selector("#mapping:not(.hidden)", timeout=20000)
        shared = page.is_checked("#shared")
        if mark_published:
            page.query_selector_all("#maptable select")[-1].select_option("published")
        page.click("#run")
        page.wait_for_selector("#results:not(.hidden)", timeout=40000)
        text = page.inner_text("#results")
        page.close()
        return text, shared

    def test_hdi_matches_the_python_engine(self):
        """n, k and the PC1 share as the Python engine reports them."""
        text, _ = self._run("hdi", mark_published=True)
        self.assertIn("193 complete rows", text)
        self.assertIn("4 sub-scores", text)
        self.assertIn("84.1%", text)          # python: 0.8409571
        self.assertEqual(self.errors, [])

    def test_hdi_weights_barely_beat_one_component(self):
        """r(published, PC1) = 0.9976 - the finding the README promises."""
        text, _ = self._run("hdi", mark_published=True)
        self.assertIn("0.9976", text)
        self.assertIn("did not change the ordering", text)

    def test_map_is_withheld_below_five_columns(self):
        """With HDI's total marked as published, four pillars remain.

        MAP is unreliable below five columns and is withheld rather than
        reported with a caveat nobody reads.
        """
        text, _ = self._run("hdi", mark_published=True)
        self.assertIn("4 sub-scores", text)
        self.assertIn("MAP is not reported", text)

    def test_an_aggregate_left_among_the_pillars_is_flagged(self):
        """The bug this page nearly shipped with.

        Left at the defaults, HDI's own published total is analysed as a
        fifth pillar - and the report then says GNI per capita and HDI are
        "the same pillar" at r = 0.963, which is an index correlating with
        its own input. The page cannot know which column is the total, so it
        argues from the data: a column correlating above 0.95 with the mean
        of the others is either the aggregate or shares construction, and
        both corrections are offered.
        """
        text, _ = self._run("hdi")           # deliberately not marked
        self.assertIn("correlates 0.983", text)
        self.assertIn("published total", text)
        self.assertIn("shared-construction box", text)

    def test_vdem_declares_shared_construction_by_default(self):
        """Four of V-Dem's five indices contain the fifth by construction."""
        _, shared = self._run("vdem")
        self.assertTrue(shared)

    def test_vdem_refuses_to_call_the_collapse_a_finding(self):
        """The whole point of the shared-construction flag.

        97.1% on one component and ten pairs above 0.90 is the most dramatic
        output this tool produces, and it is not a result at all - it is four
        columns containing the fifth. The page must show the numbers and
        refuse the inference, not hide either.
        """
        text, _ = self._run("vdem")
        self.assertIn("97.1%", text)
        self.assertIn("10 pairs", text)
        self.assertIn("not a finding", text)
        self.assertNotIn("not separable in this data", text)

    def test_epi_reports_eleven_columns_and_no_redundancy(self):
        """EPI's eleven policy objectives, with its own total marked off.

        Note what this test also documents: the aggregate detector does NOT
        fire on EPI. Its published score correlates below the 0.95 threshold
        with the mean of its pillars, because EPI is genuinely less
        one-dimensional than HDI. The detector catches a strongly aggregated
        index and misses a weakly aggregated one, which is why the page asks
        the visitor to confirm the mapping rather than relying on detection.
        """
        text, _ = self._run("epi", mark_published=True)
        self.assertIn("105 complete rows", text)
        self.assertIn("11 sub-scores", text)
        self.assertIn("45.1%", text)          # python: 0.4507225
        self.assertIn("No pair correlates at 0.90 or above", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
