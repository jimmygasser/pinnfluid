"""Domain names: an allow-list at every entry point, safe download headers and log lines.

A domain name becomes a directory name, a download file name (Content-Disposition)
and a log field. Names are letters, digits, '-', '_' and '.' (at most 64, starting
with a letter or digit); anything else is refused with 400 before it is used.
Download headers follow RFC 6266: an ASCII filename= plus filename*=UTF-8''.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp", ROOT / "pinnfluid" / "domain_prep",
             ROOT / "pinnfluid" / "input_prep", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402
import build_inputs  # noqa: E402
from pinnfluid_http import Browser, Server  # noqa: E402

INJECTION = 'p"\r\nX-Injected: yes'
BAD_NAMES = (INJECTION, "my site", 'a"b', "</title><script>alert(1)</script>", "tab\there",
             "nul\x00", "café", "-lead", "_lead", ".lead", "a" * 65, "a/b", "a\\b", "..", "a..b", "")
GOOD_NAMES = ("site", "Site-2", "site_v1.2", "a" * 64, "0lead", "A.b-c_d")


class NameRuleTests(unittest.TestCase):
    NS = "0123456789abcdef"

    def test_allowed_names(self):
        for name in GOOD_NAMES:
            self.assertEqual(app._session_name(self.NS, name), f"{self.NS}_{name}", name)

    def test_refused_names(self):
        for name in BAD_NAMES:
            with self.assertRaises(ValueError, msg=repr(name)):
                app._session_name(self.NS, name)

    def test_on_disk_names_follow_the_same_rule(self):
        app._safe_name(f"{self.NS}_site_r090")
        for name in (f"{self.NS}_{INJECTION}", f"{self.NS}_my site", "x" * 129):
            with self.assertRaises(ValueError, msg=repr(name)):
                app._safe_name(name)


class ContentDispositionTests(unittest.TestCase):
    def test_rfc6266_header(self):
        self.assertEqual(app._attachment("site_report.pdf"),
                         "attachment; filename=\"site_report.pdf\"; filename*=UTF-8''site_report.pdf")

    def test_anything_unusual_is_replaced_in_the_ascii_name_and_encoded_in_the_utf8_one(self):
        header = app._attachment('café "x"\r\n.pdf')
        self.assertNotIn("\r", header)
        self.assertNotIn("\n", header)
        ascii_part = header.split("filename=", 1)[1].split(";", 1)[0]
        self.assertEqual(ascii_part, '"caf___x___.pdf"')
        self.assertTrue(header.endswith("filename*=UTF-8''" + quote('café "x"\r\n.pdf', safe="")))


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.Handler.page_html = app._build_html()
        cls.server = Server(app)

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        work = Path(self.tmp.name)
        self.results = work / "results"
        self.results.mkdir()
        for target, attr, value in ((app, "RESULTS_DIR", self.results), (app, "DEM_ROOT", work / "dem"),
                                    (app.Handler, "dem_root", str(work / "dem")),
                                    (build_inputs, "DEM_ROOT", work / "dem"),
                                    (build_inputs, "WORKSPACE", work / "workspace")):
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        limits = patch.dict(os.environ, {"PINN_WEBAPP_MAX_ACTIVE_JOBS": "0", "PINN_WEBAPP_RATE_LIMIT_JOBS": "0",
                                         "PINN_WEBAPP_RATE_LIMIT_PREP": "0"})
        limits.start()
        self.addCleanup(limits.stop)

    def test_every_entry_point_refuses_a_bad_name(self):
        body = {"domain_size": 1000, "wind_from": 270}
        for name in (INJECTION, "my site", "</title><script>"):
            for route, payload in (("upload_dem", {"data_base64": "A" * 2000}),
                                   ("download_dem", {"west": 7.0, "south": 46.0, "east": 7.01, "north": 46.01}),
                                   ("build_inputs", body), ("predict", body), ("predict_rose", body)):
                status, data = Browser(self.server).json("POST", f"/{route}", dict(payload, domain_name=name))
                self.assertEqual(status, 400, (route, name, data))
                self.assertIn("letters, digits", data["error"])
            for route in ("download_pdf", "download_zip", "download_vtk", "map", "view_3d_rose"):
                status, _, _ = Browser(self.server).request("GET", f"/{route}?domain={quote(name)}")
                self.assertEqual(status, 400, (route, name))

    def test_no_header_can_be_injected_through_a_download(self):
        browser = Browser(self.server)
        browser.request("GET", "/")
        ns = app._cookie_namespace(browser.cookie.split("=", 1)[1])
        run = self.results / f"{ns}_{INJECTION}"  # as the old code let it be created
        run.mkdir()
        (run / "report.pdf").write_bytes(b"%PDF")
        status, headers, _ = browser.request("GET", f"/download_pdf?domain={quote(INJECTION)}")
        self.assertEqual(status, 400)
        self.assertIsNone(headers.get("X-Injected"))
        good = self.results / f"{ns}_site"
        good.mkdir()
        (good / "report.pdf").write_bytes(b"%PDF")
        status, headers, body = browser.request("GET", "/download_pdf?domain=site")
        self.assertEqual((status, body), (200, b"%PDF"))
        self.assertEqual(headers["Content-Disposition"],
                         "attachment; filename=\"site_report.pdf\"; filename*=UTF-8''site_report.pdf")

    def test_names_are_logged_escaped(self):
        out = io.StringIO()
        with redirect_stdout(out), patch.object(app, "query_stac_tiles", return_value=[]):
            Browser(self.server).json("POST", "/download_dem", {
                "domain_name": "site", "west": 7.0, "south": 46.0, "east": 7.01, "north": 46.01})
        line = next(l for l in out.getvalue().splitlines() if "[DEM]" in l)
        self.assertRegex(line, r"\[DEM\] '[0-9a-f]{16}_site':")


if __name__ == "__main__":
    unittest.main()
