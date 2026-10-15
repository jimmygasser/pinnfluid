"""Optional site integration: a shared design kit, security headers, build information.

A deployment that embeds the app in a larger site can give it the site's top bar and
design tokens (PINN_WEBAPP_KIT_DIR), its security headers (PINN_WEBAPP_CSP,
PINN_WEBAPP_ROBOTS_TAG) and its build information (PINN_WEBAPP_INFO_DIR: version.json,
the third-party notices and the source archive). Without these settings the app looks
and answers exactly as before.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp", ROOT / "pinnfluid" / "domain_prep",
             ROOT / "pinnfluid" / "input_prep", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402
from pinnfluid_http import Browser, Server  # noqa: E402

SETTINGS = ("PINN_WEBAPP_KIT_DIR", "PINN_WEBAPP_KIT_HOME", "PINN_WEBAPP_CSP", "PINN_WEBAPP_ROBOTS_TAG",
            "PINN_WEBAPP_INFO_DIR")
CSP = "default-src 'self'; frame-ancestors 'self'"


class SiteTestCase(unittest.TestCase):
    def setUp(self):
        saved = {k: os.environ.pop(k) for k in SETTINGS if k in os.environ}
        self.addCleanup(os.environ.update, saved)  # runs last: cleanups run in reverse order
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in SETTINGS])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.kit = Path(self.tmp.name) / "kit"
        self.kit.mkdir()
        (self.kit / "tokens.css").write_text(":root{--bg:#fff}")
        (self.kit / "cryosim-bar.js").write_text("customElements.define('cryosim-bar', class extends HTMLElement {});")
        (self.kit / "VERSION").write_text("0.1.0\n")
        self.info = Path(self.tmp.name) / "info"
        (self.info / "source").mkdir(parents=True)
        (self.info / "version.json").write_text('{"service": "pinnfluid"}')
        (self.info / "NOTICES.txt").write_text("Third-party notices\n")
        (self.info / "source" / "pinnfluid-abc123.tar.gz").write_bytes(b"\x1f\x8b archive")
        (Path(self.tmp.name) / "secret.txt").write_text("outside")

    def serve(self) -> Browser:
        app.Handler.page_html = app._build_html()
        self.addCleanup(setattr, app.Handler, "page_html", app._build_html())
        server = Server(app)
        self.addCleanup(server.close)
        return Browser(server)


class DefaultTests(SiteTestCase):
    def test_without_settings_nothing_changes(self):
        html = app._build_html()
        for absent in ("cryosim-bar", "static/kit/", "data-theme"):
            self.assertNotIn(absent, html)
        browser = self.serve()
        for path in ("/static/kit/tokens.css", "/version.json", "/NOTICES.txt", "/source/"):
            self.assertEqual(browser.request("GET", path)[0], 404, path)
        _, headers, _ = browser.request("GET", "/")
        self.assertIsNone(headers.get("Content-Security-Policy"))
        self.assertIsNone(headers.get("X-Robots-Tag"))

    def test_a_kit_dir_without_the_files_changes_nothing(self):
        os.environ["PINN_WEBAPP_KIT_DIR"] = str(Path(self.tmp.name) / "missing")
        self.assertNotIn("cryosim-bar", app._build_html())


class KitTests(SiteTestCase):
    def test_the_page_gets_the_bar_and_the_tokens(self):
        os.environ["PINN_WEBAPP_KIT_DIR"] = str(self.kit)
        html = app._build_html()
        self.assertIn('<link rel="stylesheet" href="static/kit/tokens.css">', html)
        self.assertIn('<script type="module" src="static/kit/cryosim-bar.js"></script>', html)
        self.assertRegex(html, r'<cryosim-bar tool="pinnfluid" home="/" skip-target="#sidebar" role="banner">')
        self.assertIn('<a href="./" aria-current="page">Predict</a>', html)
        self.assertIn('<html data-theme="light">', html)
        # the bar comes first in the body, so the skip link is the first focusable element
        self.assertRegex(html, r"<body>\s*<cryosim-bar")
        # the home link and the nav may name the site; every other URL stays relative
        no_home = re.sub(r'home="[^"]*"', "", html)
        self.assertNotRegex(no_home, r"""\b(?:href|src)\s*=\s*["']/(?!/)""")

    def test_the_home_link_is_a_setting(self):
        os.environ.update(PINN_WEBAPP_KIT_DIR=str(self.kit), PINN_WEBAPP_KIT_HOME="/portal/")
        self.assertIn('home="/portal/"', app._build_html())
        os.environ["PINN_WEBAPP_KIT_HOME"] = '"><script>'
        self.assertNotIn("<script>\"", app._build_html())
        self.assertIn('home="&quot;&gt;&lt;script&gt;"', app._build_html())

    def test_the_kit_files_are_served(self):
        os.environ["PINN_WEBAPP_KIT_DIR"] = str(self.kit)
        browser = self.serve()
        for path, ctype in (("tokens.css", "text/css"), ("cryosim-bar.js", "text/javascript"),
                            ("VERSION", "text/plain")):
            status, headers, body = browser.request("GET", f"/static/kit/{path}")
            self.assertEqual(status, 200, path)
            self.assertTrue(headers["Content-Type"].startswith(ctype), path)
            self.assertEqual(body, (self.kit / path).read_bytes())
        for path in ("../secret.txt", "other.js", "%2e%2e/secret.txt", ""):
            self.assertEqual(browser.request("GET", f"/static/kit/{path}")[0], 404, path)


class HeaderTests(SiteTestCase):
    def test_csp_and_robots_tag_on_every_response(self):
        os.environ.update(PINN_WEBAPP_CSP=CSP, PINN_WEBAPP_ROBOTS_TAG="noindex, nofollow")
        browser = self.serve()
        for path in ("/", "/status", "/static/plotly.min.js", "/nothing-here"):
            _, headers, _ = browser.request("GET", path)
            self.assertEqual(headers.get("Content-Security-Policy"), CSP, path)
            self.assertEqual(headers.get("X-Robots-Tag"), "noindex, nofollow", path)

    def test_header_values_cannot_inject_headers(self):
        os.environ["PINN_WEBAPP_CSP"] = "default-src 'self'\r\nX-Injected: yes"
        browser = self.serve()
        _, headers, _ = browser.request("GET", "/status")
        self.assertIsNone(headers.get("X-Injected"))


class InfoTests(SiteTestCase):
    def test_build_information_is_served(self):
        os.environ["PINN_WEBAPP_INFO_DIR"] = str(self.info)
        browser = self.serve()
        status, headers, body = browser.request("GET", "/version.json")
        self.assertEqual((status, body), (200, b'{"service": "pinnfluid"}'))
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        status, headers, body = browser.request("GET", "/NOTICES.txt")
        self.assertEqual((status, body), (200, b"Third-party notices\n"))
        status, _, body = browser.request("GET", "/source/")
        self.assertEqual(status, 200)
        page = body.decode()
        self.assertIn('href="pinnfluid-abc123.tar.gz"', page)
        self.assertIn('href="../NOTICES.txt"', page)
        self.assertNotRegex(page, r"""\bhref\s*=\s*["']/(?!/)""")
        status, headers, body = browser.request("GET", "/source/pinnfluid-abc123.tar.gz")
        self.assertEqual((status, body), (200, b"\x1f\x8b archive"))
        self.assertEqual(headers["Content-Type"], "application/gzip")
        for path in ("/source/../secret.txt", "/source/secret.txt", "/source/x.tar.gz", "/source"):
            self.assertIn(browser.request("GET", path)[0], (404, 308), path)

    def test_the_nav_links_the_source_when_there_is_build_information(self):
        os.environ.update(PINN_WEBAPP_KIT_DIR=str(self.kit), PINN_WEBAPP_INFO_DIR=str(self.info))
        self.assertIn('<a href="source/">Source &amp; notices</a>', app._build_html())


if __name__ == "__main__":
    unittest.main()
