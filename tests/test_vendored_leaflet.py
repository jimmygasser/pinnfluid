"""Leaflet and leaflet-draw are served by the app (static/vendor/), not from a CDN.

The page loads them by relative URLs from webapp/static/vendor/ (filled at image
build from pinned, hash-checked downloads; see static/vendor/README.md), so it
works offline, under a strict content-security policy and under a base path.
"""

from __future__ import annotations

import http.client
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "pinnfluid",
    ROOT / "pinnfluid" / "webapp",
    ROOT / "pinnfluid" / "domain_prep",
    ROOT / "pinnfluid" / "input_prep",
):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402

_OFF_ORIGIN_ASSET = re.compile(
    r"""<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*["']?(?:https?:)?//""", re.I)

VENDORED = (
    "static/vendor/leaflet/leaflet.css",
    "static/vendor/leaflet/leaflet.js",
    "static/vendor/leaflet-draw/leaflet.draw.css",
    "static/vendor/leaflet-draw/leaflet.draw.js",
)


class PageTests(unittest.TestCase):
    def test_page_loads_leaflet_from_the_app(self):
        html = app._build_html()
        self.assertNotIn("unpkg.com", html)
        self.assertEqual(_OFF_ORIGIN_ASSET.findall(html), [])
        for asset in VENDORED:
            tag = "link" if asset.endswith(".css") else "script"
            attr = "href" if tag == "link" else "src"
            self.assertRegex(html, rf'<{tag}[^>]*\b{attr}="{re.escape(asset)}"', asset)

    def test_detector_finds_off_origin_assets(self):
        for text in ('<script src="https://unpkg.com/x.js">', "<link rel=stylesheet href='//cdn/x.css'>",
                     '<img src="http://example.org/a.png">'):
            self.assertTrue(_OFF_ORIGIN_ASSET.findall(text), text)
        self.assertFalse(_OFF_ORIGIN_ASSET.findall('<a href="https://github.com/x">'))

    def test_a_changed_template_is_not_served_with_cdn_links(self):
        with patch.object(app, "BUILDER_HTML", app.BUILDER_HTML.replace("leaflet@1.9.4", "leaflet@9.9.9")):
            with self.assertRaisesRegex(RuntimeError, "Leaflet"):
                app._build_html()

    def test_the_cdn_is_used_only_when_asked_for(self):
        """PINN_WEBAPP_LEAFLET_CDN=1 keeps the template's unpkg tags (a checkout
        without the vendored files); unset, 0 or anything else means vendored."""
        saved = os.environ.pop("PINN_WEBAPP_LEAFLET_CDN", None)
        try:
            for value in (None, "", "0", "no", "false"):
                if value is not None:
                    os.environ["PINN_WEBAPP_LEAFLET_CDN"] = value
                self.assertNotIn("unpkg.com", app._build_html(), value)
            for value in ("1", "true", "yes"):
                os.environ["PINN_WEBAPP_LEAFLET_CDN"] = value
                html = app._build_html()
                for asset in VENDORED:
                    self.assertNotIn(asset, html, value)
                self.assertEqual(html.count('="https://unpkg.com/leaflet'), 4, value)
        finally:
            os.environ.pop("PINN_WEBAPP_LEAFLET_CDN", None)
            if saved is not None:
                os.environ["PINN_WEBAPP_LEAFLET_CDN"] = saved


class VendorRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        vendor = Path(cls.tmp.name) / "vendor"
        (vendor / "leaflet" / "images").mkdir(parents=True)
        (vendor / "leaflet" / "leaflet.js").write_text("/* leaflet */")
        (vendor / "leaflet" / "leaflet.css").write_text(".leaflet{}")
        (vendor / "leaflet" / "images" / "layers.png").write_bytes(b"\x89PNG\r\n")
        (vendor / "leaflet" / "LICENSE").write_text("BSD 2-Clause License")
        (vendor / "leaflet-draw" / "images").mkdir(parents=True)
        (vendor / "leaflet-draw" / "images" / "spritesheet.svg").write_text("<svg/>")
        (vendor / "leaflet" / "notes.txt").write_text("not an asset type")
        (vendor / "leaflet" / "Makefile").write_text("extensionless, not a licence")
        (Path(cls.tmp.name) / "secret.js").write_text("outside the vendor dir")
        cls.patcher = patch.object(app, "VENDOR_DIR", vendor)
        cls.patcher.start()

        class PrefixedHandler(app.Handler):
            base_path = "/pinnfluid"

        cls.servers = []
        for handler in (app.Handler, PrefixedHandler):
            server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            cls.servers.append(server)

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.shutdown()
            server.server_close()
        cls.patcher.stop()
        cls.tmp.cleanup()

    def get(self, path: str, *, prefixed: bool = False):
        host, port = self.servers[1 if prefixed else 0].server_address
        conn = http.client.HTTPConnection(host, port, timeout=10)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            conn.close()

    def test_assets_are_served_with_their_type(self):
        cases = {
            "/static/vendor/leaflet/leaflet.js": ("text/javascript", b"/* leaflet */"),
            "/static/vendor/leaflet/leaflet.css": ("text/css", b".leaflet{}"),
            "/static/vendor/leaflet/images/layers.png": ("image/png", b"\x89PNG\r\n"),
            "/static/vendor/leaflet-draw/images/spritesheet.svg": ("image/svg+xml", b"<svg/>"),
            "/static/vendor/leaflet/LICENSE": ("text/plain", b"BSD 2-Clause License"),
        }
        for path, (ctype, body) in cases.items():
            status, headers, data = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertTrue(headers["Content-Type"].startswith(ctype), (path, headers["Content-Type"]))
            self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
            self.assertEqual(data, body, path)

    def test_assets_under_the_base_path(self):
        status, _, data = self.get("/pinnfluid/static/vendor/leaflet/leaflet.js", prefixed=True)
        self.assertEqual((status, data), (200, b"/* leaflet */"))
        status, _, _ = self.get("/static/vendor/leaflet/leaflet.js", prefixed=True)
        self.assertEqual(status, 404)

    def test_nothing_else_is_reachable(self):
        for path in ("/static/vendor/../secret.js", "/static/vendor/leaflet/../../secret.js",
                     "/static/vendor/%2e%2e/secret.js", "/static/vendor//etc/passwd",
                     "/static/vendor/leaflet", "/static/vendor/leaflet/", "/static/vendor/",
                     "/static/vendor/leaflet/notes.txt", "/static/vendor/leaflet/Makefile",
                     "/static/vendor/leaflet/missing.js",
                     "/static/vendor/.hidden.js", "/static/app.py"):
            status, _, _ = self.get(path)
            self.assertEqual(status, 404, path)


if __name__ == "__main__":
    unittest.main()
