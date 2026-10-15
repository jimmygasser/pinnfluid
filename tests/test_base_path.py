"""The web app under a URL prefix (PINN_WEBAPP_BASE_PATH) with relative URLs only.

Behind a gateway that routes /pinnfluid/... to the app, the app strips the
prefix itself, answers 404 outside it, and every URL it renders is relative,
so the same HTML works at / and under any prefix.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import tempfile
import threading
import time
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
from webapp import view_3d  # noqa: E402

# A URL that starts with "/" (and not "//") in an attribute, a CSS url() or a
# quoted JavaScript/HTML string: it would leave the prefix the app is served under.
_ROOT_ABSOLUTE = [
    re.compile(r"""\b(?:href|src|action|formaction|poster)\s*=\s*["']?/(?!/)"""),
    re.compile(r"""url\(\s*["']?/(?!/)"""),
    re.compile(r"""["'`]/(?![/*\s])[A-Za-z0-9_.~-]"""),
]


def root_absolute_urls(text: str) -> list[str]:
    hits = []
    for pattern in _ROOT_ABSOLUTE:
        for m in pattern.finditer(text):
            hits.append(text[max(0, m.start() - 40):m.end() + 40])
    return hits


VALID_BODY = {
    "domain_name": "base-path-case",
    "domain_size": 1000,
    "wind_from": 270,
    "uref": 10,
    "zref": 20,
    "z0": 0.1,
    "structures": [{"yaw": 0}],
    "sampling_points": [],
    "grid": None,
}


class BasePathSettingTests(unittest.TestCase):
    def test_root_values_mean_no_prefix(self):
        for raw in (None, "", "/", "  "):
            self.assertEqual(app._normalise_base_path(raw), "")

    def test_prefix_is_normalised_without_trailing_slash(self):
        self.assertEqual(app._normalise_base_path("/pinnfluid"), "/pinnfluid")
        self.assertEqual(app._normalise_base_path("/pinnfluid/"), "/pinnfluid")
        self.assertEqual(app._normalise_base_path("/tools/pinnfluid"), "/tools/pinnfluid")

    def test_malformed_prefixes_are_refused(self):
        for raw in ("pinnfluid", "//pinnfluid", "/a//b", "/a/../b", "/./a", "/pin fluid",
                    "/a?b", "/a#b", "/a%2fb", "/a\\b"):
            with self.assertRaises(ValueError, msg=raw):
                app._normalise_base_path(raw)

    def test_read_from_the_environment(self):
        old = os.environ.get("PINN_WEBAPP_BASE_PATH")
        try:
            os.environ["PINN_WEBAPP_BASE_PATH"] = "/pinnfluid/"
            self.assertEqual(app._base_path_from_env(), "/pinnfluid")
            os.environ.pop("PINN_WEBAPP_BASE_PATH")
            self.assertEqual(app._base_path_from_env(), "")
        finally:
            if old is None:
                os.environ.pop("PINN_WEBAPP_BASE_PATH", None)
            else:
                os.environ["PINN_WEBAPP_BASE_PATH"] = old


class BasePathHttpTests(unittest.TestCase):
    """The HTTP boundary under /pinnfluid, as the gateway forwards it (prefix kept)."""

    @classmethod
    def setUpClass(cls):
        app.Handler.page_html = app._build_html()

        class PrefixedHandler(app.Handler):
            base_path = "/pinnfluid"

        cls.server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), PrefixedHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        with app._JOBS_LOCK:
            app._JOBS.clear()
            app._JOB_SUBMISSIONS.clear()

    def request(self, method: str, path: str, body: bytes | None = None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_page_under_the_prefix(self):
        for path in ("/pinnfluid/", "/pinnfluid/index.html"):
            status, _, body = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertIn("pinnfluid - Wind and pressure prediction", body.decode("utf-8"))

    def test_prefix_without_slash_redirects_keeping_the_query(self):
        status, headers, _ = self.request("GET", "/pinnfluid")
        self.assertEqual(status, 308)
        self.assertEqual(headers.get("Location"), "/pinnfluid/")
        status, headers, _ = self.request("GET", "/pinnfluid?x=1")
        self.assertEqual(status, 308)
        self.assertEqual(headers.get("Location"), "/pinnfluid/?x=1")

    def test_everything_outside_the_prefix_is_404(self):
        for path in ("/", "/index.html", "/status", "/static/plotly.min.js",
                     "/download_zip?domain=x", "/pinnfluidx/", "/pinnfluid-other/status"):
            status, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
        status, _, _ = self.request("POST", "/predict", json.dumps(VALID_BODY).encode())
        self.assertEqual(status, 404)

    def test_routes_below_the_prefix(self):
        status, _, body = self.request("GET", "/pinnfluid/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")
        status, headers, body = self.request("GET", "/pinnfluid/static/plotly.min.js")
        self.assertEqual(status, 200)
        self.assertTrue(headers.get("Content-Type", "").startswith("text/javascript"))
        self.assertGreater(len(body), 100_000)
        status, _, body = self.request("POST", "/pinnfluid/upload_dem", b"not-json")
        self.assertEqual(status, 400)
        self.assertIn(b"must be valid JSON", body)

    def test_predict_and_job_status_under_the_prefix(self):
        def fake_predict(body, progress=None):
            progress("running the fake model")
            return {"success": True, "domain": body["domain_name"]}

        with patch.object(app, "_run_predict", side_effect=fake_predict):
            status, _, body = self.request(
                "POST", "/pinnfluid/predict", json.dumps(VALID_BODY).encode())
            self.assertEqual(status, 200, body)
            job_id = json.loads(body)["job_id"]
            job = {}
            for _ in range(100):
                status, _, body = self.request("GET", f"/pinnfluid/job_status?id={job_id}")
                self.assertEqual(status, 200)
                job = json.loads(body)
                if job["state"] in ("done", "error"):
                    break
                time.sleep(0.05)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(job["result"], {"success": True, "domain": "base-path-case"})


class RelativeUrlTests(unittest.TestCase):
    def test_detector_finds_root_absolute_urls(self):
        for text in ('<a href="/runs">', "<iframe src='/map?domain=x'>", "fetch('/build_inputs', {",
                     "el.href = '/download_pdf?domain=' + d;", "background: url(/static/x.png)",
                     '<script src="/static/plotly.min.js"></script>'):
            self.assertTrue(root_absolute_urls(text), text)
        for text in ('<a href="runs">', "fetch('build_inputs')", "x.replace(/_/g,' ')",
                     '<a href="https://example.org/a">', "'m/s'", "'</div>'"):
            self.assertFalse(root_absolute_urls(text), text)

    def test_main_page_has_no_root_absolute_url(self):
        self.assertEqual(root_absolute_urls(app._build_html()), [])

    def test_generated_pages_have_no_root_absolute_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            rose = results / "rose-case"
            rose.mkdir()
            (rose / "rose_summary.json").write_text(json.dumps({
                "worst_domain": "rose-case_d270",
                "sectors": [{"domain": "rose-case_d270", "dir_deg": 270},
                            {"domain": "rose-case_d0", "dir_deg": 0}],
            }))
            run = results / "plain-case"
            (run / "plots").mkdir(parents=True)
            (run / "report.pdf").write_bytes(b"%PDF")
            with patch.object(app, "RESULTS_DIR", results), \
                    patch.object(app, "has_saved_inputs", return_value=True):
                pages = {
                    "rose map": app._rose_map_wrapper_html("rose-case"),
                    "rose 3d": app._rose_3d_wrapper_html("rose-case", structure=False),
                    "rose structure": app._rose_3d_wrapper_html("rose-case", structure=True),
                    "runs index": app._runs_index_html(),
                    "roi chooser": app._structure_roi_chooser_html("plain-case", ["roi_0", "roi_1"]),
                }
        for name, html in pages.items():
            self.assertEqual(root_absolute_urls(html), [], name)
            self.assertIn("domain=", html, name)

    def test_3d_viewers_load_plotly_relative(self):
        import plotly.graph_objects as go  # type: ignore

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(view_3d, "build_3d_figure", return_value=(go.Figure(), {})), \
                patch.object(view_3d, "build_structure_3d_figure", return_value=(go.Figure(), {})), \
                patch.object(view_3d, "_streamline_js_hook", return_value=""):
            for writer in (view_3d.write_3d_html, view_3d.write_structure_3d_html):
                out = writer(Path(tmp) / f"{writer.__name__}.html", saved_inputs={}, domain_name="x")
                html = out.read_text(encoding="utf-8")
                self.assertIn('src="static/plotly.min.js"', html, writer.__name__)
                self.assertEqual(root_absolute_urls(html), [], writer.__name__)

    def test_no_root_absolute_url_literal_in_the_page_sources(self):
        """The map writer needs a full prediction to run; its source and every other
        HTML-producing module are checked for a root-absolute URL literal instead."""
        sources = [ROOT / "pinnfluid" / "domain_prep" / "domain_builder.py"]
        sources += sorted((ROOT / "pinnfluid" / "webapp").glob("*.py"))
        for src in sources:
            text = src.read_text(encoding="utf-8")
            if src.name == "domain_builder.py":
                text = text.split("class Handler", 1)[0]  # the standalone server's own routes
            if src.name == "app.py":
                text = text.split("class Handler", 1)[0]  # the routes themselves
            self.assertEqual(root_absolute_urls(text), [], src.name)


if __name__ == "__main__":
    unittest.main()
