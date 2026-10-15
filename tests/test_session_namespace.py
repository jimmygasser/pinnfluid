"""Each browser session sees only its own domains, runs and jobs.

The app gives every browser a random 128-bit session id in an HttpOnly cookie
(pinnfluid_session). Domains live under a per-session namespace on disk, so two
people who both call their domain "test" never see or overwrite each other's
terrain, inputs or results, and /job_status answers only the session that
started the job. Anything else gets the same 404 as a name that does not exist.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
import zipfile
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
import build_inputs  # noqa: E402

COOKIE_RE = re.compile(
    r"^pinnfluid_session=([0-9a-f]{32}); Path=/pinnfluid/; HttpOnly; SameSite=Lax(; Secure)?$")

BODY = {
    "domain_name": "shared",
    "domain_size": 1000,
    "wind_from": 270,
    "uref": 10,
    "zref": 20,
    "z0": 0.1,
    "structures": [],
    "sampling_points": [],
    "grid": None,
}

RESULT_ROUTES = ("download_pdf", "download_zip", "download_vtk", "download_npz", "map",
                 "view_3d", "download_3d", "view_3d_structure", "download_3d_structure",
                 "map_rose", "view_3d_rose", "view_3d_structure_rose")


class Browser:
    """An HTTP client with a cookie jar of one cookie, like one browser profile."""

    def __init__(self, server, cookie: str | None = None):
        self.host, self.port = server.server_address
        self.cookie = cookie

    def request(self, method: str, path: str, body: dict | None = None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=20)
        headers = {"Content-Type": "application/json"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, body=data, headers=headers)
            r = conn.getresponse()
            payload = r.read()
            set_cookie = r.getheader("Set-Cookie")
            if set_cookie:
                self.cookie = set_cookie.split(";", 1)[0]
            return r.status, dict(r.getheaders()), payload
        finally:
            conn.close()

    @property
    def session_id(self) -> str:
        return self.cookie.split("=", 1)[1]


class SessionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.Handler.page_html = app._build_html()

        class PrefixedHandler(app.Handler):
            base_path = "/pinnfluid"

        cls.server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), PrefixedHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        work = Path(self.tmp.name)
        self.results = work / "results"
        self.dem = work / "dem"
        self.results.mkdir()
        self.dem.mkdir()
        for target, attr, value in ((app, "RESULTS_DIR", self.results),
                                    (app, "DEM_ROOT", self.dem),
                                    (app.Handler, "dem_root", str(self.dem)),
                                    (build_inputs, "DEM_ROOT", self.dem),
                                    (build_inputs, "WORKSPACE", work / "workspace")):
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        with app._JOBS_LOCK:
            app._JOBS.clear()
            app._JOB_SUBMISSIONS.clear()

    def browser(self, cookie: str | None = None) -> Browser:
        return Browser(self.server, cookie)

    def wait_for(self, browser: Browser, job_id: str) -> dict:
        for _ in range(200):
            status, _, body = browser.request("GET", f"/pinnfluid/job_status?id={job_id}")
            self.assertEqual(status, 200, body)
            job = json.loads(body)
            if job["state"] in ("done", "error"):
                return job
            time.sleep(0.02)
        self.fail(f"job {job_id} did not finish")


class CookieTests(SessionTestCase):
    def test_a_request_without_a_cookie_gets_a_new_session(self):
        # the default (no Secure), whatever the environment of the test run says
        self.enterContext(patch.dict(os.environ, {"PINN_WEBAPP_SECURE_COOKIE": "0"}))
        first, second = self.browser(), self.browser()
        for browser in (first, second):
            status, headers, _ = browser.request("GET", "/pinnfluid/")
            self.assertEqual(status, 200)
            self.assertRegex(headers.get("Set-Cookie", ""), COOKIE_RE)
            self.assertNotIn("Secure", headers["Set-Cookie"])
        self.assertNotEqual(first.cookie, second.cookie)
        # the browser sends it back: same session, no new cookie
        status, headers, _ = first.request("GET", "/pinnfluid/")
        self.assertEqual(status, 200)
        self.assertNotIn("Set-Cookie", headers)

    def test_secure_with_the_setting(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_SECURE_COOKIE": "1"}):
            _, headers, _ = self.browser().request("GET", "/pinnfluid/")
        self.assertRegex(headers["Set-Cookie"], COOKIE_RE)
        self.assertTrue(headers["Set-Cookie"].endswith("; Secure"))

    def test_a_malformed_cookie_is_replaced(self):
        for value in ("../../etc", "ABCDEF0123456789ABCDEF0123456789", "0" * 31, "0" * 33, ""):
            browser = self.browser(f"pinnfluid_session={value}")
            _, headers, _ = browser.request("GET", "/pinnfluid/")
            self.assertRegex(headers.get("Set-Cookie", ""), COOKIE_RE, value)

    def test_the_cookie_is_found_among_others(self):
        sid = "0123456789abcdef0123456789abcdef"
        browser = self.browser(f"other=1; pinnfluid_session={sid}; theme=dark")
        _, headers, _ = browser.request("GET", "/pinnfluid/")
        self.assertNotIn("Set-Cookie", headers)

    def test_status_and_assets_set_no_cookie(self):
        for path in ("/pinnfluid/status", "/pinnfluid/static/plotly.min.js"):
            status, headers, _ = self.browser().request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertNotIn("Set-Cookie", headers, path)


class IsolationTests(SessionTestCase):
    @staticmethod
    def fake_predict(body, progress=None):
        """Writes a run like _run_predict does, under the name it is given."""
        name = body["domain_name"]
        run = app.RESULTS_DIR / name
        (run / "plots").mkdir(parents=True, exist_ok=True)
        (run / "plots" / "speed.png").write_bytes(b"png of " + body["owner"].encode())
        (run / "report.pdf").write_bytes(b"%PDF of " + body["owner"].encode())
        return {"success": True, "domain_name": name, "worst_domain": f"{name}_r270",
                "sectors": [{"domain": f"{name}_r270"}]}

    def predict(self, browser: Browser, owner: str) -> tuple[str, dict]:
        with patch.object(app, "_run_predict", side_effect=self.fake_predict):
            status, _, body = browser.request("POST", "/pinnfluid/predict", dict(BODY, owner=owner))
            self.assertEqual(status, 200, body)
            job_id = json.loads(body)["job_id"]
            return job_id, self.wait_for(browser, job_id)

    def test_two_sessions_with_the_same_domain_name_are_isolated(self):
        alice, bob = self.browser(), self.browser()
        _, job_a = self.predict(alice, "alice")
        _, job_b = self.predict(bob, "bob")
        self.assertEqual(job_a["state"], "done", job_a)
        self.assertEqual(job_b["state"], "done", job_b)
        # Responses name the domain as the user typed it.
        for job in (job_a, job_b):
            self.assertEqual(job["result"]["domain_name"], "shared")
            self.assertEqual(job["result"]["worst_domain"], "shared_r270")
            self.assertEqual(job["result"]["sectors"], [{"domain": "shared_r270"}])
        # Two runs on disk, neither of them called "shared".
        runs = sorted(p.name for p in self.results.iterdir())
        self.assertEqual(len(runs), 2, runs)
        self.assertNotIn("shared", runs)
        self.assertTrue(all(r.endswith("_shared") for r in runs), runs)
        for browser, owner in ((alice, b"alice"), (bob, b"bob")):
            status, headers, body = browser.request("GET", "/pinnfluid/download_pdf?domain=shared")
            self.assertEqual(status, 200)
            self.assertEqual(body, b"%PDF of " + owner)
            self.assertEqual(headers["Content-Disposition"], 'attachment; filename="shared_report.pdf"')
            status, _, body = browser.request("GET", "/pinnfluid/download_zip?domain=shared")
            self.assertEqual(status, 200)
            with zipfile.ZipFile(io.BytesIO(body)) as zf:
                self.assertEqual(zf.read("speed.png"), b"png of " + owner)

    def test_results_of_another_session_are_not_found(self):
        alice = self.browser()
        self.predict(alice, "alice")
        on_disk = next(self.results.iterdir()).name
        for intruder in (self.browser(), self.browser("pinnfluid_session=" + "f" * 32)):
            for route in RESULT_ROUTES:
                for name in ("shared", on_disk):
                    status, _, body = intruder.request("GET", f"/pinnfluid/{route}?domain={name}")
                    self.assertEqual(status, 404, (route, name, body))
                    self.assertNotIn(b"of alice", body)
                    if name == "shared":  # nothing reveals alice's namespace
                        self.assertNotIn(on_disk.encode(), body)
        status, _, _ = alice.request("GET", "/pinnfluid/download_pdf?domain=shared")
        self.assertEqual(status, 200)

    def test_names_the_app_rendered_resolve_for_their_owner_only(self):
        """Pages built from on-disk names (the wind-rose selector, the ROI chooser)
        link with the namespaced name; it must resolve for the owner, once."""
        alice = self.browser()
        self.predict(alice, "alice")
        on_disk = next(self.results.iterdir()).name
        status, _, body = alice.request("GET", f"/pinnfluid/download_pdf?domain={on_disk}")
        self.assertEqual((status, body), (200, b"%PDF of alice"))
        rose = self.results / f"{on_disk}_rose"
        rose.mkdir()
        (rose / "rose_summary.json").write_text(json.dumps({
            "worst_domain": f"{on_disk}_r270",
            "sectors": [{"domain": f"{on_disk}_r270", "dir_deg": 270}],
        }))
        status, _, body = alice.request("GET", "/pinnfluid/map_rose?domain=shared_rose")
        self.assertEqual(status, 200, body)
        page = body.decode()
        self.assertIn("— shared_rose", page)  # the title shows the user's name
        self.assertNotIn(f"— {on_disk}", page)
        status, _, _ = self.browser().request("GET", "/pinnfluid/map_rose?domain=shared_rose")
        self.assertEqual(status, 404)

    def test_job_status_answers_only_the_session_that_started_the_job(self):
        alice = self.browser()
        job_id, job = self.predict(alice, "alice")
        self.assertEqual(job["state"], "done")
        self.assertNotIn("session", job)
        _, _, unknown = alice.request("GET", "/pinnfluid/job_status?id=000000000000")
        for intruder in (self.browser(), self.browser("pinnfluid_session=" + "e" * 32)):
            status, headers, body = intruder.request("GET", f"/pinnfluid/job_status?id={job_id}")
            self.assertEqual(status, 404)
            self.assertEqual(body, unknown)
        status, headers, _ = self.browser().request("GET", f"/pinnfluid/job_status?id={job_id}")
        self.assertEqual(status, 404)
        self.assertRegex(headers.get("Set-Cookie", ""), COOKIE_RE)

    def test_terrain_uploads_are_namespaced(self):
        seen = []

        def fake_register(name, dem_bytes, dem_root):
            seen.append((name, dem_root))
            return {"width_m": 1.0, "height_m": 1.0, "bounds_crs": [0, 0, 1, 1],
                    "crs": "EPSG:32632", "png_base64": ""}

        alice, bob = self.browser(), self.browser()
        with patch.object(app, "register_custom_dem", side_effect=fake_register):
            for browser in (alice, bob):
                status, _, body = browser.request("POST", "/pinnfluid/upload_dem", {
                    "domain_name": "shared", "data_base64": "A" * 2000})
                self.assertEqual(status, 200, body)
        names = [name for name, _ in seen]
        self.assertEqual(len(set(names)), 2, names)
        self.assertTrue(all(n.endswith("_shared") and n != "shared" for n in names), names)

    def test_build_inputs_and_predict_use_the_namespaced_name(self):
        seen = []
        alice = self.browser()
        with patch.object(app, "_run_build_inputs",
                          side_effect=lambda b: seen.append(b["domain_name"]) or {
                              "success": True, "domain_name": b["domain_name"]}):
            status, _, body = alice.request("POST", "/pinnfluid/build_inputs", BODY)
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["domain_name"], "shared")
        ns = app._session_namespace(alice.session_id)
        self.assertEqual(seen, [f"{ns}_shared"])

    def test_the_run_index_lists_only_the_sessions_own_runs(self):
        alice, bob = self.browser(), self.browser()
        self.predict(alice, "alice")
        self.predict(bob, "bob")
        with patch.dict(os.environ, {"PINN_WEBAPP_ENABLE_RUN_INDEX": "1"}):
            status, _, body = alice.request("GET", "/pinnfluid/runs")
        self.assertEqual(status, 200)
        page = body.decode()
        self.assertEqual(page.count("download_pdf?domain="), 1, page)
        ns_bob = app._session_namespace(bob.session_id)
        self.assertNotIn(ns_bob, page)


class NamespaceHelperTests(unittest.TestCase):
    def test_the_namespace_is_derived_from_but_does_not_reveal_the_id(self):
        sid = "0123456789abcdef0123456789abcdef"
        ns = app._session_namespace(sid)
        self.assertRegex(ns, r"^[0-9a-f]{16}$")
        self.assertNotIn(ns, sid)
        self.assertEqual(ns, app._session_namespace(sid))
        self.assertNotEqual(ns, app._session_namespace("f" + sid[1:]))

    def test_session_names(self):
        ns = "0123456789abcdef"
        self.assertEqual(app._session_name(ns, "site"), f"{ns}_site")
        self.assertEqual(app._session_name(ns, f"{ns}_site"), f"{ns}_site")
        other = "fedcba9876543210"
        self.assertEqual(app._session_name(ns, f"{other}_site"), f"{ns}_{other}_site")
        for bad in ("", "..", "../x", ".hidden"):
            with self.assertRaises(ValueError):
                app._session_name(ns, bad)

    def test_display_names(self):
        ns = "0123456789abcdef"
        self.assertEqual(app._display_name(f"{ns}_site_r090"), "site_r090")
        self.assertEqual(app._display_name("site"), "site")


if __name__ == "__main__":
    unittest.main()
