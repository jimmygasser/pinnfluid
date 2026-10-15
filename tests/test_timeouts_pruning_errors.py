"""Timeouts, pruning and client errors.

- Terrain subprocesses, tile downloads and idle client connections have time
  limits, and a step that runs out of time ends with a clear error.
- Finished jobs, old DEMs, left-over workspaces and old wind-rose summaries are
  pruned, so a long-running server does not grow without bound.
- Clients get validation messages, or a generic message with a reference for
  anything else; tracebacks and server paths stay in the log.
- The in-process prediction limit is global (one counter for every client) and
  configurable.
"""

from __future__ import annotations

import ast
import contextlib
import http.client
import io
import json
import os
import socket
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
import build_inputs  # noqa: E402
import domain_builder  # noqa: E402

BODY = {"domain_name": "site", "domain_size": 1000, "wind_from": 270}


def leaks(text: str) -> list[str]:
    """Server-side details that must not reach a client."""
    return [s for s in ("Traceback", "/app/", "/tmp/", "/src/", ".py", "OSError",
                        "RuntimeError", "File \"") if s in text]


class StepTimeoutTests(unittest.TestCase):
    def test_a_stuck_subprocess_ends_with_a_clear_error(self):
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, r"dem_prep did not finish within 1 s"):
            domain_builder._run_step([sys.executable, "-c", "import time; time.sleep(60)"],
                                     "dem_prep", timeout=1)
        self.assertLess(time.monotonic() - started, 20)

    def test_the_step_timeout_is_configurable(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_STEP_TIMEOUT_S": "1"}):
            with self.assertRaisesRegex(RuntimeError, "within 1 s"):
                domain_builder._run_step([sys.executable, "-c", "import time; time.sleep(60)"],
                                         "finalize_domain")

    def test_a_step_reports_its_own_validation_message(self):
        script = "raise ValueError('Input DEM mosaic is not large enough\\nPivot: (1.0, 2.0)')"
        with self.assertRaises(ValueError) as caught:
            domain_builder._run_step([sys.executable, "-c", script], "dem_prep")
        message = str(caught.exception)
        self.assertIn("dem_prep: Input DEM mosaic is not large enough", message)
        self.assertIn("Pivot: (1.0, 2.0)", message)
        self.assertEqual(leaks(message), [])

    def test_other_step_failures_are_internal(self):
        with self.assertRaises(RuntimeError) as caught:
            domain_builder._run_step([sys.executable, "-c", "import sys; sys.exit(3)"],
                                     "place_structure")
        self.assertIn("place_structure failed", str(caught.exception))

    def test_every_subprocess_has_a_timeout(self):
        tree = ast.parse((ROOT / "pinnfluid" / "domain_prep" / "domain_builder.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess"]
        self.assertTrue(calls)
        for call in calls:
            self.assertIn("timeout", [k.arg for k in call.keywords], call.lineno)

    def test_tile_downloads_have_a_timeout(self):
        seen = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(url, timeout=None):
            seen.append(timeout)
            return Response(b"x" * 2000)

        href = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/a/a_2_2056_5728.tif"
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(domain_builder, "_stac_open", side_effect=fake_urlopen):
            downloaded, failed = domain_builder.download_tiles([{"name": "t.tif", "href": href}], tmp)
            self.assertEqual((downloaded, failed), (1, 0))
            self.assertEqual((Path(tmp) / "t.tif").read_bytes(), b"x" * 2000)
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0])
        self.assertLessEqual(seen[0], 120)


class SocketTimeoutTests(unittest.TestCase):
    def test_an_idle_connection_is_closed(self):
        class QuickHandler(app.Handler):
            timeout = 0.5

        server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuickHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            sock = socket.create_connection(server.server_address, timeout=10)
            sock.sendall(b"GET /status HTTP/1.1\r\n")  # and never the rest
            started = time.monotonic()
            self.assertEqual(sock.recv(1024), b"")  # closed by the server
            self.assertLess(time.monotonic() - started, 5)
            sock.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_the_socket_timeout_has_a_default_and_a_setting(self):
        self.assertEqual(app.Handler.timeout, 60)
        with patch.dict(os.environ, {"PINN_WEBAPP_SOCKET_TIMEOUT_S": "15"}):
            self.assertEqual(app._socket_timeout_from_env(), 15)
        with patch.dict(os.environ, {"PINN_WEBAPP_SOCKET_TIMEOUT_S": "0"}):
            self.assertIsNone(app._socket_timeout_from_env())


class JobPruningTests(unittest.TestCase):
    def setUp(self):
        limits = patch.dict(os.environ, {"PINN_WEBAPP_MAX_ACTIVE_JOBS": "0",
                                         "PINN_WEBAPP_RATE_LIMIT_JOBS": "0",
                                         "PINN_WEBAPP_RATE_LIMIT_PREP": "0"})
        limits.start()
        self.addCleanup(limits.stop)
        with app._JOBS_LOCK:
            app._JOBS.clear()
            app._JOB_SUBMISSIONS.clear()
        self.now = 1_000_000.0
        patcher = patch.object(app, "_clock", side_effect=lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(app._JOBS.clear)

    def make_job(self, state: str) -> str:
        jid, _, _ = app._job_admit_and_create("predict", session="s")
        if state in ("done", "error"):
            app._job_finish(jid, state, result={"big": "x" * 100} if state == "done" else None,
                            error=None if state == "done" else "failed")
        elif state == "running":
            app._job_update(jid, state="running")
        return jid

    def test_finished_jobs_expire_after_the_ttl(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_JOB_TTL_S": "3600"}):
            done, failed, running = (self.make_job("done"), self.make_job("error"),
                                     self.make_job("running"))
            self.now += 3599
            self.assertIsNotNone(app._job_get(done))
            self.now += 2
            self.assertIsNone(app._job_get(done))
            self.assertIsNone(app._job_get(failed))
            self.assertIsNotNone(app._job_get(running))  # never while it runs
            newest = self.make_job("done")
            self.assertEqual(set(app._JOBS), {running, newest})

    def test_the_number_of_finished_jobs_is_capped(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_JOB_TTL_S": "0"}):
            ids = []
            for _ in range(app._MAX_FINISHED_JOBS + 5):
                ids.append(self.make_job("done"))
                self.now += 1
            self.make_job("queued")
        self.assertEqual(sum(j["state"] == "done" for j in app._JOBS.values()),
                         app._MAX_FINISHED_JOBS)
        self.assertNotIn(ids[0], app._JOBS)
        self.assertIn(ids[-1], app._JOBS)


class WorkDirPruningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        work = Path(self.tmp.name)
        self.dem, self.workspace, self.results = work / "dem", work / "workspace", work / "results"
        for target, attr, value in ((app, "DEM_ROOT", self.dem), (app, "WORKSPACE", self.workspace),
                                    (app, "RESULTS_DIR", self.results),
                                    (app.Handler, "dem_root", str(self.dem)),
                                    (build_inputs, "DEM_ROOT", self.dem),
                                    (build_inputs, "WORKSPACE", self.workspace)):
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.now = time.time()

    def entry(self, root: Path, name: str, age_s: float, *, child: str = "x") -> Path:
        d = root / name
        d.mkdir(parents=True)
        (d / child).mkdir() if child == "inputs" else (d / child).write_text("x")
        t = self.now - age_s
        os.utime(d, (t, t))
        return d

    def test_old_entries_go_and_recent_or_busy_ones_stay(self):
        cases = self.workspace / "cases" / "singlestructures"
        cfd = self.workspace / "cfd" / "singlestructures"
        old_dem = self.entry(self.dem, "ns_old", 7 * 3600)
        new_dem = self.entry(self.dem, "ns_new", 60)
        busy_dem = self.entry(self.dem, "ns_busy", 7 * 3600)
        busy_sector = self.entry(self.dem, "ns_busy_r090", 7 * 3600)
        old_case = self.entry(cases, "ns_old", 7 * 3600)
        old_cfd = self.entry(cfd, "ns_old", 7 * 3600)
        old_rose = self.entry(self.results, "ns_rose", 7 * 3600)  # summary only, no inputs/
        old_run = self.entry(self.results, "ns_run", 7 * 3600, child="inputs")  # retention's job
        with patch.dict(os.environ, {"PINN_WEBAPP_WORK_TTL_S": "21600"}), app._busy("ns_busy"):
            app._prune_work_dirs()
        for gone in (old_dem, old_case, old_cfd, old_rose):
            self.assertFalse(gone.exists(), gone)
        for kept in (new_dem, busy_dem, busy_sector, old_run):
            self.assertTrue(kept.exists(), kept)

    def test_the_number_of_dems_is_capped(self):
        dems = [self.entry(self.dem, f"ns_{i}", 600 - i) for i in range(6)]  # dems[0] oldest
        with patch.dict(os.environ, {"PINN_WEBAPP_MAX_DEMS": "4", "PINN_WEBAPP_WORK_TTL_S": "0"}):
            app._prune_work_dirs()
        self.assertEqual([d.exists() for d in dems], [False, False, True, True, True, True])

    def test_compute_requests_prune(self):
        old_dem = self.entry(self.dem, "ns_old", 7 * 3600)
        server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            conn = http.client.HTTPConnection(*server.server_address, timeout=20)
            with patch.object(app, "register_custom_dem", return_value={
                    "width_m": 1, "height_m": 1, "bounds_crs": [0, 0, 1, 1], "crs": "x",
                    "png_base64": ""}):
                conn.request("POST", "/upload_dem", body=json.dumps(
                    {"domain_name": "site", "data_base64": "A" * 2000}))
                self.assertEqual(conn.getresponse().status, 200)
        finally:
            conn.close()
            server.shutdown()
            server.server_close()
        self.assertFalse(old_dem.exists())


class ClientErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.Handler.page_html = app._build_html()
        cls.server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        limits = patch.dict(os.environ, {"PINN_WEBAPP_MAX_ACTIVE_JOBS": "0",
                                         "PINN_WEBAPP_RATE_LIMIT_JOBS": "0",
                                         "PINN_WEBAPP_RATE_LIMIT_PREP": "0"})
        limits.start()
        self.addCleanup(limits.stop)
        self.cookie = ""
        with app._JOBS_LOCK:
            app._JOBS.clear()
            app._JOB_SUBMISSIONS.clear()
            app._PREP_SUBMISSIONS.clear()
            app._PREP_ACTIVE = 0

    def request(self, method: str, path: str, body: dict | None = None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=20)
        headers = {"Content-Type": "application/json"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        try:
            conn.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
            r = conn.getresponse()
            if r.getheader("Set-Cookie"):
                self.cookie = r.getheader("Set-Cookie").split(";", 1)[0]
            return r.status, json.loads(r.read())
        finally:
            conn.close()

    def test_an_internal_error_answers_with_a_reference_only(self):
        boom = OSError("[Errno 2] /app/pinnfluid/webapp/workspace/cfd/x/meta.json: lost")
        log = io.StringIO()
        with patch.object(app, "_run_build_inputs", side_effect=boom), \
                contextlib.redirect_stderr(log):
            status, payload = self.request("POST", "/build_inputs", BODY)
        self.assertEqual(status, 500, payload)
        self.assertFalse(payload["success"])
        self.assertRegex(payload["reference"], r"^[0-9a-f]{10}$")
        self.assertIn(payload["reference"], payload["error"])
        self.assertEqual(leaks(json.dumps(payload)), [], payload)
        logged = log.getvalue()
        self.assertIn(payload["reference"], logged)
        self.assertIn("Traceback", logged)
        self.assertIn("/app/pinnfluid/webapp/workspace", logged)

    def test_validation_messages_reach_the_client_without_paths(self):
        with patch.object(app, "_run_build_inputs",
                          side_effect=ValueError("DEM at /tmp/w/dem/ns_site/dem_cropped.tif has no CRS")):
            status, payload = self.request("POST", "/build_inputs", BODY)
        self.assertEqual(status, 400, payload)
        self.assertEqual(payload["error"], "DEM at <path> has no CRS")

    def test_a_failed_job_reports_a_reference_only(self):
        def fail(body, progress=None):
            raise RuntimeError("Traceback (most recent call last):\n  File \"/app/x.py\"")

        with patch.object(app, "_run_predict", side_effect=fail), \
                contextlib.redirect_stderr(io.StringIO()):
            status, payload = self.request("POST", "/predict", BODY)
            self.assertEqual(status, 200, payload)
            job = {}
            for _ in range(200):
                _, job = self.request("GET", f"/job_status?id={payload['job_id']}")
                if job["state"] in ("done", "error"):
                    break
                time.sleep(0.02)
        self.assertEqual(job["state"], "error")
        self.assertIn("reference", job["error"])
        self.assertEqual(leaks(json.dumps(job)), [], job)

    def test_responses_carry_no_server_paths(self):
        def fake_predict(body, progress=None):
            return {"success": True, "domain_name": body["domain_name"],
                    "saved_dir": "/w/results/x/plots", "case_dir": "/w/case", "cfd_dir": "/w/cfd",
                    "plots": {"speed": {"path": "/w/results/x/plots/speed.png",
                                        "filename": "speed.png", "png_base64": "AAAA"}}}

        with patch.object(app, "_run_predict", side_effect=fake_predict):
            _, payload = self.request("POST", "/predict", BODY)
            job = {}
            for _ in range(200):
                _, job = self.request("GET", f"/job_status?id={payload['job_id']}")
                if job["state"] in ("done", "error"):
                    break
                time.sleep(0.02)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(job["result"], {"success": True, "domain_name": "site",
                                         "plots": {"speed": {"filename": "speed.png",
                                                             "png_base64": "AAAA"}}})
        with patch.object(app, "_run_build_inputs", return_value={
                "success": True, "domain_name": "x", "case_dir": "/w/case", "cfd_dir": "/w/cfd",
                "grid_shape": [1, 2, 3]}):
            status, payload = self.request("POST", "/build_inputs", BODY)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload, {"success": True, "domain_name": "x", "grid_shape": [1, 2, 3]})

    def test_a_missing_run_answers_404_without_details(self):
        for route in ("map", "view_3d", "download_pdf", "download_vtk", "map_rose"):
            status, payload = self.request("GET", f"/{route}?domain=nothing-here")
            self.assertEqual(status, 404, route)
            self.assertEqual(payload["error"], "not found", route)

    def test_bad_coordinates_are_a_client_error(self):
        status, payload = self.request("GET", "/lv95_to_wgs84?e=abc&n=1")
        self.assertEqual(status, 400)
        self.assertEqual(leaks(json.dumps(payload)), [])


class GlobalLimitTests(unittest.TestCase):
    """PINN_WEBAPP_RATE_LIMIT_JOBS is one counter for every client (per-client limits
    belong in front of the app, where the client address is known)."""

    def setUp(self):
        with app._JOBS_LOCK:
            app._JOBS.clear()
            app._JOB_SUBMISSIONS.clear()
        self.addCleanup(app._JOBS.clear)
        self.addCleanup(app._JOB_SUBMISSIONS.clear)

    def test_the_global_prediction_limit_is_configurable_and_shared(self):
        server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        statuses = []
        try:
            with patch.dict(os.environ, {"PINN_WEBAPP_RATE_LIMIT_JOBS": "2",
                                         "PINN_WEBAPP_RATE_LIMIT_WINDOW": "3600",
                                         "PINN_WEBAPP_MAX_ACTIVE_JOBS": "0"}), \
                    patch.object(app, "_job_start"):
                for i in range(3):  # three browsers
                    conn = http.client.HTTPConnection(*server.server_address, timeout=20)
                    try:
                        conn.request("POST", "/predict", body=json.dumps(dict(BODY, domain_name=f"d{i}")),
                                     headers={"X-Forwarded-For": f"10.0.0.{i}"})
                        r = conn.getresponse()
                        statuses.append((r.status, json.loads(r.read())))
                    finally:
                        conn.close()
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual([s for s, _ in statuses], [200, 200, 429])
        self.assertGreater(statuses[2][1]["retry_after"], 0)


if __name__ == "__main__":
    unittest.main()
