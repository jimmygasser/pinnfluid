"""Slow clients and capacity: a request deadline, a connection cap, admission before
the body, and a /status without software versions.

- A request (line, headers and body) must arrive within PINN_WEBAPP_REQUEST_DEADLINE_S
  (default 60 s) in total; a client trickling bytes is cut off. The work after the
  request has arrived is not limited by it.
- The server (LimitedHTTPServer, used by main) holds at most PINN_WEBAPP_MAX_CONNECTIONS
  connections (default 64) and answers 503 to more.
- A terrain or prediction request that would not be admitted gets 429 before its body
  (up to 32 MiB) is read.
- /status reports the device, not the torch version (PINN_WEBAPP_STATUS_DETAIL=1 adds it).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp", ROOT / "pinnfluid" / "domain_prep",
             ROOT / "pinnfluid" / "input_prep", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402
from pinnfluid_http import Browser, Server  # noqa: E402


def closed_within(sock: socket.socket, seconds: float, trickle: bytes = b"") -> float | None:
    """Send `trickle` one byte every 0.2 s until the server closes the connection;
    the seconds it took, or None if it stayed open for `seconds`."""
    sock.settimeout(0.2)
    started = time.monotonic()
    i = 0
    while time.monotonic() - started < seconds:
        try:
            if sock.recv(4096) == b"":
                return time.monotonic() - started
        except socket.timeout:
            pass
        except OSError:
            return time.monotonic() - started
        if i < len(trickle):
            try:
                sock.sendall(trickle[i:i + 1])
            except OSError:
                return time.monotonic() - started
            i += 1
    return None


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        app.Handler.page_html = app._build_html()
        self.server = Server(app)
        self.addCleanup(self.server.close)
        env = patch.dict(os.environ, {"PINN_WEBAPP_REQUEST_DEADLINE_S": "1"})
        env.start()
        self.addCleanup(env.stop)

    def connect(self) -> socket.socket:
        sock = socket.create_connection(self.server.address, timeout=5)
        self.addCleanup(sock.close)
        return sock

    def test_trickled_headers_are_cut_off(self):
        took = closed_within(self.connect(), 6, b"GET /status HTTP/1.1\r\nHost: x\r\nX-Slow: " + b"a" * 100)
        self.assertIsNotNone(took, "a trickling client kept the connection")
        self.assertLess(took, 4)

    def test_a_trickled_body_is_cut_off(self):
        sock = self.connect()
        sock.sendall(b"POST /upload_dem HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: 1000\r\n\r\n")
        took = closed_within(sock, 6, b"{" + b" " * 999)
        self.assertIsNotNone(took, "a trickling body kept the connection")
        self.assertLess(took, 4)

    def test_slow_work_after_the_request_is_not_cut_off(self):
        def slow(name):
            time.sleep(2)  # longer than the deadline
            raise FileNotFoundError(name)

        with patch.object(app, "_ensure_map_html", side_effect=slow):
            status, _, body = Browser(self.server).request("GET", "/map?domain=site", timeout=10)
        self.assertEqual(status, 404, body)

    def test_the_deadline_has_a_default(self):
        os.environ.pop("PINN_WEBAPP_REQUEST_DEADLINE_S")
        self.assertEqual(app._request_deadline_s(), 60)


class ConnectionCapTests(unittest.TestCase):
    def test_connections_beyond_the_cap_get_503(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_MAX_CONNECTIONS": "2"}):
            httpd = app.LimitedHTTPServer(("127.0.0.1", 0), app.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        idle = [socket.create_connection(httpd.server_address, timeout=5) for _ in range(2)]
        time.sleep(0.3)
        third = socket.create_connection(httpd.server_address, timeout=5)
        third.sendall(b"GET /status HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertTrue(third.recv(1024).startswith(b"HTTP/1.1 503"))
        third.close()
        idle[0].close()
        idle[1].close()
        time.sleep(0.5)
        fourth = socket.create_connection(httpd.server_address, timeout=5)
        fourth.sendall(b"GET /status HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        self.assertTrue(fourth.recv(1024).startswith(b"HTTP/1.1 200"))
        fourth.close()

    def test_main_uses_the_limited_server(self):
        self.assertIn("LimitedHTTPServer((args.host", (ROOT / "pinnfluid" / "webapp" / "app.py").read_text())


class AdmissionBeforeBodyTests(unittest.TestCase):
    def setUp(self):
        self.server = Server(app)
        self.addCleanup(self.server.close)
        with app._JOBS_LOCK:
            app._PREP_SUBMISSIONS.clear()
            app._JOB_SUBMISSIONS.clear()
            app._JOBS.clear()
            app._PREP_ACTIVE = 0
        self.addCleanup(app._PREP_SUBMISSIONS.clear)
        self.addCleanup(app._JOB_SUBMISSIONS.clear)

    def post_headers_only(self, route: str) -> bytes:
        sock = socket.create_connection(self.server.address, timeout=5)
        self.addCleanup(sock.close)
        sock.sendall(f"POST {route} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {30 * 1024 * 1024}\r\n\r\n".encode())
        return sock.recv(4096)  # the body never comes: an answer must not wait for it

    def test_prep_requests_over_the_limit_are_refused_before_the_body(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_RATE_LIMIT_PREP": "1", "PINN_WEBAPP_MAX_ACTIVE_JOBS": "0"}):
            self.assertTrue(app._prep_admit()[0])
            app._prep_release()
            for route in ("/upload_dem", "/download_dem", "/build_inputs"):
                answer = self.post_headers_only(route)
                self.assertTrue(answer.startswith(b"HTTP/1.1 429"), (route, answer[:80]))

    def test_predictions_over_the_limit_are_refused_before_the_body(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_RATE_LIMIT_JOBS": "1", "PINN_WEBAPP_MAX_ACTIVE_JOBS": "0"}):
            self.assertIsNotNone(app._job_admit_and_create("predict")[0])
            for route in ("/predict", "/predict_rose"):
                answer = self.post_headers_only(route)
                self.assertTrue(answer.startswith(b"HTTP/1.1 429"), (route, answer[:80]))

    def test_an_admitted_request_still_reads_its_body(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_RATE_LIMIT_PREP": "5"}):
            status, payload = Browser(self.server).json("POST", "/upload_dem", {"domain_name": "x"})
        self.assertEqual(status, 400)
        self.assertIn("data_base64", payload["error"])


class StatusTests(unittest.TestCase):
    def test_status_has_no_software_versions(self):
        server = Server(app)
        self.addCleanup(server.close)
        status, body = Browser(server).json("GET", "/status")
        self.assertEqual(status, 200)
        self.assertEqual(set(body["runtime"]) - {"gpu"}, {"requested", "resolved", "cuda_available"})
        with patch.dict(os.environ, {"PINN_WEBAPP_STATUS_DETAIL": "1"}):
            _, body = Browser(server).json("GET", "/status")
        self.assertIn("torch", body["runtime"])


if __name__ == "__main__":
    unittest.main()
