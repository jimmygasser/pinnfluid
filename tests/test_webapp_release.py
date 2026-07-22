from __future__ import annotations

import contextlib
import http.client
import os
import sys
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "pinnfluid",
    ROOT / "pinnfluid" / "webapp",
    ROOT / "pinnfluid" / "domain_prep",
    ROOT / "pinnfluid" / "input_prep",
):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402


@contextlib.contextmanager
def environment(**values):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class WindRoseGeometryTests(unittest.TestCase):
    def test_automatic_sectors_start_at_reference_and_obey_cap(self):
        with environment(PINN_WEBAPP_MAX_ROSE_SECTORS=8):
            directions = app._parse_rose_directions({
                "wind_from": 270,
                "rose_sectors": 16,
            })
        self.assertEqual(directions, [270.0, 315.0, 0.0, 45.0, 90.0, 135.0, 180.0, 225.0])

    def test_custom_directions_over_cap_are_rejected(self):
        with environment(PINN_WEBAPP_MAX_ROSE_SECTORS=4):
            with self.assertRaisesRegex(ValueError, "limited to 4"):
                app._parse_rose_directions({"rose_directions": "0,45,90,135,180"})

    def test_single_and_grid_remain_fixed_in_geographic_frame(self):
        body = {
            "domain_name": "fixed",
            "wind_from": 270,
            "structures": [{"yaw": 15.0, "crs_x": 2600000.0, "crs_y": 1200000.0}],
            "grid": {
                "grid_yaw": 20.0,
                "struct_yaw": 5.0,
                "center": {"crs_x": 2600100.0, "crs_y": 1200100.0},
            },
        }
        single_invariant = None
        grid_invariant = None
        struct_invariant = None
        for direction in (270.0, 0.0, 90.0, 180.0):
            sector = app._rose_sector_body(
                body,
                sector_name=f"fixed_{int(direction)}",
                sector_wind_from=direction,
                geometry_reference_wind_from=270.0,
            )
            structure = sector["structures"][0]
            grid = sector["grid"]
            self.assertEqual(structure["crs_x"], body["structures"][0]["crs_x"])
            self.assertEqual(structure["crs_y"], body["structures"][0]["crs_y"])
            self.assertEqual(grid["center"], body["grid"]["center"])

            current_single = (structure["yaw"] - direction) % 360.0
            current_grid = (grid["grid_yaw"] - direction) % 360.0
            current_struct = (grid["struct_yaw"] - direction) % 360.0
            single_invariant = current_single if single_invariant is None else single_invariant
            grid_invariant = current_grid if grid_invariant is None else grid_invariant
            struct_invariant = current_struct if struct_invariant is None else struct_invariant
            self.assertAlmostEqual(current_single, single_invariant)
            self.assertAlmostEqual(current_grid, grid_invariant)
            self.assertAlmostEqual(current_struct, struct_invariant)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        with app._JOBS_LOCK:
            app._JOBS.clear()
            app._JOB_SUBMISSIONS.clear()
            app._PREP_SUBMISSIONS.clear()
            app._PREP_ACTIVE = 0

    def tearDown(self):
        with app._JOBS_LOCK:
            app._JOBS.clear()
            app._JOB_SUBMISSIONS.clear()
            app._PREP_SUBMISSIONS.clear()
            app._PREP_ACTIVE = 0

    def test_preparation_and_prediction_share_active_limit(self):
        with environment(
            PINN_WEBAPP_MAX_ACTIVE_JOBS=1,
            PINN_WEBAPP_RATE_LIMIT_PREP=0,
            PINN_WEBAPP_RATE_LIMIT_JOBS=0,
        ):
            admitted, _, _ = app._prep_admit()
            self.assertTrue(admitted)
            job_id, message, _ = app._job_admit_and_create("predict")
            self.assertIsNone(job_id)
            self.assertIn("already running", message)
            app._prep_release()

            job_id, _, _ = app._job_admit_and_create("predict")
            self.assertIsNotNone(job_id)

    def test_preparation_rate_limit(self):
        with environment(
            PINN_WEBAPP_MAX_ACTIVE_JOBS=1,
            PINN_WEBAPP_RATE_LIMIT_PREP=2,
            PINN_WEBAPP_RATE_LIMIT_WINDOW=3600,
        ):
            for _ in range(2):
                admitted, _, _ = app._prep_admit()
                self.assertTrue(admitted)
                app._prep_release()
            admitted, message, retry_after = app._prep_admit()
            self.assertFalse(admitted)
            self.assertIn("limit", message)
            self.assertGreater(retry_after, 0)


class InputValidationTests(unittest.TestCase):
    def test_browser_sized_request_is_valid(self):
        app._validate_compute_request({
            "domain_size": 1000,
            "wind_from": 270,
            "uref": 10,
            "zref": 20,
            "z0": 0.1,
            "structures": [{"yaw": 0}],
            "sampling_points": [],
            "grid": None,
        })

    def test_excessive_vertical_extent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "z_top_offset"):
            app._validate_compute_request({
                "domain_size": 1000,
                "z_top_offset": 100000,
            })

    def test_excessive_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "10 x 10"):
            app._validate_compute_request({
                "domain_size": 1000,
                "grid": {"rows": 11, "cols": 10},
            })


class HttpBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.Handler.page_html = app._build_html()
        cls.server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method: str, path: str, body: bytes | None = None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_branding_and_support_contact(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("X-Frame-Options"), "SAMEORIGIN")
        text = body.decode("utf-8")
        self.assertIn("pinnfluid - Wind and pressure prediction", text)
        self.assertIn("jimmy.gasser@epfl.ch", text)
        self.assertIn("https://github.com/jimmygasser/pinnfluid", text)
        self.assertNotIn("Past runs", text)

    def test_run_index_is_hidden_by_default(self):
        with environment(PINN_WEBAPP_ENABLE_RUN_INDEX=0):
            status, _, _ = self.request("GET", "/runs")
        self.assertEqual(status, 404)

    def test_oversized_request_is_rejected(self):
        with environment(PINN_WEBAPP_MAX_REQUEST_MB=1):
            conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
            try:
                conn.putrequest("POST", "/upload_dem")
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(1024 * 1024 + 1))
                conn.endheaders()
                response = conn.getresponse()
                status = response.status
                body = response.read()
            finally:
                conn.close()
        self.assertEqual(status, 413)
        self.assertIn(b"exceeds the 1 MiB limit", body)

    def test_malformed_json_is_rejected(self):
        status, _, body = self.request("POST", "/upload_dem", b"not-json")
        self.assertEqual(status, 400)
        self.assertIn(b"must be valid JSON", body)


if __name__ == "__main__":
    unittest.main()
