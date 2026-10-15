"""The VTK and NPZ exports are built once per run, atomically, behind an admission check.

Each download used to export every field again into the same directory and zip it,
so parallel downloads overwrote each other's files mid-zip (truncated archives with
200) and every click cost seconds of CPU. Now the first request builds the archive in
a temporary place and renames it into exports/; requests for the same archive wait for
that build and get the same file; later requests are served from the cache. Starting
a build is limited per session and globally (429 when busy); a cached archive is
always served.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp", ROOT / "pinnfluid" / "domain_prep",
             ROOT / "pinnfluid" / "input_prep", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402
from pinnfluid_http import Browser, Server  # noqa: E402


class FakeExporter:
    """Stands in for export_prediction_vtk / export_prediction_npz: slow, and writes
    1 MB files whose content names the call."""

    def __init__(self, suffix: str, gate: threading.Event | None = None):
        self.suffix, self.gate, self.calls = suffix, gate, 0
        self.lock = threading.Lock()

    def __call__(self, out_dir, *, domain_name, **_):
        with self.lock:
            self.calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for part in ("global", "roi_000"):
            f = out / f"{domain_name}_{part}.{self.suffix}"
            with f.open("wb") as fh:  # written slowly, like a real export
                for i in range(16):
                    fh.write(bytes([i]) * 65536)
                    time.sleep(0.01)
            files.append(str(f))
        return {"files": files, "skipped": []}


class ExportCacheTests(unittest.TestCase):
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
        self.results = Path(self.tmp.name) / "results"
        self.results.mkdir()
        self.vtk, self.npz = FakeExporter("vtk"), FakeExporter("npy")
        for target, attr, value in ((app, "RESULTS_DIR", self.results),
                                    (app, "has_saved_inputs", lambda root, name: (root / name / "inputs").is_dir()),
                                    (app, "load_saved_inputs", lambda root, name: {
                                        "bundle": None, "pred_flow": None, "roi_bundles": {}, "roi_preds": {}}),
                                    (app, "export_prediction_vtk", self.vtk),
                                    (app, "export_prediction_npz", self.npz)):
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {"PINN_WEBAPP_MAX_EXPORT_BUILDS": "2",
                                      "PINN_WEBAPP_MAX_EXPORT_BUILDS_PER_SESSION": "1"})
        env.start()
        self.addCleanup(env.stop)

    def browser_with_runs(self, *names) -> Browser:
        browser = Browser(self.server)
        browser.request("GET", "/")
        ns = app._session_namespace(browser.cookie.split("=", 1)[1])
        for name in names:
            (self.results / f"{ns}_{name}" / "inputs").mkdir(parents=True)
        return browser

    def parallel(self, browser: Browser, path: str, n: int = 8):
        with ThreadPoolExecutor(n) as pool:
            return list(pool.map(lambda _: browser.request("GET", path), range(n)))

    def check_parallel(self, route: str, exporter: FakeExporter, suffix: str):
        browser = self.browser_with_runs("site")
        answers = self.parallel(browser, f"/{route}?domain=site")
        self.assertEqual([a[0] for a in answers], [200] * 8, [a[2][:200] for a in answers])
        digests = {hashlib.sha256(a[2]).hexdigest() for a in answers}
        self.assertEqual(len(digests), 1, "the parallel downloads differ")
        self.assertEqual(exporter.calls, 1, "the export ran more than once")
        with zipfile.ZipFile(io.BytesIO(answers[0][2])) as zf:
            self.assertIsNone(zf.testzip())
            sizes = {i.filename: i.file_size for i in zf.infolist() if i.filename.endswith(suffix)}
        self.assertEqual(sorted(sizes.values()), [16 * 65536, 16 * 65536], sizes)
        # later downloads come from the cache
        status, _, again = browser.request("GET", f"/{route}?domain=site")
        self.assertEqual((status, hashlib.sha256(again).hexdigest()), (200, digests.pop()))
        self.assertEqual(exporter.calls, 1)
        # nothing temporary is left next to the run
        leftovers = [p for p in (self.results).rglob("*") if ".tmp" in p.name or p.name.startswith(".")]
        self.assertEqual(leftovers, [])

    def test_eight_parallel_vtk_downloads(self):
        self.check_parallel("download_vtk", self.vtk, ".vtk")

    def test_eight_parallel_npz_downloads(self):
        self.check_parallel("download_npz", self.npz, ".npy")

    def test_a_new_prediction_invalidates_the_cache(self):
        browser = self.browser_with_runs("site")
        self.assertEqual(browser.request("GET", "/download_vtk?domain=site")[0], 200)
        ns = app._session_namespace(browser.cookie.split("=", 1)[1])
        app._invalidate_3d_cache(f"{ns}_site")
        self.assertEqual(browser.request("GET", "/download_vtk?domain=site")[0], 200)
        self.assertEqual(self.vtk.calls, 2)

    def test_starting_builds_is_limited_per_session_and_globally(self):
        gate = threading.Event()
        self.vtk.gate = gate
        alice = self.browser_with_runs("one", "two")
        bob = self.browser_with_runs("one")
        carol = self.browser_with_runs("one")
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(alice.request, "GET", "/download_vtk?domain=one")
            for _ in range(100):  # until alice's build has started
                if self.vtk.calls:
                    break
                time.sleep(0.02)
            # alice's second run: her one build slot is taken
            status, _, body = alice.request("GET", "/download_vtk?domain=two")
            self.assertEqual(status, 429, body)
            self.assertIn(b"retry_after", body)
            second = pool.submit(bob.request, "GET", "/download_vtk?domain=one")
            for _ in range(100):
                if self.vtk.calls == 2:
                    break
                time.sleep(0.02)
            # two builds run: the global limit (2) is reached for carol
            status, _, _ = carol.request("GET", "/download_vtk?domain=one")
            self.assertEqual(status, 429)
            gate.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        # a cached archive is served whatever is running
        self.assertEqual(alice.request("GET", "/download_vtk?domain=one")[0], 200)
        self.assertEqual(carol.request("GET", "/download_vtk?domain=one")[0], 200)

    def test_a_failed_build_leaves_nothing_and_answers_500(self):
        browser = self.browser_with_runs("site")
        with patch.object(app, "export_prediction_vtk", side_effect=RuntimeError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()):  # the logged traceback
            status, _, body = browser.request("GET", "/download_vtk?domain=site")
        self.assertEqual(status, 500, body)
        self.assertEqual([p.name for p in self.results.rglob("*.zip")], [])


if __name__ == "__main__":
    unittest.main()
