"""Saved runs are kept per session, with a global size cap that evicts the least
recently used runs first.

The upstream retention (PINN_WEBAPP_MAX_RUNS newest runs of all users) let one
session's predictions push out another session's results, and an 8-sector wind
rose pushed out its own first sectors. Now each session keeps its newest
PINN_WEBAPP_MAX_RUNS_PER_SESSION runs, all sessions together stay below
PINN_WEBAPP_MAX_GB (least recently used first, across sessions), a wind rose and its
sectors count as one run, and nothing in use is removed.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp", ROOT / "pinnfluid" / "domain_prep",
             ROOT / "pinnfluid" / "input_prep", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402
from webapp import results_io  # noqa: E402
from pinnfluid_http import Browser, Server  # noqa: E402

A, B = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"
MB = 1024 * 1024


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results = Path(self.tmp.name) / "results"
        self.results.mkdir()
        patcher = patch.object(app, "RESULTS_DIR", self.results)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.now = time.time()
        self.env = {"PINN_WEBAPP_MAX_RUNS_PER_SESSION": "10", "PINN_WEBAPP_MAX_GB": "15"}

    def run_dir(self, ns: str, name: str, age_s: float, *, size: int = 1000, inputs: bool = True,
                rose: bool = False) -> Path:
        d = self.results / f"{ns}_{name}"
        d.mkdir()
        if inputs:
            (d / "inputs").mkdir()
            (d / "inputs" / "flow.npz").write_bytes(b"x" * size)
        if rose:
            (d / "rose_summary.json").write_text("{}")
        t = self.now - age_s
        os.utime(d, (t, t))
        return d

    def enforce(self, keep: str):
        with patch.dict(os.environ, self.env):
            app._enforce_retention(keep)

    def names(self) -> set:
        return {p.name for p in self.results.iterdir()}

    def test_each_session_keeps_its_newest_runs(self):
        a_runs = [self.run_dir(A, f"r{i}", 1000 - i) for i in range(12)]  # r0 oldest
        b_runs = [self.run_dir(B, f"r{i}", 5000 - i) for i in range(3)]  # older than all of A's
        self.enforce(keep=a_runs[-1].name)
        self.assertEqual([d.exists() for d in a_runs], [False, False] + [True] * 10)
        self.assertTrue(all(d.exists() for d in b_runs), "another session's runs were removed")

    def test_a_wind_rose_counts_as_one_run_and_never_evicts_its_sectors(self):
        self.env["PINN_WEBAPP_MAX_RUNS_PER_SESSION"] = "3"
        older = [self.run_dir(A, f"old{i}", 900 - i) for i in range(3)]
        base = self.run_dir(A, "rose", 100, inputs=False, rose=True)
        sectors = [self.run_dir(A, f"rose_r{d:03d}", 200 - i) for i, d in enumerate(range(0, 360, 45))]
        self.enforce(keep=base.name)
        self.assertTrue(base.exists())
        self.assertTrue(all(s.exists() for s in sectors), "a rose evicted its own sectors")
        # the rose is one run: of the three older runs, the newest two stay
        self.assertEqual([d.exists() for d in older], [False, True, True])

    def test_a_rose_in_progress_is_kept(self):
        self.env["PINN_WEBAPP_MAX_RUNS_PER_SESSION"] = "1"
        sectors = [self.run_dir(A, f"rose_r{d:03d}", 300 + d) for d in (0, 90, 180)]  # no summary yet
        other = self.run_dir(A, "newer", 10)
        with app._busy(f"{A}_rose"):
            self.enforce(keep=other.name)
        self.assertTrue(all(s.exists() for s in sectors))

    def test_the_size_cap_evicts_the_least_recently_used_runs_of_any_session(self):
        self.env["PINN_WEBAPP_MAX_GB"] = str(3.5 * MB / 1024 ** 3)  # room for three runs
        b_old = self.run_dir(B, "old", 500, size=MB)
        a_mid = self.run_dir(A, "mid", 400, size=MB)
        b_new = self.run_dir(B, "new", 300, size=MB)
        a_kept = self.run_dir(A, "current", 600, size=MB)  # oldest, but the run just saved
        self.enforce(keep=a_kept.name)
        self.assertEqual(self.names(), {a_mid.name, b_new.name, a_kept.name})
        self.assertFalse(b_old.exists())

    def test_runs_in_use_are_kept(self):
        self.env["PINN_WEBAPP_MAX_RUNS_PER_SESSION"] = "1"
        busy = self.run_dir(A, "busy", 900)
        newer = self.run_dir(A, "newer", 10)
        with app._busy(busy.name):
            self.enforce(keep=newer.name)
        self.assertTrue(busy.exists())

    def test_work_dir_pruning_leaves_roses_to_retention(self):
        base = self.run_dir(A, "rose", 10 * 3600, inputs=False, rose=True)
        with patch.object(app, "DEM_ROOT", self.results / "none"), \
                patch.object(app, "WORKSPACE", self.results / "none"), \
                patch.dict(os.environ, {"PINN_WEBAPP_WORK_TTL_S": "3600"}):
            app._prune_work_dirs()
        self.assertTrue(base.exists())

    def test_opening_a_run_counts_as_using_it(self):
        server = Server(app)
        self.addCleanup(server.close)
        browser = Browser(server)
        browser.request("GET", "/")
        ns = app._session_namespace(browser.cookie.split("=", 1)[1])
        old = self.run_dir(ns, "old", 900)
        (old / "report.pdf").write_bytes(b"%PDF")
        newer = self.run_dir(ns, "newer", 500)
        current = self.run_dir(ns, "current", 1)
        self.assertEqual(browser.request("GET", "/download_pdf?domain=old")[0], 200)
        self.env["PINN_WEBAPP_MAX_RUNS_PER_SESSION"] = "2"
        self.enforce(keep=current.name)
        self.assertTrue(old.exists(), "the run just opened was evicted")
        self.assertFalse(newer.exists())


class WiringTests(unittest.TestCase):
    def test_the_app_saves_without_the_global_retention(self):
        tree = ast.parse((ROOT / "pinnfluid" / "webapp" / "app.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "save_inputs_and_predictions"]
        self.assertTrue(calls)
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            self.assertIn("retention", kw, call.lineno)
            self.assertIs(kw["retention"].value, False)
        enforced = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "_enforce_retention"]
        self.assertGreaterEqual(len(enforced), 3)  # after a prediction, a wind rose and an export

    def test_save_can_skip_the_global_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfd = root / "cfd"
            cfd.mkdir()
            (cfd / "meta.json").write_text("{}")
            np.savez_compressed(cfd / "flow.npz", **{k: np.zeros((1, 1, 1), np.float32)
                                                     for k in ("Ux", "Uy", "Uz", "p", "is_fluid")})
            out = {"bundle": SimpleNamespace(is_fluid=np.ones((1, 1, 1), np.float32)),
                   "pred_flow": np.zeros((1, 1, 1, 4), np.float32), "roi_bundles": {}, "roi_preds": {}}
            with patch.object(results_io, "enforce_results_retention") as retention:
                results_io.save_inputs_and_predictions(root / "results", "x", case_dir=root / "case",
                                                       cfd_dir=cfd, predict_out=out,
                                                       pressure_reference_kinematic=0.0, retention=False)
                retention.assert_not_called()
                results_io.save_inputs_and_predictions(root / "results", "y", case_dir=root / "case",
                                                       cfd_dir=cfd, predict_out=out,
                                                       pressure_reference_kinematic=0.0)
                retention.assert_called_once()


if __name__ == "__main__":
    unittest.main()
