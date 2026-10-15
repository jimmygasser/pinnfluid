"""Everything the web app writes goes below PINN_WEBAPP_WORK_DIR.

With PINN_WEBAPP_WORK_DIR set, the DEM cache (dem/), the build workspace
(workspace/) and, unless PINN_WEBAPP_RESULTS_DIR says otherwise, the saved runs
(results/) live below it, so the app runs on a read-only source tree (a
container with a read-only root and one writable directory). Without it the
layout is the historical one, inside the checkout.

The paths are read once at import, so each test runs the app in a fresh
interpreter with its own environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "pinnfluid" / "webapp" / "checkpoints"
HYBRID = ("hybrid-stage1-292d.pth", "hybrid-stage2-292d.pth")

_PRELUDE = textwrap.dedent(f"""
    import json, sys
    from pathlib import Path
    ROOT = Path({str(ROOT)!r})
    for p in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp",
              ROOT / "pinnfluid" / "domain_prep", ROOT / "pinnfluid" / "input_prep"):
        sys.path.insert(0, str(p))
""")

_PATHS = _PRELUDE + textwrap.dedent("""
    from webapp import app
    import build_inputs
    paths = build_inputs._case_paths("case")
    print(json.dumps({
        "results": str(app.RESULTS_DIR),
        "dem_root": str(app.Handler.dem_root),
        "workspace": str(build_inputs.WORKSPACE),
        "case_dem": str(paths["dem_dir"]),
        "case_dir": str(paths["case_dir"]),
        "cfd_dir": str(paths["cfd_dir"]),
    }))
""")

# The whole workflow through the HTTP boundary: upload a small projected GeoTIFF,
# build the inputs for one structure, predict with the hybrid model (about 30 s on
# a CPU), and open the map and the 3D view (rendered and cached on first open).
_FLOW = _PRELUDE + textwrap.dedent("""
    import base64, http.client, threading, time
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from webapp import app

    x0, y0, px, n = 500000.0, 5200000.0, 2.0, 200
    yy, xx = np.mgrid[0:n, 0:n].astype("float32")
    z = 1000.0 + 0.05 * xx + 3.0 * np.sin(xx / 25.0) * np.cos(yy / 30.0)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=n, height=n, count=1, dtype="float32",
                      crs="EPSG:32632", transform=from_origin(x0, y0, px, px),
                      # tiled: dem_prep rewrites the raster with tiled=True and the
                      # source's block size, which must then be a multiple of 16
                      tiled=True, blockxsize=128, blockysize=128) as dst:
            dst.write(z, 1)
        tif = mem.read()

    server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    jar = {}  # one browser session: the cookie of the first response

    def send(method, path, payload=None, timeout=600):
        conn = http.client.HTTPConnection(*server.server_address, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if jar:
            headers["Cookie"] = jar["cookie"]
        conn.request(method, path, body=None if payload is None else json.dumps(payload),
                     headers=headers)
        r = conn.getresponse()
        if r.getheader("Set-Cookie"):
            jar["cookie"] = r.getheader("Set-Cookie").split(";", 1)[0]
        return r.status, r.read()

    def post(path, payload):
        status, body = send("POST", path, payload)
        return status, json.loads(body)

    out = {}
    out["upload"] = post("/upload_dem", {
        "domain_name": "wd-case",
        "data_base64": base64.b64encode(tif).decode("ascii"),
    })
    cx, cy = x0 + n * px / 2, y0 - n * px / 2
    BUILD_BODY = {
        "domain_name": "wd-case", "domain_size": 200, "wind_from": 270,
        "flat_terrain": False, "uref": 10, "zref": 20, "z0": 0.1,
        "flatten_ground": True, "grid": None, "sampling_points": [],
        "custom_center_crs": {"x": cx, "y": cy},
        "structures": [{"type": "cube.stl", "yaw": 0, "crs_x": cx, "crs_y": cy}],
    }
    out["build"] = post("/build_inputs", BUILD_BODY)
    out["predict"] = post("/predict", BUILD_BODY)
    job = {}
    for _ in range(1200):
        _, body = send("GET", "/job_status?id=" + out["predict"][1].get("job_id", ""), timeout=30)
        job = json.loads(body)
        if job.get("state") in ("done", "error"):
            break
        time.sleep(0.5)
    out["job"] = [job.get("state"), job.get("error")]
    # the viewers render on first open and are cached next to the run
    for view in ("map", "view_3d"):
        status, body = send("GET", f"/{view}?domain=wd-case")
        out[view] = [status, 'src="static/plotly.min.js"' in body.decode("utf-8", "replace")]
    out["namespace"] = app._session_namespace(jar["cookie"].split("=", 1)[1])
    server.shutdown()
    for step in ("upload", "build"):
        status, body = out[step]
        out[step] = [status, {k: v for k, v in body.items() if k != "png_base64"}]
    print(json.dumps(out))
""")


def run_app(script: str, **env_values) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PINN_WEBAPP_WORK_DIR", "PINN_WEBAPP_RESULTS_DIR")}
    env.update({k: str(v) for k, v in env_values.items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-c", script], env=env, cwd=str(ROOT),
                          capture_output=True, text=True, timeout=900)


def snapshot(root: Path) -> dict:
    """Every file of the tree (but .git and byte-code caches) with size and mtime."""
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in filenames:
            p = Path(dirpath) / name
            st = p.stat()
            files[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return files


def last_json_line(proc: subprocess.CompletedProcess) -> dict:
    if proc.returncode != 0:
        raise AssertionError(f"app process failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# Importing the app loads the model registry, which needs checkpoints (as for
# every test module that imports it).
_NO_CHECKPOINTS = "" if any(CHECKPOINTS.glob("*.pth")) else (
    f"needs model checkpoints in {CHECKPOINTS}; run pinnfluid/webapp/fetch_checkpoints.py")


@unittest.skipIf(_NO_CHECKPOINTS, _NO_CHECKPOINTS)
class WorkDirPathTests(unittest.TestCase):
    def test_writable_paths_are_below_the_work_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            paths = last_json_line(run_app(_PATHS, PINN_WEBAPP_WORK_DIR=work))
        self.assertEqual(paths["dem_root"], str(work / "dem"))
        self.assertEqual(paths["workspace"], str(work / "workspace"))
        self.assertEqual(paths["results"], str(work / "results"))
        self.assertEqual(paths["case_dem"], str(work / "dem" / "case"))
        for key in ("case_dir", "cfd_dir"):
            self.assertTrue(paths[key].startswith(str(work / "workspace") + os.sep), paths[key])

    def test_an_explicit_results_dir_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = last_json_line(run_app(
                _PATHS, PINN_WEBAPP_WORK_DIR=Path(tmp) / "work",
                PINN_WEBAPP_RESULTS_DIR=Path(tmp) / "runs"))
        self.assertEqual(paths["results"], str(Path(tmp) / "runs"))
        self.assertEqual(paths["dem_root"], str(Path(tmp) / "work" / "dem"))

    def test_without_a_work_dir_the_checkout_layout_is_kept(self):
        paths = last_json_line(run_app(_PATHS))
        self.assertEqual(paths["dem_root"], str(ROOT / "dem"))
        self.assertEqual(paths["workspace"], str(ROOT / "pinnfluid" / "webapp" / "workspace"))
        self.assertEqual(paths["results"], str(ROOT / "pinnfluid" / "webapp" / "results"))

    def test_a_relative_work_dir_is_refused(self):
        proc = run_app(_PATHS, PINN_WEBAPP_WORK_DIR="relative/work")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PINN_WEBAPP_WORK_DIR must be an absolute path", proc.stderr)


def _flow_prerequisites() -> str:
    """Why the workflow test cannot run here, or '' when it can."""
    missing = [name for name in HYBRID if not (CHECKPOINTS / name).is_file()]
    if missing:
        return (f"needs the hybrid checkpoints ({', '.join(missing)} missing in "
                f"{CHECKPOINTS}); run pinnfluid/webapp/fetch_checkpoints.py")
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return "needs rasterio (requirements-webapp.txt)"
    return ""


@unittest.skipIf(_flow_prerequisites(), _flow_prerequisites())
class WorkDirFlowTests(unittest.TestCase):
    def test_the_workflow_writes_only_below_the_work_dir(self):
        before = snapshot(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            out = last_json_line(run_app(_FLOW, PINN_WEBAPP_WORK_DIR=work))
            written = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
        self.assertEqual(out["upload"][0], 200, out)
        self.assertTrue(out["upload"][1]["success"], out)
        self.assertEqual(out["build"][0], 200, out)
        self.assertTrue(out["build"][1]["success"], out)
        self.assertTrue(out["build"][1]["cfd_dir"].startswith(str(work / "workspace")), out)
        case = f"{out['namespace']}_wd-case"  # the session's name for it
        self.assertIn(os.path.join("dem", case, "dem_cropped.tif"), written)
        # the build happened in workspace/ (cfd_dir above), which predict cleans up
        self.assertEqual([p for p in written if p.startswith("workspace" + os.sep)], [], written)
        self.assertEqual(out["job"], ["done", None], out)
        self.assertIn(os.path.join("results", case, "report.pdf"), written)
        self.assertIn(os.path.join("results", case, "map.html"), written)
        self.assertIn(os.path.join("results", case, "view_3d.html"), written)
        self.assertEqual(out["map"], [200, True], out)
        self.assertEqual(out["view_3d"], [200, True], out)
        self.assertEqual(snapshot(ROOT), before, "files written or changed in the source tree")


if __name__ == "__main__":
    unittest.main()
