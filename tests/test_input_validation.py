"""Input validation at the public boundary.

- User files are opened as GeoTIFF only: GDAL would otherwise pick the format
  from the content, and a VRT upload could read local files into the preview PNG
  or fetch /vsicurl/ URLs.
- Structure and grid types are names of the shipped single_stl/*.stl files.
- An uploaded raster's pixel count is checked from its header, before any pixel
  is decoded (a small compressed file can expand to gigabytes).
- DEM tiles come only from the swisstopo STAC host.
- Uploaded GeoTIFFs with any block layout (striped, as QGIS writes them, or tile
  sizes that are not multiples of 16) go through the terrain preparation.
"""

from __future__ import annotations

import ast
import base64
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
import dem_prep  # noqa: E402
import domain_builder  # noqa: E402

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:  # pragma: no cover - rasterio is a web-app requirement
    rasterio = None

BAD_TYPES = ("../../../etc/passwd", "/etc/passwd", "nosuch.stl", "README.md",
             "ABL_domainsize_possibilities.txt", "single_stl/cube.stl", "cube", "", 7)


def geotiff_bytes(width: int, height: int, *, pixel: float = 2.0, dtype: str = "float32",
                  **creation) -> bytes:
    data = (1000 + np.arange(width * height, dtype="float64").reshape(height, width) % 97)
    with rasterio.io.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=width, height=height, count=1, dtype=dtype,
                      crs="EPSG:32632", transform=from_origin(500000.0, 5200000.0, pixel, pixel),
                      **creation) as dst:
            dst.write(data.astype(dtype), 1)
        return mem.read()


class StructureTypeTests(unittest.TestCase):
    BODY = {"domain_size": 1000, "wind_from": 270}

    def test_shipped_types_are_accepted(self):
        app._validate_compute_request(dict(self.BODY, structures=[{"type": "cube.stl"}]))
        app._validate_compute_request(dict(self.BODY, grid={"type": "flatpanel.stl", "rows": 2,
                                                            "cols": 2}))
        self.assertIn("cube.stl", domain_builder.STRUCTURE_TYPES)
        self.assertEqual(domain_builder.structure_stl_path("cube.stl"),
                         ROOT / "single_stl" / "cube.stl")

    def test_other_types_are_refused(self):
        for bad in BAD_TYPES:
            with self.assertRaisesRegex(ValueError, "structure type", msg=bad):
                app._validate_compute_request(dict(self.BODY, structures=[{"type": bad}]))
            with self.assertRaisesRegex(ValueError, "structure type", msg=bad):
                app._validate_compute_request(dict(self.BODY, grid={"type": bad, "rows": 1,
                                                                    "cols": 1}))
            with self.assertRaises(ValueError, msg=bad):
                domain_builder.structure_stl_path(bad)

    def test_domain_names_have_a_length_limit(self):
        ns = "0123456789abcdef"
        self.assertEqual(app._session_name(ns, "a" * 64), f"{ns}_" + "a" * 64)
        with self.assertRaisesRegex(ValueError, "domain name"):
            app._session_name(ns, "a" * 65)
        # on-disk names (namespace, name, wind-rose suffix) stay within the general cap
        app._safe_name(f"{ns}_" + "a" * 64 + "_r090")
        with self.assertRaisesRegex(ValueError, "domain name"):
            app._safe_name("a" * 129)


class HttpValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.Handler.page_html = app._build_html()
        cls.server = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dem = Path(self.tmp.name) / "dem"
        self.dem.mkdir()
        for target, attr, value in ((app.Handler, "dem_root", str(self.dem)),
                                    (build_inputs, "DEM_ROOT", self.dem)):
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        with app._JOBS_LOCK:
            app._PREP_SUBMISSIONS.clear()
            app._PREP_ACTIVE = 0

    def post(self, path: str, body: dict):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=60)
        try:
            conn.request("POST", path, body=json.dumps(body),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, json.loads(r.read())
        finally:
            conn.close()

    def upload(self, data: bytes):
        return self.post("/upload_dem", {"domain_name": "upload",
                                         "data_base64": base64.b64encode(data).decode()})

    def test_bad_structure_types_give_400(self):
        body = {"domain_name": "t", "domain_size": 1000, "wind_from": 270}
        for route in ("/build_inputs", "/predict", "/predict_rose"):
            for bad in ("../../../etc/passwd", "/etc/passwd", "nosuch.stl"):
                status, payload = self.post(route, dict(body, structures=[{"type": bad}]))
                self.assertEqual(status, 400, (route, bad, payload))
                self.assertIn("structure type", payload["error"])
                status, _ = self.post(route, dict(body, grid={"type": bad, "rows": 1, "cols": 1}))
                self.assertEqual(status, 400, (route, bad))

    @unittest.skipIf(rasterio is None, "needs rasterio")
    def test_a_vrt_upload_is_refused(self):
        """A VRT that pulls a local raster (here a GeoTIFF on the server's disk) must
        not be opened, so no preview of the local file comes back."""
        local = Path(self.tmp.name) / "server-local.tif"
        local.write_bytes(geotiff_bytes(64, 64))
        vrt = f"""<VRTDataset rasterXSize="64" rasterYSize="64">
  <SRS>EPSG:32632</SRS>
  <GeoTransform>500000, 2, 0, 5200000, 0, -2</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{local}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
<!-- {"padding " * 200} -->
""".encode()
        status, payload = self.upload(vrt)
        self.assertEqual(status, 400, {k: v for k, v in payload.items() if k != "png_base64"})
        self.assertNotIn("png_base64", payload)
        self.assertFalse(payload["success"])
        self.assertNotIn(str(local), payload["error"])

    @unittest.skipIf(rasterio is None, "needs rasterio")
    def test_a_compressed_raster_over_the_pixel_cap_is_refused_before_decoding(self):
        bomb = geotiff_bytes(3000, 3000, pixel=1.0, compress="deflate", tiled=True,
                             blockxsize=512, blockysize=512)
        self.assertLess(len(bomb), 20 * 1024 * 1024)  # well inside the upload limit
        with patch.object(rasterio.io.DatasetReader, "read",
                          side_effect=AssertionError("decoded")):
            status, payload = self.upload(bomb)
        self.assertEqual(status, 400, payload)
        self.assertIn("pixel limit", payload["error"])
        self.assertNotIn("png_base64", payload)

    @unittest.skipIf(rasterio is None, "needs rasterio")
    def test_the_pixel_cap_is_configurable(self):
        small = geotiff_bytes(100, 100)
        with patch.dict(os.environ, {"PINN_WEBAPP_MAX_DEM_PIXELS": "9999"}):
            status, payload = self.upload(small)
        self.assertEqual(status, 400, payload)
        with patch.dict(os.environ, {"PINN_WEBAPP_MAX_DEM_PIXELS": "10000"}):
            status, payload = self.upload(small)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["png_base64"])


class GdalEnvironmentTests(unittest.TestCase):
    def test_safe_gdal_defaults(self):
        self.assertEqual(os.environ.get("GDAL_DISABLE_READDIR_ON_OPEN"), "EMPTY_DIR")
        self.assertEqual(os.environ.get("CPL_VSIL_CURL_ALLOWED_EXTENSIONS"), ".none")

    def test_every_raster_read_names_the_gtiff_driver(self):
        """Every rasterio.open() that reads in the terrain pipeline passes driver="GTiff"
        (writes give the driver in their profile)."""
        files = [ROOT / "pinnfluid" / "domain_prep" / name
                 for name in ("domain_builder.py", "dem_prep.py", "place_structure.py")]
        files += [ROOT / "pinnfluid" / "input_prep" / "common.py",
                  ROOT / "pinnfluid" / "vis" / "domain_report.py"]
        reads = 0
        for path in files:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "open" and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "rasterio"):
                    continue
                mode = node.args[1].value if len(node.args) > 1 and isinstance(
                    node.args[1], ast.Constant) else "r"
                if mode != "r":
                    continue
                reads += 1
                drivers = [k.value.value for k in node.keywords
                           if k.arg == "driver" and isinstance(k.value, ast.Constant)]
                self.assertEqual(drivers, ["GTiff"], f"{path.name}:{node.lineno}")
        self.assertGreaterEqual(reads, 15)


class StacHostTests(unittest.TestCase):
    @staticmethod
    def stac_page(hrefs, next_href=None):
        features = [{"assets": {f"swissalti3d_2019_{2600 + i}-1200_2_2056_5728.tif": {"href": h}}}
                    for i, h in enumerate(hrefs)]
        links = [{"rel": "next", "href": next_href}] if next_href else []
        return json.dumps({"features": features, "links": links}).encode()

    def test_only_swisstopo_tiles_are_used(self):
        good = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/a/a_2_2056_5728.tif"
        pages = [self.stac_page([good, "https://evil.example/x.tif",
                                 "http://data.geo.admin.ch/plain-http.tif",
                                 "https://data.geo.admin.ch.evil.example/x.tif",
                                 "file:///etc/passwd", "/vsicurl/https://evil.example/x"],
                                next_href="https://evil.example/stac?page=2")]
        opened = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            opened.append(req.full_url)
            return Response(pages[len(opened) - 1])

        with patch.object(domain_builder.urllib.request, "urlopen", side_effect=fake_urlopen):
            tiles = domain_builder.query_stac_tiles(7.0, 46.0, 7.01, 46.01)
        self.assertEqual([t["href"] for t in tiles], [good])
        self.assertEqual(len(opened), 1, opened)  # the foreign "next" page is not fetched

    def test_downloads_refuse_other_hosts(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(domain_builder.urllib.request, "urlretrieve") as retrieve:
            downloaded, failed = domain_builder.download_tiles(
                [{"name": "swissalti3d_x.tif", "href": "https://evil.example/x.tif"}], tmp)
        self.assertEqual((downloaded, failed), (0, 1))
        retrieve.assert_not_called()


@unittest.skipIf(rasterio is None, "needs rasterio")
class BlockLayoutTests(unittest.TestCase):
    """dem_prep rewrote the merged DEM with tiled=True and the source's block size:
    a striped GeoTIFF (QGIS and GDAL write strips by default) has blocks as wide as
    the image and a few rows high, which are not valid tile sizes (multiples of 16),
    and failed with "Bad value ... for TileWidth/TileLength"."""

    def check_merge(self, data: bytes):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.tif"
            src.write_bytes(data)
            out = Path(tmp) / "merged.tif"
            dem_prep.merge_tifs([str(src)], str(out))
            with rasterio.open(src) as a, rasterio.open(out) as b:
                np.testing.assert_allclose(b.read(1), a.read(1).astype("float32"))
                self.assertEqual(b.transform, a.transform)
                self.assertEqual(b.crs, a.crs)
                self.assertEqual(b.dtypes[0], "float32")

    def test_a_striped_geotiff(self):
        striped = geotiff_bytes(150, 120, tiled=False, blockysize=8)
        with rasterio.io.MemoryFile(striped) as mem, mem.open() as src:
            self.assertEqual(src.block_shapes[0], (8, 150))  # strips, 150 wide
        self.check_merge(striped)

    def test_one_row_strips(self):
        self.check_merge(geotiff_bytes(64, 40, tiled=False, blockysize=1))

    def test_an_integer_dem_with_nodata(self):
        self.check_merge(geotiff_bytes(90, 90, dtype="int16", nodata=-32768, tiled=True,
                                       blockxsize=16, blockysize=16))


if __name__ == "__main__":
    unittest.main()
