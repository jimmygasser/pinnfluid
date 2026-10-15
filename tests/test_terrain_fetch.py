"""Terrain fetches and rasters: redirects stay on the swisstopo host, and unusual
nodata values and block layouts go through the merges.

- STAC pages and tiles are fetched with an opener that refuses a redirect to another
  host or to http (the host check covers only the URLs the app starts from).
- A float64 DEM whose nodata lies outside the float32 range (ArcGIS writes
  -1.797e308) merges, with its nodata mapped to -9999.
- merge_and_crop (the swisstopo tiles) writes fresh float32 profiles, like merge_tifs.
"""

from __future__ import annotations

import ast
import email.message
import io
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp", ROOT / "pinnfluid" / "domain_prep",
             ROOT / "pinnfluid" / "input_prep", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

import dem_prep  # noqa: E402
import domain_builder  # noqa: E402

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:  # pragma: no cover
    rasterio = None

DBL_MAX = 1.7976931348623157e308


class RedirectTests(unittest.TestCase):
    def redirect(self, newurl: str):
        handler = domain_builder._StacRedirectHandler()
        req = urllib.request.Request("https://data.geo.admin.ch/api/stac/v1/collections/x/items")
        return handler.redirect_request(req, io.BytesIO(), 302, "Found", email.message.Message(), newurl)

    def test_redirects_to_another_host_or_to_http_are_refused(self):
        for target in ("https://evil.example/x.tif", "http://data.geo.admin.ch/x.tif",
                       "https://data.geo.admin.ch.evil.example/x", "https://data.geo.admin.ch:8443/x",
                       "file:///etc/passwd"):
            with self.assertRaises(urllib.error.HTTPError, msg=target):
                self.redirect(target)

    def test_redirects_on_the_host_are_followed(self):
        new = self.redirect("https://data.geo.admin.ch/ch.swisstopo.swissalti3d/y.tif")
        self.assertEqual(new.full_url, "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/y.tif")

    def test_every_swisstopo_fetch_uses_that_opener(self):
        self.assertTrue(any(isinstance(h, domain_builder._StacRedirectHandler)
                            for h in domain_builder._STAC_OPENER.handlers))
        tree = ast.parse((ROOT / "pinnfluid" / "domain_prep" / "domain_builder.py").read_text())
        direct = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                  and n.attr in ("urlopen", "urlretrieve")]
        self.assertEqual(direct, [], "urllib.request.urlopen/urlretrieve follow any redirect")


def write_tif(path: Path, data: np.ndarray, *, crs: str, nodata=None, **creation) -> None:
    h, w = data.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype=data.dtype,
                       crs=crs, transform=from_origin(2600000.0, 1200000.0, 2.0, 2.0), nodata=nodata,
                       **creation) as dst:
        dst.write(data, 1)


@unittest.skipIf(rasterio is None, "needs rasterio")
class NodataAndProfileTests(unittest.TestCase):
    def test_a_float64_dem_with_nodata_outside_float32(self):
        data = (1000 + np.arange(40 * 30, dtype="float64").reshape(30, 40) % 50)
        data[0, :5] = -DBL_MAX
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in.tif", Path(tmp) / "merged.tif"
            write_tif(src, data, crs="EPSG:32632", nodata=-DBL_MAX)
            dem_prep.merge_tifs([str(src)], str(out))
            with rasterio.open(out) as merged:
                self.assertEqual(merged.nodata, -9999.0)
                values = merged.read(1)
        np.testing.assert_array_equal(values[0, :5], -9999.0)
        np.testing.assert_allclose(values[1:], data[1:].astype("float32"))

    def test_merge_and_crop_takes_striped_tiles(self):
        data = (500 + np.arange(150 * 120, dtype="float32").reshape(120, 150) % 70)
        with tempfile.TemporaryDirectory() as tmp:
            write_tif(Path(tmp) / "swissalti3d_2019_2600-1200_2_2056_5728.tif", data, crs="EPSG:2056",
                      tiled=False, blockysize=8)
            path, w_m, h_m = domain_builder.merge_and_crop(tmp, 2600010.0, 1199810.0, 2600210.0, 1199990.0)
            with rasterio.open(path) as cropped:
                self.assertEqual(cropped.dtypes[0], "float32")
                self.assertEqual((cropped.width, cropped.height), (100, 90))
                np.testing.assert_allclose(cropped.read(1), data[5:95, 5:105])
        self.assertEqual((w_m, h_m), (200.0, 180.0))


if __name__ == "__main__":
    unittest.main()
