# Vendored map libraries

The web app serves Leaflet and leaflet-draw itself, from this directory, at
`static/vendor/...` (relative to the page), instead of loading them from the
unpkg CDN. The page then works offline, under a strict content-security policy
and under `PINN_WEBAPP_BASE_PATH`.

The files are not in the repository. A deployment (the container image) puts
them here from pinned downloads whose hashes it checks; for a local checkout,
fetch the same files by hand. Expected layout and sources:

| file | source |
| --- | --- |
| `leaflet/leaflet.js` | https://unpkg.com/leaflet@1.9.4/dist/leaflet.js |
| `leaflet/leaflet.css` | https://unpkg.com/leaflet@1.9.4/dist/leaflet.css |
| `leaflet/images/layers.png`, `layers-2x.png`, `marker-icon.png`, `marker-icon-2x.png`, `marker-shadow.png` | https://unpkg.com/leaflet@1.9.4/dist/images/ |
| `leaflet/LICENSE` | https://unpkg.com/leaflet@1.9.4/LICENSE (BSD 2-Clause) |
| `leaflet-draw/leaflet.draw.js` | https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js |
| `leaflet-draw/leaflet.draw.css` | https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css |
| `leaflet-draw/images/spritesheet.png`, `spritesheet-2x.png`, `spritesheet.svg` | https://unpkg.com/leaflet-draw@1.0.4/dist/images/ |
| `leaflet-draw/LICENSE` | `MIT-LICENSE.md` of https://github.com/Leaflet/Leaflet.draw at tag v1.0.4 (MIT) |

The CSS files load their images from `images/` next to themselves, and Leaflet
finds its marker icons the same way. Only `.js`, `.css`, `.png`, `.svg` and
`LICENSE` files below this directory are served. The app prints a warning at
startup when the four `.js`/`.css` files are missing.

Without the files, `PINN_WEBAPP_LEAFLET_CDN=1` makes the page load Leaflet and
leaflet-draw from the unpkg CDN instead, as the page did before (off by default).
