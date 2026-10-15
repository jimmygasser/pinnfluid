"""Optional integration into a larger site; everything is off unless configured.

A deployment that embeds the app in a larger site can give it:

    PINN_WEBAPP_KIT_DIR    a directory with the site's design kit: tokens.css (design
                           tokens as CSS custom properties) and cryosim-bar.js (a
                           <cryosim-bar> custom element, the site's top bar), plus an
                           optional VERSION. The page then shows that bar above the map,
                           with its own links in the bar's nav (a bottom tab bar on
                           phones), and the files are served at static/kit/.
    PINN_WEBAPP_KIT_HOME   where the bar's home link goes (default /).
    PINN_WEBAPP_CSP        a Content-Security-Policy header for every response.
    PINN_WEBAPP_ROBOTS_TAG an X-Robots-Tag header for every response (e.g. noindex).
    PINN_WEBAPP_INFO_DIR   a directory with build information, served as version.json,
                           NOTICES.txt (third-party notices) and source/ (the source
                           archives, *.tar.gz in its source/ directory, with an index).

Without these settings the app looks and answers exactly as without this module.
Every URL it adds is relative, like the rest of the app, so it works under
PINN_WEBAPP_BASE_PATH.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Optional

KIT_FILES = {
    "tokens.css": "text/css; charset=utf-8",
    "cryosim-bar.js": "text/javascript; charset=utf-8",
    "VERSION": "text/plain; charset=utf-8",
}
_ARCHIVE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz")

# The page under the bar: the map, the sidebar and the report panel start below it.
# The bar is --appbar-h high plus its 1 px border (the kit's tokens). Below 640 px
# the bar's nav becomes a bottom tab bar, as on the site's other pages.
SHELL_CSS = """
:root { --site-bar-h: calc(var(--appbar-h, 56px) + 1px); --site-tabs-h: 0px; color-scheme: light; }
#sidebar { position:fixed; }  /* its phone drawer waits off-screen to the right: no sideways scroll */
cryosim-bar { position:fixed; top:0; left:0; right:0; z-index:2200; }
#map, #sidebar { top:var(--site-bar-h); height:calc(100vh - var(--site-bar-h) - var(--site-tabs-h)); }
#sb-toggle { top:calc(50% + var(--site-bar-h) / 2); }
#predict-panel { top:var(--site-bar-h) !important; height:calc(100vh - var(--site-bar-h) - var(--site-tabs-h)) !important; }
#status { bottom:calc(12px + var(--site-tabs-h)); }
#sidebar-header { background:var(--primary, #1565C0); }
.btn-primary { background:var(--primary, #1976D2); }
.btn-primary:hover { background:var(--primary-hover, #1565C0); }
.pnav a { display:inline-flex; align-items:center; min-height:2.25rem; padding:0 var(--space-3, 12px);
  border-radius:var(--radius, 8px); color:var(--muted, #56636f); font-weight:500; text-decoration:none;
  white-space:nowrap; }
.pnav a:hover { background:var(--surface, #f5f8fb); color:var(--fg, #14212b); }
.pnav a[aria-current=page] { background:var(--primary-soft, #e8f3fb); color:var(--primary, #1565C0); font-weight:600; }
:focus-visible { outline:2px solid var(--focus, #1565C0); outline-offset:2px; }
@media (max-width:640px) {
  :root { --site-tabs-h: var(--bottomnav-h, 64px); }
  cryosim-bar::part(nav) { display:contents; }
  .pnav { position:fixed; z-index:2200; inset:auto 0 0 0; height:var(--site-tabs-h); display:flex; gap:0;
    align-items:stretch; background:var(--raised, #fff); border-top:1px solid var(--border, #d5dde5); }
  .pnav a { flex:1; justify-content:center; min-height:0; padding:0 var(--space-1, 4px); border-radius:0;
    font-size:var(--text-xs, 12px); white-space:normal; text-align:center; }
}
"""


def _dir_from_env(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def kit_dir() -> Optional[Path]:
    """The kit directory, if it is set and holds the stylesheet and the bar."""
    d = _dir_from_env("PINN_WEBAPP_KIT_DIR")
    if d is None or not all((d / f).is_file() for f in ("tokens.css", "cryosim-bar.js")):
        return None
    return d


def info_dir() -> Optional[Path]:
    d = _dir_from_env("PINN_WEBAPP_INFO_DIR")
    return d if d is not None and d.is_dir() else None


def apply_kit(page: str, *, github_url: str) -> str:
    """The page with the site's bar and tokens, or unchanged without a kit."""
    if kit_dir() is None:
        return page
    home = html.escape(os.environ.get("PINN_WEBAPP_KIT_HOME", "").strip() or "/", quote=True)
    links = ['<a href="./" aria-current="page">Predict</a>']
    if info_dir() is not None:
        links.append('<a href="source/">Source &amp; notices</a>')
    links.append(f'<a href="{html.escape(github_url, quote=True)}" target="_blank" rel="noopener">GitHub</a>')
    bar = (f'<cryosim-bar tool="pinnfluid" home="{home}" skip-target="#sidebar" role="banner">\n'
           '  <nav slot="nav" class="pnav" aria-label="pinnfluid">\n    '
           + "\n    ".join(links) + "\n  </nav>\n</cryosim-bar>\n")
    head = ('<link rel="stylesheet" href="static/kit/tokens.css">\n'
            '<script type="module" src="static/kit/cryosim-bar.js"></script>\n'
            f'<style id="site-shell">{SHELL_CSS}</style>\n')
    # the app's pages are drawn for a light ground: the bar follows (data-theme)
    page = page.replace("<html>", '<html data-theme="light">', 1)
    page = page.replace("</head>", head + "</head>", 1)
    return re.sub(r"<body[^>]*>", lambda m: m.group(0) + "\n" + bar, page, count=1)


def extra_headers() -> list[tuple[str, str]]:
    """The configured security headers (a value with a line break is ignored)."""
    headers = []
    for name, env in (("Content-Security-Policy", "PINN_WEBAPP_CSP"),
                      ("X-Robots-Tag", "PINN_WEBAPP_ROBOTS_TAG")):
        value = os.environ.get(env, "").strip()
        if value and "\r" not in value and "\n" not in value:
            headers.append((name, value))
    return headers


def _source_index(archives: list[str]) -> bytes:
    items = "".join(f'<li><a href="{html.escape(a, quote=True)}">{html.escape(a)}</a></li>' for a in archives)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>pinnfluid - source</title>"
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:36px;max-width:48rem}</style>"
        "</head><body><h1>Source</h1>"
        "<p>The source code this server runs, with its changes, under the licences it contains:</p>"
        f"<ul>{items or '<li>none</li>'}</ul>"
        '<p><a href="../NOTICES.txt">Third-party notices</a> &middot; '
        '<a href="../version.json">Build information</a></p>'
        "</body></html>"
    ).encode("utf-8")


def get(path: str) -> Optional[tuple[str, bytes]]:
    """(content type, body) for a request path (below the base path) that this module
    serves, or None."""
    rel = path.lstrip("/")
    kit = kit_dir()
    if kit is not None and rel.startswith("static/kit/"):
        name = rel[len("static/kit/"):]
        if name in KIT_FILES and (kit / name).is_file():
            return KIT_FILES[name], (kit / name).read_bytes()
        return None
    info = info_dir()
    if info is None:
        return None
    if rel == "version.json" and (info / "version.json").is_file():
        return "application/json; charset=utf-8", (info / "version.json").read_bytes()
    if rel == "NOTICES.txt" and (info / "NOTICES.txt").is_file():
        return "text/plain; charset=utf-8", (info / "NOTICES.txt").read_bytes()
    source = info / "source"
    if rel == "source/":
        archives = sorted(p.name for p in source.glob("*.tar.gz") if p.is_file()) if source.is_dir() else []
        return "text/html; charset=utf-8", _source_index(archives)
    if rel.startswith("source/") and _ARCHIVE_RE.fullmatch(rel[len("source/"):]):
        archive = source / rel[len("source/"):]
        if archive.is_file():
            return "application/gzip", archive.read_bytes()
    return None
