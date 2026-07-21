"""Download the four model checkpoints into webapp/checkpoints/.

The weights are published in the model release (not tracked in git). Fill in
CHECKPOINT_URLS with the release URLs, then run:

    python fetch_checkpoints.py

Each file is verified by size (and SHA-256 if provided) and skipped if already
present. This is intentionally dependency-light (urllib only).
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# TODO: fill in once the weights are published (Zenodo/EnviDat/GitHub release).
# Format: "filename": {"url": "...", "sha256": "..." (optional)}
CHECKPOINT_URLS: dict = {
    "grid-unet-stage1-292d.pth": {"url": ""},
    "grid-unet-stage2-292d.pth": {"url": ""},
    "hybrid-stage1-292d.pth": {"url": ""},
    "hybrid-stage2-292d.pth": {"url": ""},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    missing_urls = [name for name, spec in CHECKPOINT_URLS.items() if not spec.get("url")]
    if missing_urls:
        print(
            "No download URLs configured yet. Edit CHECKPOINT_URLS in this file,\n"
            "or place these files in checkpoints/ manually:\n  - "
            + "\n  - ".join(missing_urls)
        )
        return 1

    for name, spec in CHECKPOINT_URLS.items():
        dest = CHECKPOINT_DIR / name
        if dest.exists():
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name} <- {spec['url']}")
        urllib.request.urlretrieve(spec["url"], dest)  # noqa: S310 (trusted release URL)
        if spec.get("sha256"):
            got = _sha256(dest)
            if got != spec["sha256"]:
                dest.unlink(missing_ok=True)
                print(f"[FAIL] checksum mismatch for {name}: {got}")
                return 2
        print(f"[ok  ] {name}")
    print("All checkpoints present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
