"""Download the four model checkpoints into webapp/checkpoints/.

The weights are GitHub Release assets rather than git objects. Run:

    python fetch_checkpoints.py

Each file is verified by size (and SHA-256 if provided) and skipped if already
present. This is intentionally dependency-light (urllib only).
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
RELEASE_BASE = "https://github.com/jimmygasser/pinnfluid/releases/download/v0.1.0-beta"

CHECKPOINT_URLS: dict = {
    "grid-unet-stage1-292d.pth": {
        "url": f"{RELEASE_BASE}/grid-unet-stage1-292d.pth",
        "sha256": "93c342c512e83587308829ad68974f3af8f07c035d9e3b92c54652aba92a6aec",
    },
    "grid-unet-stage2-292d.pth": {
        "url": f"{RELEASE_BASE}/grid-unet-stage2-292d.pth",
        "sha256": "f25d863364fea388591dd436e62e188db8060dc7cf90993a51320473d6301979",
    },
    "hybrid-stage1-292d.pth": {
        "url": f"{RELEASE_BASE}/hybrid-stage1-292d.pth",
        "sha256": "e5f7b04c4da817bac94d71a4f6572fcb823a61f9bbc16476ffa4b3b46700cb14",
    },
    "hybrid-stage2-292d.pth": {
        "url": f"{RELEASE_BASE}/hybrid-stage2-292d.pth",
        "sha256": "fe0e9929f6a544ef940129e36c3f44780c84d5ace7d668d80050258a63fe0e9b",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in CHECKPOINT_URLS.items():
        dest = CHECKPOINT_DIR / name
        if dest.exists():
            got = _sha256(dest)
            if got == spec["sha256"]:
                print(f"[skip] {name} already present and verified")
                continue
            print(f"[warn] replacing {name}: checksum was {got}")
        print(f"[get ] {name} <- {spec['url']}")
        temp = dest.with_suffix(dest.suffix + ".part")
        temp.unlink(missing_ok=True)
        try:
            urllib.request.urlretrieve(spec["url"], temp)  # noqa: S310 (trusted release URL)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        got = _sha256(temp)
        if got != spec["sha256"]:
            temp.unlink(missing_ok=True)
            print(f"[FAIL] checksum mismatch for {name}: {got}")
            return 2
        temp.replace(dest)
        print(f"[ok  ] {name}")
    print("All checkpoints present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
