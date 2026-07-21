# Model weights

The four checkpoints for the two final cascades are **not tracked in git**
(they total ~220 MB and one file exceeds GitHub's 100 MB limit). They are
published in the model release and downloaded here.

| file                        | model                       | ~size |
|-----------------------------|-----------------------------|-------|
| `grid-unet-stage1-292d.pth` | 3D U-Net cascade — Stage 1  | 131 MB |
| `grid-unet-stage2-292d.pth` | 3D U-Net cascade — Stage 2  | 65 MB |
| `hybrid-stage1-292d.pth`    | Hybrid cascade — Stage 1    | 14 MB |
| `hybrid-stage2-292d.pth`    | Hybrid cascade — Stage 2    | 7 MB |

## Get them

Run the fetch helper, or download the four files manually into this folder:

```bash
python fetch_checkpoints.py
```

Each checkpoint stores the exact training config, input/output scalers, and
weights, so the app rebuilds the right architecture from the file alone.

> TODO: publish the weights (Zenodo/EnviDat model release or a GitHub release)
> and put the download URLs in `fetch_checkpoints.py`.
