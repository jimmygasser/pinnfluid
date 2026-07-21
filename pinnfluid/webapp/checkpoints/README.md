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

These are the seed-42 Iter5 checkpoints trained with y-reflection augmentation
and turbulent-viscosity momentum residuals:

- `H292_ymir_s1_phys_nut` and `H292_ymir_s2_phys_nut_on_ymir_s1`
- `G292_ymir_s1_phys_nut` and `G292_ymir_s2_phys_nut_highspeed_on_ymir_s1`

SHA-256 checksums:

```text
93c342c512e83587308829ad68974f3af8f07c035d9e3b92c54652aba92a6aec  grid-unet-stage1-292d.pth
f25d863364fea388591dd436e62e188db8060dc7cf90993a51320473d6301979  grid-unet-stage2-292d.pth
e5f7b04c4da817bac94d71a4f6572fcb823a61f9bbc16476ffa4b3b46700cb14  hybrid-stage1-292d.pth
fe0e9929f6a544ef940129e36c3f44780c84d5ace7d668d80050258a63fe0e9b  hybrid-stage2-292d.pth
```

## Get them

Run the fetch helper, or download the four files manually into this folder:

```bash
python fetch_checkpoints.py
```

Each checkpoint stores the exact training config, input/output scalers, and
weights, so the app rebuilds the right architecture from the file alone.

> TODO: publish the weights (Zenodo/EnviDat model release or a GitHub release)
> and put the download URLs in `fetch_checkpoints.py`.
