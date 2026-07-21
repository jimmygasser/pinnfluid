
"""Hybrid two-branch model for the unified terrain-structure pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CASCADE_STAGE,
    CASCADE_USE_ABL_VELOCITY_BASELINE,
    CASCADE_ZERO_INIT_HEAD,
    DEPTH,
    DROPOUT,
    ENCODER_DEPTH,
    GLOBAL_ENCODER_DEPTH,
    FOURIER_SIGMA,
    GLOBAL_ENCODER_WIDTH,
    GLOBAL_ENCODER_DILATIONS,
    GLOBAL_INPUT_COLS,
    GRID_UNET_BASE_WIDTH,
    GRID_UNET_DROPOUT,
    GRID_UNET_LEVELS,
    GRID_UNET_ROI_STRUCTURE_MODE,
    GRID_UNET_TERRAIN_CONTEXT_DEPTH,
    GRID_UNET_TERRAIN_CONTEXT_DILATIONS,
    GRID_UNET_TERRAIN_CONTEXT_WIDTH,
    GRID_UNET_USE_TERRAIN_CONTEXT,
    HIDDEN_DIM,
    NUM_FOURIER_FEATURES,
    OUTPUT_COLS,
    ROI_ENCODER_DEPTH,
    ROI_ENCODER_WIDTH,
    ROI_ENCODER_DILATIONS,
    ROI_INPUT_COLS,
    CASCADE_STAGE2_REFINER_KIND,
    resolve_structure_channel_mode,
    structure_channel_count,
    STRUCTURE_ENCODER_WIDTH,
    STRUCTURE_ENCODER_DEPTH,
    STRUCTURE_ENCODER_DILATIONS,
    STRUCTURE_ENCODER_INPUT_MODE,
    USE_STRUCTURE_ENCODER,
)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 3, n_freq: int = NUM_FOURIER_FEATURES, sigma: float = FOURIER_SIGMA):
        super().__init__()
        self.n_freq = int(n_freq)
        if self.n_freq > 0:
            B = torch.randn(in_dim, self.n_freq) * float(sigma)
            self.register_buffer('B', B)
            self.out_dim = 2 * self.n_freq
        else:
            self.register_buffer('B', torch.zeros(in_dim, 0))
            self.out_dim = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_freq <= 0:
            return x.new_zeros((x.shape[0], 0))
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = DROPOUT):
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.lin1(x))
        h = self.drop(h)
        h = self.lin2(h)
        return self.act(x + h)


def _resolve_dilations(dilations, *, depth: int) -> list[int]:
    total_layers = int(depth) + 1
    if dilations is None:
        return [1] * total_layers
    if isinstance(dilations, (int, float)):
        return [max(1, int(dilations))] * total_layers
    values = [max(1, int(v)) for v in dilations]
    if not values:
        return [1] * total_layers
    if len(values) == 1:
        return values * total_layers
    if len(values) != total_layers:
        raise ValueError(f"Expected {total_layers} encoder dilations, got {len(values)}: {values}")
    return values


def _structure_in_channels() -> int:
    return int(structure_channel_count(STRUCTURE_ENCODER_INPUT_MODE))


def _grid_roi_in_channels() -> int:
    return int(len(ROI_INPUT_COLS) + structure_channel_count(GRID_UNET_ROI_STRUCTURE_MODE))


def _normalize_model_kind(model_kind: str | None) -> str:
    kind = str(model_kind or "hybrid").strip().lower()
    if kind == "unet":
        kind = "grid_unet"
    if kind == "cascade":
        stage = str(CASCADE_STAGE or "stage1").strip().lower()
        if stage == "stage2":
            kind = "cascade_stage2"
        else:
            kind = "cascade_stage1"
    if kind not in {"hybrid", "grid_unet", "cascade_stage1", "cascade_stage2"}:
        raise ValueError(f"Unsupported model kind: {model_kind!r}")
    return kind


class TerrainEncoder2D(nn.Module):
    def __init__(self, in_ch: int = 4, width: int = GLOBAL_ENCODER_WIDTH, depth: int = ENCODER_DEPTH, dilations=None):
        super().__init__()
        dilation_schedule = _resolve_dilations(dilations, depth=int(depth))
        first_dilation = int(dilation_schedule[0])
        blocks = [nn.Conv2d(in_ch, width, kernel_size=3, padding=first_dilation, dilation=first_dilation), nn.SiLU()]
        for dilation in dilation_schedule[1:]:
            dilation = int(dilation)
            blocks += [nn.Conv2d(width, width, kernel_size=3, padding=dilation, dilation=dilation), nn.SiLU()]
        self.net = nn.Sequential(*blocks)
        self.out_ch = int(width)
        self.dilation_schedule = tuple(int(v) for v in dilation_schedule)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0.0):
        super().__init__()
        groups = max(1, min(8, out_ch))
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout3d(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3DCore(nn.Module):
    def __init__(self, in_ch: int, *, base_width: int = GRID_UNET_BASE_WIDTH, levels: int = GRID_UNET_LEVELS, out_ch: int = len(OUTPUT_COLS), dropout: float = GRID_UNET_DROPOUT):
        super().__init__()
        levels = max(2, int(levels))
        widths = [int(base_width) * (2 ** i) for i in range(levels)]
        self.enc_blocks = nn.ModuleList()
        prev = int(in_ch)
        for width in widths:
            self.enc_blocks.append(ConvBlock3D(prev, width, dropout=dropout))
            prev = width
        self.pools = nn.ModuleList([nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True) for _ in range(levels - 1)])
        self.up_transpose = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for level in range(levels - 1, 0, -1):
            self.up_transpose.append(nn.ConvTranspose3d(widths[level], widths[level - 1], kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock3D(widths[level - 1] * 2, widths[level - 1], dropout=dropout))
        self.out_conv = nn.Conv3d(widths[0], out_ch, kernel_size=1)
        self.levels = levels
        self.widths = widths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for idx, block in enumerate(self.enc_blocks):
            h = block(h)
            if idx < len(self.enc_blocks) - 1:
                skips.append(h)
                h = self.pools[idx](h)
        for up, block in zip(self.up_transpose, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-3:] != skip.shape[-3:]:
                h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)
        return self.out_conv(h)


class GridUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_kind = "grid_unet"
        self.use_structure_encoder = False
        self.grid_unet_roi_structure_mode = str(resolve_structure_channel_mode(GRID_UNET_ROI_STRUCTURE_MODE))
        self.uses_grid_terrain_context = bool(GRID_UNET_USE_TERRAIN_CONTEXT)
        if self.uses_grid_terrain_context:
            self.grid_terrain_context_encoder = TerrainEncoder2D(
                in_ch=4,
                width=GRID_UNET_TERRAIN_CONTEXT_WIDTH,
                depth=GRID_UNET_TERRAIN_CONTEXT_DEPTH,
                dilations=GRID_UNET_TERRAIN_CONTEXT_DILATIONS,
            )
            terrain_context_ch = int(self.grid_terrain_context_encoder.out_ch)
        else:
            self.grid_terrain_context_encoder = None
            terrain_context_ch = 0
        self.global_unet = UNet3DCore(in_ch=len(GLOBAL_INPUT_COLS) + terrain_context_ch)
        self.roi_unet = UNet3DCore(in_ch=_grid_roi_in_channels() + terrain_context_ch)
        self.model_hparams = {
            "global_in_ch": int(len(GLOBAL_INPUT_COLS)),
            "roi_in_ch": int(_grid_roi_in_channels()),
            "grid_unet_base_width": int(GRID_UNET_BASE_WIDTH),
            "grid_unet_levels": int(GRID_UNET_LEVELS),
            "grid_unet_dropout": float(GRID_UNET_DROPOUT),
            "grid_unet_roi_structure_mode": str(self.grid_unet_roi_structure_mode),
            "grid_unet_use_terrain_context": bool(self.uses_grid_terrain_context),
            "grid_unet_terrain_context_width": int(GRID_UNET_TERRAIN_CONTEXT_WIDTH),
            "grid_unet_terrain_context_depth": int(GRID_UNET_TERRAIN_CONTEXT_DEPTH),
            "grid_unet_terrain_context_dilations": [] if self.grid_terrain_context_encoder is None else list(self.grid_terrain_context_encoder.dilation_schedule),
        }

    @staticmethod
    def _augment_volume_with_context(x_volume_scaled: torch.Tensor, terrain_context_2d: torch.Tensor | None) -> torch.Tensor:
        if terrain_context_2d is None:
            return x_volume_scaled
        depth = int(x_volume_scaled.shape[-1])
        ctx3d = terrain_context_2d.unsqueeze(-1).expand(-1, -1, -1, -1, depth)
        return torch.cat([x_volume_scaled, ctx3d], dim=1)

    def encode_grid_terrain_context(self, terrain_2d: torch.Tensor) -> torch.Tensor | None:
        if not self.uses_grid_terrain_context or self.grid_terrain_context_encoder is None:
            return None
        return self.grid_terrain_context_encoder(terrain_2d)

    def forward_global_grid(self, x_volume_scaled: torch.Tensor, *, terrain_context_2d: torch.Tensor | None = None) -> torch.Tensor:
        return self.global_unet(self._augment_volume_with_context(x_volume_scaled, terrain_context_2d))

    def forward_roi_grid(self, x_volume_scaled: torch.Tensor, *, terrain_context_2d: torch.Tensor | None = None) -> torch.Tensor:
        return self.roi_unet(self._augment_volume_with_context(x_volume_scaled, terrain_context_2d))


def _sample_feature_map(feat_map: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    grid = xy_norm.view(1, -1, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return sampled


class PointHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        depth: int = DEPTH,
        out_dim: int = len(OUTPUT_COLS),
        *,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.ff = FourierFeatures(in_dim=3, n_freq=NUM_FOURIER_FEATURES, sigma=FOURIER_SIGMA)
        total_in = in_dim + self.ff.out_dim
        self.input = nn.Sequential(nn.Linear(total_in, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(int(depth))])
        self.output = nn.Linear(hidden_dim, out_dim)
        if bool(zero_init_output):
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ff = self.ff(x[:, :3])
        h = self.input(torch.cat([x, ff], dim=-1))
        h = self.blocks(h)
        return self.output(h)


class UnifiedHybridPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_kind = "hybrid"
        self.use_structure_encoder = bool(USE_STRUCTURE_ENCODER)
        self.global_encoder = TerrainEncoder2D(
            in_ch=4,
            width=GLOBAL_ENCODER_WIDTH,
            depth=GLOBAL_ENCODER_DEPTH,
            dilations=GLOBAL_ENCODER_DILATIONS,
        )
        self.roi_encoder = TerrainEncoder2D(
            in_ch=4,
            width=ROI_ENCODER_WIDTH,
            depth=ROI_ENCODER_DEPTH,
            dilations=ROI_ENCODER_DILATIONS,
        )
        if self.use_structure_encoder:
            self.structure_encoder_input_mode = str(STRUCTURE_ENCODER_INPUT_MODE)
            self.structure_encoder = TerrainEncoder2D(
                in_ch=_structure_in_channels(),
                width=STRUCTURE_ENCODER_WIDTH,
                depth=STRUCTURE_ENCODER_DEPTH,
                dilations=STRUCTURE_ENCODER_DILATIONS,
            )
            structure_dim = int(self.structure_encoder.out_ch)
        else:
            self.structure_encoder = None
            self.structure_encoder_input_mode = str(STRUCTURE_ENCODER_INPUT_MODE)
            structure_dim = 0
        self.global_head = PointHead(in_dim=len(GLOBAL_INPUT_COLS) + self.global_encoder.out_ch)
        self.roi_head = PointHead(
            in_dim=len(ROI_INPUT_COLS) + self.global_encoder.out_ch + self.roi_encoder.out_ch + structure_dim
        )
        self.model_hparams = {
            'global_encoder_width': int(GLOBAL_ENCODER_WIDTH),
            'roi_encoder_width': int(ROI_ENCODER_WIDTH),
            'structure_encoder_width': int(STRUCTURE_ENCODER_WIDTH),
            'encoder_depth': int(ENCODER_DEPTH),
            'global_encoder_depth': int(GLOBAL_ENCODER_DEPTH),
            'roi_encoder_depth': int(ROI_ENCODER_DEPTH),
            'structure_encoder_depth': int(STRUCTURE_ENCODER_DEPTH),
            'global_encoder_dilations': list(self.global_encoder.dilation_schedule),
            'roi_encoder_dilations': list(self.roi_encoder.dilation_schedule),
            'structure_encoder_dilations': [] if self.structure_encoder is None else list(self.structure_encoder.dilation_schedule),
            'hidden_dim': int(HIDDEN_DIM),
            'depth': int(DEPTH),
            'dropout': float(DROPOUT),
            'num_fourier_features': int(NUM_FOURIER_FEATURES),
            'use_structure_encoder': bool(self.use_structure_encoder),
            'structure_encoder_input_mode': str(self.structure_encoder_input_mode),
        }

    def encode_global(self, terrain: torch.Tensor) -> torch.Tensor:
        return self.global_encoder(terrain)

    def encode_roi(self, roi_terrain: torch.Tensor) -> torch.Tensor:
        return self.roi_encoder(roi_terrain)

    def encode_structure(self, structure_terrain: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_structure_encoder or self.structure_encoder is None or structure_terrain is None:
            return None
        return self.structure_encoder(structure_terrain)

    def forward_global_from_encoded(self, feat: torch.Tensor, x_scaled: torch.Tensor, xy_local: torch.Tensor) -> torch.Tensor:
        terrain_feat = _sample_feature_map(feat, xy_local)
        return self.global_head(torch.cat([x_scaled, terrain_feat], dim=1))

    def forward_global(self, terrain: torch.Tensor, x_scaled: torch.Tensor, xy_local: torch.Tensor) -> torch.Tensor:
        feat = self.encode_global(terrain)
        return self.forward_global_from_encoded(feat, x_scaled, xy_local)

    def forward_roi_from_encoded(
        self,
        g_feat: torch.Tensor,
        r_feat: torch.Tensor,
        x_scaled: torch.Tensor,
        xy_global: torch.Tensor,
        xy_local: torch.Tensor,
        *,
        s_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            x_scaled,
            _sample_feature_map(g_feat, xy_global),
            _sample_feature_map(r_feat, xy_local),
        ]
        if self.use_structure_encoder and s_feat is not None:
            parts.append(_sample_feature_map(s_feat, xy_local))
        return self.roi_head(torch.cat(parts, dim=1))

    def forward_roi(
        self,
        global_terrain: torch.Tensor,
        roi_terrain: torch.Tensor,
        x_scaled: torch.Tensor,
        xy_global: torch.Tensor,
        xy_local: torch.Tensor,
        structure_terrain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        g_feat = self.encode_global(global_terrain)
        r_feat = self.encode_roi(roi_terrain)
        s_feat = self.encode_structure(structure_terrain)
        return self.forward_roi_from_encoded(g_feat, r_feat, x_scaled, xy_global, xy_local, s_feat=s_feat)


class CascadeStage1PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_kind = "cascade_stage1"
        self.use_structure_encoder = False
        self.uses_abl_velocity_baseline = bool(CASCADE_USE_ABL_VELOCITY_BASELINE)
        self.zero_init_output_head = bool(CASCADE_ZERO_INIT_HEAD)
        self.global_encoder = TerrainEncoder2D(
            in_ch=4,
            width=GLOBAL_ENCODER_WIDTH,
            depth=GLOBAL_ENCODER_DEPTH,
            dilations=GLOBAL_ENCODER_DILATIONS,
        )
        self.global_head = PointHead(
            in_dim=len(GLOBAL_INPUT_COLS) + self.global_encoder.out_ch,
            zero_init_output=self.zero_init_output_head,
        )
        self.model_hparams = {
            "global_encoder_width": int(GLOBAL_ENCODER_WIDTH),
            "global_encoder_depth": int(GLOBAL_ENCODER_DEPTH),
            "global_encoder_dilations": list(self.global_encoder.dilation_schedule),
            "hidden_dim": int(HIDDEN_DIM),
            "depth": int(DEPTH),
            "dropout": float(DROPOUT),
            "num_fourier_features": int(NUM_FOURIER_FEATURES),
            "cascade_stage": "stage1",
            "cascade_uses_abl_velocity_baseline": bool(self.uses_abl_velocity_baseline),
            "cascade_zero_init_head": bool(self.zero_init_output_head),
        }

    def encode_global(self, terrain: torch.Tensor) -> torch.Tensor:
        return self.global_encoder(terrain)

    def encode_roi(self, roi_terrain: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cascade_stage1 does not define an ROI branch")

    def encode_structure(self, structure_terrain: torch.Tensor | None) -> torch.Tensor | None:
        return None

    def forward_global_from_encoded(self, feat: torch.Tensor, x_scaled: torch.Tensor, xy_local: torch.Tensor) -> torch.Tensor:
        terrain_feat = _sample_feature_map(feat, xy_local)
        return self.global_head(torch.cat([x_scaled, terrain_feat], dim=1))

    def forward_global(self, terrain: torch.Tensor, x_scaled: torch.Tensor, xy_local: torch.Tensor) -> torch.Tensor:
        feat = self.encode_global(terrain)
        return self.forward_global_from_encoded(feat, x_scaled, xy_local)


class CascadeStage2Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_kind = "cascade_stage2"
        self.use_structure_encoder = True
        self.uses_abl_velocity_baseline = False
        self.zero_init_output_head = bool(CASCADE_ZERO_INIT_HEAD)
        self.roi_encoder = TerrainEncoder2D(
            in_ch=4,
            width=ROI_ENCODER_WIDTH,
            depth=ROI_ENCODER_DEPTH,
            dilations=ROI_ENCODER_DILATIONS,
        )
        self.structure_encoder_input_mode = str(STRUCTURE_ENCODER_INPUT_MODE)
        self.structure_encoder = TerrainEncoder2D(
            in_ch=_structure_in_channels(),
            width=STRUCTURE_ENCODER_WIDTH,
            depth=STRUCTURE_ENCODER_DEPTH,
            dilations=STRUCTURE_ENCODER_DILATIONS,
        )
        structure_dim = int(self.structure_encoder.out_ch)
        self.roi_head = PointHead(
            in_dim=len(ROI_INPUT_COLS) + 4 + self.roi_encoder.out_ch + structure_dim,
            zero_init_output=self.zero_init_output_head,
        )
        self.model_hparams = {
            "roi_encoder_width": int(ROI_ENCODER_WIDTH),
            "structure_encoder_width": int(STRUCTURE_ENCODER_WIDTH),
            "roi_encoder_depth": int(ROI_ENCODER_DEPTH),
            "structure_encoder_depth": int(STRUCTURE_ENCODER_DEPTH),
            "roi_encoder_dilations": list(self.roi_encoder.dilation_schedule),
            "structure_encoder_dilations": list(self.structure_encoder.dilation_schedule),
            "hidden_dim": int(HIDDEN_DIM),
            "depth": int(DEPTH),
            "dropout": float(DROPOUT),
            "num_fourier_features": int(NUM_FOURIER_FEATURES),
            "cascade_stage": "stage2",
            "cascade_zero_init_head": bool(self.zero_init_output_head),
            "structure_encoder_input_mode": str(self.structure_encoder_input_mode),
        }

    def encode_global(self, terrain: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cascade_stage2 expects a frozen stage1 conditioner, not an internal global encoder")

    def encode_roi(self, roi_terrain: torch.Tensor) -> torch.Tensor:
        return self.roi_encoder(roi_terrain)

    def encode_structure(self, structure_terrain: torch.Tensor | None) -> torch.Tensor | None:
        if structure_terrain is None:
            return None
        return self.structure_encoder(structure_terrain)

    def forward_roi_from_encoded(
        self,
        r_feat: torch.Tensor,
        x_scaled: torch.Tensor,
        xy_local: torch.Tensor,
        bg_scaled: torch.Tensor,
        *,
        s_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            x_scaled,
            bg_scaled,
            _sample_feature_map(r_feat, xy_local),
        ]
        if s_feat is not None:
            parts.append(_sample_feature_map(s_feat, xy_local))
        return self.roi_head(torch.cat(parts, dim=1))


class CascadeStage2GridUNetRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_kind = "cascade_stage2"
        self.cascade_stage2_refiner_kind = "grid_unet"
        self.uses_cascade_grid_refiner = True
        self.use_structure_encoder = False
        self.zero_init_output_head = bool(CASCADE_ZERO_INIT_HEAD)
        self.grid_unet_roi_structure_mode = str(resolve_structure_channel_mode(GRID_UNET_ROI_STRUCTURE_MODE))
        in_ch = int(_grid_roi_in_channels() + len(OUTPUT_COLS))
        self.roi_unet = UNet3DCore(in_ch=in_ch)
        if self.zero_init_output_head:
            nn.init.zeros_(self.roi_unet.out_conv.weight)
            nn.init.zeros_(self.roi_unet.out_conv.bias)
        self.model_hparams = {
            "cascade_stage": "stage2",
            "cascade_stage2_refiner_kind": "grid_unet",
            "cascade_zero_init_head": bool(self.zero_init_output_head),
            "roi_in_ch": int(_grid_roi_in_channels()),
            "background_in_ch": int(len(OUTPUT_COLS)),
            "grid_unet_base_width": int(GRID_UNET_BASE_WIDTH),
            "grid_unet_levels": int(GRID_UNET_LEVELS),
            "grid_unet_dropout": float(GRID_UNET_DROPOUT),
            "grid_unet_roi_structure_mode": str(self.grid_unet_roi_structure_mode),
        }

    def encode_global(self, terrain: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("cascade_stage2 grid refiner expects a frozen stage1 conditioner")

    def encode_roi(self, roi_terrain: torch.Tensor) -> None:
        return None

    def encode_structure(self, structure_terrain: torch.Tensor | None) -> None:
        return None

    def forward_roi_grid(self, x_volume_scaled: torch.Tensor, bg_volume_scaled: torch.Tensor) -> torch.Tensor:
        if x_volume_scaled.ndim != 5 or bg_volume_scaled.ndim != 5:
            raise ValueError("cascade Stage-2 grid refiner expects 5D BCHW[D] tensors")
        if x_volume_scaled.shape[0] != bg_volume_scaled.shape[0] or x_volume_scaled.shape[-3:] != bg_volume_scaled.shape[-3:]:
            raise ValueError(
                f"Stage-2 grid refiner input shape mismatch: x={tuple(x_volume_scaled.shape)} bg={tuple(bg_volume_scaled.shape)}"
            )
        return self.roi_unet(torch.cat([x_volume_scaled, bg_volume_scaled], dim=1))


def create_model(device: str = 'cuda', *, model_kind: str = 'hybrid') -> nn.Module:
    kind = _normalize_model_kind(model_kind)
    if kind == "hybrid":
        return UnifiedHybridPINN().to(device)
    if kind == "grid_unet":
        return GridUNet3D().to(device)
    if kind == "cascade_stage1":
        return CascadeStage1PINN().to(device)
    if kind == "cascade_stage2":
        refiner_kind = str(CASCADE_STAGE2_REFINER_KIND or "point").strip().lower()
        if refiner_kind in {"grid", "grid_unet", "unet", "3d_unet"}:
            return CascadeStage2GridUNetRefiner().to(device)
        return CascadeStage2Refiner().to(device)
    raise ValueError(f"Unsupported model kind: {model_kind!r}")
