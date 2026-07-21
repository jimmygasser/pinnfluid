"""Finite-difference utilities for the unified terrain-structure model."""

from __future__ import annotations

import torch

from config import NU


class NonUniformGridPhysics:
    def __init__(self, x_coords: torch.Tensor, y_coords: torch.Tensor, z_levels: torch.Tensor):
        self.x = x_coords
        self.y = y_coords
        self.z = z_levels
        dx = torch.diff(self.x)
        dy = torch.diff(self.y)
        self.dx = torch.clamp(torch.median(dx), min=1e-6) if len(dx) else torch.tensor(1.0, device=self.x.device)
        self.dy = torch.clamp(torch.median(dy), min=1e-6) if len(dy) else torch.tensor(1.0, device=self.y.device)

    def grad_x(self, f: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(f)
        if f.shape[0] < 2:
            return out
        dx = self.dx
        if f.shape[0] >= 3:
            out[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
        out[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
        out[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
        return out

    def grad_y(self, f: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(f)
        if f.shape[1] < 2:
            return out
        dy = self.dy
        if f.shape[1] >= 3:
            out[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        out[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
        out[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
        return out

    def grad_z(self, f: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(f)
        if f.shape[2] < 2 or self.z.shape[0] < 2:
            return out
        z = self.z
        if f.shape[2] == 2:
            dz = torch.clamp(z[1] - z[0], min=1e-6)
            out[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
            out[:, :, 1] = out[:, :, 0]
            return out
        zim1 = z[:-2]
        zi = z[1:-1]
        zip1 = z[2:]
        a = torch.clamp(zi - zim1, min=1e-6).view(1, 1, -1)
        b = torch.clamp(zip1 - zi, min=1e-6).view(1, 1, -1)
        c = torch.clamp(zip1 - zim1, min=1e-6).view(1, 1, -1)
        out[:, :, 1:-1] = ((a * a) * (f[:, :, 2:] - f[:, :, 1:-1]) + (b * b) * (f[:, :, 1:-1] - f[:, :, :-2])) / (a * b * c)
        out[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / torch.clamp(z[1] - z[0], min=1e-6)
        out[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / torch.clamp(z[-1] - z[-2], min=1e-6)
        return out

    def divergence(self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> torch.Tensor:
        return self.grad_x(ux) + self.grad_y(uy) + self.grad_z(uz)

    def momentum_residual(self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, p: torch.Tensor, nu: float = NU):
        dux_dx = self.grad_x(ux)
        dux_dy = self.grad_y(ux)
        dux_dz = self.grad_z(ux)
        duy_dx = self.grad_x(uy)
        duy_dy = self.grad_y(uy)
        duy_dz = self.grad_z(uy)
        duz_dx = self.grad_x(uz)
        duz_dy = self.grad_y(uz)
        duz_dz = self.grad_z(uz)
        conv_x = ux * dux_dx + uy * dux_dy + uz * dux_dz
        conv_y = ux * duy_dx + uy * duy_dy + uz * duy_dz
        conv_z = ux * duz_dx + uy * duz_dy + uz * duz_dz
        dp_dx = self.grad_x(p)
        dp_dy = self.grad_y(p)
        dp_dz = self.grad_z(p)
        diff_x = nu * (self.grad_x(dux_dx) + self.grad_y(dux_dy) + self.grad_z(dux_dz))
        diff_y = nu * (self.grad_x(duy_dx) + self.grad_y(duy_dy) + self.grad_z(duy_dz))
        diff_z = nu * (self.grad_x(duz_dx) + self.grad_y(duz_dy) + self.grad_z(duz_dz))
        rx = conv_x + dp_dx - diff_x
        ry = conv_y + dp_dy - diff_y
        rz = conv_z + dp_dz - diff_z
        return rx, ry, rz

    def momentum_residual_with_nu_field(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
        p: torch.Tensor,
        nu_eff: torch.Tensor,
    ):
        nu_eff = torch.clamp(nu_eff, min=1e-9)
        dux_dx = self.grad_x(ux)
        dux_dy = self.grad_y(ux)
        dux_dz = self.grad_z(ux)
        duy_dx = self.grad_x(uy)
        duy_dy = self.grad_y(uy)
        duy_dz = self.grad_z(uy)
        duz_dx = self.grad_x(uz)
        duz_dy = self.grad_y(uz)
        duz_dz = self.grad_z(uz)
        div_u = dux_dx + duy_dy + duz_dz
        conv_x = ux * dux_dx + uy * dux_dy + uz * dux_dz
        conv_y = ux * duy_dx + uy * duy_dy + uz * duy_dz
        conv_z = ux * duz_dx + uy * duz_dy + uz * duz_dz
        dp_dx = self.grad_x(p)
        dp_dy = self.grad_y(p)
        dp_dz = self.grad_z(p)

        two_thirds = 2.0 / 3.0
        tau_xx = nu_eff * (2.0 * dux_dx - two_thirds * div_u)
        tau_yy = nu_eff * (2.0 * duy_dy - two_thirds * div_u)
        tau_zz = nu_eff * (2.0 * duz_dz - two_thirds * div_u)
        tau_xy = nu_eff * (dux_dy + duy_dx)
        tau_xz = nu_eff * (dux_dz + duz_dx)
        tau_yz = nu_eff * (duy_dz + duz_dy)

        diff_x = self.grad_x(tau_xx) + self.grad_y(tau_xy) + self.grad_z(tau_xz)
        diff_y = self.grad_x(tau_xy) + self.grad_y(tau_yy) + self.grad_z(tau_yz)
        diff_z = self.grad_x(tau_xz) + self.grad_y(tau_yz) + self.grad_z(tau_zz)
        rx = conv_x + dp_dx - diff_x
        ry = conv_y + dp_dy - diff_y
        rz = conv_z + dp_dz - diff_z
        return rx, ry, rz
