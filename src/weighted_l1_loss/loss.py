from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


class WeightedL1Loss(nn.Module):
    """Resolution-aware L1 + gradient loss for 3D volumes.

    Supports uniform spacing for the whole batch or per-sample spacing.

    Parameters
    ----------
    reduction : {"mean", "sum"}
        Reduction applied to the final loss across the batch if spacing is per-sample.
    dtype : Optional[torch.dtype]
        Optional dtype to cast spacing when provided as Python sequence.
    """

    def __init__(self, reduction: str = "mean", dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self._spacing_dtype = dtype

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        spacing: torch.Tensor | Iterable[float],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Shape [B, C, D, H, W]
        target : Tensor
            Shape [B, C, D, H, W]
        spacing : Tensor or sequence
            Either shape [3] or [B, 3], order (sx, sy, sz)
        """
        if input.ndim != 5 or target.ndim != 5:
            raise ValueError("input and target must be 5D tensors [B, C, D, H, W]")
        if input.shape != target.shape:
            raise ValueError("input and target must have the same shape")

        if isinstance(spacing, (list, tuple)):
            spacing = torch.tensor(
                spacing,
                device=input.device,
                dtype=self._spacing_dtype or input.dtype,
            )

        if not torch.is_tensor(spacing):
            raise TypeError("spacing must be a Tensor or a sequence of floats")

        if spacing.ndim == 1:
            if spacing.numel() != 3:
                raise ValueError("1D spacing must have exactly 3 elements: (sx, sy, sz)")
            sx, sy, sz = spacing
            wx, wy, wz = 1.0 / sx, 1.0 / sy, 1.0 / sz
            voxel_volume = sx * sy * sz

            l1_voxel = torch.abs(input - target).mean() * voxel_volume

            dz_in = input[:, :, 1:, :, :] - input[:, :, :-1, :, :]
            dz_tg = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
            dy_in = input[:, :, :, 1:, :] - input[:, :, :, :-1, :]
            dy_tg = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
            dx_in = input[:, :, :, :, 1:] - input[:, :, :, :, :-1]
            dx_tg = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

            l1_z = torch.abs(dz_in - dz_tg).mean() * wz
            l1_y = torch.abs(dy_in - dy_tg).mean() * wy
            l1_x = torch.abs(dx_in - dx_tg).mean() * wx

            total_loss = 0.5 * l1_voxel + 0.5 * (l1_x + l1_y + l1_z) / 3.0
            return total_loss

        elif spacing.ndim == 2:
            if spacing.shape[1] != 3 or spacing.shape[0] != input.shape[0]:
                raise ValueError("2D spacing must be shape [B, 3] matching batch size")
            sx, sy, sz = spacing[:, 0], spacing[:, 1], spacing[:, 2]
            wx, wy, wz = 1.0 / sx, 1.0 / sy, 1.0 / sz
            voxel_volume = sx * sy * sz
            batch_size = input.shape[0]

            l1_voxel = torch.abs(input - target).view(batch_size, -1).mean(dim=1) * voxel_volume

            dz = torch.abs(
                (input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
                - (target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
            ).view(batch_size, -1).mean(dim=1) * wz
            dy = torch.abs(
                (input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
                - (target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
            ).view(batch_size, -1).mean(dim=1) * wy
            dx = torch.abs(
                (input[:, :, :, :, 1:] - input[:, :, :, :, :-1])
                - (target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
            ).view(batch_size, -1).mean(dim=1) * wx

            per_sample = 0.5 * l1_voxel + 0.5 * (dx + dy + dz) / 3.0
            if self.reduction == "mean":
                return per_sample.mean()
            else:
                return per_sample.sum()
        else:
            raise ValueError("spacing must be 1D [3] or 2D [B, 3]")