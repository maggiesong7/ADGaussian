from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        mode: str,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.mode = mode
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        shift_trans: Float[Tensor, "batch view 4 4"] | None = None,
    ) -> DecoderOutput:
        h, w = image_shape
        b, v, _, _ = extrinsics.shape
        color, accum_weights, accum_area, area_max = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        )
        
        if shift_trans is not None:
            means = []
            covariances = []
            harmonics = []
            opacities = []
            for idx in range(b):
                prune_mask = accum_area[idx]>0
                means_batch = gaussians.means[idx][prune_mask]
                covariances_batch = gaussians.covariances[idx][prune_mask]
                harmonics_batch = gaussians.harmonics[idx][prune_mask]
                opacities_batch = gaussians.opacities[idx][prune_mask]
                
                means.append(means_batch)
                covariances.append(covariances_batch)
                harmonics.append(harmonics_batch)
                opacities.append(opacities_batch)
            
            output_gaussian =  Gaussians(
                torch.stack(means),
                torch.stack(covariances),
                torch.stack(harmonics),
                torch.stack(opacities),
            )
            color, _, _, _ = render_cuda(
                rearrange(extrinsics @ shift_trans, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(output_gaussian.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(output_gaussian.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(output_gaussian.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(output_gaussian.opacities, "b g -> (b v) g", v=v),
            )
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        return DecoderOutput(
            color,
            None
            if depth_mode is None
            else self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
