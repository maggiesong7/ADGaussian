from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians

from .mast3r.mast3r import model as MAST3RBackbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder, EncoderOutput
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_geofdn_cfg import EncoderVisualizerGeoFDNCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding
from .backbone.unimatch.geometry import coords_grid


def get_downsampled_dense_depths(raw_depths, near, far, depth_downscale_factor=1):
    b, v, h, w = raw_depths.shape
    depths = raw_depths.view(
        b*v,
        h // depth_downscale_factor,
        depth_downscale_factor,
        w // depth_downscale_factor,
        depth_downscale_factor,
        1,
    )
    depths = depths.permute(0, 1, 3, 5, 2, 4).contiguous()
    depths = depths.view(-1, depth_downscale_factor * depth_downscale_factor)
    depths_tmp = torch.where(depths == 0.0,
                                1e5 * torch.ones_like(depths),
                                depths)
    depths = torch.min(depths_tmp, dim=-1).values
    depths = depths.view(b, v, (h // depth_downscale_factor)*(w // depth_downscale_factor))

    near_ = repeat(near, "b v -> b v (h w)", h=h // depth_downscale_factor, w=w // depth_downscale_factor)
    far_ = repeat(far, "b v -> b v (h w)", h=h // depth_downscale_factor, w=w // depth_downscale_factor)
    depths = torch.where(
        (depths <= far_) & (depths >= near_),
        depths, torch.zeros_like(depths))

    return depths.float()


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class MAST3RCfg:
    pos_embed: str
    patch_embed_cls: str
    img_size: list[int]
    head_type: str
    output_mode: str
    enc_embed_dim: int
    enc_depth: int
    enc_num_heads: int
    dec_embed_dim: int
    dec_depth: int
    dec_num_heads: int
    two_confs: bool
    use_offsets: bool
    pretrained_mast3r_path: str | None


@dataclass
class EncoderGeoFDNCfg:
    name: Literal["geofdn"]
    num_surfaces: int
    visualizer: EncoderVisualizerGeoFDNCfg
    mast3r_backbone: MAST3RCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    downscale_factor: int
    shim_patch_size: int
    wo_depthpos: bool
    wo_depthenc: bool


class EncoderGeoFDN(Encoder[EncoderGeoFDNCfg]):
    backbone: MAST3RBackbone
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderGeoFDNCfg) -> None:
        super().__init__(cfg)
        
        self.wo_depthpos = cfg.wo_depthpos
        self.wo_depthenc = cfg.wo_depthenc

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # multi-view Transformer backbone
        self.backbone = MAST3RBackbone.AsymmetricMASt3R(
            pos_embed=cfg.mast3r_backbone.pos_embed,
            patch_embed_cls=cfg.mast3r_backbone.patch_embed_cls,
            img_size=cfg.mast3r_backbone.img_size,
            head_type=cfg.mast3r_backbone.head_type,
            output_mode=cfg.mast3r_backbone.output_mode,
            depth_mode=('exp', -MAST3RBackbone.inf, MAST3RBackbone.inf),
            conf_mode=('exp', 1, MAST3RBackbone.inf),
            enc_embed_dim=cfg.mast3r_backbone.enc_embed_dim,
            enc_depth=cfg.mast3r_backbone.enc_depth,
            enc_num_heads=cfg.mast3r_backbone.enc_num_heads,
            dec_embed_dim=cfg.mast3r_backbone.dec_embed_dim,
            dec_depth=cfg.mast3r_backbone.dec_depth,
            dec_num_heads=cfg.mast3r_backbone.dec_num_heads,
            two_confs=cfg.mast3r_backbone.two_confs,
            use_offsets=cfg.mast3r_backbone.use_offsets,
            sh_degree=self.gaussian_adapter.d_sh
        )
        self.backbone.downstream_head1.dpt.requires_grad_(False)
        self.backbone.downstream_head2.dpt.requires_grad_(False)
        if cfg.mast3r_backbone.pretrained_mast3r_path is not None:
            ckpt = torch.load(cfg.mast3r_backbone.pretrained_mast3r_path)
            self.backbone.load_state_dict(ckpt['model'], strict=False)

    def normalize_images(
        self,
        images: Float[Tensor, "batch view c h w"],
    ):
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std
    
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None
    ) -> EncoderOutput:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        
        images = self.normalize_images(context["image"])

        view1 = {
            "img": images[:, 0],
            "instance": torch.range(0, 1)
            }
        view2 = {
            "img": context["depth"][:, 0].repeat(1, 3, 1, 1),
            "instance": torch.range(0, 1)
            }
        
        lidar_masks = rearrange((context["depth"]!=0).float(), "b v h w -> b v () h w")
        
        if self.wo_depthpos:    
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.backbone._encode_symmetrized(view1, view2)
        else:     
            s_depth = get_downsampled_dense_depths(context["depth"], context["near"], context["far"], depth_downscale_factor=16)
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.backbone._encode_symmetrized(view1, view2, s_depth[:, 0], s_depth[:, 0])
        dec1, dec2 = self.backbone._decoder(feat1, pos1, feat2, pos2)
        
        prompt = {
            "img": images[:, 0],
            "depth": context["depth"][:, 0],
            "near": context["near"][:, 0],
            "far": context["far"][:, 0],
            "wo_depthenc": self.wo_depthenc
        }
        
        head_input = [[tok.float() for tok in dec1], [tok.float() for tok in dec2]]
        pred_gaussians = self.backbone._downstream_head(1, head_input, shape1, prompt)
        raw_gaussians = torch.stack([pred_gaussians], dim=1)
        # pred_gaussians2 = self.backbone._downstream_head(2, [tok.float() for tok in dec2], shape2, prompt2)
        # raw_gaussians = torch.stack([pred_gaussians1, pred_gaussians2], dim=1)
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "b v h w (srf c) -> b v (h w) srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        
        depths = rearrange(context["near"], "b v -> b v () () ()") + \
            (rearrange(context["far"], "b v -> b v () () ()") - rearrange(context["near"], "b v -> b v () () ()")) * gaussians[..., 2:3].sigmoid()
        
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(gaussians[..., 3:4].sigmoid(), global_step) / gpp,
            rearrange(
                gaussians[..., 4:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            
        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        output_gaussian =  Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        return EncoderOutput(
            output_gaussian,
            lidar_masks,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
