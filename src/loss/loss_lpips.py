from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: Float[Tensor, "batch view c h w"],
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        mask: None | Float[Tensor, "batch view 1 h w"] = None,
        depth: None | Float[Tensor, "batch view h w"] = None
    ) -> Float[Tensor, ""]:
        if mask is not None:
            image = batch["context"]["image"] #* mask
            prediction_color = prediction #* mask
            weight = self.cfg.weight / 2
        else:
            image = batch["target"]["image"]
            prediction_color = prediction
            weight = self.cfg.weight

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        loss = self.lpips.forward(
            rearrange(prediction_color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            normalize=True,
        )
        return weight * loss.mean()
