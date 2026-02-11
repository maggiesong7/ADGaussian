from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
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
            delta = (prediction - batch["context"]["image"]) * mask
            return self.cfg.weight/2 * (delta**2).mean()
        else:
            delta = prediction - batch["target"]["image"]
            return self.cfg.weight * (delta**2).mean()
