from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians


@dataclass
class EncoderOutput:
    gaussian: Gaussians
    lidar_mask: Float[Tensor, "batch view 1 h w"]
    
    
T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> EncoderOutput:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
