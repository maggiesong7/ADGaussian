import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
def generate_shift_transformation(
    t: Float[Tensor, " time_step"],
    max_shift: int = 1,
) -> Float[Tensor, "*batch time_step 4 4"]:
    # Generate a translation in the image plane.
    tf = torch.eye(4, dtype=torch.float32, device=t.device)
    tf = tf.broadcast_to((t.shape[0], 4, 4)).clone()
    
    # shift along x axis
    steps = t * max_shift
    tf[..., 0, 3] += steps
    return tf


@torch.no_grad()
def generate_shift(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    t: Float[Tensor, " time_step"],
    max_shift: int,
) -> Float[Tensor, "*batch time_step 4 4"]:
    tf = generate_shift_transformation(t, max_shift)
    return rearrange(extrinsics, "... i j -> ... () i j") @ tf
