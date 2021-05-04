"""Linear algebra."""
import torch
Tensor = torch.Tensor


def cov(xx: Tensor, bias: bool = False) -> Tensor:
    """Return covarience."""
    if xx.dim() > 2:
        raise ValueError('Input should be 1D or 2D tensor.')

    xx = xx - xx.mean(-1)
    factor = 1 / (xx.shape[-1] - int(not bool(bias)))

    return factor * xx @ xx.T
