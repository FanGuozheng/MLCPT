"""Base machine learning module."""
from typing import Union
import numpy as np
import torch

ndarray = np.ndarray
Tensor = torch.Tensor


class BaseMLModel:
    """Base of machine learning model."""

    def __init__(self, x: Union[ndarray, Tensor], y: Union[ndarray, Tensor]):
        self.x, self.y = self._check(x, y)

    def _check(self, x, y):
        """Check the dimension of x and y."""
        if isinstance(x, ndarray):
            assert isinstance(y, ndarray)
            if x.ndim == 1:
                x = np.expand_dims(x, 1)
            elif x.ndim > 2:
                raise ValueError('dimension of x should be one or two')

            if y.ndim > 2:
                raise ValueError('dimension of x should be one or two')
            x, y = torch.from_numpy(x), torch.from_numpy(y)

        elif isinstance(x, Tensor):
            assert isinstance(y, Tensor)
            if x.ndim == 1:
                x.unsqueeze(1)
            elif x.ndim() > 2:
                raise ValueError('dimension of x should be one or two')
            # if y.ndim() == 1:
            #     y.unsqueeze(1)
            if y.ndim() > 2:
                raise ValueError('dimension of x should be one or two')

        assert x.shape[0] == y.shape[0]

        return x, y
