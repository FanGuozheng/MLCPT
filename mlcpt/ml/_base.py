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

            if y.ndim < 1:
                raise ValueError(f'dimension of y should be 1 or 2, get {y.ndim}')
            elif y.ndim == 1:
                y = np.expand_dims(y, 1)
            elif y.ndim > 2:
                raise ValueError(f'dimension of y should be 1 or 2, get {y.ndim}')
            x, y = torch.from_numpy(x), torch.from_numpy(y)

        elif isinstance(x, Tensor):
            assert isinstance(y, Tensor)
            if x.dim == 1:
                x = x.unsqueeze(1)
            elif x.dim() > 2:
                raise ValueError('dimension of x should be one or two')
            if y.dim() < 1:
                raise ValueError(f'dimension of y should be 1 or 2, get {y.dim()}')
            elif y.dim() == 1:
                y = y.unsqueeze(1)
            elif y.dim() > 2:
                raise ValueError(f'dimension of y should be 1 or 2, get {y.dim()}')

        assert x.shape[0] == y.shape[0], f'batch size of x({x.shape[0]}),' + \
            f' and y({y.shape[0]}) are not same.'

        return x, y

    def split(self, train_ratio: float = 0.5, min_size: int = 3):
        """Split data into train and test."""
        n_train = int(self.x.shape[0] * train_ratio)
        n_test = self.x.shape[0] - n_train
        assert n_train >= min_size, f'train size {n_train} is smaller than {min_size}'
        assert n_test >= min_size, f'test size {n_test} is smaller than {min_size}'

        self.x_train = self.x[: n_train]
        self.x_test = self.x[n_test:]
        self.y_train = self.x[: n_train]
        self.y_test = self.x[n_test:]

    def __check__(self, xnew):
        """Check for the test input."""
        if isinstance(xnew, ndarray):
            if xnew.ndim == 1:
                xnew = np.expand_dims(xnew, 1)
            elif xnew.ndim > 2:
                raise ValueError('dimension of xnew should be one or two')
            xnew = torch.from_numpy(xnew)
        elif isinstance(xnew, Tensor):
            if xnew.dim == 1:
                xnew = xnew.unsqueeze(1)
            elif xnew.dim() > 2:
                raise ValueError('dimension of xnew should be one or two')

        assert xnew.shape[1:] == self.x.shape[1:], \
            f'The feature size should be consitent, but get {xnew.shape[1:]} and {self.x.shape[1:]}'

        return xnew
