"""Gaussian discriminant analysis."""
import torch
import numpy as np
from _base import BaseMLModel
from _math.linalg import cov
pi = np.pi

class GDAClassifier(BaseMLModel):
    """Gaussian discriminant analysis classifier.

    Parameters
    ----------
    x: Fearues, array-like of shape, 1D or 2D.

    y: Targets.

    """

    def __init__(self, x, y):
        super().__init__(x, y)

    def __call__(self):
        self.cov = cov(self.x)
        _y = torch.unique(self.y)

        n_label = len(_y)

        _prob = []
        for ii, ilabel in enumerate(n_label):
            ix = self.x[self.y == ilabel, :]
            _cov = cov(ix)
            _ix = ix - torch.mean(ix)
            _prob.append(1 / (np.linalg.det(2 * pi * _cov) ** 0.5) *
                         np.exp(-0.5 * _ix.T @ torch.inv(_cov) @ _ix))

    def pred(self):
        pass