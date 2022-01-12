"""Linear machine learning module."""
import torch
from mlcpt.ml._base import BaseMLModel
from mlcpt.ml.loss.loss import Loss


class LinearRegression(BaseMLModel):
    """Linear optimization class.

    Parameters
    ----------
    x: Features, 1D or 2D array.
    y: Targets, 1D or 2D array.

    Returns
    -------
    w: Coefficients with bias.

    """

    def __init__(self, x, y):
        super().__init__(x, y)

    def __call__(self, lr=1, tol=1E-4, loss_update=''):
        """Return parameters with least squares."""
        return torch.inverse(self.x.T @ self.x) @ self.x.T @ self.y


class LinearClassifier(BaseMLModel):
    """Linear optimization class.

    Parameters
    ----------
    x: Features, 1D or 2D array.
    y: Targets, 1D or 2D array.

    Returns
    -------
    w: Coefficients with bias.

    """

    def __init__(self, x, y):
        super().__init__(x, y)

    def __call__(self, lr=1, tol=1E-4, loss_update=''):
        """Run linear optimization."""
        loss = Loss(self.x)

        for istep in range(10):
            self._w = loss._w
            _pred = self.pred()
            loss(_pred, self.y)

            # if (_pred == self.y).all() break

    def pred(self):
        return torch.where(self.x @ self._w[1:] + self._w[0] > 0.0, 1, -1)
