"""Test linear machine learning module."""
import numpy as np
import torch
from linear.linear import LinearRegression, LinearClassifier
torch.set_default_dtype(torch.float64)


def test_linear_reg():
    x = np.array([[0.0, 1.0, 1.0], [2.0, 2.0, 3.0], [6.0, 4.0, 5.0]])
    y = np.array([1.0, 3.0, 8.0])
    linear = LinearRegression(x, y)
    assert (abs(linear(lr=1) - torch.tensor([0.75, 1.5, -0.5])) < 1E-13).all()


def test_linear_clf():
    x = np.array([[0.0, 1.0, 1.0], [2.0, 2.0, 3.0], [6.0, 4.0, 5.0], [2.0, 3.0, 8.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    linear = LinearClassifier(x, y)
    linear()
    assert (abs(linear.pred() - torch.from_numpy(y)) < 1E-14).all()
