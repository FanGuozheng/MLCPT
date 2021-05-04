"""Test linear machine learning module."""
import numpy as np
import torch
from gaussian.gda import GDAClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

def test_gda_clf():
    x = np.array([[0.0, 1.0, 1.0], [2.0, 2.0, 3.0], [6.0, 4.0, 5.0], [2.0, 3.0, 8.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    linear = GDAClassifier(x, y)
    linear()
    assert (abs(linear.pred() - torch.from_numpy(y)) < 1E-14).all()
test_gda_clf()