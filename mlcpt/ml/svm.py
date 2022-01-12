"""Support vector machine with cvxopt toolkit."""
import logging
from cvxopt import matrix, solvers
import numpy as np
import cvxopt.solvers

from mlcpt.ml._base import BaseMLModel
from mlcpt.math.kernel import Kernel
from mlcpt.ml._loss import Loss


class SVMRegression(BaseMLModel):

    def __init__(self):
        pass

    def __call__(self):
        pass

    def _lagrangian(self, x, y):
        pass


class SVMClassifier(BaseMLModel):

    def __init__(self, x, y, c=0.1, **kwargs):
        self.x = x
        self.y = y
        self.c = c
        self.multiplier = self.lagrangian()

    def __call__(self, xnew, tol=1E-4):
        return self._pred(xnew, tol)

    def lagrangian(self):
        """Solve quadratic program with solvers.qp().

        The functions in cvxopt is:
        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b.

        """
        n_batch, n_features = self.x.shape
        # P = <x, x> * <y, y> = K_x * K_y
        P = matrix((self.x @ self.x.T) * (self.y @ self.y.T))
        q = matrix(-1 * np.ones(n_batch))

        # s.t. 0 < \alpha < c ==> -\alpha < 0 and \alpha < c
        G = matrix(np.vstack((-np.diag(np.ones(n_batch)),
                             np.diag(np.ones(n_batch)))))
        h = matrix(np.concatenate((
            np.zeros(n_batch), self.c * np.ones(n_batch))).T)

        A = matrix(self.y * np.diag(np.ones(n_batch)))
        A = cvxopt.matrix(self.y, (1, n_batch))
        b = matrix(0.0)

        return np.ravel(solvers.qp(P, q, G, h, A, b)['x'])


    def _pred(self, x, tol):
        # select support vector
        mask = self.multiplier > tol
        multiplier = self.multiplier[mask]

        x_svm = self.x[mask]
        y_svm= self.y[mask]

        result = np.zeros(x.shape[0])
        for ix in x:
            for im, ix, iy in zip(multiplier, x_svm, y_svm):
                result += im * iy * ix @ x.T

        bias = (max(result) + min(result)) / 2.0
        result = result + bias

        return np.sign(result)
