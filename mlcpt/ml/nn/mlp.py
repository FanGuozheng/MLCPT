#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:08:16 2021

@author: gz_fan
"""
from typing import Union, Optional, Literal
import torch
from torch.nn import Sigmoid
from torch import Tensor
from numpy import ndarray
from mlcpt.ml.nn.nn import NN


class MultiLayerPerceptron(NN):
    """Multi-layer perceptron method.

    Arguments:
        x: Input features, a 2D torch.Tensor or numpy.ndarray.
        y: Input targets, a 2D torch.Tensor or numpy.ndarray.
        n_hidden_layer: Number of hidden layer.
        lr: Learning rate in MLP.
        initialization: How to initialize neural network parameters.

    References:

    """

    def __init__(self,
                 x: Union[Tensor, ndarray],
                 y: Union[Tensor, ndarray],
                 n_hidden_layer: list = [3, 3],
                 lr: Optional[float] = 0.05,
                 max_iter: int = 1000,
                 activation: Literal['logistic', 'relu'] ='logistic',
                 initialization: Literal['random', 'zero'] = 'random'
                 ):
        # Check data type, size and dimension
        super().__init__(x, y)
        self.n_batch, self.n_feature = self.x.shape
        self.n_layer = len(n_hidden_layer)

        # add output feature shape to n_hidden_layer
        n_hidden_layer.append(self.y.shape[1])
        self.layer_params = n_hidden_layer
        self.lr = lr
        self.max_iter = max_iter
        self.activation = activation
        self.is_regression = True
        # self.initialization = initialization
        super().initialization(initialization)

        # self._initialization()
        self._fit()

    def _fit(self):
        """Train the mlti-layer perceptron model."""
        for it in range(self.max_iter):
            a = self.forward(self.x)
            self._error(self.y, a[-1])
            self.backward(a)

            if abs(a[-1] - self.y).sum() < 1E-5:
                break

    def __call__(self, xnew):
        """Run mlti-layer perceptron predictions."""
        xnew = super().__check__(xnew)
        a = self.forward(xnew)
        return a[-1].squeeze()

    def _hidden(self):
        pass

    def _activation(self, xi: Tensor):
        if self.is_regression:
            return xi
        else:
            _m = Sigmoid()
            return _m(xi)

    def optimizer(self, grad, coefficient):
        return [ic - self.lr * ig for ig, ic in zip(grad, coefficient)]

    def backward(self, activation):
        """Calculate backward propogation gradients.

        Arguments:
            activation: A list of activation results of each layer.

        Notes:
            This function

        Reference:

        """
        # last layer gradient
        _delta = [activation[-1] - self.y]
        # activation_grad = [self.sigmoid_grad(activation[-1])]

        w_grad = [self._gradient(self.n_layer, activation, _delta[0])]
        bias_grad = [torch.mean(_delta[0], 0)]

        for il in range(self.n_layer - 1, -1, -1):
            # activation_grad.extend(self.sigmoid_grad(activation[il]))
            _delta.insert(0, _delta[0] @ self.w[il + 1].T)

            # update coefficient self.w
            w_grad.insert(0, self._gradient(il, activation, _delta[0]))
            bias_grad.insert(0, torch.mean(_delta[0], 0))

        self.w = self.optimizer(w_grad, self.w)
        self.bias = self.optimizer(bias_grad, self.bias)

    def _gradient(self, layer, activation, delta):
        co = activation[layer].T @ delta
        co += self.w[layer] * 1E-5
        return co / self.n_batch

    def sigmoid_grad(self, activation):
        return activation * (1.0 - activation)

    def forward(self, x):
        """Calculate forward functions.

        Arguments:
            x: Input tensor with shape: [n_batch, n_feature].

        Returns:
            z: Dot product results of input and weights.
            activation: The activation is the output of each layer, the size
                is [[n_feature, n_hidden_0], [n_hidden_0, n_hidden_1] ...
                [n_hidden_last, n_output]].
        """
        # z is dot product result, a is activation result
        z, activation = [], [x]

        # first hidden layer: [n_batch, n_feature] @ [n_feature, n_hidden_0]
        z.append(x @ self.w[0] + self.bias[0])

        activation.append(self._activation(z[-1]))

        # Calculate the rest hidden layer and output layer
        for ih in range(self.n_layer):
            z.append(activation[-1] @ self.w[ih + 1] + self.bias[ih + 1])
            activation.append(self._activation(z[-1]))

        return activation

    def _error(self, y, pred):
        return -(y * torch.log(pred) + (1 - y) * torch.log(1 - pred)).sum()
