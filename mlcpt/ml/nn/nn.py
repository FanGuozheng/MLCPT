#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 20:09:42 2021

@author: gz_fan
"""
from typing import Union, Optional, Literal
import torch
from torch.nn import Sigmoid
from torch import Tensor
from numpy import ndarray
from mlcpt.ml._base import BaseMLModel
from torch import Tensor


class NN(BaseMLModel):
    """Neural network algorithm.

    Arguments:
        x: Input features, a 2D torch.Tensor or numpy.ndarray.
        y: Input targets, a 2D torch.Tensor or numpy.ndarray.
        n_hidden_layer: Number of hidden layer.
        lr: Learning rate in MLP.
        initializer: How to initialize neural network parameters.

    References:

    """

    def __init__(self,
                 x: Union[Tensor, ndarray],
                 y: Union[Tensor, ndarray]):
        super().__init__(x, y)

    def __check__(self, xnew):
        """Check the input for prediction."""
        return super().__check__(xnew)

    def initialization(self, initialization: str):
        """Initialize parameters in neural network."""
        if initialization == 'xavier':
            # normalized xavier weight implementation
            n_node1 = torch.tensor([self.n_feature + self.layer_params[0]])
            low, up = -(1 / torch.sqrt(n_node1)), (1 / torch.sqrt(n_node1))
            self.w = [torch.randn(self.n_feature, self.layer_params[0])]
            self.w[0] = low + self.w[0] * (up - low)
            self.bias = [torch.zeros(self.layer_params[0])]

            for ii, jj in zip(self.layer_params[:-1], self.layer_params[1:]):
                n_nodeij = torch.tensor([ii + jj])
                low, up = -(1 / torch.sqrt(n_nodeij)), (1 / torch.sqrt(n_nodeij))
                self.w.extend([torch.randn(ii, jj)])
                self.w[-1] = low + self.w[-1] * (up - low)

            self.bias.extend([torch.zeros(jj) for jj in self.layer_params[1:]])
            import matplotlib.pyplot as plt
            plt.plot(torch.arange(len(self.w[0].flatten())), self.w[0].flatten(), 'rx')
            print(self.w[-1])
        elif initialization == 'random':
            # Initialize input layer
            self.w = [torch.randn(self.n_feature, self.layer_params[0])]
            self.bias = [torch.randn(self.layer_params[0])]

            # Initialize hidden layer and output layer
            self.w.extend([torch.randn(ii, jj) for ii, jj in zip(
                self.layer_params[:-1], self.layer_params[1:])])

            self.bias.extend([torch.randn(jj) for jj in self.layer_params[1:]])

        elif initialization == 'zeros':
            # Initialize input layer
            self.w = [torch.zeros(self.n_feature, self.layer_params[0])]
            self.bias = [torch.zeros(self.layer_params[0])]

            # Initialize hidden layer and output layer
            self.w.extend([torch.zeros(ii, jj) for ii, jj in zip(
                self.layer_params[:-1], self.layer_params[1:])])

            self.bias.extend([torch.zeros(jj) for jj in self.layer_params[1:]])

        else:
            raise NotImplementedError(f'{initialization} is not implemented.')

    def activation(self, activation: str):
        """Activation functions in neural network."""

    def optimization(self, optimization: str):
        """Optimization methods for neural network."""
