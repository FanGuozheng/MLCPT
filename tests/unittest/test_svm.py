#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:59:52 2021

@author: gz_fan
"""
import numpy as np
from mlcpt.ml import SVMClassifier


def test_cls_np():
    n_batch, n_feature = 10, 4
    x = np.random.randn(n_batch, n_feature)
    y = np.sign(x.sum(1))
    svm = SVMClassifier(x, y)
    pred = svm(x)
    print(y, '\n', pred)
    print(y == pred)


def test_reg_np():
    pass
