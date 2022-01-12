#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:45:29 2022

@author: gz_fan
"""
import numpy as np
import torch
import functools


def fix_seed(function):
    """This is a function which fix the seed, the same in TBMaLT."""

    # Use functools.wraps to maintain the original function's docstring
    @functools.wraps(function)
    def wrapper(*args, **kwargs):

        # Set both numpy's and pytorch's seed to zero
        np.random.seed(0)
        torch.manual_seed(0)

        # Call the function and return its result
        return function(*args, **kwargs)

    return wrapper
