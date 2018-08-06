from __future__ import absolute_import, division, print_function

import argparse
import logging
import math

import numpy as np
import pandas as pd
import torch

import pyro


def f(x):

    a = torch.ones((3, 3))

    return torch.matmul(x , a), x + a[:, 1]




x = torch.ones((3, 2))
x[:, 1] *= 2
rangelis = range(2)
output = list(map(lambda i: f(x[:, i]), rangelis))
print(output[0])
# y.backward()
# print(z.grad.data)