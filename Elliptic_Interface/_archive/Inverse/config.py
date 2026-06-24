import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

global device
device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
torch.manual_seed(121); np.random.seed(0)

# Fixed collocation grid + boundary + test (geometry-INDEPENDENT points). For the inverse problem the
# geometry is unknown, so only the point CLASSIFICATION (inside/outside) and interface points change
# per candidate; the wavelet matrices at these fixed points are precomputed once (see Wfamily.py).
class DataConfig:
    def __init__(self):
        self.lo, self.hi = -1.0, 1.0
        self.K = 90
    def generate_points(self):
        g = np.linspace(self.lo, self.hi, self.K); X, Y = np.meshgrid(g, g)
        xs, ys = X.ravel(), Y.ravel()
        nb = np.linspace(self.lo, self.hi, self.K)
        x_bc = np.concatenate([nb, nb, -np.ones(self.K), np.ones(self.K)])
        y_bc = np.concatenate([-np.ones(self.K), np.ones(self.K), nb, nb])
        return xs, ys, x_bc, y_bc

config = DataConfig()
XS, YS, x_bc, y_bc = config.generate_points()
insq = (np.abs(XS) < 1) & (np.abs(YS) < 1)
