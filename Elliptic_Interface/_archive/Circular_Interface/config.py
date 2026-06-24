import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Global device configuration
global device
device = torch.device('cpu')          # 1D/2D interface problems are small; CPU + float64 for clean errors
torch.set_default_dtype(torch.float64)
torch.manual_seed(121)
np.random.seed(0)


class DataConfig:
    """Geometry + collocation points for the 2D circular-interface problem on (-1,1)^2."""
    def __init__(self):
        # Domain
        self.x_lower, self.x_upper = -1.0, 1.0
        self.y_lower, self.y_upper = -1.0, 1.0
        # Interface: circle of radius r0 centred at the origin
        self.r0 = 0.5
        # Sampling
        self.K = 90          # interior grid resolution per axis
        self.M = 240         # interface collocation points
        self.Ktest = 220     # test grid resolution per axis

    def generate_points(self):
        # interior grid, split by the interface into Omega^- (inside) and Omega^+ (outside)
        g = np.linspace(self.x_lower, self.x_upper, self.K)
        Xg, Yg = np.meshgrid(g, g)
        xs, ys = Xg.ravel(), Yg.ravel()
        r = np.sqrt(xs**2 + ys**2)
        inside = r < self.r0 - 1e-9
        outside = (r > self.r0 + 1e-9) & (np.abs(xs) < 1) & (np.abs(ys) < 1)

        # interface points on the circle + outward radial normals
        th = np.linspace(0, 2*np.pi, self.M, endpoint=False)
        x_if, y_if = self.r0*np.cos(th), self.r0*np.sin(th)
        nx_if, ny_if = np.cos(th), np.sin(th)

        # outer boundary points (square edges) -- all lie in Omega^+
        nb = np.linspace(self.x_lower, self.x_upper, self.K)
        x_bc = np.concatenate([nb, nb, -np.ones(self.K), np.ones(self.K)])
        y_bc = np.concatenate([-np.ones(self.K), np.ones(self.K), nb, nb])

        # test / plotting grid
        gt = np.linspace(self.x_lower+1e-3, self.x_upper-1e-3, self.Ktest)
        Xt, Yt = np.meshgrid(gt, gt)

        return {
            'r0': self.r0,
            'inside':  (xs[inside],  ys[inside]),
            'outside': (xs[outside], ys[outside]),
            'interface': (x_if, y_if, nx_if, ny_if),
            'boundary': (x_bc, y_bc),
            'test': (Xt, Yt),
        }


config = DataConfig()
pts = config.generate_points()

r0 = pts['r0']
x_in,  y_in  = pts['inside']
x_out, y_out = pts['outside']
x_if, y_if, nx_if, ny_if = pts['interface']
x_bc, y_bc = pts['boundary']
Xt, Yt = pts['test']
x_test, y_test = Xt.ravel(), Yt.ravel()
