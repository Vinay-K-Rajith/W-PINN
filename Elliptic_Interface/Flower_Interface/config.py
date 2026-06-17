import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

global device
device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
torch.manual_seed(121)
np.random.seed(0)


# ---- flower (3-lobed) interface defined as the zero level set of a harmonic-cubic phi ----
# phi = x^2 + y^2 - r0^2 + eps*(x^3 - 3 x y^2)   (Laplacian = 4 -> source f = -4)
R0, EPS = 0.5, 0.30

def phi(x, y):   return x*x + y*y - R0*R0 + EPS*(x**3 - 3*x*y*y)
def gradphi(x, y): return 2*x + 3*EPS*(x*x - y*y), 2*y - 6*EPS*x*y


class DataConfig:
    def __init__(self):
        self.x_lower, self.x_upper = -1.0, 1.0
        self.y_lower, self.y_upper = -1.0, 1.0
        self.K = 110
        self.M = 300
        self.Ktest = 240

    def _interface(self):
        th = np.linspace(0, 2*np.pi, self.M, endpoint=False); rs = np.full(self.M, R0)
        for _ in range(60):                       # Newton solve phi(r,theta)=0 along rays
            c3 = np.cos(3*th); g = rs*rs - R0*R0 + EPS*rs**3*c3; gp = 2*rs + 3*EPS*rs*rs*c3
            rs = rs - g/gp
        xg, yg = rs*np.cos(th), rs*np.sin(th)
        gx, gy = gradphi(xg, yg); nrm = np.hypot(gx, gy)
        return xg, yg, gx/nrm, gy/nrm

    def generate_points(self):
        g = np.linspace(self.x_lower, self.x_upper, self.K)
        Xg, Yg = np.meshgrid(g, g); xs, ys = Xg.ravel(), Yg.ravel()
        p = phi(xs, ys)
        inside  = p < -1e-6
        outside = (p > 1e-6) & (np.abs(xs) < 1) & (np.abs(ys) < 1)
        x_if, y_if, nx_if, ny_if = self._interface()
        nb = np.linspace(self.x_lower, self.x_upper, self.K)
        x_bc = np.concatenate([nb, nb, -np.ones(self.K), np.ones(self.K)])
        y_bc = np.concatenate([-np.ones(self.K), np.ones(self.K), nb, nb])
        gt = np.linspace(self.x_lower+1e-3, self.x_upper-1e-3, self.Ktest)
        Xt, Yt = np.meshgrid(gt, gt)
        return {'inside': (xs[inside], ys[inside]), 'outside': (xs[outside], ys[outside]),
                'interface': (x_if, y_if, nx_if, ny_if), 'boundary': (x_bc, y_bc), 'test': (Xt, Yt)}


config = DataConfig()
pts = config.generate_points()
x_in,  y_in  = pts['inside']
x_out, y_out = pts['outside']
x_if, y_if, nx_if, ny_if = pts['interface']
x_bc, y_bc = pts['boundary']
Xt, Yt = pts['test']
x_test, y_test = Xt.ravel(), Yt.ravel()
