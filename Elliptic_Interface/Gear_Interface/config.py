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


# ---- gear ("settings-icon") interface defined as the zero level set of a WINDOWED smooth phi ----
# phi = (x^2+y^2) - R0^2 + B*Re((x+iy)^N)/R0^N * exp(-s*((x^2+y^2)-R0^2)^2)
#   Re((x+iy)^N)/R0^N = (r/R0)^N cos(N theta): smooth, -> 0 at origin (u well defined there);
#   the Gaussian window restores full tooth amplitude at r=R0 (deep teeth, depth set by B) and
#   decays away from Gamma so the level set stays a single closed curve.  u = phi/a per region
#   => [[u]]=0, [[a d_n u]]=0.  A harmonic phi (flower) cannot make real teeth (r^N suppressed for
#   r<1); the cost here is that the source f = -Delta phi is spatially VARYING (computed exactly by
#   autograd in EllipticInterface.py -- one-time problem data; the wavelet solve stays AD-free).
N_TEETH, R0, B, S = 8, 0.55, 0.04, 8.0          # 8 teeth, ~44% radial depth, single valid curve

def phi(x, y):                                   # numpy level-set (point classification, interface)
    x = np.asarray(x, float); y = np.asarray(y, float)
    r2 = x*x + y*y; rez = ((x + 1j*y)**N_TEETH).real
    return r2 - R0*R0 + B*rez/(R0**N_TEETH)*np.exp(-S*(r2 - R0*R0)**2)

def _phi_t(x, y):                                # torch level-set (for autograd grad & Laplacian)
    r2 = x*x + y*y; rez = ((x + 1j*y)**N_TEETH).real
    return r2 - R0*R0 + B*rez/(R0**N_TEETH)*torch.exp(-S*(r2 - R0*R0)**2)

def lap_grad(x, y):                              # exact Laplacian (-> f) and gradient (-> normals)
    xt = torch.tensor(np.asarray(x, float), requires_grad=True)
    yt = torch.tensor(np.asarray(y, float), requires_grad=True)
    p = _phi_t(xt, yt)
    gx, = torch.autograd.grad(p.sum(), xt, create_graph=True)
    gy, = torch.autograd.grad(p.sum(), yt, create_graph=True)
    gxx, = torch.autograd.grad(gx.sum(), xt, retain_graph=True)
    gyy, = torch.autograd.grad(gy.sum(), yt, retain_graph=True)
    return (gxx + gyy).detach().numpy(), gx.detach().numpy(), gy.detach().numpy()


class DataConfig:
    def __init__(self):
        self.x_lower, self.x_upper = -1.0, 1.0
        self.y_lower, self.y_upper = -1.0, 1.0
        self.K = 140
        self.M = 720
        self.Ktest = 300

    def _interface(self):
        th = np.linspace(0, 2*np.pi, self.M, endpoint=False)
        rlo = np.full(self.M, 0.15); rhi = np.full(self.M, 1.05)
        xr = lambda r: (r*np.cos(th), r*np.sin(th)); flo = phi(*xr(rlo))
        for _ in range(70):                       # bisection: phi(r,theta)=0 along each ray
            rm = 0.5*(rlo + rhi); fm = phi(*xr(rm)); left = np.sign(fm) == np.sign(flo)
            rlo = np.where(left, rm, rlo); flo = np.where(left, fm, flo); rhi = np.where(left, rhi, rm)
        rs = 0.5*(rlo + rhi); xg, yg = rs*np.cos(th), rs*np.sin(th)
        _, gx, gy = lap_grad(xg, yg); nrm = np.hypot(gx, gy)
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
