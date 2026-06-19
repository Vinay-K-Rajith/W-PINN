from config import *
import math

# Tensor-product Gaussian-derivative (Mexican-hat) wavelet family, same family as the repo's
# Helmholtz/Maxwell examples, but with BANDED multiresolution refinement: coarse levels are placed
# globally, while the finest levels are placed ONLY in the annular band around the gear teeth
# (where the centre (kx/jx, ky/jy) lies in r in BAND).  This resolves the sharp teeth cheaply --
# a few hundred extra functions instead of ~5000 for a global finest level -- and is exactly the
# adaptive multiresolution behaviour a fixed-coordinate vanilla PINN cannot provide.

COARSE = [0, 1, 2, 3]      # placed everywhere
FINE   = [4, 5]            # placed only inside BAND (the teeth annulus)
BAND   = (0.42, 0.86)

def wavelet_family(coarse=COARSE, fine=FINE, band=BAND, lo=-1.0, hi=1.0, pad=0.5):
    def lvl1d(levels):
        out = []
        for l in levels:
            j = 2.0**l
            for k in range(int(np.floor((lo-pad)*j)), int(np.ceil((hi+pad)*j)) + 1):
                out.append((j, float(k)))
        return out
    f1c = lvl1d(coarse)
    fam = [(jx, kx, jy, ky) for (jx, kx) in f1c for (jy, ky) in f1c]      # coarse: global product
    rlo, rhi = band
    for l in fine:                                                       # fine: banded by centre
        j = 2.0**l
        ks = [float(k) for k in range(int(np.floor((lo-pad)*j)), int(np.ceil((hi+pad)*j)) + 1)]
        for kx in ks:
            for ky in ks:
                if rlo <= math.hypot(kx/j, ky/j) <= rhi:
                    fam.append((j, kx, j, ky))
    arr = torch.tensor(fam, dtype=torch.float64, device=device)
    return len(arr), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def gaussian(x, y, jx, kx, jy, ky):
    X = jx[None, :]*x[:, None] - kx[None, :]
    Y = jy[None, :]*y[:, None] - ky[None, :]
    return X * Y * torch.exp(-(X**2 + Y**2)/2)

def laplacian(x, y, jx, kx, jy, ky):
    X = jx[None, :]*x[:, None] - kx[None, :]
    Y = jy[None, :]*y[:, None] - ky[None, :]
    E = torch.exp(-(X**2 + Y**2)/2)
    return -(jx[None, :]**2)*X*Y*(3 - X**2)*E - (jy[None, :]**2)*X*Y*(3 - Y**2)*E

def Dxgaussian(x, y, jx, kx, jy, ky):
    X = jx[None, :]*x[:, None] - kx[None, :]
    Y = jy[None, :]*y[:, None] - ky[None, :]
    return jx[None, :]*(1 - X**2)*Y*torch.exp(-(X**2 + Y**2)/2)

def Dygaussian(x, y, jx, kx, jy, ky):
    X = jx[None, :]*x[:, None] - kx[None, :]
    Y = jy[None, :]*y[:, None] - ky[None, :]
    return jy[None, :]*(1 - Y**2)*X*torch.exp(-(X**2 + Y**2)/2)


def _t(a):
    return torch.as_tensor(a, dtype=torch.float64, device=device)

len_family, jx, kx, jy, ky = wavelet_family()
print("family_len:", len_family)

# operational matrices on each point set ----------------------------------------------------
xin, yin   = _t(x_in),  _t(y_in)
xout, yout = _t(x_out), _t(y_out)
xif, yif   = _t(x_if),  _t(y_if)
nxif, nyif = _t(nx_if), _t(ny_if)
xbc, ybc   = _t(x_bc),  _t(y_bc)
xte, yte   = _t(x_test), _t(y_test)

# interior: value + Laplacian (for the PDE residual)
Win,  Lin  = gaussian(xin, yin, jx, kx, jy, ky),   laplacian(xin, yin, jx, kx, jy, ky)
Wout, Lout = gaussian(xout, yout, jx, kx, jy, ky), laplacian(xout, yout, jx, kx, jy, ky)

# interface: value (continuity) + normal derivative (flux)
Wif = gaussian(xif, yif, jx, kx, jy, ky)
DnIf = nxif[:, None]*Dxgaussian(xif, yif, jx, kx, jy, ky) + nyif[:, None]*Dygaussian(xif, yif, jx, kx, jy, ky)

# boundary + test value matrices
Wbc  = gaussian(xbc, ybc, jx, kx, jy, ky)
Wtest = gaussian(xte, yte, jx, kx, jy, ky)
