from config import *

# Tensor-product Gaussian-derivative (Mexican-hat) wavelet family, same family as the repo's
# Helmholtz/Maxwell examples. Parsimonious coarse levels keep the operational matrices well
# conditioned (the solution is smooth in each subdomain; the kink comes from the decomposition).

LEVELS = [0, 1, 2]

def wavelet_family(levels=LEVELS, lo=-1.0, hi=1.0, pad=0.5):
    f1 = []
    for l in levels:
        j = 2.0**l
        for k in range(int(np.floor((lo-pad)*j)), int(np.ceil((hi+pad)*j)) + 1):
            f1.append((j, float(k)))
    fam = [(jx, kx, jy, ky) for (jx, kx) in f1 for (jy, ky) in f1]
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
