from config import *
from InverseProblem import phi, interface, a_minus, fval

# Tensor Gaussian-derivative wavelet family + value/Laplacian matrices precomputed ONCE at the fixed
# grid (geometry-independent). The forward() solver below selects inside/outside rows per candidate
# geometry and re-evaluates only the (cheap) interface matrices -> the inverse outer-loop stays fast.
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

len_family, JX, KX, JY, KY = wavelet_family()

def W_mats(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    X = JX[None, :]*x[:, None] - KX[None, :]; Y = JY[None, :]*y[:, None] - KY[None, :]
    E = torch.exp(-(X**2 + Y**2)/2)
    Psi = X*Y*E
    lap = -(JX[None, :]**2)*X*Y*(3 - X**2)*E - (JY[None, :]**2)*X*Y*(3 - Y**2)*E
    dx  = JX[None, :]*(1 - X**2)*Y*E
    dy  = JY[None, :]*(1 - Y**2)*X*E
    return Psi, lap, dx, dy

# precompute on the fixed grid + boundary
P_all, L_all, _, _ = W_mats(XS, YS)
P_b, _, _, _ = W_mats(x_bc, y_bc)
NF = len_family

def forward(rho, phi_grid, iface, ub, wi=10., wb=10., tik=1e-10):
    """AD-free least-squares forward solve for given contrast + geometry. Returns (cM,cP,bM,bP)."""
    import math
    xg, yg, nx, ny = iface
    inside = phi_grid < -1e-9; outside = (phi_grid > 1e-9) & insq
    Pg, _, dxg, dyg = W_mats(xg, yg)
    dnG = torch.tensor(nx)[:, None]*dxg + torch.tensor(ny)[:, None]*dyg
    n = 2*NF + 2
    def blk(rows, cM=None, cP=None, bM=None, bP=None):
        B = torch.zeros(rows, n)
        if cM is not None: B[:, :NF] = cM
        if cP is not None: B[:, NF:2*NF] = cP
        if bM is not None: B[:, 2*NF] = bM
        if bP is not None: B[:, 2*NF+1] = bP
        return B
    nin, nout, nif = int(inside.sum()), int(outside.sum()), len(xg)
    A = torch.cat([blk(nin, cM=-a_minus*L_all[inside]), blk(nout, cP=-rho*L_all[outside]),
                   math.sqrt(wi)*blk(nif, cP=Pg, cM=-Pg, bP=1., bM=-1.),
                   math.sqrt(wi)*blk(nif, cP=rho*dnG, cM=-a_minus*dnG),
                   math.sqrt(wb)*blk(len(ub), cP=P_b, bP=1.)], 0)
    b = torch.cat([torch.full((nin,), fval), torch.full((nout,), fval),
                   torch.zeros(nif), torch.zeros(nif), math.sqrt(wb)*torch.tensor(ub)]).unsqueeze(1)
    s = A.norm(dim=0).clamp_min(1e-30)
    th = (torch.linalg.lstsq(torch.cat([A/s, math.sqrt(tik)*torch.eye(n)]),
                             torch.cat([b, torch.zeros(n, 1)]), driver='gelsd').solution.squeeze(1))/s
    return th[:NF], th[NF:2*NF], th[2*NF], th[2*NF+1]

def predict_at(c, b, P): return (P@c + b).numpy()
