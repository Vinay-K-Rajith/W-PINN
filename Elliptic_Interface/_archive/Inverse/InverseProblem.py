from config import *

# General level-set interface model:  phi = x^2+y^2 - r0^2 + c2*(x^2-y^2) + c3*(x^3-3xy^2)
# (Delta phi = 4 -> f = -4).  c2 = 2-fold (ellipse) term, c3 = 3-fold (flower) term.
# Forward map: given coefficient field a (contrast rho) and geometry (r0,c2,c3), solve for u.
# Inverse: recover the unknown physical parameters from sparse, noisy interior measurements of u.
fval = -4.0
a_minus = 1.0

def phi(x, y, r0, c2, c3): return x*x + y*y - r0*r0 + c2*(x*x - y*y) + c3*(x**3 - 3*x*y*y)
def gradphi(x, y, r0, c2, c3):
    return 2*x + 2*c2*x + 3*c3*(x*x - y*y), 2*y - 2*c2*y - 6*c3*x*y

def interface(r0, c2, c3, M=240):
    th = np.linspace(0, 2*np.pi, M, endpoint=False); rs = np.full(M, max(r0, 0.3))
    for _ in range(80):
        c2t, c3t = np.cos(2*th), np.cos(3*th)
        g = rs*rs*(1 + c2*c2t) + c3*c3t*rs**3 - r0*r0
        gp = 2*rs*(1 + c2*c2t) + 3*c3*c3t*rs*rs
        rs = rs - g/np.where(np.abs(gp) < 1e-9, 1e-9, gp)
    xg, yg = rs*np.cos(th), rs*np.sin(th)
    gx, gy = gradphi(xg, yg, r0, c2, c3); nrm = np.hypot(gx, gy)
    return xg, yg, gx/nrm, gy/nrm

def truth_field(r0, c2, c3, rho):
    a_out = rho*a_minus
    def u(x, y):
        p = phi(x, y, r0, c2, c3); return np.where(p < 0, p/a_minus, p/a_out)
    return u

def radius_at(r0, c2, c3, th):          # interface radius at angles (for geometric error / plotting)
    rs = np.full_like(th, max(r0, 0.3))
    for _ in range(80):
        c2t, c3t = np.cos(2*th), np.cos(3*th)
        g = rs*rs*(1 + c2*c2t) + c3*c3t*rs**3 - r0*r0
        gp = 2*rs*(1 + c2*c2t) + 3*c3*c3t*rs*rs
        rs = rs - g/np.where(np.abs(gp) < 1e-9, 1e-9, gp)
    return rs
