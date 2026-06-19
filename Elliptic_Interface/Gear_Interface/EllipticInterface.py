from config import *

# discontinuous coefficient across the gear interface
global rho
rho   = 1000.0
a_in  = 1.0
a_out = rho * a_in

# manufactured exact solution via the level set: u = phi/a per region
# => [[u]]=0, [[a d_n u]]=0.  Unlike the flower (harmonic phi, constant f=-4), the windowed gear
# phi is NOT harmonic, so the source f = -Delta phi VARIES in space -- computed exactly by autograd.
def exact(x, y):
    p = phi(x, y)
    return np.where(p < 0, p / a_in, p / a_out)

inside_test = phi(x_test, y_test) < 0
u_bc        = exact(x_bc, y_bc)
u_exact_te  = exact(x_test, y_test)

# spatially-varying source f = -Delta phi at the interior collocation points (problem data)
_lap_in,  _gx_in,  _gy_in  = lap_grad(x_in,  y_in)
_lap_out, _gx_out, _gy_out = lap_grad(x_out, y_out)
f_in  = -_lap_in
f_out = -_lap_out

# exact normal-derivative jump on Gamma varies as |grad phi|*(1/a_out - 1/a_in)
_gx, _gy = lap_grad(x_if, y_if)[1:]
jump_dn  = np.hypot(_gx, _gy) * (1.0/a_out - 1.0/a_in)
