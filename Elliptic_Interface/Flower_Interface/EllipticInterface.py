from config import *

# discontinuous coefficient across the flower interface
global rho
rho   = 1000.0
a_in  = 1.0
a_out = rho * a_in

# manufactured exact solution via the level set: u = phi/a per region
# => [[u]]=0, [[a d_n u]]=0, source f = -Delta phi = -4; flux-kink VARIES along the interface.
fval = -4.0

def exact(x, y):
    p = phi(x, y)
    return np.where(p < 0, p / a_in, p / a_out)

inside_test = phi(x_test, y_test) < 0
u_bc        = exact(x_bc, y_bc)
u_exact_te  = exact(x_test, y_test)

# exact normal-derivative jump on Gamma varies as |grad phi|*(1/a_out - 1/a_in)
_gx, _gy = gradphi(x_if, y_if)
jump_dn  = np.hypot(_gx, _gy) * (1.0/a_out - 1.0/a_in)
