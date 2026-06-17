from config import *

# ---- discontinuous coefficient: a = a_in inside the circle, a_out outside ----
global rho
rho   = 1000.0                 # contrast a_out / a_in  (sweep this for the contrast study)
a_in  = 1.0
a_out = rho * a_in

# ---- manufactured exact solution (level-set construction u = phi/a, phi = x^2+y^2-r0^2) ----
# gives homogeneous interface jumps [[u]]=0, [[a d_n u]]=0  and constant source f = -Delta phi = -4,
# with a genuine flux-kink across the interface: [[d_n u]] = 2 r0 (1/a_out - 1/a_in).
fval = -4.0

def phi(x, y):
    return x*x + y*y - r0*r0

def exact(x, y):
    p = phi(x, y)
    return np.where(p < 0, p / a_in, p / a_out)

# known data for the solver
u_bc       = exact(x_bc, y_bc)          # Dirichlet boundary values
u_exact_te = exact(x_test, y_test)      # reference for error reporting
jump_dn    = 2*r0*(1.0/a_out - 1.0/a_in)   # exact normal-derivative jump on the interface
