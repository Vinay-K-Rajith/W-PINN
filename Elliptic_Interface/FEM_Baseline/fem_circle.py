"""
Conforming-FEM reference for the circular elliptic-interface problem
====================================================================
The classical "gold standard" baseline for  -div(a grad u) = f  on Omega=(-1,1)^2 with a jumping
by rho across the circle Gamma = { x^2+y^2 = r0^2 }.

Why conforming FEM is the right reference
-----------------------------------------
A continuous-Galerkin FEM on a mesh whose element edges lie ON Gamma satisfies BOTH interface
conditions for free:  [[u]]=0 because the elements share nodes on Gamma (CG is continuous), and
[[a d_n u]]=0 is the natural condition of the weak form.  No special interface handling is needed.
We mesh the square and the disk together (gmsh OCC fragment) so the circle is an internal edge set.

Manufactured solution (same as the wavelet runs)
------------------------------------------------
phi = x^2+y^2-r0^2,  u = phi/a per region  =>  [[u]]=0, [[a d_n u]]=0,  f = -Delta phi = -4.
Dirichlet data on the outer square is the exact u (the square lies in the outside region).

Outputs accuracy (rel L2, Linf), degrees of freedom, and wall-time vs mesh size -> the reference
row and the FEM convergence rate to compare against the AD-free wavelet solver.
"""
import time, numpy as np, gmsh, meshio
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm, asm, condense, solve
from skfem.helpers import dot, grad

R0 = 0.5

def make_mesh(h):
    """Square (-1,1)^2 with embedded disk r=R0, conforming (circle = internal edges)."""
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    rect = gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
    disk = gmsh.model.occ.addDisk(0, 0, 0, R0, R0)
    gmsh.model.occ.fragment([(2, rect)], [(2, disk)])   # split so the circle is a shared boundary
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.model.mesh.generate(2)
    gmsh.write("_circ.msh"); gmsh.finalize()
    m = meshio.read("_circ.msh")
    pts = m.points[:, :2].T
    tris = np.vstack([c.data for c in m.cells if c.type == "triangle"]).T
    return MeshTri(pts, tris)

def solve_fem(rho, h):
    a_in, a_out = 1.0, rho
    mesh = make_mesh(h)
    basis = Basis(mesh, ElementTriP1())

    @BilinearForm
    def stiff(u, v, w):                              # a is piecewise const; mesh conforms to Gamma
        inside = (w.x[0]**2 + w.x[1]**2) < R0*R0
        a = np.where(inside, a_in, a_out)
        return a * dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        return -4.0 * v                              # f = -Delta phi = -4

    A = asm(stiff, basis); b = asm(load, basis)

    x = basis.doflocs                                 # exact u at the P1 nodes (Dirichlet + error)
    phi = x[0]**2 + x[1]**2 - R0*R0
    u_ex = np.where(phi < 0, phi/a_in, phi/a_out)

    D = basis.get_dofs()                              # exterior boundary = square edges only
    t0 = time.time()
    uh = solve(*condense(A, b, x=u_ex, D=D))          # Dirichlet = exact u on the outer square
    dt = time.time() - t0

    M = asm(BilinearForm(lambda u, v, w: u*v), basis) # mass matrix for the L2 norm
    err = uh - u_ex
    relL2 = np.sqrt(err @ (M @ err) / (u_ex @ (M @ u_ex)))
    Linf = np.max(np.abs(err))
    return dict(ndof=basis.N, h=h, relL2=relL2, Linf=Linf, sec=dt)

if __name__ == "__main__":
    for rho in [10.0, 1000.0]:
        print(f"\n===== conforming FEM (P1), circle r0={R0}, rho={rho:g} =====")
        print(f"{'h':>8} {'nDoF':>8} {'relL2':>11} {'Linf':>11} {'solve_s':>8}")
        for h in [0.1, 0.05, 0.025, 0.0125]:
            m = solve_fem(rho, h)
            print(f"{m['h']:>8.4f} {m['ndof']:>8} {m['relL2']:>11.3e} {m['Linf']:>11.3e} {m['sec']:>8.3f}",
                  flush=True)
