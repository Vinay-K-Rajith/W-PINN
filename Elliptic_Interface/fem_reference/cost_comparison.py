"""
Total-cost head-to-head for GEOMETRY-VARYING problems (the mesh-free thesis test)
=====================================================================================================
Scenario: an inverse / design sweep visits N different geometries (here: a circle whose radius changes).
  * FEM must build a CONFORMING mesh for every geometry (gmsh kept initialised; we time only the
    per-geometry remesh + assemble + solve -- the init/finalise overhead is amortised, fair to FEM).
  * The wavelet method precomputes the geometry-INDEPENDENT matrix block ONCE, then per geometry only
    rebuilds the moving inclusion block + interface and re-solves the (fixed-basis) least squares.
Both compared at MATCHED accuracy (~1.5e-3 rel-L2).  Honest: report whoever wins.
Problem: circle r=R, u=phi/a (phi=x^2+y^2-R^2), f=-4, rho=10, homogeneous jumps, Dirichlet=exact u.
"""
import sys, os, time, math, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "multi_inclusion"))
from multi_inclusion import solve_physical, build_basis, matrix_precomp
import gmsh, meshio
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm, asm, condense, solve as femsolve
from skfem.helpers import dot, grad
torch.set_default_dtype(torch.float64)

RHO = 10.0
F = lambda x, y: np.full_like(np.asarray(x, float), -4.0)

# ---------------- FEM (gmsh kept initialised; time per-geometry remesh+assemble+solve) ----------------
def fem_once(R0, h):
    t0 = time.time()
    gmsh.clear()
    rect = gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2); disk = gmsh.model.occ.addDisk(0, 0, 0, R0, R0)
    gmsh.model.occ.fragment([(2, rect)], [(2, disk)]); gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.model.mesh.generate(2); gmsh.write("_c.msh")
    m = meshio.read("_c.msh"); pts = m.points[:, :2].T
    tris = np.vstack([c.data for c in m.cells if c.type == "triangle"]).T
    mesh = MeshTri(pts, tris); t_mesh = time.time() - t0
    basis = Basis(mesh, ElementTriP1())
    @BilinearForm
    def stiff(u, v, w):
        a = np.where((w.x[0]**2 + w.x[1]**2) < R0*R0, 1.0, RHO); return a*dot(grad(u), grad(v))
    @LinearForm
    def load(v, w): return -4.0*v
    A = asm(stiff, basis); b = asm(load, basis)
    x = basis.doflocs; ph = x[0]**2 + x[1]**2 - R0*R0
    u_ex = np.where(ph < 0, ph/1.0, ph/RHO)
    uh = femsolve(*condense(A, b, x=u_ex, D=basis.get_dofs()))
    M = asm(BilinearForm(lambda u, v, w: u*v), basis); err = uh - u_ex
    relL2 = np.sqrt(err @ (M @ err) / (u_ex @ (M @ u_ex)))
    return dict(relL2=relL2, ndof=basis.N, t=time.time()-t0, t_mesh=t_mesh)

# ---------------- wavelet (precompute matrix block once; per-geometry re-solve) ----------------
def wav_once(R0, fam_out, precomp):
    incl = [(0.0, 0.0, R0)]
    fams_in = build_basis(incl, coarse=(0, 1, 2, 3), band_cells=0.0, n_in_levels=3, alpha=4.0)[1]
    G = lambda x, y: (x*x + y*y - R0*R0)/RHO
    t0 = time.time()
    pred = solve_physical(incl, [1.0], RHO, F, G, basis=(fam_out, fams_in), precomp=precomp, K=140, M=400)
    t = time.time() - t0
    gt = np.linspace(-0.999, 0.999, 200); Xt, Yt = np.meshgrid(gt, gt); xt = Xt.ravel(); yt = Yt.ravel()
    u = pred(xt, yt); ph = xt*xt + yt*yt - R0*R0; ue = np.where(ph < 0, ph, ph/RHO)
    return dict(relL2=np.linalg.norm(u-ue)/np.linalg.norm(ue), NF=fam_out[0].numel()+fams_in[0][0].numel(), t=t)

if __name__ == "__main__":
    radii = np.linspace(0.35, 0.65, 12)            # the N geometries an inverse/design sweep visits
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    print("FEM (conforming remesh per geometry, h=0.1):")
    fem = [fem_once(R, 0.1) for R in radii]
    gmsh.finalize()
    print(f"  per-geom: mesh {np.mean([f['t_mesh'] for f in fem]):.3f}s  total "
          f"{np.mean([f['t'] for f in fem]):.3f}s  relL2 {np.mean([f['relL2'] for f in fem]):.2e}  "
          f"ndof {int(np.mean([f['ndof'] for f in fem]))}")

    fam_out = build_basis([(0.0, 0.0, 0.5)], coarse=(0, 1, 2, 3), band_cells=0.0)[0]
    t0 = time.time(); precomp = matrix_precomp(fam_out, 140); t_pre = time.time() - t0
    print(f"\nwavelet (precompute matrix block once: {t_pre:.2f}s, then re-solve per geometry):")
    wav = [wav_once(R, fam_out, precomp) for R in radii]
    print(f"  per-geom: solve {np.mean([w['t'] for w in wav]):.3f}s  relL2 "
          f"{np.mean([w['relL2'] for w in wav]):.2e}  NF {wav[0]['NF']}")

    N = len(radii)
    fem_total = sum(f['t'] for f in fem)
    wav_total = t_pre + sum(w['t'] for w in wav)
    print(f"\n=== TOTAL for N={N} geometries (matched ~1.5e-3 accuracy) ===")
    print(f"  FEM     : {fem_total:6.2f}s  ({fem_total/N:.3f}s/geom, all remesh)")
    print(f"  wavelet : {wav_total:6.2f}s  ({t_pre:.2f}s once + {(wav_total-t_pre)/N:.3f}s/geom)")
    print(f"  ratio FEM/wavelet = {fem_total/wav_total:.2f}  (>1 => wavelet wins total cost)")
