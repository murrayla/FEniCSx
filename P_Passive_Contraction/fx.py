"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: fx.py
        Contraction over volume from EM data informed anisotropy
"""

# +==+==+==+
# Setup
# += Imports
from dolfinx import log, io,  default_scalar_type 
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from _meshGen import msh_
from mpi4py import MPI
import pandas as pd
import numpy as np
import ufl
import os
# += Parameters
DIM = 3
INC = 10
I_0 = 0
F_0 = 0.0 
TOL = 1e-6
ZDISC = 5 
ZLINE = 14
ORDER = 2
QUADRATURE = 4
X, Y, Z = 0, 1, 2
CONSTIT_MYO = [1, 1, 1, 1]
MNR_CONS = [1, 2]
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
constitutive = 1

# +==+==+==+
# cauchy_tensor
#   Inputs: 
#       u  | obj | displacement
#       p  | obj | pressure
#       x  | obj | degrees of freedoms
#       azi | np.array | azimuth angle values
#       ele | np.array | elevation angle values
#   Outputs:
#       cauchy stress values
def cauchy_tensor(u, p, x, azi, ele, depth):
    depth += 1
    print("\t" * depth + "~> Calculate the values of cauchy stress tensor")

    # += Indices
    i, j, k = ufl.indices(3)
    # += Curvilinear mapping dependent on subdomain values
    Push = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi), ufl.cos(azi), 0],
        [0, 0, 1]
    ]) * ufl.as_matrix([
        [1, 0, 0],
        [0, ufl.cos(ele), -ufl.sin(ele)],
        [0, ufl.sin(ele), ufl.cos(ele)]
    ])
    # += Subdomain dependent rotations of displacement and coordinates
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # += Metric tensors
    Z_co = ufl.grad(x_nu).T * ufl.grad(x_nu)
    Z_ct = ufl.inv(Z_co)
    z_co = ufl.grad(nu).T * ufl.grad(nu)
    # += Kinematics variables
    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u_nu)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(z_co[i,j] - Z_co[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
    if not(constitutive):
        # += Material Setup | Guccione
        Q = (
            CONSTIT_MYO[1] * E[0,0]**2 + 
            CONSTIT_MYO[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
            CONSTIT_MYO[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
        )
        piola = CONSTIT_MYO[0]/2 * ufl.exp(Q) * ufl.as_matrix([
            [4*CONSTIT_MYO[1]*E[0,0], 2*CONSTIT_MYO[3]*(E[1,0] + E[0,1]), 2*CONSTIT_MYO[3]*(E[2,0] + E[0,2])],
            [2*CONSTIT_MYO[3]*(E[0,1] + E[1,0]), 4*CONSTIT_MYO[2]*E[1,1], 2*CONSTIT_MYO[2]*(E[2,1] + E[1,2])],
            [2*CONSTIT_MYO[3]*(E[0,2] + E[2,0]), 2*CONSTIT_MYO[2]*(E[1,2] + E[2,1]), 4*CONSTIT_MYO[3]*E[2,2]],
        ]) - p * Z_co
    else:
        # += Material Setup | Mooney-Rivlin
        Ic = ufl.variable(ufl.tr(C))
        IIc = ufl.variable((Ic**2 - ufl.inner(C, C))/2)
        psi = MNR_CONS[0] * (Ic - 3) + MNR_CONS[1] *(IIc - 3) 
        term1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
        term2 = -ufl.diff(psi, IIc)
        piola = 2 * F * (term1*I + term2*C) + p * ufl.inv(Z_co) * J * ufl.inv(F).T
    # += Calculate tensor
    sig = 1/J * F*piola*F.T
    return sig
    
# +==+==+==+
# cauchy_tensor
#   Inputs: 
#       u  | obj | displacement
#   Outputs:
#       green strain values
def green_tensor(u, depth):
    depth += 1
    print("\t" * depth + "~> Calculate the values of green strain tensor")

    # += Determine kinematics
    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    E = ufl.variable(0.5*(C-I))
    # += Assign tensor
    eps = ufl.as_tensor([
        [E[0, 0], E[0, 1], E[0, 2]], 
        [E[1, 0], E[1, 1], E[1, 2]], 
        [E[2, 0], E[2, 1], E[2, 2]]
    ])
    return eps

"""
dir_bc
  Inputs: 
      mix_vs  | obj | mixed vector space
      Vx, Vy, Vz | obj | collapsed vector spaces
      ft | np_array | facet tag data
      du | float | displacement value
  Outputs:
      numpy array of boudnary condition assignment data
"""
def dir_bc(mix_vs, Vx, Vy, Vz, ft, n_tg, l_tg, du, depth):
    depth += 1
    print("\t" * depth + "~> Apply boundary conditions")

    # += Locate subdomain dofs
    xx0_dofs, xx1_dofs, yx0_dofs, yx1_dofs, zx0_dofs, zx1_dofs = (
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x0")[0][0]])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x1")[0][0]])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x0")[0][0]])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x1")[0][0]])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x0")[0][0]])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x1")[0][0]])),
    )
    # += Interpolate 
    uxx0, uxx1, uyx0, uyx1, uzx0, uzx1 = Function(Vx), Function(Vx), Function(Vy), Function(Vy), Function(Vz), Function(Vz)
    uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-du)))
    uyx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uyx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uzx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uzx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # += Dirichlet Boundary Conditions
    bc_UxX0 = dirichletbc(value=uxx0, dofs=xx0_dofs, V=mix_vs.sub(0).sub(X))
    bc_UxX1 = dirichletbc(value=uxx1, dofs=xx1_dofs, V=mix_vs.sub(0).sub(X))
    bc_UyX0 = dirichletbc(value=uyx0, dofs=yx0_dofs, V=mix_vs.sub(0).sub(Y))
    bc_UyX1 = dirichletbc(value=uyx1, dofs=yx1_dofs, V=mix_vs.sub(0).sub(Y))
    bc_UzX0 = dirichletbc(value=uzx0, dofs=zx0_dofs, V=mix_vs.sub(0).sub(Z))
    bc_UzX1 = dirichletbc(value=uzx1, dofs=zx1_dofs, V=mix_vs.sub(0).sub(Z))
    # += Assign
    bc = [bc_UxX0, bc_UxX1, bc_UyX0, bc_UyX1, bc_UzX0, bc_UzX1]
    return bc

"""
laplacian
  Inputs: 
      mix_vs  | obj | mixed vector space
      Vx, Vy, Vz | obj | collapsed vector spaces
      ft | np_array | facet tag data
      du | float | displacement value
  Outputs:
      numpy array of boudnary condition assignment data
"""
def laplacian(coords, angles, radius, depth):
    depth += 1
    print("\t" * depth + "~> Apply Laplacian smoothing")

    # += create new values
    s_data = np.zeros_like(angles)

    # += Iterate through coordinates
    for i in range(len(coords)):
        dist = np.linalg.norm(coords - coords[i], axis=1)
        kern = np.where(dist <= radius)[0]
        s_data[i] = np.mean(angles[kern])

    return s_data

def anistropic(tnm, x_n, depth):
    depth += 1
    print("\t" * depth + "~> Load and apply anistropic fibre orientations")

    # += Load
    nmb = tnm.split("_")[1]
    ang_df = pd.read_csv("P_Passive_Contraction/_csv/vals_{}.csv".format(nmb))
    n_list = []
    f = "/Users/murrayla/Documents/main_PhD/FEniCSx/FEniCSx/P_Passive_Contraction/_msh/_mesh.nodes"

    # += Generate node coordinates
    for line in open(f, 'r'):
        n_list.append(line.strip().replace('\t', ' ').split(' '))
    node = np.array(n_list[1:]).astype(np.float64)
    cart = node[:, 1:]
    new_id = []

    # += Smooth
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])

    # += Reorganise data to match appropriate node order
    for i in range(len(coords)):
        pos = coords[i]
        dis = np.linalg.norm(pos - cart, axis=1)
        new_id.append(np.argmin(dis))

    # += Assign values
    azi_vals = ang_df["a"].to_numpy()[new_id]
    ele_vals = ang_df["e"].to_numpy()[new_id]

    # += 
    azi_vals = laplacian(coords, azi_vals, 5000, depth)
    ele_vals = laplacian(coords, ele_vals, 5000, depth)

    return azi_vals, ele_vals

# +==+==+==+
# fx_
#   Inputs: 
#       tnm  | str | test name
#       file | str | file name
#       tg_c | dict | cytosol physical element data
#       tg_S | dict | sarcomere physical element data
#   Outputs:
#       .bp folder of deformation
def fx_(tnm, file, n_tg, l_tg, depth):
    depth += 1
    print("\t" * depth + "+= Begin FE")

    # += Domain Setup
    print("\t" * depth + "+= Load Mesh and Setup Vector Spaces")
    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    V2 = ufl.VectorElement(family="Lagrange", cell=domain.ufl_cell(), degree=ORDER, quad_scheme="default")
    V1 = ufl.FiniteElement(family="Lagrange", cell=domain.ufl_cell(), degree=ORDER-1, quad_scheme="default")
    Mxs = FunctionSpace(mesh=domain, element=ufl.MixedElement([V2, V1]))
    Tes = FunctionSpace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))
    Dcs = FunctionSpace(mesh=domain, element=("Discontinuous Lagrange", 0))

    # += Extract subdomains for dofs
    print("\t" * depth + "+= Extract Subdomains")
    V, _ = Mxs.sub(0).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()
    Vz, dofsZ = V.sub(Z).collapse()
    
    # += Anistropic setup
    print("\t" * depth + "+= Setup Anistropy")
    x = ufl.SpatialCoordinate(domain)
    x_n = Function(V)
    azi, ele = Function(Vx), Function(Vx)
    if tnm.split("_")[1] == "test":
        azi.x.array[:] = np.full_like(azi.x.array[:], F_0, dtype=default_scalar_type)
        ele.x.array[:] = np.full_like(azi.x.array[:], F_0, dtype=default_scalar_type)
    else:
        azi.x.array[:], ele.x.array[:] = anistropic(tnm, x_n, depth)
        
    # += Variables
    print("\t" * depth + "+= Setup Variables")
    v, q = ufl.TestFunctions(Mxs)
    mx = Function(Mxs)
    u, p = ufl.split(mx)
    i, j, k, a, b = ufl.indices(5)

    # += Kinematics
    print("\t" * depth + "+= Setup Kinematics and Calculate Deformation")
    Push = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi), ufl.cos(azi), 0],
        [0, 0, 1]
    ]) * ufl.as_matrix([
        [1, 0, 0],
        [0, ufl.cos(ele), -ufl.sin(ele)],
        [0, ufl.sin(ele), ufl.cos(ele)]
    ])
    # += Subdomain dependent rotations of displacement and coordinates
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # += Metric tensors
    Z_co = ufl.grad(x_nu).T * ufl.grad(x_nu)
    Z_ct = ufl.inv(Z_co)
    z_co = ufl.grad(nu).T * ufl.grad(nu)
    z_ct = ufl.inv(z_co)
    # += Christoffel Symbol | Γ^{i}_{j, a}
    gamma = ufl.as_tensor((
        0.5 * Z_ct[k, a] * (
            ufl.grad(Z_co)[a, i, j] + ufl.grad(Z_co)[a, j, i] - ufl.grad(Z_co)[i, j, a]
        )
    ), [k, i, j])
    # += Covariant Derivative
    covDev = ufl.grad(v) - ufl.as_tensor(v[k]*gamma[k, i, j], [i, j])
    # += Kinematics variables
    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u_nu)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(z_co[i,j] - Z_co[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
    if not(constitutive):
        # += Material Setup | Guccione
        Q = (
            CONSTIT_MYO[1] * E[0,0]**2 + 
            CONSTIT_MYO[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
            CONSTIT_MYO[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
        )
        piola = CONSTIT_MYO[0]/2 * ufl.exp(Q) * ufl.as_matrix([
            [4*CONSTIT_MYO[1]*E[0,0], 2*CONSTIT_MYO[3]*(E[1,0] + E[0,1]), 2*CONSTIT_MYO[3]*(E[2,0] + E[0,2])],
            [2*CONSTIT_MYO[3]*(E[0,1] + E[1,0]), 4*CONSTIT_MYO[2]*E[1,1], 2*CONSTIT_MYO[2]*(E[2,1] + E[1,2])],
            [2*CONSTIT_MYO[3]*(E[0,2] + E[2,0]), 2*CONSTIT_MYO[2]*(E[1,2] + E[2,1]), 4*CONSTIT_MYO[3]*E[2,2]],
        ]) - p * ufl.inv(Z_co) * J * ufl.inv(F).T #- p * Z_co
    else:
        # += Material Setup | Mooney-Rivlin
        Ic = ufl.variable(ufl.tr(C))
        IIc = ufl.variable((Ic**2 - ufl.inner(C, C))/2)
        psi = MNR_CONS[0] * (Ic - 3) + MNR_CONS[1] *(IIc - 3) 
        term1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
        term2 = -ufl.diff(psi, IIc)
        piola = 2 * F * (term1*I + term2*C) + p * ufl.inv(Z_co) * J * ufl.inv(F).T

    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])

    # += Solver Setup
    print("\t" * depth + "+= Setup Solver")
    metadata = {"quadrature_degree": QUADRATURE}
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata=metadata)
    R = term * dx + q * (J - 1) * dx

    # log.set_log_level(log.LogLevel.INFO)

    # += Boundary Conditions
    print("\t" * depth + "+= Setup Boundary Conditions")
    disp = EDGE[0] * int(tnm.split("_")[3])/100
    bc = dir_bc(Mxs, Vx, Vy, Vz, ft, n_tg, l_tg, disp, depth)

    # += Nonlinear Solver
    print("\t" * depth + "+= Solve ...")
    problem = NonlinearProblem(R, mx, bc)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = TOL
    solver.rtol = TOL
    solver.convergence_criterion = "incremental"

    # +==+==+
    # Solution and Output
    # += Solve
    try:
        num_its, converged = solver.solve(mx)
        print("\t" * depth + " ... converged in {} its".format(num_its))
    except:
        print("\t" * depth + " ... not converged")
        return None
    
    # += Data functions for exporting with setup
    print("\t" * depth + "+= Setup Export Functions for Data Storage")
    Vu_sol, up_to_u_sol = Mxs.sub(0).collapse() 
    dis = Function(Vu_sol) 
    sig = Function(Tes)
    eps = Function(Tes)
    # += Label Functions
    dis.name = "U - Displacement"
    eps.name = "E - Green Strain"
    sig.name = "S - Cauchy Stress"
    # += File setup
    file = os.path.dirname(os.path.abspath(__file__)) + "/_bp/" + tnm
    dis_file = io.VTXWriter(MPI.COMM_WORLD, file + "_DIS" + ".bp", dis, engine="BP4")
    azi_file = io.VTXWriter(MPI.COMM_WORLD, file + "_AZI" + ".bp", azi, engine="BP4")
    ele_file = io.VTXWriter(MPI.COMM_WORLD, file + "_ELE" + ".bp", ele, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, file + "_SIG" + ".bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, file + "_EPS" + ".bp", eps, engine="BP4")
    
    # += Evaluate values
    print("\t" * depth + "+= Evaluate Tensors")
    u_eval = mx.sub(0).collapse()
    p_eval = mx.sub(1).collapse()
    dis.interpolate(u_eval)
    # += Setup tensor space for stress tensor interpolation
    cauchy = Expression(
        e=cauchy_tensor(u_eval, p_eval, x, azi, ele, depth), 
        X=Tes.element.interpolation_points()
    )
    epsilon = Expression(
        e=green_tensor(u_eval, depth), 
        X=Tes.element.interpolation_points()
    )
    sig.interpolate(cauchy)
    eps.interpolate(epsilon)
    # += Reposition tensors for output
    n_comps = 9
    sig_arr, eps_arr = sig.x.array, eps.x.array
    n_nodes = len(sig_arr) // n_comps
    r_sig = sig_arr.reshape((n_nodes, n_comps))
    r_eps = eps_arr.reshape((n_nodes, n_comps))
    # += Data storage
    pd.DataFrame(
        data = {
            "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
            "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
            "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
            "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5],
            "Azimuth": azi.x.array[:], "Elevation": ele.x.array[:],
        }
    ).to_csv(os.path.dirname(os.path.abspath(__file__)) + "/_csv/" + tnm + ".csv")

    # += Finish
    print("\t" * depth + "+= Close and End")
    # += Write files
    dis_file.write(0)
    azi_file.write(0)
    ele_file.write(0)
    sig_file.write(0)
    eps_file.write(0)
    # += Close files
    dis_file.close()
    azi_file.close()
    ele_file.close()
    sig_file.close()
    eps_file.close()

    return num_its

# +==+==+==+
# main
#   Inputs: 
#       tnm  | str | test name
#       msh  | bool | indicator of mesh generation style
#   Outputs:
#       .bp folder of deformation
def main(emfs, msh, depth):
    depth += 1
    # += Mesh generation 
    file, n_tg, l_tg = msh_("RotCube", depth)
    # += Run Mechanics
    s, f = [], []
    title = "GC_15"
    for emf in emfs:
        tnm = "_".join([emf, title])
        # += Run
        print("\t" * depth + "!! BEGIN TEST: " + tnm + " !!")
        # += Enter FENICS
        its = fx_(tnm, file, n_tg, l_tg, depth)
        if its:
            s.append(emf)
            print("\t" * depth + "!! TEST PASS: " + tnm + " !!")
        else:
            print("\t" * depth + "!! TEST FAIL: " + tnm + " !!")
            f.append(emf)
            continue
    print("\t" * depth + "!! END !!") 
    print("\t" * depth + " ~> Pass: {}".format(s))
    print("\t" * depth + " ~> Fail: {}".format(f))

# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    # += Arguments
    msh = True
    emfs = ["seg_" + x for x in ["test"] + [str(y) for y in range(0, 36, 1)]]
    main(emfs, msh, depth)
    