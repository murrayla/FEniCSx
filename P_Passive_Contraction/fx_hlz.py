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
from dolfinx import log, io,  default_scalar_type, geometry
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological, Expression, locate_dofs_geometrical
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
MNR_CONS = [1, 1]
HLZ_CONS = [x for x in [0.059, 8.023, 18.472, 16.026, 2.481, 11.120, 0.216, 11.436]]
# HLZ_CONS = [x for x in [1, 1, 1, 1, 1, 1, 1, 1]]
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

    # Rotation matrices (azimuth then elevation)
    R_azi = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi),  ufl.cos(azi), 0],
        [0,             0,            1]
    ])

    R_ele = ufl.as_matrix([
        [1, 0,             0],
        [0, ufl.cos(ele), -ufl.sin(ele)],
        [0, ufl.sin(ele),  ufl.cos(ele)]
    ])

    # Total rotation (first elevation, then azimuth)
    Push = R_azi * R_ele  
    # Apply transformation to displacement field
    u_nu = Push * u

    # Deformation gradient
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u_nu)  

    # Kinematic tensors
    C = ufl.variable(F.T * F)  # Right Cauchy-Green tensor
    B = ufl.variable(F * F.T)  # Left Cauchy-Green tensor

    # e1 = ufl.as_vector([[
    #     ufl.cos(azi) * ufl.cos(ele),
    #     ufl.sin(azi) * ufl.cos(ele),
    #     ufl.sin(ele)
    # ]])
    e1 = ufl.as_tensor([[1.0, 0.0, 0.0]]) 
    e2 = ufl.as_tensor([[0.0, 1.0, 0.0]]) 
    I4e1 = ufl.inner(e1 * C, e1)
    I4e2 = ufl.inner(e2 * C, e2)
    I8e1e2 = ufl.inner(e1 * C, e2)  

    reg = 1e-6  # Small stabilizer
    
    cond = lambda a: ufl.conditional(a > reg + 1, a, 0)

    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0]) +
        2 * HLZ_CONS[4] * cond(I4e2 - 1) * (ufl.exp(HLZ_CONS[5] * cond(I4e2 - 1) ** 2) - 1) * ufl.outer(e2[0], e2[0]) +
        HLZ_CONS[6] * I8e1e2 * ufl.exp(HLZ_CONS[7] * (I8e1e2**2)) * (ufl.outer(e1[0], e2[0]) + ufl.outer(e2[0], e1[0]))
    )
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

    # += Tags
    tgs_x0 = ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x0")[0][0]])
    tgs_x1 = ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x1")[0][0]])

    # += Locate subdomain dofs
    xx0_dofs = locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=tgs_x0)
    xx1_dofs = locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=tgs_x1)
    yx0_dofs = locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=tgs_x0)
    yx1_dofs = locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=tgs_x1)
    zx0_dofs = locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=tgs_x0)
    zx1_dofs = locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=tgs_x1)

    # += Interpolate 
    uxx0, uxx1, uyx0, uyx1, uzx0, uzx1 = Function(Vx), Function(Vx), Function(Vy), Function(Vy), Function(Vz), Function(Vz)
    uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du)))
    uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
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
def gauss_smooth(coords, angles, depth):
    depth += 1
    print("\t" * depth + "~> Apply Gaussian smoothing with anisotropy")

    # += create new values
    s_data = np.zeros_like(angles)

    # += Standard deviations
    sigma_x = 2000 / 2.0 
    sigma_y = 500 / 2.0  
    sigma_z = 500 / 2.0

    # += Iterate through coordinates
    for i in range(len(coords)):
        dist_x = coords[:, 0] - coords[i, 0]
        dist_y = coords[:, 1] - coords[i, 1]
        dist_z = coords[:, 2] - coords[i, 2]
        weights = np.exp(-0.5 * ((dist_x / sigma_x) ** 2 +
                                 (dist_y / sigma_y) ** 2 +
                                 (dist_z / sigma_z) ** 2))
        weights /= weights.sum()
        s_data[i] = np.sum(weights * angles)

    return s_data


def anistropic(tnm, msh_ref, azi_vals, ele_vals, x_n, depth):
    depth += 1
    print("\t" * depth + "~> Load and apply anistropic fibre orientations")

    # += Load
    nmb = tnm.split("_")[0]
    ang_df = pd.read_csv("P_Passive_Contraction/_csv/vals_{}_{}.csv".format("EMGEO_" + str(msh_ref), nmb))
    n_list = []
    # f = "/Users/murrayla/Documents/main_PhD/FEniCSx/FEniCSx/P_Passive_Contraction/_msh/" + "EMGEO_" + str(msh_ref) + "_" + str(nmb) + "_mesh.nodes"
    f = "/Users/murrayla/Documents/main_PhD/FEniCSx/FEniCSx/P_Passive_Contraction/_msh/" + "EMGEO_" + str(msh_ref) + "_mesh.nodes"

    # += Generate node coordinates
    for line in open(f, 'r'):
        n_list.append(line.strip().replace('\t', ' ').split(' '))
    node = np.array(n_list[1:]).astype(np.float64)
    cart = node[:, 1:]
    new_id = []

    # += Capture coordinates
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    a = ang_df["a"].to_numpy()
    e = ang_df["e"].to_numpy()

    # += Reorganise data to match appropriate node order
    for i in range(len(coords)):
        pos = coords[i]
        dis = np.linalg.norm(pos - cart, axis=1)
        idx = np.argmin(dis)
        azi_vals[i] = a[idx]
        ele_vals[i] = e[idx]

    # # += Smooth
    azi_vals = gauss_smooth(coords, azi_vals, depth)
    ele_vals = gauss_smooth(coords, ele_vals, depth)

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
def fx_(tnm, file, msh_ref, n_tg, l_tg, pct, depth):
    depth += 1
    print("\t" * depth + "+= Begin FE")

    # += Domain Setup
    print("\t" * depth + "+= Load Mesh and Setup Vector Spaces")
    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    V2 = ufl.VectorElement(family="Lagrange", cell=domain.ufl_cell(), degree=ORDER, quad_scheme="default")
    V1 = ufl.FiniteElement(family="Lagrange", cell=domain.ufl_cell(), degree=ORDER-1, quad_scheme="default")
    Mxs = FunctionSpace(mesh=domain, element=ufl.MixedElement([V2, V1]))
    Tes = FunctionSpace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))

    # print(Tes.element.interpolation_points())

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
    azi, ele = Function(Vx), Function(Vy)
    azi_vals = np.full_like(azi.x.array[:], F_0, dtype=default_scalar_type)
    ele_vals = np.full_like(ele.x.array[:], F_0, dtype=default_scalar_type)
    if tnm.split("_")[0] == "test":
        azi.x.array[:] = azi_vals
        ele.x.array[:] = ele_vals
    else:
        azi.x.array[:], ele.x.array[:] = anistropic(tnm, msh_ref, azi_vals, ele_vals, x_n, depth)

    # Rotation matrices (azimuth then elevation)
    R_azi = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi),  ufl.cos(azi), 0],
        [0,             0,            1]
    ])

    R_ele = ufl.as_matrix([
        [1, 0,             0],
        [0, ufl.cos(ele), -ufl.sin(ele)],
        [0, ufl.sin(ele),  ufl.cos(ele)]
    ])

    # Total rotation (first elevation, then azimuth)
    Push = R_azi * R_ele  

    # += Variables
    print("\t" * depth + "+= Setup Variables")
    v, q = ufl.TestFunctions(Mxs)
    mx = Function(Mxs)
    u, p = ufl.split(mx)
    i, j, k, l, a, b = ufl.indices(6)  # Increased indices for clarity

    # Apply transformation to displacement field
    u_nu = Push * u

    # Deformation gradient
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u_nu)  

    # Covariant basis vectors (undeformed)
    A1 = ufl.as_vector([
        ufl.cos(azi) * ufl.cos(ele),
        ufl.sin(azi) * ufl.cos(ele),
        ufl.sin(ele)
    ])
    A2 = ufl.as_vector([0.0, 1.0, 0.0])  
    A3 = ufl.as_vector([0.0, 0.0, 1.0])  

    # A_v = ufl.as_tensor([A1, A2, A3])
    # a_v = ufl.as_tensor(F * A_v[i] for i in range(3))  # Deformed basis vectors

    # Metric tensors
    # G_v = ufl.as_tensor(ufl.dot(A_v[i], A_v[j]), (i, j))  # Reference metric
    G_v = ufl.as_tensor([
        [ufl.dot(A1, A1), ufl.dot(A1, A2), ufl.dot(A1, A3)],
        [ufl.dot(A2, A1), ufl.dot(A2, A2), ufl.dot(A2, A3)],
        [ufl.dot(A3, A1), ufl.dot(A3, A2), ufl.dot(A3, A3)]
    ])  # Reference metric
    g_v = ufl.as_tensor([
        [ufl.dot(F * A1, F * A1), ufl.dot(F * A1, F * A2), ufl.dot(F * A1, F * A3)],
        [ufl.dot(F * A2, F * A1), ufl.dot(F * A2, F * A2), ufl.dot(F * A2, F * A3)],
        [ufl.dot(F * A3, F * A1), ufl.dot(F * A3, F * A2), ufl.dot(F * A3, F * A3)]
    ])  # Reference metric
    # g_v = ufl.as_tensor(ufl.dot(a_v[i], a_v[j]), (i, j))  # Deformed metric

    G_v_inv = ufl.inv(G_v)  
    g_v_inv = ufl.inv(g_v)

    # Christoffel symbols (using index notation)
    Gamma = ufl.as_tensor(
        0.5 * G_v_inv[k, l] * (ufl.grad(G_v[j, l])[i] + ufl.grad(G_v[i, l])[j] - ufl.grad(G_v[i, j])[l]),
        (i, j, k)
    )

    # Covariant derivative of test function
    covDev = ufl.as_tensor(ufl.grad(v)[i, j] + Gamma[i, k, j] * v[k], (i, j))

    # Kinematic tensors
    C = ufl.variable(F.T * F)  # Right Cauchy-Green tensor
    B = ufl.variable(F * F.T)  # Left Cauchy-Green tensor
    E_v = 0.5 * (g_v - G_v)    # Green-Lagrange strain
    J = ufl.det(F)             # Jacobian determinant

    # e1 = ufl.as_vector([[
    #     ufl.cos(azi) * ufl.cos(ele),
    #     ufl.sin(azi) * ufl.cos(ele),
    #     ufl.sin(ele)
    # ]])
    e1 = ufl.as_tensor([[1.0, 0.0, 0.0]]) 
    e2 = ufl.as_tensor([[0.0, 1.0, 0.0]]) 
    I4e1 = ufl.inner(e1 * C, e1)
    I4e2 = ufl.inner(e2 * C, e2)
    I8e1e2 = ufl.inner(e1 * C, e2)  

    reg = 1e-6  # Small stabilizer
    
    cond = lambda a: ufl.conditional(a > reg + 1, a, 0)

    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0]) +
        2 * HLZ_CONS[4] * cond(I4e2 - 1) * (ufl.exp(HLZ_CONS[5] * cond(I4e2 - 1) ** 2) - 1) * ufl.outer(e2[0], e2[0]) +
        HLZ_CONS[6] * I8e1e2 * ufl.exp(HLZ_CONS[7] * (I8e1e2**2)) * (ufl.outer(e1[0], e2[0]) + ufl.outer(e2[0], e1[0]))
    )
    piola = J * sig * ufl.inv(F.T) + p * ufl.inv(G_v) * J * ufl.inv(F.T)
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])

    # += Solver Setup
    print("\t" * depth + "+= Setup Solver")
    metadata = {"quadrature_degree": QUADRATURE}
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata=metadata)
    R = term * dx + q * (J - 1) * dx

    # log.set_log_level(log.LogLevel.INFO)

    # += Boundary Conditions
    print("\t" * depth + "+= Setup Boundary Conditions")

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
    file = os.path.dirname(os.path.abspath(__file__)) + "/_bp/"
    dis_file = io.VTXWriter(MPI.COMM_WORLD, file + "/DISP/_" + tnm + "_" + str(msh_ref) + "_" + str(pct) + ".bp", dis, engine="BP4")
    azi_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_AZI/_" + tnm + "_" + str(msh_ref) + ".bp", azi, engine="BP4")
    # ele_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_ELE/_" + tnm + "_" + str(msh_ref) + ".bp", ele, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_SIG/_" + tnm + "_" + str(msh_ref) + "_" + str(pct) + ".bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_EPS/_" + tnm + "_" + str(msh_ref) + "_" + str(pct) + ".bp", eps, engine="BP4")

    # disp = CUBE["x"] * PXLS["x"] * (pct / 100) 
    # bc = dir_bc(Mxs, Vx, Vy, Vz, ft, n_tg, l_tg, disp, depth)

    # # += Nonlinear Solver
    # print("\t" * depth + "+= Solve ...")
    # problem = NonlinearProblem(R, mx, bc)
    # solver = NewtonSolver(domain.comm, problem)
    # solver.atol = TOL
    # solver.rtol = TOL
    # solver.convergence_criterion = "incremental"

    # # +==+==+
    # # Solution and Output
    # # += Solve
    # try:
    #     num_its, converged = solver.solve(mx)
    #     print("\t" * depth + " ... converged in {} its".format(num_its))
    # except:
    #     print("\t" * depth + " ... not converged")
    #     return None
    
    # # += Evaluate values
    # print("\t" * depth + "+= Evaluate Tensors")
    # u_eval = mx.sub(0).collapse()
    # p_eval = mx.sub(1).collapse()
    # dis.interpolate(u_eval)
    # # += Setup tensor space for stress tensor interpolation
    # cauchy = Expression(
    #     e=cauchy_tensor(u_eval, p_eval, x, azi, ele, depth), 
    #     X=Tes.element.interpolation_points()
    # )
    # epsilon = Expression(
    #     e=green_tensor(u_eval, depth), 
    #     X=Tes.element.interpolation_points()
    # )
    # sig.interpolate(cauchy)
    # eps.interpolate(epsilon)
    # # += Reposition tensors for output
    # n_comps = 9
    # sig_arr, eps_arr = sig.x.array, eps.x.array
    # n_nodes = len(sig_arr) // n_comps
    # r_sig = sig_arr.reshape((n_nodes, n_comps))
    # r_eps = eps_arr.reshape((n_nodes, n_comps))

    # # += Coordinates
    # coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    # df = pd.DataFrame(
    #     data={
    #         "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
    #         "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
    #         "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
    #         "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
    #         "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5],
    #         "Azimuth": azi.x.array[:], "Elevation": ele.x.array[:],
    #     }
    # )

    # # Save CSV
    # csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv", f"{tnm}_{msh_ref}_15.csv")
    # df.to_csv(csv_path)   

    # # += Finish
    # print("\t" * depth + "+= Close and End")
    # # += Write files
    # dis_file.write(0)
    # azi_file.write(0)
    # # ele_file.write(0)
    # sig_file.write(0)
    # eps_file.write(0)
    v = np.arange(0,21,1)
    for k, p in enumerate(v):
        disp = CUBE["x"] * PXLS["x"] * (p / 100) 
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
            continue
        
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

        # += Coordinates
        coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
        df = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
                "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
                "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
                "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5],
                "Azimuth": azi.x.array[:], "Elevation": ele.x.array[:],
            }
        )

        # Save CSV
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv", f"{tnm}_{msh_ref}_{p}.csv")
        df.to_csv(csv_path)  

        # points = np.array([[9000, 9000, 4000], [9000, 2000, 1000], [2000, 9000, 4000], [2000, 2000, 1000]])
        # bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        # cells = []
        # points_on_proc = []
        # # Find cells whose bounding-box collide with the the points
        # cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        # # Choose one of the cells that contains the point
        # colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
        # for i, point in enumerate(points):
        #     if len(colliding_cells.links(i)) > 0:
        #         points_on_proc.append(point)
        #         cells.append(colliding_cells.links(i)[0])

        # r_sig = sig.eval(points_on_proc, cells)
        # r_eps = eps.eval(points_on_proc, cells)

        # df = pd.DataFrame(
        #     data={
        #         "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
        #         "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
        #         "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
        #         "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5]
        #     }
        # )

        # # Save CSV
        # csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv", f"{tnm}_{msh_ref}_rt.csv")
        # df.to_csv(csv_path)  

        # += Finish
        print("\t" * depth + "+= Close and End")
        # += Write files
        dis_file.write(k)
        azi_file.write(k)
        # ele_file.write(0)
        sig_file.write(k)
        eps_file.write(k)
    # += Close files
    dis_file.close()
    azi_file.close()
    # ele_file.close()
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
def main(emfs, msh_ref, depth):
    depth += 1
    # += Run Mechanics\
    l_tg = {
        0: ['Point_x0y0z1', 'Point_x0y0z0', 'Point_x0y1z1', 'Point_x0y1z0', 'Point_x1y0z1', 'Point_x1y0z0', 'Point_x1y1z1', 'Point_x1y1z0'], 
        1: ['Line_x0y0z', 'Line_x0yz1', 'Line_x0y1z', 'Line_x0yz0', 'Line_x1y0z', 'Line_x1yz1', 'Line_x1y1z', 'Line_x1yz0', 'Line_xy0z0', 'Line_xy0z1', 'Line_xy1z0', 'Line_xy1z1'], 
        2: ['Surface_x0', 'Surface_x1', 'Surface_y0', 'Surface_y1', 'Surface_z0', 'Surface_z1'], 
        3: ['Volume']}
    n_tg = {
        0: [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008], 
        1: [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512], 
        2: [1110, 1112, 1101, 1121, 1011, 1211], 
        3: [5, 6]
    }
    s, f, v = [], [], []
    for emf in emfs:
        # += Run
        print("\t" * depth + "!! BEGIN TEST: " + emf + " !!")
        # += Enter FENICS
        converged = 0
        i = 0
        # file = f'/Users/murrayla/Documents/main_PhD/FEniCSx/FEniCSx/P_Passive_Contraction/_msh/EMGEO_{msh_ref}_{emf}.msh'
        file = f'/Users/murrayla/Documents/main_PhD/FEniCSx/FEniCSx/P_Passive_Contraction/_msh/EMGEO_{msh_ref}.msh'
        # its = fx_(emf, file, msh_ref, n_tg, l_tg, 10, depth)

        while converged != 1:
            pct = (20 - i)
            if i > 20:
                print("\t" * depth + "!! TEST FAIL: " + emf + " !!")
                f.append(emf)
                converged = 1
                break
            its = fx_(emf, file, msh_ref, n_tg, l_tg, pct, depth)
            if its:
                s.append(emf)
                print("\t" * depth + "!! TEST PASS: " + emf + " !!")
                converged = 1
            else:
                i += 1
        v.append(pct)
    print("\t" * depth + "!! END !!") 
    print("\t" * depth + " ~> Pass: {}".format(s))
    print("\t" * depth + " ~> Fail: {}".format(f))
    print(v)

# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    # += Arguments
    # msh_ref = 2200
    msh_ref = [500]
    # emfs = [x for x in [str(y) for y in range(0, 36, 1)]]
    # emfs = [x for x in ["0", "2", "5"]]# + [str(y) for y in [0, 2, 5, 6, 7, 9, 10, 11, 18, 26, 29, 31]]]
    # emfs = ["seg_" + x for x in [str(y) for y in range(0, 36, 1)]]
    emfs = [x for x in [str(y) for y in ["test"]]]
    for i in msh_ref:
        main(emfs, i, depth)
    