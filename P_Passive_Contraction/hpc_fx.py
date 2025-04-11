"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: hpc_fx.py
        Contraction over volume from EM data informed anisotropy
"""

# +==+==+==+
# Setup
# += Imports
from dolfinx import log, io,  default_scalar_type
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, locate_dofs_geometrical, Expression, element, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element, mixed_element
from mpi4py import MPI
import pandas as pd
import numpy as np
import argparse
# import basix
import ufl
import os
# += Parameters
DIM = 3
INC = 10
I_0 = 0
F_0 = 0.0
TOL = 1e-5
ZDISC = 5 
ZLINE = 14
ORDER = 2 
QUADRATURE = 4
X, Y, Z = 0, 1, 2
CONSTIT_MYO = [1, 1, 1, 1]
MNR_CONS = [1, 1]
HLZ_CONS = [x for x in [0.059, 8.023, 18.472, 16.026, 2.481, 11.120, 0.216, 11.436]]
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]

# += Calculate Cauchy Stress
def cauchy_tensor(u, p, x, azi, ele, depth):
    depth += 1
    print("\t" * depth + "~> Calculate the values of cauchy stress tensor")

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
    Push = R_azi * R_ele  
    u_nu = Push * u

    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u_nu) 
    C = ufl.variable(F.T * F) 
    B = ufl.variable(F * F.T) 

    e1 = ufl.as_tensor([[1.0, 0.0, 0.0]]) 
    I4e1 = ufl.inner(e1 * C, e1)

    reg = 1e-6 
    cond = lambda a: ufl.conditional(a > reg + 1, a, 0)
    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0])
    )

    return sig
    
# += Calculate Strain
def green_tensor(u, depth):
    depth += 1
    print("\t" * depth + "~> Calculate the values of green strain tensor")

    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    E = ufl.variable(0.5*(C-I))
    eps = ufl.as_tensor([
        [E[0, 0], E[0, 1], E[0, 2]], 
        [E[1, 0], E[1, 1], E[1, 2]], 
        [E[2, 0], E[2, 1], E[2, 2]]
    ])

    return eps

# += Smooth Data
def gauss_smooth(coords, angles, depth):
    depth += 1
    print("\t" * depth + "~> Apply Gaussian smoothing with anisotropy")

    s_data = np.zeros_like(angles)
    sigma_x = 2000 / 2.0 
    sigma_y = 500 / 2.0  
    sigma_z = 500 / 2.0

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

# += Apply fibre orientation values
def anistropic(tnm, msh_ref, azi_vals, ele_vals, x_n, depth):
    depth += 1
    print("\t" * depth + "~> Load and apply anistropic fibre orientations")

    nmb = tnm.split("_")[0]
    ang_df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "_csv/vals_{}_{}.csv".format("EMGEO_" + str(msh_ref), nmb))
    n_list = []
    f = os.path.dirname(os.path.abspath(__file__)) + "_msh/" + "EMGEO_" + str(msh_ref) + "_mesh.nodes"

    for line in open(f, 'r'):
        n_list.append(line.strip().replace('\t', ' ').split(' '))
    node = np.array(n_list[1:]).astype(np.float64)
    cart = node[:, 1:]

    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    a = ang_df["a"].to_numpy()
    e = ang_df["e"].to_numpy()

    for i in range(len(coords)):
        pos = coords[i]
        dis = np.linalg.norm(pos - cart, axis=1)
        idx = np.argmin(dis)
        azi_vals[i] = a[idx]
        ele_vals[i] = e[idx]

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
    P2 = element("Lagrange", domain.basix_cell(), ORDER, shape=(domain.geometry.dim,))
    P1 = element("Lagrange", domain.basix_cell(), ORDER-1)
    Mxs = functionspace(domain, mixed_element([P2, P1]))
    Tes = functionspace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))

    # += Extract subdomains for dofs
    print("\t" * depth + "+= Extract Subdomains")
    V, _ = Mxs.sub(0).collapse()
    V0 = Mxs.sub(0)
    V0x, _ = V0.sub(X).collapse()
    V0y, _ = V0.sub(Y).collapse()
    V0z, _ = V0.sub(Z).collapse()
    
    # += Anistropic setup
    print("\t" * depth + "+= Setup Anistropy")
    x = ufl.SpatialCoordinate(domain)
    x_n = Function(V)
    ori = Function(V)
    azi, ele = Function(V0x), Function(V0y)
    azi_vals = np.full_like(azi.x.array[:], F_0, dtype=default_scalar_type)
    ele_vals = np.full_like(ele.x.array[:], F_0, dtype=default_scalar_type)
    if tnm == "test":
        azi.x.array[:] = azi_vals
        ele.x.array[:] = ele_vals
    else:
        azi.x.array[:], ele.x.array[:] = anistropic(tnm, msh_ref, azi_vals, ele_vals, x_n, depth)

    # += Compute unit vector from azi/ele
    ori_arr = ori.x.array.reshape(-1, 3)
    ori_arr[:, 0] = np.cos(ele.x.array) * np.cos(azi.x.array) 
    ori_arr[:, 1] = np.cos(ele.x.array) * np.sin(azi.x.array) 
    ori_arr[:, 2] = np.sin(ele.x.array) 
    ori.x.array[:] = ori_arr.reshape(-1)

    # += Create Push matrix
    R_azi = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi),  ufl.cos(azi), 0],
        [0,             0,            1]
    ])
    R_ele = ufl.as_matrix([
        [1, 0, 0],
        [0, ufl.cos(ele), -ufl.sin(ele)],
        [0, ufl.sin(ele),  ufl.cos(ele)]
    ])
    Push = R_azi * R_ele  

    # += Variables
    print("\t" * depth + "+= Setup Variables")
    v, q = ufl.TestFunctions(Mxs)
    mx = Function(Mxs)
    u, p = ufl.split(mx)
    i, j, k, l, a, b = ufl.indices(6)  
    u_nu = Push * u

    # += Kinematics
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u_nu)  

    # += [UNDERFORMED] Covariant basis vectors 
    A1 = ufl.as_vector([
        ufl.cos(azi) * ufl.cos(ele),
        ufl.sin(azi) * ufl.cos(ele),
        ufl.sin(ele)
    ])
    A2 = ufl.as_vector([0.0, 1.0, 0.0])  
    A3 = ufl.as_vector([0.0, 0.0, 1.0])

    # += [UNDERFORMED] Metric tensors
    G_v = ufl.as_tensor([
        [ufl.dot(A1, A1), ufl.dot(A1, A2), ufl.dot(A1, A3)],
        [ufl.dot(A2, A1), ufl.dot(A2, A2), ufl.dot(A2, A3)],
        [ufl.dot(A3, A1), ufl.dot(A3, A2), ufl.dot(A3, A3)]
    ]) 
    G_v_inv = ufl.inv(G_v)  

    # += [DEFORMED] Metric covariant tensors
    g_v = ufl.as_tensor([
        [ufl.dot(F * A1, F * A1), ufl.dot(F * A1, F * A2), ufl.dot(F * A1, F * A3)],
        [ufl.dot(F * A2, F * A1), ufl.dot(F * A2, F * A2), ufl.dot(F * A2, F * A3)],
        [ufl.dot(F * A3, F * A1), ufl.dot(F * A3, F * A2), ufl.dot(F * A3, F * A3)]
    ])
    g_v_inv = ufl.inv(g_v)

    # += Christoffel symbols 
    Gamma = ufl.as_tensor(
        0.5 * G_v_inv[k, l] * (ufl.grad(G_v[j, l])[i] + ufl.grad(G_v[i, l])[j] - ufl.grad(G_v[i, j])[l]),
        (i, j, k)
    )

    # += Covariant derivative
    covDev = ufl.as_tensor(ufl.grad(v)[i, j] + Gamma[i, k, j] * v[k], (i, j))

    # += Kinematics 
    C = ufl.variable(F.T * F)  
    B = ufl.variable(F * F.T)  
    E_v = 0.5 * (g_v - G_v)    
    J = ufl.det(F)             

    # += Basis for Cauchy
    e1 = ufl.as_tensor([[1.0, 0.0, 0.0]]) 
    e2 = ufl.as_tensor([[0.0, 1.0, 0.0]]) 
    I4e1 = ufl.inner(e1 * C, e1)
    I4e2 = ufl.inner(e2 * C, e2)
    I8e1e2 = ufl.inner(e1 * C, e2)  

    # += Stretch condition
    reg = 1e-6  
    cond = lambda a: ufl.conditional(a > reg + 1, a, 0)

    # # += Stress
    # sig = (
    #     HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
    #     2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0]) +
    #     2 * HLZ_CONS[4] * cond(I4e2 - 1) * (ufl.exp(HLZ_CONS[5] * cond(I4e2 - 1) ** 2) - 1) * ufl.outer(e2[0], e2[0]) +
    #     HLZ_CONS[6] * I8e1e2 * ufl.exp(HLZ_CONS[7] * (I8e1e2**2)) * (ufl.outer(e1[0], e2[0]) + ufl.outer(e2[0], e1[0]))
    # )

    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0])
    )
    
    piola = J * sig * ufl.inv(F.T) + p * ufl.inv(G_v) * J * ufl.inv(F.T)

    # += Residual and Solver
    print("\t" * depth + "+= Setup Solver and Residual")
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUADRATURE})
    R = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx

    # log.set_log_level(log.LogLevel.INFO)

    # += Data functions for exporting with setup
    print("\t" * depth + "+= Setup Export Functions for Data Storage")
    dis = Function(V) 
    sig = Function(Tes)
    eps = Function(Tes)
    # += Label Functions
    dis.name = "U - Displacement"
    eps.name = "E - Green Strain"
    sig.name = "S - Cauchy Stress"
    # += File setup
    file = os.path.dirname(os.path.abspath(__file__)) + "/_bp/"
    dis_file = io.VTXWriter(MPI.COMM_WORLD, file + "/DISP/_" + tnm + "_" + str(msh_ref) + "_" + str(pct) + ".bp", dis, engine="BP4")
    ori_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_ANG/_" + tnm + "_" + str(msh_ref) + "_" + str(pct) + ".bp", ori, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_SIG/_" + tnm + "_" + str(msh_ref) + "_" + str(pct) + ".bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_EPS/_" + tnm + "_" + str(msh_ref) + "_" + str(pct) + ".bp", eps, engine="BP4")

    tgs_x0 = ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x0")[0][0]])
    tgs_x1 = ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x1")[0][0]])
    xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x0)
    xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x1)
    yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x0)
    yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x1)
    zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x0)
    zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x1)

    # += Shape iteration
    for k in range(0, pct+1, 1):

        du = CUBE["x"] * PXLS["x"] * (k / 100)
        
        d_xx0 = dirichletbc(Constant(domain, default_scalar_type(du//2)), xx0, Mxs.sub(0).sub(X))
        d_xx1 = dirichletbc(Constant(domain, default_scalar_type(-du//2)), xx1, Mxs.sub(0).sub(X))
        d_yx0 = dirichletbc(Constant(domain, default_scalar_type(0)), yx0, Mxs.sub(0).sub(Y))
        d_yx1 = dirichletbc(Constant(domain, default_scalar_type(0)), yx1, Mxs.sub(0).sub(Y))
        d_zx0 = dirichletbc(Constant(domain, default_scalar_type(0)), zx0, Mxs.sub(0).sub(Z))
        d_zx1 = dirichletbc(Constant(domain, default_scalar_type(0)), zx1, Mxs.sub(0).sub(Z))
        bc = [d_xx0, d_yx0, d_zx0, d_xx1, d_yx1, d_zx1]

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
        num_its, _ = solver.solve(mx)
        print("\t" * depth + " ... converged in {} its".format(num_its))
        
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
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv", f"{tnm}_{int(msh_ref)}_{k}_stretch.csv")
        df.to_csv(csv_path)  

        # += Finish
        print("\t" * depth + "+= Close and End")
        # += Write files
        dis_file.write(k)
        sig_file.write(k)
        eps_file.write(k)

    ori_file.write(0)
    # += Close files
    dis_file.close()
    ori_file.close()
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
    # += Run Mechanics
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

    s, f = [], []
    for emf in emfs:
        # += Run
        print("\t" * depth + "!! BEGIN TEST: " + emf + " !!")
        # += Enter FENICS
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_msh/EMGEO_" + str(msh_ref) + ".msh")
        try:
            its = fx_(emf, file, msh_ref, n_tg, l_tg, 20, depth)
            s.append(emf)
            print("\t" * depth + "!! TEST PASS: " + emf + " !!")
        except:
            f.append(emf)
            print("\t" * depth + "!! TEST FAIL: " + emf + " !!")
            continue
                
    print("\t" * depth + "!! END !!") 
    print("\t" * depth + " ~> Pass: {}".format(s))
    print("\t" * depth + " ~> Fail: {}".format(f))

# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    # += Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--test_num",type=str)
    parser.add_argument("-a", "--all_test",type=int)
    parser.add_argument("-r", "--ref_level",type=int)
    args = parser.parse_args()
    n = args.test_num
    a = args.all_test
    r = args.ref_level
    if a == 0:
        emfs = [n]
    elif a == 1:
        emfs = [x for x in [str(y) for y in range(0, 36, 1)]]
    main(emfs, r, depth)
    