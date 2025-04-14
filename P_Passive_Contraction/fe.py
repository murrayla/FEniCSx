"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: fe.py
       consolidation script to run all components of fe simulation
"""

# ∆ Raw
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ∆ Dolfin
import ufl
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx import log, io,  default_scalar_type
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, locate_dofs_geometrical, Expression, element, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# ∆ Seed
random.seed(17081993)

# ∆ Global Constants
DIM = 3
ORDER = 2 
TOL = 1e-5
QUADRATURE = 4
X, Y, Z = 0, 1, 2
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]

# ∆ Material Properties
# HLZ_CONS = [x for x in [0.059, 8.023, 18.472, 16.026, 2.481, 11.120, 0.216, 11.436]] # raw
# HLZ_CONS = [5.02557295e-01, 1.04201471e+01, 1.00000000e-03, 4.26048750e+01] # 1
# HLZ_CONS = [0.10100884, 13.47767073, 1.92510268, 10.57663791] # 2
HLZ_CONS = [0.62213483, 10.45545208, 0.40308552, 50] # 3
# HLZ_CONS = [0.12767144, 18.80398525, 0.09138111, 8.61983857] # 4

# ∆ Stretch Plotting
def stretch(test, r, depth):
    depth += 1
    sns.set_style("whitegrid")
    print("\t" * depth + "+= Plot Stretch response")

    # ∆ Load
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv/")
    files = [file for file in os.listdir(path) if (((str(test) + "_" + str(int(r)) + "_") in file)) and ("stretch" in file)]
    pnt_sig = dict() 

    # ∆ Iterate files
    for file in files:

        # ∆ Strain value
        eps = int(file.split("_")[-2])

        # ∆ Select data
        file_df = pd.read_csv(path + file)
        f_df = file_df[
            (file_df["X"] >= 50) & (file_df["X"] <= EDGE[0] - 50) & 
            (file_df["Y"] >= 50) & (file_df["Y"] <= EDGE[1] - 50) & 
            (file_df["Z"] >= 50) & (file_df["Z"] <= EDGE[2] - 50)
        ].copy()
        
        # ∆ Isolate
        for _, row in f_df.iterrows():
            pnt_id = (round(row["X"]), round(row["Y"]), round(row["Z"]))
            if pnt_id not in pnt_sig:
                pnt_sig[pnt_id] = []
            pnt_sig[pnt_id].append((eps, row["sig_xx"]))

    # ∆ Sample
    a_keys = list(pnt_sig.keys())
    samp = random.sample(a_keys, min(50, len(a_keys)))

    # ∆ Plot setup
    _, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Major Axis Normal Stress [σ_ff] - Stretch")
    ax.set_xlabel("Strain, ΔL/L [%]")
    ax.set_ylabel("Stress [mN/mm^2]")

    # ∆ Sample
    for key in samp:
        point_series = sorted(pnt_sig[key], key=lambda x: x[0])
        strains = [s[0] for s in point_series]
        stresses = [s[1] for s in point_series]
        ax.plot(strains, stresses, alpha=0.6, linewidth=1)

    # ∆ Plot experimental values
    # µ Li
    exp_eps = [0, 5, 10, 15, 20]
    exp_sig = [0, 0.3857, 1.1048, 1.8023, 2.6942] 
    exp_sem = [0, 0.0715, 0.2257, 0.3251, 0.3999]    
    ax.errorbar(exp_eps, exp_sig, yerr=exp_sem, fmt="--o", label="Li et al. (2023)", color="tab:red")
    # µ Caporizzo
    exp_eps = [0, 2.5, 5, 7.5, 10, 11]
    exp_sig = [0, 0.24, 0.375, 0.625, 1.1, 1.27] 
    ax.plot(exp_eps, exp_sig, "--g", marker="o", label="Caporizzo et al. (2018)")

    # ∆ Save
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + f"/_png/{test}_{int(r)}_EXPXX.png")
    plt.close()

# ∆ Plot stress strain curves
def plot_(n, r, s, depth):
    depth += 1
    # += Stress Strain
    print("\t" * depth + "+= Plot Stress-Strain")
    stretch(n, r, depth)

# ∆ Cauchy
def cauchy_tensor(u, p, x, azi, ele, depth):
    depth += 1
    print("\t" * depth + "~> Calculate the values of cauchy stress tensor")
    
    # ∆ Kinematics
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u)  
    C = ufl.variable(F.T * F)  
    B = ufl.variable(F * F.T) 

    # ∆ Constitutive 
    ff = ufl.as_tensor([[1.0, 0.0, 0.0]]) 
    I4e1 = ufl.inner(ff * C, ff)
    reg = 1e-6 
    cond = lambda a: ufl.conditional(a > reg + 1, a, 0)

    # ∆ Cauchy
    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(ff[0], ff[0])
    )

    return sig
    
# ∆ Strain
def green_tensor(u, depth):
    depth += 1
    print("\t" * depth + "~> Calculate the values of green strain tensor")

    # ∆ Kinematics
    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    E = ufl.variable(0.5*(C-I))

    # ∆ Large strain
    eps = ufl.as_tensor([
        [E[0, 0], E[0, 1], E[0, 2]], 
        [E[1, 0], E[1, 1], E[1, 2]], 
        [E[2, 0], E[2, 1], E[2, 2]]
    ])

    return eps

# ∆ Gaussian smoothing
def gauss_smooth(coords, angles, zs, depth):
    depth += 1
    print("\t" * depth + "~> Apply Gaussian smoothing with anisotropy")

    # ∆ Standard deviations to smooth over
    s_data = np.zeros_like(angles)
    sd_x, sd_y, sd_z = 1000, 200, 200

    # ∆ Smooth 
    for i in range(len(coords)):
        dist_x = coords[zs, 0] - coords[i, 0]
        dist_y = coords[zs, 1] - coords[i, 1]
        dist_z = coords[zs, 2] - coords[i, 2]
        weights = np.exp(-0.5 * ((dist_x / sd_x) ** 2 +
                                 (dist_y / sd_y) ** 2 +
                                 (dist_z / sd_z) ** 2))
        weights /= weights.sum()
        s_data[i] = np.sum(weights * angles)

    return s_data

# ∆ Assign anisotropy
def anistropic(tnm, msh_ref, azi_vals, ele_vals, x_n, zs_nodes, depth):
    depth += 1
    print("\t" * depth + "~> Load and apply anistropic fibre orientations")

    # ∆ Load angle data and mesh
    nmb = tnm.split("_")[0]
    ang_df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "_csv/vals_{}_{}.csv".format("EMGEO_" + str(msh_ref), nmb))
    n_list = []
    f = os.path.dirname(os.path.abspath(__file__)) + "_msh/" + "EMGEO_" + str(msh_ref) + "_mesh.nodes"

    # ∆ Load coordinate data
    for line in open(f, 'r'):
        n_list.append(line.strip().replace('\t', ' ').split(' '))
    node = np.array(n_list[1:]).astype(np.float64)
    cart = node[:, 1:]

    # ∆ Load angles
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    a = ang_df["a"].to_numpy()
    e = ang_df["e"].to_numpy()

    # ∆ Assign angles to nodes
    zs = []
    for i in range(len(coords)):
        pos = coords[i]
        dis = np.linalg.norm(pos - cart, axis=1)
        idx = np.argmin(dis)
        if min(dis) < 500:
            azi_vals[i] = a[idx]
            ele_vals[i] = e[idx]
            zs_nodes[i] = 1
            zs.append(i)

    # ∆ Smoothing
    azi_vals = gauss_smooth(coords, azi_vals, zs, depth)
    ele_vals = gauss_smooth(coords, ele_vals, zs, depth)

    return azi_vals, ele_vals, zs_nodes

# ∆ Simulation
def fx_(tnm, file, r, pct, s, depth):
    depth += 1
    print("\t" * depth + "+= Begin FE")

    # ∆ Domain
    print("\t" * depth + "+= Load Mesh and Setup Vector Spaces")
    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    P2 = element("Lagrange", domain.basix_cell(), ORDER, shape=(domain.geometry.dim,))
    P1 = element("Lagrange", domain.basix_cell(), ORDER-1)
    Mxs = functionspace(domain, mixed_element([P2, P1]))
    Tes = functionspace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))

    # ∆ Subdomains
    print("\t" * depth + "+= Extract Subdomains")
    V, _ = Mxs.sub(0).collapse()
    V0 = Mxs.sub(0)
    V0x, _ = V0.sub(X).collapse()
    V0y, _ = V0.sub(Y).collapse()
    V0z, _ = V0.sub(Z).collapse()
    
    # ∆ Fibre field
    print("\t" * depth + "+= Setup Anistropy")
    x = ufl.SpatialCoordinate(domain)
    x_n, ori = Function(V), Function(V)
    zsn = Function(V0x)
    azi, ele, ang = Function(V0x), Function(V0y), Function(V0z)
    azi_vals = np.full_like(azi.x.array[:], 0, dtype=default_scalar_type)
    ele_vals = np.full_like(ele.x.array[:], 0, dtype=default_scalar_type)
    ang_vals = np.full_like(ang.x.array[:], 0, dtype=default_scalar_type)
    zs_nodes = np.full_like(zsn.x.array[:], 0, dtype=default_scalar_type)
    if tnm == "test":
        azi.x.array[:] = azi_vals
        ele.x.array[:] = ele_vals
    else:
        azi.x.array[:], ele.x.array[:], zsn.x.array[:] = anistropic(tnm, r, azi_vals, ele_vals, x_n, zs_nodes, depth)

    # ∆ Create angular orientation vector
    ori_arr = ori.x.array.reshape(-1, 3)
    ori_arr[:, 0] = np.cos(ele.x.array) * np.cos(azi.x.array) 
    ori_arr[:, 1] = np.cos(ele.x.array) * np.sin(azi.x.array) 
    ori_arr[:, 2] = np.sin(ele.x.array) 
    ori.x.array[:] = ori_arr.reshape(-1)
    v_x = np.array([1.0, 0.0, 0.0])
    ang_vals = np.rad2deg(np.arccos(np.clip(np.dot(ori_arr, v_x), -1.0, 1.0)))
    ang.x.array[:] = ang_vals.reshape(-1)

    # ∆ Save characteristics files
    print("\t" * depth + "+= Store geometry files")
    zsn.name = "Z-Disc Node Locations"
    ori.name = "Orientation Vectors"
    ang.name = "Angular Orientation [DEG]"
    file = os.path.dirname(os.path.abspath(__file__)) + "/_bp/"
    zsn_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_ZSN/" + tnm + "_" + str(r) + ".bp", zsn, engine="BP4")
    ori_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_ANG/" + tnm + "_" + str(r) + ".bp", ori, engine="BP4")
    sph_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_SPH/" + tnm + "_" + str(r) + ".bp", ang, engine="BP4")
    zsn_file.write(0)
    ori_file.write(0)
    sph_file.write(0)
    zsn_file.close()
    ori_file.close()
    sph_file.close()

    # ∆ Variational terms
    print("\t" * depth + "+= Setup Variables")
    mx = Function(Mxs)
    v, q = ufl.TestFunctions(Mxs)
    u, p = ufl.split(mx)

    # ∆ Fibre orientation push
    Push = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi),  ufl.cos(azi), 0],
        [0,             0,            1]
    ]) * ufl.as_matrix([
        [1, 0, 0],
        [0, ufl.cos(ele), -ufl.sin(ele)],
        [0, ufl.sin(ele),  ufl.cos(ele)]
    ])
    u_nu = Push * u

    # ∆ Kinematics Setup
    i, j, k, l, a, b = ufl.indices(6)  
    I = ufl.Identity(DIM)  
    F = ufl.variable(I + ufl.grad(u_nu))

    # ∆ Metric tensors
    # µ [UNDERFORMED] Covariant basis vectors 
    A1 = ufl.as_vector([
        ufl.cos(azi) * ufl.cos(ele),
        ufl.sin(azi) * ufl.cos(ele),
        ufl.sin(ele)
    ])
    A2 = ufl.as_vector([0.0, 1.0, 0.0])  
    A3 = ufl.as_vector([0.0, 0.0, 1.0])
    # µ [UNDERFORMED] Metric tensors
    G_v = ufl.as_tensor([
        [ufl.dot(A1, A1), ufl.dot(A1, A2), ufl.dot(A1, A3)],
        [ufl.dot(A2, A1), ufl.dot(A2, A2), ufl.dot(A2, A3)],
        [ufl.dot(A3, A1), ufl.dot(A3, A2), ufl.dot(A3, A3)]
    ]) 
    G_v_inv = ufl.inv(G_v)  
    # µ [DEFORMED] Metric covariant tensors
    g_v = ufl.as_tensor([
        [ufl.dot(F * A1, F * A1), ufl.dot(F * A1, F * A2), ufl.dot(F * A1, F * A3)],
        [ufl.dot(F * A2, F * A1), ufl.dot(F * A2, F * A2), ufl.dot(F * A2, F * A3)],
        [ufl.dot(F * A3, F * A1), ufl.dot(F * A3, F * A2), ufl.dot(F * A3, F * A3)]
    ])
    g_v_inv = ufl.inv(g_v)

    # ∆ Christoffel symbols 
    Gamma = ufl.as_tensor(
        0.5 * G_v_inv[k, l] * (ufl.grad(G_v[j, l])[i] + ufl.grad(G_v[i, l])[j] - ufl.grad(G_v[i, j])[l]),
        (i, j, k)
    )

    # ∆ Covariant derivative
    covDev = ufl.as_tensor(ufl.grad(v)[i, j] + Gamma[i, k, j] * v[k], (i, j))

    # ∆ Kinematics Tensors
    C = ufl.variable(F.T * F)  
    B = ufl.variable(F * F.T)  
    E = ufl.as_tensor(0.5 * (g_v - G_v))   
    J = ufl.det(F)   

    # ∆ Constitutive setup
    e1 = ufl.as_tensor([[
        ufl.cos(azi) * ufl.cos(ele),
        ufl.sin(azi) * ufl.cos(ele),
        ufl.sin(ele)
    ]])
    I4e1 = ufl.inner(e1 * C, e1)
    cond = lambda a: ufl.conditional(a > 1, a, 0)

    # ∆ Sigma
    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0])
    )

    # ∆ Second Piola-Kirchoff with Pressure term
    s_piola = J * ufl.inv(F) * sig * ufl.inv(F.T) + J * ufl.inv(F) * p * ufl.inv(G_v) * ufl.inv(F.T)

    # ∆ Residual
    print("\t" * depth + "+= Setup Solver and Residual")
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUADRATURE})
    R = ufl.as_tensor(s_piola[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx

    log.set_log_level(log.LogLevel.INFO)

    # ∆ Data sttore
    print("\t" * depth + "+= Setup Export Functions for Data Storage")
    dis = Function(V) 
    sig = Function(Tes)
    eps = Function(Tes)
    dis.name = "U - Displacement"
    eps.name = "E - Green Strain"
    sig.name = "S - Cauchy Stress"
    file = os.path.dirname(os.path.abspath(__file__)) + "/_bp/"
    dis_file = io.VTXWriter(MPI.COMM_WORLD, file + "/DISP/_" + tnm + "_" + str(r) + "_" + str(pct) + ".bp", dis, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_SIG/_" + tnm + "_" + str(r) + "_" + str(pct) + ".bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, file + "/_EPS/_" + tnm + "_" + str(r) + "_" + str(pct) + ".bp", eps, engine="BP4")

    # ∆ Setup boundary terms
    print("\t" * depth + "+= Boundary Conditions")
    tgs_x0 = ft.find(1110)
    tgs_x1 = ft.find(1112)
    xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x0)
    xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x1)
    yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x0)
    yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x1)
    zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x0)
    zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x1)

    # ∆ Iterate strain
    for k in range(0, pct+1, 1):

        # ∆ Apply displacement
        du = CUBE["x"] * PXLS["x"] * (k / 100)
        if s:
            d_xx0 = dirichletbc(Constant(domain, default_scalar_type(-du//2)), xx0, Mxs.sub(0).sub(X))
            d_xx1 = dirichletbc(Constant(domain, default_scalar_type(du//2)), xx1, Mxs.sub(0).sub(X))
        else:
            d_xx0 = dirichletbc(Constant(domain, default_scalar_type(du//2)), xx0, Mxs.sub(0).sub(X))
            d_xx1 = dirichletbc(Constant(domain, default_scalar_type(-du//2)), xx1, Mxs.sub(0).sub(X))
        d_yx0 = dirichletbc(Constant(domain, default_scalar_type(0)), yx0, Mxs.sub(0).sub(Y))
        d_yx1 = dirichletbc(Constant(domain, default_scalar_type(0)), yx1, Mxs.sub(0).sub(Y))
        d_zx0 = dirichletbc(Constant(domain, default_scalar_type(0)), zx0, Mxs.sub(0).sub(Z))
        d_zx1 = dirichletbc(Constant(domain, default_scalar_type(0)), zx1, Mxs.sub(0).sub(Z))
        bc = [d_xx0, d_yx0, d_zx0, d_xx1, d_yx1, d_zx1]

        # ∆ Solver
        print("\t" * depth + "+= Solve ...")
        problem = NonlinearProblem(R, mx, bc)
        solver = NewtonSolver(domain.comm, problem)
        solver.atol = TOL
        solver.rtol = TOL
        solver.convergence_criterion = "incremental"

        # ∆ Solve
        num_its, _ = solver.solve(mx)
        print("\t" * depth + " ... converged in {} its".format(num_its))
        
        # ∆ Evaluation
        print("\t" * depth + "+= Evaluate Tensors")
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        # µ Evaluate stress
        cauchy = Expression(
            e=cauchy_tensor(u_eval, p_eval, x, azi, ele, depth), 
            X=Tes.element.interpolation_points()
        )
        # µ Evaluate strain
        green = Expression(
            e=green_tensor(u_eval, depth), 
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        eps.interpolate(green)
       
        # ∆ Format for saving
        n_comps = 9
        sig_arr, eps_arr = sig.x.array, eps.x.array
        n_nodes = len(sig_arr) // n_comps
        r_sig = sig_arr.reshape((n_nodes, n_comps))
        r_eps = eps_arr.reshape((n_nodes, n_comps))

        # ∆ Store data
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

        # ∆ Save CSV
        if s:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv", f"{tnm}_{int(r)}_{k}_stretch.csv")
        else:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv", f"{tnm}_{int(r)}_{k}.csv")
        df.to_csv(csv_path)  

        # ∆ Write files
        dis_file.write(k)
        sig_file.write(k)
        eps_file.write(k)

    # ∆ Close files
    dis_file.close()
    sig_file.close()
    eps_file.close()

    return num_its

def main(args):
    depth = 1
    print("\t" * depth + "!! BEGIN FE !!")

    # ∆ Tests
    if args.all_test == 0:
        emfs = [args.test_num]
    elif args.all_test == 1:
        emfs = [x for x in [str(y) for y in range(0, 36, 1)]]

    # ∆ Iterate tests and simulate
    for emf in emfs:
        print("\t" * depth + " ~> Test: {}".format(emf))
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_msh/EMGEO_" + str(args.ref_level) + ".msh")
        fx_(emf, file, args.ref_level, args.def_level, args.stretch_test, depth)

    # ∆ Plot
    plot_(args.test_num, args.ref_level, args.stretch_test, depth)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--test_num", type=str)
    # parser.add_argument("-a", "--all_test", type=int)
    # parser.add_argument("-r", "--ref_level", type=int)
    # parser.add_argument("-p", "--def_level", type=int)
    # parser.add_argument("-s", "--stretch_test", type=int)
    # args = parser.parse_args()
    
    class args:
        def __init__(self, n, a, r, p, s, depth):
            self.test_num = n
            self.all_test = a
            self.ref_level = r
            self.b = 0
            self.def_level = p
            self.stretch_test = s
            self.depth = depth
    vals = args("test", 0, 1000, 20, 0, 1)
    main(vals)