"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: fe_opti.py
       parameter optimiser for fe script
"""

# ∆ Raw
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

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
    
# ∆ Cauchy
def cauchy_tensor(u, HLZ_CONS, p, x, azi, ele, depth):
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
    
# ∆ Run simulation
def run_simulation(HLZ_CONS, r, pct, s, tnm, env):
    depth = 1

    # ∆ Retain from environment setup
    domain = env["domain"]
    ft = env["ft"]
    Mxs = env["Mxs"]
    Tes = env["Tes"]

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
    x_n = Function(V)
    azi, ele = Function(V0x), Function(V0y)
    azi_vals = np.full_like(azi.x.array[:], 0, dtype=default_scalar_type)
    ele_vals = np.full_like(ele.x.array[:], 0, dtype=default_scalar_type)
    azi.x.array[:] = azi_vals
    ele.x.array[:] = ele_vals

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

    # log.set_log_level(log.LogLevel.INFO)

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
    sig_xx_mean = []
    for k in [0, 5, 10, 15, 20]:

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
        # µ Evaluate stress
        cauchy = Expression(
            e=cauchy_tensor(u_eval, HLZ_CONS, p_eval, x, azi, ele, depth), 
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)
       
        # ∆ Format for saving
        n_comps = 9
        sig_arr = sig.x.array
        n_nodes = len(sig_arr) // n_comps
        r_sig = sig_arr.reshape((n_nodes, n_comps))

        # ∆ Store data
        coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
        df = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
                "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5]
            }
        )

        # ∆ Retain mean value
        sig_xx_mean.append(df.loc[:, 'sig_xx'].mean())

    # ∆ Update text files on progress 
    # µ Mean data
    f = open(f"m_{r}_.txt", "a")
    for x in sig_xx_mean:
        f.write(str(x)  +  "  ")
    f.close()
    # µ Constitutive datas
    f = open(f"c_{r}_.txt", "a")
    f.write(str(HLZ_CONS) + "  ")
    f.close()

    return sig_xx_mean

# ∆ Error calculation
def error_function(HLZ_CONS, strain_exp, stress_exp, r, pct, s, tnm, env):

    sig_sim = run_simulation(HLZ_CONS, r, pct, s, tnm, env)
    if sig_sim is None:
        return 1e10

    eps_sim = np.linspace(0, pct / 100, len(sig_sim))
    interp = np.interp(strain_exp, eps_sim, sig_sim)

    mse = np.mean((interp - stress_exp) ** 2)
    return mse

# ∆ Simulation environment
def setup_simulation_environment(file, order=2, quadrature=4):

    # ∆ Domain
    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    P2 = element("Lagrange", domain.basix_cell(), order, shape=(domain.geometry.dim,))
    P1 = element("Lagrange", domain.basix_cell(), order - 1)
    Mxs = functionspace(domain, mixed_element([P2, P1]))
    Tes = functionspace(mesh=domain, element=("Lagrange", order, (DIM, DIM)))

    return {
        "domain": domain,
        "ct": ct,
        "ft": ft,
        "Mxs": Mxs,
        "Tes": Tes
    }

def my_callback(xk, convergence):
    print(f"Current best parameters: {xk}")
    print(f"Convergence: {convergence}")
    return False  # Return True to stop the optimization early


# ∆ Initiate
if __name__ == '__main__':
    
    # ∆ Test setup
    tnm = "test"  # Or your test name
    r = 1000
    file = os.path.dirname(os.path.abspath(__file__)) + f"/_msh/EMGEO_{r}.msh"  # Replace with your mesh file
    pct = 20 
    s = 1

    # ∆ Experimental data to converge to
    eps_exp = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    sig_exp = np.array([0, 0.3857, 1.1048, 1.8023, 2.6942])
    bounds = [(0.001, 50), (0.001, 50), (0.001, 50), (0.001, 50)]
    init_hlz = [0.059, 8.023, 18.472, 16.026] 

    # ∆ Setup
    env = setup_simulation_environment(file=file)
    n_cpu = multiprocessing.cpu_count()
    result = differential_evolution(
        error_function,
        bounds=bounds,
        args=(eps_exp, sig_exp, r, pct, s, tnm, env),
        strategy='best1bin',
        maxiter=50,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        workers=-1, 
        updating='deferred',
        polish=True,
        callback=my_callback
    )

    # Save final result
    np.savetxt("best_parameters.txt", result.x)
    print("Optimization complete. Best parameters:")
    print(result.x)

    # # ∆ Optimise
    # opti_param = result.x
    # print(f" += Optimized HLZ_CONS: {opti_param}")
    # print(f" += Minimum Error: {result.fun}")

    # # ∆ Run
    # opti_sig = run_simulation(opti_param, r, pct, s, file, tnm)
    # if opti_sig is not None:
    #     # ∆ Plot data values
    #     strain_sim = np.linspace(0, pct / 100, len(opti_sig))
    #     plt.plot(eps_exp, sig_exp, 'o', label='Experimental Data')
    #     plt.plot(strain_sim, opti_sig, '-', label='Optimized Simulation')
    #     plt.xlabel('Strain')
    #     plt.ylabel('Stress')
    #     plt.legend()
    #     plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/_png/" + f"{r}_mean_opti.png")
    #     plt.close()
    # else:
    #     print("Simulation failed with optimized parameters.")
        
    # print(opti_param)

    

