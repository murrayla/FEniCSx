from dolfinx import log, io, default_scalar_type
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, Expression, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element, mixed_element
from mpi4py import MPI
import pandas as pd
import numpy as np
import scipy.optimize as opt
import ufl
import os
    
# += Calculate Strain
def green_tensor(u, depth):
    depth += 1
    print("\t" * depth + "~> Calculate the values of green strain tensor")

    I = ufl.variable(ufl.Identity(3))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    E = ufl.variable(0.5*(C-I))
    eps = ufl.as_tensor([
        [E[0, 0], E[0, 1], E[0, 2]], 
        [E[1, 0], E[1, 1], E[1, 2]], 
        [E[2, 0], E[2, 1], E[2, 2]]
    ])

    return eps

# Modified cauchy_tensor function to accept HLZ_CONS as an argument
def cauchy_tensor(u, HLZ_CONS, azi, ele, p, depth):
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

    I = ufl.Identity(3)  
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

    # += [DEFORMED] Metric covariant tensors
    g_v = ufl.as_tensor([
        [ufl.dot(F * A1, F * A1), ufl.dot(F * A1, F * A2), ufl.dot(F * A1, F * A3)],
        [ufl.dot(F * A2, F * A1), ufl.dot(F * A2, F * A2), ufl.dot(F * A2, F * A3)],
        [ufl.dot(F * A3, F * A1), ufl.dot(F * A3, F * A2), ufl.dot(F * A3, F * A3)]
    ])

    
    C = ufl.variable(F.T * F) 
    # E = ufl.as_tensor(0.5 * (g_v - G_v))  
    B = ufl.variable(F * F.T) 
    J = ufl.det(F)

    # e1 = ufl.as_tensor([[1.0, 0.0, 0.0]]) 
    # I4e1 = ufl.inner(e1 * C, e1)

    # reg = 1e-6 
    # cond = lambda a: ufl.conditional(a > reg + 1, a, 0)
    # sig = (
    #     HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
    #     2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0])
    # )

    # Psi = (
    #     HLZ_CONS[0]/(2*HLZ_CONS[1]) * (ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) - 1) + 
    #     HLZ_CONS[2]/(2*HLZ_CONS[3]) * (ufl.exp(HLZ_CONS[3] * (cond(I4e1 - 1) ** 2)) - 1)
    # )

    e1 = e1 = ufl.as_tensor([[
        ufl.cos(azi) * ufl.cos(ele),
        ufl.sin(azi) * ufl.cos(ele),
        ufl.sin(ele)
    ]])
    I4e1 = ufl.inner(e1 * C, e1)

    reg = 1e-6 
    cond = lambda a: ufl.conditional(a > reg + 1, a, 0)

    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0])
    ) #- p * ufl.inv(G_v)

    return sig

def run_simulation(HLZ_CONS, r, pct, s, file, tnm):
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
    """Encapsulates the Dolfinx simulation and returns stress values."""
    DIM = 3
    ORDER = 2
    QUADRATURE = 4
    X, Y, Z = 0, 1, 2
    PXLS = {"x": 11, "y": 11, "z": 50}
    CUBE = {"x": 1000, "y": 1000, "z": 100}
    EDGE = [PXLS[d] * CUBE[d] for d in ["x", "y", "z"]]
    depth = 0

    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    P2 = element("Lagrange", domain.basix_cell(), ORDER, shape=(domain.geometry.dim,))
    P1 = element("Lagrange", domain.basix_cell(), ORDER - 1)
    Mxs = functionspace(domain, mixed_element([P2, P1]))
    Tes = functionspace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))

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
    azi_vals = np.full_like(azi.x.array[:], 0.0, dtype=default_scalar_type)
    ele_vals = np.full_like(ele.x.array[:], 0.0, dtype=default_scalar_type)
    azi.x.array[:] = azi_vals
    ele.x.array[:] = ele_vals

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
    E = ufl.as_tensor(0.5 * (g_v - G_v))   
    J = ufl.det(F) 

    # Q = (
    #     CONSTIT_MYO[1] * E[0,0]**2 + 
    #     CONSTIT_MYO[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
    #     CONSTIT_MYO[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
    # )
    # s_piola = CONSTIT_MYO[0]/4 * ufl.exp(Q) * ufl.as_matrix([
    #     [4*CONSTIT_MYO[1]*E[0,0], 2*CONSTIT_MYO[3]*(E[1,0] + E[0,1]), 2*CONSTIT_MYO[3]*(E[2,0] + E[0,2])],
    #     [2*CONSTIT_MYO[3]*(E[0,1] + E[1,0]), 4*CONSTIT_MYO[2]*E[1,1], 2*CONSTIT_MYO[2]*(E[2,1] + E[1,2])],
    #     [2*CONSTIT_MYO[3]*(E[0,2] + E[2,0]), 2*CONSTIT_MYO[2]*(E[1,2] + E[2,1]), 4*CONSTIT_MYO[3]*E[2,2]],
    # ]) - p * ufl.inv(G_v)
    # f_piola = F * s_piola #- p * ufl.inv(G_v) * J * ufl.inv(F).T            

    # += Basis for Cauchy
    # e1 = ufl.as_tensor([[1.0, 0.0, 0.0]]) 
    e1 = ufl.as_tensor([[
        ufl.cos(azi) * ufl.cos(ele),
        ufl.sin(azi) * ufl.cos(ele),
        ufl.sin(ele)
    ]]) 
    # e2 = ufl.as_tensor([[0.0, 1.0, 0.0]]) 
    I4e1 = ufl.inner(e1 * C, e1)
    # I4e2 = ufl.inner(e2 * C, e2)
    # I8e1e2 = ufl.inner(e1 * C, e2)  

    # += Stretch condition
    # reg = 1e-6  
    cond = lambda a: ufl.conditional(a > 1, a, 0)

    # # # += Stress
    # # sig = (
    # #     HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
    # #     2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0]) +
    # #     2 * HLZ_CONS[4] * cond(I4e2 - 1) * (ufl.exp(HLZ_CONS[5] * cond(I4e2 - 1) ** 2) - 1) * ufl.outer(e2[0], e2[0]) +
    # #     HLZ_CONS[6] * I8e1e2 * ufl.exp(HLZ_CONS[7] * (I8e1e2**2)) * (ufl.outer(e1[0], e2[0]) + ufl.outer(e2[0], e1[0]))
    # # )

    # Psi = (
    #     HLZ_CONS[0]/(2*HLZ_CONS[1]) * (ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) - 1) + 
    #     HLZ_CONS[2]/(2*HLZ_CONS[3]) * (ufl.exp(HLZ_CONS[3] * (cond(I4e1 - 1) ** 2)) - 1)
    # )

    sig = (
        HLZ_CONS[0] * ufl.exp(HLZ_CONS[1] * (ufl.tr(C) - 3)) * B +
        2 * HLZ_CONS[2] * cond(I4e1 - 1) * (ufl.exp(HLZ_CONS[3] * cond(I4e1 - 1) ** 2) - 1) * ufl.outer(e1[0], e1[0])
    )

    # s_piola = J * ufl.inv(F) * sig * ufl.inv(F.T) - p * ufl.inv(G_v) * J * ufl.inv(F.T)
    s_piola = J * ufl.inv(F) * sig * ufl.inv(F.T) - p * ufl.inv(G_v + 2 * E)
    # piola = J * df.diff(Psi, F) * df.inv(F.T) + p * ufl.inv(G_v) * J * df.inv(F.T)
    
    # piola = J * sig * ufl.inv(F.T) + p * ufl.inv(G_v) * J * ufl.inv(F.T)

    # += Residual and Solver
    print("\t" * depth + "+= Setup Solver and Residual")
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUADRATURE})
    R = ufl.as_tensor(s_piola[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx

    # log.set_log_level(log.LogLevel.INFO)7

    tgs_x0 = ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x0")[0][0]])
    tgs_x1 = ft.find(n_tg[2][np.where(np.array(l_tg[2]) == "Surface_x1")[0][0]])
    xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x0)
    xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x1)
    yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x0)
    yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x1)
    zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x0)
    zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x1)
    sig = Function(Tes)
    stress_values = []
    stress_mean = []
    # += Shape iteration
    for k in [0, 5, 10, 15, 20]: #range(0, pct+1, 2):

        du = CUBE["x"] * PXLS["x"] * (k / 100)
        
        if s:
            d_xx0 = dirichletbc(Constant(domain, default_scalar_type(du//2)), xx0, Mxs.sub(0).sub(X))
            d_xx1 = dirichletbc(Constant(domain, default_scalar_type(-du//2)), xx1, Mxs.sub(0).sub(X))
        else:
            d_xx0 = dirichletbc(Constant(domain, default_scalar_type(-du//2)), xx0, Mxs.sub(0).sub(X))
            d_xx1 = dirichletbc(Constant(domain, default_scalar_type(du//2)), xx1, Mxs.sub(0).sub(X))
        d_yx0 = dirichletbc(Constant(domain, default_scalar_type(0)), yx0, Mxs.sub(0).sub(Y))
        d_yx1 = dirichletbc(Constant(domain, default_scalar_type(0)), yx1, Mxs.sub(0).sub(Y))
        d_zx0 = dirichletbc(Constant(domain, default_scalar_type(0)), zx0, Mxs.sub(0).sub(Z))
        d_zx1 = dirichletbc(Constant(domain, default_scalar_type(0)), zx1, Mxs.sub(0).sub(Z))
        bc = [d_xx0, d_yx0, d_zx0, d_xx1, d_yx1, d_zx1]

        # += Nonlinear Solver
        print("\t" * depth + "+= Solve ...")
        problem = NonlinearProblem(R, mx, bc)
        solver = NewtonSolver(domain.comm, problem)
        solver.atol = 1e-5
        solver.rtol = 1e-5
        solver.convergence_criterion = "incremental"


        # +==+==+
        # Solution and Output
        # += Solve
        try:
            num_its, converged = solver.solve(mx)
        except Exception as e:
            print(f"Dolfinx error encountered during solve: {e}")
            return None  # Return None to indicate failure

        if not converged:
            print(f"Solver did not converge for pct = {k}")
            return None  # Return None to indicate failure

        print("\t" * depth + " ... converged in {} its".format(num_its))

        # Extract the stress at the center of the domain.  We'll assume
        # the center is at (0, 0, Z/2).  You may need to adjust this
        # depending on your mesh.
        # center = np.array([[CUBE["x"] / 2, CUBE["y"] / 2, CUBE["z"] / 2]])
        print("\t" * depth + "+= Evaluate Tensors")
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        cauchy = Expression(
            e=cauchy_tensor(u_eval, HLZ_CONS, azi, ele, p_eval, depth), 
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        n_comps = 9
        sig_arr = sig.x.array
        n_nodes = len(sig_arr) // n_comps
        r_sig = sig_arr.reshape((n_nodes, n_comps))
        stress_values.append(np.max(r_sig[:, 0]))  # Get xx component
        stress_mean.append(np.mean(r_sig[:, 0]))

    print(sig_arr)
    print(stress_mean)
    print(HLZ_CONS)

    # f = open("stress_vals_pos.txt", "a")
    # for x in stress_values:
    #     f.write(str(x))
    # f.close()

    f = open("m_1000_p_.txt", "a")
    for x in stress_mean:
        f.write(str(x)  +  "  ")
    f.close()

    f = open("c_1000_p_.txt", "a")
    f.write(str(HLZ_CONS) + "  ")
    f.close()
    return stress_mean


def error_function(HLZ_CONS, strain_exp, stress_exp, r, pct, s, file, tnm):
    """Calculates the Mean Squared Error between simulation and experiment."""
    stress_sim = run_simulation(HLZ_CONS, r, pct, s, file, tnm)
    if stress_sim is None:
        return 1e10  # Return a very large error value on solver failure

    # Interpolate simulation stress to experimental strain points
    strain_sim = np.linspace(0, pct / 100, len(stress_sim))
    interp_stress_sim = np.interp(strain_exp, strain_sim, stress_sim)

    mse = np.mean((interp_stress_sim - stress_exp) ** 2)
    return mse

if __name__ == '__main__':
    # Example usage:
    tnm = "test"  # Or your test name
    file = os.path.dirname(os.path.abspath(__file__)) + "/_msh/EMGEO_1000.msh"  # Replace with your mesh file
    r = 1000  # Or your r value
    pct = 20  # Max strain percentage
    s = 1 #Or True
    # Load your experimental data.  Make sure strain_exp is between 0 and 0.2
    strain_exp = np.array([0.0, 0.05, 0.10, 0.15, 0.20])  # Example experimental strain points
    # exp_strain = [0, 5, 10, 15, 20]
    # exp_stress =  
    stress_exp = np.array([0, 0.3857, 1.1048, 1.8023, 2.6942])  # Example experimental stress points
    bounds = [(0.87475258, 0.87475258), (8.023, 8.023), (1, 50), (16.026, 16.026)]
    # initial_parameters = [0.059, 8.023, 18.472, 16.026] 
    initial_parameters = [0.87475258, 8.023, 18.472, 16.026] 
    result = opt.minimize(
        error_function,
        initial_parameters,
        args=(strain_exp, stress_exp, r, pct, s, file, tnm),
        method='Nelder-Mead',
        bounds=bounds
    )

    optimized_parameters = result.x
    print("Optimized HLZ_CONS:", optimized_parameters)
    print("Minimum Error:", result.fun)

    # Run the simulation with the optimized parameters and plot the results
    optimized_stress = run_simulation(optimized_parameters, r, pct, s, file, tnm)
    if optimized_stress is not None:
        strain_sim = np.linspace(0, pct / 100, len(optimized_stress))
        import matplotlib.pyplot as plt
        plt.plot(strain_exp, stress_exp, 'o', label='Experimental Data')
        plt.plot(strain_sim, optimized_stress, '-', label='Optimized Simulation')
        plt.xlabel('Strain')
        plt.ylabel('Stress')
        plt.legend()
        plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/_png/" + "p1000_mean_opti.png")
        plt.close()
    else:
        print("Simulation failed with optimized parameters.")
        
    print(optimized_parameters)

    

