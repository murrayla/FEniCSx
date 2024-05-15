# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 10/05/2024
# Code: nodeBasedAnisIter_2D.py
#   Simple contraction with iteration approach on node based anistropic behaviour.
# +==+==+==+==+

# +==+==+==+
# Setup
# += Imports
from dolfinx import io,  default_scalar_type
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological, Expression, form
from dolfinx.fem.petsc import NonlinearProblem, set_bc, assemble, create_matrix, create_vector, assemble_matrix, assemble_vector
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import ufl
# += Parameters
ITS = 5
ROT = np.arctan(0.3/1)
NULL = 0.0
UNIT = 1.0
X, Y = 0, 1
GEOM_DIM = 2
MESH_DIM = 2
TOLERANCE = 1e-5
DISPLACEMENT = 0.10
MATERIAL_CONSTANTS = [1, 1, 0.5, 0.5]
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "area": 5}
GROUP_IDS = {"Cytosol": 1, "Straight": 2, "Incline": 3, "Decline": 4, "x0": 5, "x1": 6}

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, elem_order, quad_order, ID):
    # +==+ Domain Setup
    # += Load mesh data
    file = "P_Branch_Contraction/gmsh_msh/" + test_name + ".msh"
    domain, ct, _ = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=GEOM_DIM)
    # += Create mixed element spaces
    Ve = ufl.VectorElement(family="CG", cell=domain.ufl_cell(), degree=elem_order)
    Vp = ufl.FiniteElement(family="CG", cell=domain.ufl_cell(), degree=elem_order-1)  
    W = FunctionSpace(mesh=domain, element=ufl.MixedElement([Ve, Vp]))
    w = Function(W)
    # +== Extract subdomains for dofs
    V, _ = W.sub(0).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()

    # +==+ Boundary Condition Setup
    def boundary_conditions(domain, W, Vx, Vy, du):
        # += Facet assignment
        fdim = MESH_DIM - 1
        x0_ft = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], min(x[0])))
        x1_ft = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], max(x[0])))
        y0_ft = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], min(x[1])))
        mfacets = np.hstack([x0_ft, y0_ft])
        # += Assign boundaries IDs and sort
        mvalues = np.hstack([
            np.full_like(x0_ft, FACET_TAGS["x0"]), 
            np.full_like(x1_ft, FACET_TAGS["x0"]), 
            np.full_like(y0_ft, FACET_TAGS["x1"])
        ])
        sfacets = np.argsort(mfacets)
        ft = meshtags(mesh=domain, dim=fdim, entities=mfacets[sfacets], values=mvalues[sfacets])
        # += Locate subdomain dofs
        x_dofs_at_x0 = locate_dofs_topological(V=(W.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=x0_ft)
        # y_dofs_at_x0 = locate_dofs_topological(V=(W.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=x0_ft)
        x_dofs_at_x1 = locate_dofs_topological(V=(W.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=x1_ft)
        # y_dofs_at_x1 = locate_dofs_topological(V=(W.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=x1_ft)
        # x_dofs_at_y1 = locate_dofs_topological(V=(W.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=y0_ft)
        y_dofs_at_y1 = locate_dofs_topological(V=(W.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=y0_ft)
        # += Interpolate 
        ux_at_x0, uy_at_x0, ux_at_x1, uy_at_x1 = Function(Vx), Function(Vy), Function(Vx), Function(Vy)
        ux_at_x0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(NULL)))
        uy_at_x0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(NULL)))
        ux_at_x1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-du)))
        uy_at_x1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(NULL)))
        # += Create Dirichlet over subdomains
        bc_UxX0 = dirichletbc(value=ux_at_x0, dofs=x_dofs_at_x0, V=W.sub(0).sub(X))
        bc_UyX0 = dirichletbc(value=uy_at_x0, dofs=y_dofs_at_y1, V=W.sub(0).sub(Y))
        bc_UxX1 = dirichletbc(value=ux_at_x1, dofs=x_dofs_at_x1, V=W.sub(0).sub(X))
        # bc_UyX1 = dirichletbc(value=uy_at_x1, dofs=y_dofs_at_x1, V=W.sub(0).sub(Y))
        # += Assign
        return [bc_UxX0, bc_UyX0, bc_UxX1], ft
    
    # += Extract boundary conditions and facet tags
    bc, ft = boundary_conditions(domain, W, Vx, Vy, DISPLACEMENT)
    
    # +==+ Subdomain Setup
    V_1DG = FunctionSpace(domain, ("DG", 0))
    def subdomain_assignments(domain, ct, V_1DG):
        # += Locate cells of interest
        str_myo = ct.find(GROUP_IDS["Straight"])
        inc_myo = ct.find(GROUP_IDS["Incline"])
        dec_myo = ct.find(GROUP_IDS["Decline"])
        cytosol = ct.find(GROUP_IDS["Cytosol"])
        # += Create 1-DOF space for assignment
        fibre_rot, fibre_val = Function(V_1DG), Function(V_1DG)
        # += Conditionally assign rotation and material value
        if len(str_myo):
            fibre_rot.x.array[str_myo] = np.full_like(str_myo, NULL, dtype=default_scalar_type)
            fibre_val.x.array[str_myo] = np.full_like(str_myo, MATERIAL_CONSTANTS[1], dtype=default_scalar_type)
        if len(inc_myo):
            fibre_rot.x.array[inc_myo] = np.full_like(inc_myo, ROT, dtype=default_scalar_type)
            fibre_val.x.array[inc_myo] = np.full_like(inc_myo, MATERIAL_CONSTANTS[1], dtype=default_scalar_type)
        if len(dec_myo):
            fibre_rot.x.array[dec_myo] = np.full_like(dec_myo, -ROT, dtype=default_scalar_type)
            fibre_val.x.array[dec_myo] = np.full_like(dec_myo, MATERIAL_CONSTANTS[1], dtype=default_scalar_type)
        if len(cytosol):
            fibre_rot.x.array[cytosol] = np.full_like(cytosol, NULL, dtype=default_scalar_type)
            fibre_val.x.array[cytosol] = np.full_like(cytosol, 0.5, dtype=default_scalar_type)
        return fibre_rot, fibre_val
    
    # += Define rotation and material value
    fibre_rot, fibre_val = subdomain_assignments(domain, ct, V_1DG)

    # +==+ Variational Problem Setup
    # += Test and Trial Parameters
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    # += Coordinate values
    x = Function(V)
    x.interpolate(lambda x: (x[0], x[1]))
    # += Tensor Indices
    i, j, k, a, b = ufl.indices(5)
    # += Curvilinear mapping dependent on subdomain values
    Push = ufl.as_matrix([
        [ufl.cos(fibre_rot), -ufl.sin(fibre_rot)],
        [ufl.sin(fibre_rot), ufl.cos(fibre_rot)]
    ])
    # += Subdomain dependent rotations of displacement and coordinates
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # += Metric tensors
    Z_un = ufl.grad(x_nu).T * ufl.grad(x_nu)
    Z_co = ufl.grad(nu).T * ufl.grad(nu)
    Z_ct = ufl.inv(Z_co)
    # += Christoffel Symbol | Γ^{i}_{j, a}
    gamma = ufl.as_tensor((
        0.5 * Z_ct[k, a] * (
            ufl.grad(Z_co)[a, i, j] + ufl.grad(Z_co)[a, j, i] - ufl.grad(Z_co)[i, j, a]
        )
    ), [k, i, j])
    # += Covariant Derivative
    covDev = ufl.grad(v) - ufl.as_tensor(v[k]*gamma[k, i, j], [i, j])
    # += Kinematics variables
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
    # += Material Setup | Guccione
    Q = (
        fibre_val * E[0,0]**2 + 
        MATERIAL_CONSTANTS[2] * (E[1,1]**2) + 
        MATERIAL_CONSTANTS[3] * (2*E[0,1]*E[1,0])
    )
    piola = MATERIAL_CONSTANTS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*fibre_val*E[0,0], 2*MATERIAL_CONSTANTS[3]*(E[1,0] + E[0,1])],
        [2*MATERIAL_CONSTANTS[3]*(E[0,1] + E[1,0]), 4*MATERIAL_CONSTANTS[2]*E[1,1]],
    ]) - p * Z_un
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    
    # +==+ Solver Setup
    # += Measure assignment
    metadata = {"quadrature_degree": quad_order}
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata=metadata)
    # += Residual equation
    R = term * dx + q * (J - 1) * dx

    # +==+ Setup stress output
    def cauchy_tensor(u, p, x, r):
        # += Tensor Indices
        i, j = ufl.indices(2)
        # += Curvilinear Mapping
        Push = ufl.as_matrix([
            [ufl.cos(r), -ufl.sin(r)],
            [ufl.sin(r), ufl.cos(r)]
        ])
        x_nu = ufl.inv(Push) * x
        u_nu = ufl.inv(Push) * u
        nu = ufl.inv(Push) * (x + u_nu)
        # += Metric Tensors
        Z_un = ufl.grad(x_nu).T * ufl.grad(x_nu)
        Z_co = ufl.grad(nu).T * ufl.grad(nu)
        # += Kinematics
        I = ufl.variable(ufl.Identity(MESH_DIM))
        F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
        E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
        J = ufl.variable(ufl.det(F))
        # += Material Setup | Guccione
        Q = (
            fibre_val * E[0,0]**2 + 
            MATERIAL_CONSTANTS[2] * (E[1,1]**2) + 
            MATERIAL_CONSTANTS[3] * (2*E[0,1]*E[1,0])
        )
        piola = MATERIAL_CONSTANTS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
            [4*fibre_val*E[0,0], 2*MATERIAL_CONSTANTS[3]*(E[1,0] + E[0,1])],
            [2*MATERIAL_CONSTANTS[3]*(E[0,1] + E[1,0]), 4*MATERIAL_CONSTANTS[2]*E[1,1]],
        ]) - p * Z_un
        sig = 1/J * F*piola*F.T
        return sig
    
    # +==+ Setup strain output
    def green_tensor(u, p, x, r):
        # += Tensor Indices
        i, j = ufl.indices(2)
        # += Curvilinear Mapping
        Push = ufl.as_matrix([
            [ufl.cos(r), -ufl.sin(r)],
            [ufl.sin(r), ufl.cos(r)]
        ])
        x_nu = ufl.inv(Push) * x
        u_nu = ufl.inv(Push) * u
        nu = ufl.inv(Push) * (x + u_nu)
        # += Metric Tensors
        Z_un = ufl.grad(x_nu).T * ufl.grad(x_nu)
        Z_co = ufl.grad(nu).T * ufl.grad(nu)
        # += Kinematics
        I = ufl.variable(ufl.Identity(MESH_DIM))
        F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
        E = ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j])
        return ufl.as_tensor([[E[0, 0], E[0, 1]], [E[0, 1], E[1, 1]]])

    # +==+ Setup Iteration Loop variables
    # += Preset displacements per iteration
    phase_ext = np.linspace(DISPLACEMENT/ITS, DISPLACEMENT, ITS)
    phase_con = np.linspace(DISPLACEMENT, -DISPLACEMENT, ITS*2)
    phase_exp = np.linspace(-DISPLACEMENT, 0, ITS)
    bc_u_iter = np.concatenate([phase_ext, phase_con, phase_exp])
    # += Export data
    Vu_sol, _ = W.sub(0).collapse() 
    disp = Function(Vu_sol) 
    disp.name = "U - Displacement"
    TS = FunctionSpace(mesh=domain, element=("CG", elem_order, (2,2)))
    sig = Function(TS)
    sig.name = "S - Cauchy Stress"
    eps = Function(TS)
    eps.name = "E - Green Strain"
    # test_name = "TESTING_ITERATION"
    # += Setup file writers
    dis_file = io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ID + "_DISP.bp", disp, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ID + "_EPS.bp", eps, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ID + "_SIG.bp", sig, engine="BP4")

    # +==+ Loop deformations
    for i, d in enumerate(bc_u_iter):
        # += Attain BC
        print("     += Iteration: {}".format(i))
        bc, _ = boundary_conditions(domain, W, Vx, Vy, d)
        
        # += Nonlinear Solver
        problem = NonlinearProblem(R, w, bc)
        solver = NewtonSolver(domain.comm, problem)
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"

        # += Solve
        num_its, _ = solver.solve(w)
        print(f"Converged in {num_its} iterations.")

        # += Store displacement results
        u_eval = w.sub(0).collapse()
        disp.interpolate(u_eval)
        # += Setup tensor space for stress tensor interpolation
        cauchy = Expression(
            e=cauchy_tensor(u_eval, w.sub(1).collapse(), x, fibre_rot), 
            X=TS.element.interpolation_points()
        )
        epsilon = Expression(
            e=green_tensor(u_eval, None, x, fibre_rot), 
            X=TS.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        eps.interpolate(epsilon)

        np.save("P_Branch_Contraction/numpy_data/disp_x_" + test_name + "_" + ID + "_" + str(i) + ".npy", disp.x.array[dofsX])
        np.save("P_Branch_Contraction/numpy_data/disp_y_" + test_name + "_" + ID + "_" + str(i) + ".npy", disp.x.array[dofsY])
        np.save("P_Branch_Contraction/numpy_data/sig_x_" + test_name + "_" + ID + "_" + str(i) + ".npy", sig.x.array[dofsX])
        np.save("P_Branch_Contraction/numpy_data/sig_y_" + test_name + "_" + ID + "_" + str(i) + ".npy", sig.x.array[dofsY])
        np.save("P_Branch_Contraction/numpy_data/eps_x_" + test_name + "_" + ID + "_" + str(i) + ".npy", eps.x.array[dofsX])
        np.save("P_Branch_Contraction/numpy_data/eps_y_" + test_name + "_" + ID + "_" + str(i) + ".npy", eps.x.array[dofsY])

        # += Write files 
        for j in range(1, 10, 1): dis_file.write(i*10 + j)
        for j in range(1, 10, 1): eps_file.write(i*10 + j)
        for j in range(1, 10, 1): sig_file.write(i*10 + j)

    # += Close files
    dis_file.close()
    eps_file.close()
    sig_file.close()
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test = [
        "SINGLE_MIDDLE", "SINGLE_DOUBLE", "SINGLE_ACROSS", 
        "DOUBLE_ACROSS", "BRANCH_ACROSS", "BRANCH_MIDDLE", 
        "TRANSFER_DOUBLE", "CYTOSOL"
    ]
    # test = [
    #     "SINGLE_MIDDLE"
    # ]
    # test_name = "SUB_REDQUAD_XX_TRANSFER_DOUBLE"
    test_name = ["SUB_REDQUAD_XX_" + x for x in test]
    # += Element order
    elem_order = 2
    # += Quadature Degree
    quad_order = 4
    # += Feed Main()
    count = 0
    # main(test_name[0], elem_order, quad_order, ID="_")
    for name in test_name:
        try:
            main(name, elem_order, quad_order, ID="_10PctIter")
            count += 1
        except:
            count -= 1
            continue
            
    print(" WE COMPLETED: {}".format(count))