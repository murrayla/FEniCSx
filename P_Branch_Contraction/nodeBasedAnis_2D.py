# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 17/01/2024
# Code: nodeBasedAnis.py
#   Simple contraction with iteration approach on node based anistropic behaviour.
# +==+==+==+==+

# +==+==+==+
# Setup
# += Imports
from dolfinx import mesh, io, la, default_scalar_type
from dolfinx.fem import (
    Function, FunctionSpace, dirichletbc, form,
    locate_dofs_geometrical, locate_dofs_topological, 
    Expression
)
from dolfinx.fem.petsc import (
    NonlinearProblem, LinearProblem, apply_lifting, 
    assemble_matrix, assemble_vector, create_matrix, 
    create_vector, set_bc
)
from dolfinx.mesh import (create_unit_square, create_unit_cube, CellType, locate_entities_boundary, locate_entities, meshtags)
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
#
import matplotlib.pyplot as plt
import numpy as np
import gmsh
import ufl
# += Parameters
MESH_DIM = 2
X, Y = 0, 1
X_ELS = 5
Y_ELS = 5
ROT = 0 #np.pi/4
LAMBDA = 0.02 # 10% extension
MAX_ITS = 5
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "area": 7}
GEO_DIM = 2
BASE_MS = 10
# Guccione
GCC_CONS = [0.5, 10, 1, 1]

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, elem_order, quad_order):
    # +==+==+
    # Setup problem space
    file = "P_Branch_Contraction/gmsh_msh/" + test_name + ".msh"
    domain, _, ft = io.gmshio.read_from_msh(file, MPI.COMM_WORLD, 0, gdim=GEO_DIM)
    Ve = ufl.VectorElement(family="CG", cell=domain.ufl_cell(), degree=elem_order)
    Vp = ufl.FiniteElement(family="CG", cell=domain.ufl_cell(), degree=elem_order-1)  
    W = FunctionSpace(domain, ufl.MixedElement([Ve, Vp]))
    w = Function(W)

    # +==+==+ 
    # Extract subdomains for dofs
    V, _ = W.sub(0).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()

    # +==+==+
    # Facet assignment
    fdim = MESH_DIM - 1
    # += Locate Facets
    x0_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 0))
    x1_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 1))
    y0_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 0))
    y1_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 1))
    # += Collate facets into stack
    mfacets = np.hstack([x0_facets, x1_facets, y0_facets, y1_facets])
    # += Assign boundaries IDs in stack
    mvalues = np.hstack([
        np.full_like(x0_facets, FACET_TAGS["x0"]), 
        np.full_like(x1_facets, FACET_TAGS["x1"]),
        np.full_like(y0_facets, FACET_TAGS["y0"]), 
        np.full_like(y1_facets, FACET_TAGS["y1"]),
    ])
    # += Sort and assign all tags
    sfacets = np.argsort(mfacets)
    ft = meshtags(mesh=domain, dim=fdim, entities=mfacets[sfacets], values=mvalues[sfacets])

    # +==+==+
    # BC: Base [x0]
    # += Locate subdomain dofs
    x0_dofs_x = locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x0_facets)
    x0_dofs_y = locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, x0_facets)
    # += Interpolate 
    u0_bc_x = Function(Vx)
    u0_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    u0_bc_y = Function(Vy)
    u0_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    # += Create Dirichlet over subdomains
    bc_z0_x = dirichletbc(u0_bc_x, x0_dofs_x, W.sub(0).sub(X))
    bc_z0_y = dirichletbc(u0_bc_y, x0_dofs_y, W.sub(0).sub(Y))

    # +==+==+
    # BC: Base [x1]
    # += Locate subdomain dofs
    x1_dofs_x = locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x1_facets)
    x1_dofs_y = locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, x1_facets)
    # += Interpolate 
    u1_bc_x = Function(Vx)
    u1_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-LAMBDA)))
    u1_bc_y = Function(Vy)
    u1_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    # += Create Dirichlet over subdomains
    bc_z1_x = dirichletbc(u1_bc_x, x1_dofs_x, W.sub(0).sub(X))
    bc_z1_y = dirichletbc(u1_bc_y, x1_dofs_y, W.sub(0).sub(Y))

    # +==+ BC Concatenate
    bc = [bc_z0_x, bc_z0_y, bc_z1_x, bc_z1_y]

    # += Fibre alignment positions
    def Fibril(x):
        return ((x[0] <= 1) and (x[0] >= 0) and (x[1] <= 0.6) and (x[1] >= 0.4))
    def Cytosol(x):
        return ((x[0] > 1) and (x[0] < 0) and (x[1] > 0.6) and (x[1] < 0.4))
    
    kappa = Function(Q)
    cell_myo = locate_entities(domain, domain.topology.dim, Fibril)
    cell_cyt = locate_entities(domain, domain.topology.dim, Cytosol)
    kappa.x.array[cell_myo] = np.full_like(cell_myo, 1, dtype=default_scalar_type)
    kappa.x.array[cell_cyt] = np.full_like(cell_cyt, 0.1, dtype=default_scalar_type)

    # +==+==+
    # Variational Problem Setup
    # += Test and Trial Parameters
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    # += Tensor Indices
    i, j, k, a, b = ufl.indices(5)
    # += Curvilinear Mapping
    Push = ufl.as_matrix([
        [ufl.cos(ROT), -ufl.sin(ROT)],
        [ufl.sin(ROT), ufl.cos(ROT)]
    ])
    # += Curvilinear Coordinates
    x = Function(V)
    x.interpolate(lambda x: (x[0], x[1]))
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # += Metric Tensors
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
    # += Kinematics
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
    # += Material Setup | Guccione
    Q = (
        GCC_CONS[1] * E[0,0]**2 + 
        GCC_CONS[2] * (E[1,1]**2) + 
        GCC_CONS[3] * (2*E[0,1]*E[1,0])
    )
    piola = GCC_CONS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*GCC_CONS[1]*E[0,0], 2*GCC_CONS[3]*(E[1,0] + E[0,1])],
        [2*GCC_CONS[3]*(E[0,1] + E[1,0]), 4*GCC_CONS[2]*E[1,1]],
    ]) - p * Z_un
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    metadata = {"quadrature_degree": quad_order}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    R = term * dx + q * (J - 1) * dx 

    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    #    (0): Quadrature order
    #    (1): Integration domains
    #    (2): Residual equation
    metadata = {"quadrature_degree": quad_order}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    R = term * dx + q * (J - 1) * dx 

    lam = [0]

    Vu_sol, up_to_u_sol = W.sub(0).collapse() 
    u_sol = Function(Vu_sol) 

    Vp_sol, up_to_p_sol = W.sub(1).collapse() 
    p_sol = Function(Vp_sol) 

    u_sol.name = "disp"
    p_sol.name = "pressure"
    
    eps_file = io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + "_eps.bp", u_sol, engine="BP4")
    
    # += Nonlinear Solver
    problem = NonlinearProblem(R, w, bc)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    # +==+==+
    # Solution and Output
    # += Solve
    num_its, converged = solver.solve(w)
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged after {num_its} iterations.")

    u_eval = w.sub(0).collapse()
    u_sol.interpolate(u_eval)

    eps_file.write(0)

    eps_file.close()
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "QUAD_XX_SINGLE_MIDDLE"
    # += Element order
    elem_order = 2
    # += Quadature Degree
    quad_order = 4
    # += Feed Main()
    main(test_name, elem_order, quad_order)