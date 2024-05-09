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
import numpy as np
import ufl
# += Parameters
MESH_DIM = 2
X, Y = 0, 1
X_ELS = 5
Y_ELS = 5
ROT = 0 #np.pi/4
LAMBDA = 0.03 # 10% extension
MAX_ITS = 5
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "area": 7}
GEO_DIM = 2
BASE_MS = 10
# Guccione
GCC_CONS = [0.5, 5, 1, 1]
# += Group IDs
CYTOSOL = 1
MYO_STRAIGHT = 2
MYO_UPANGLED = 3
MYO_DNANGLED = 4
LEFT_SIDE = 5
RIGHT_SIDE = 6

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, elem_order, quad_order, ID):
    # +==+==+
    # Setup problem space
    file = "P_Branch_Contraction/gmsh_msh/" + test_name + ".msh"
    domain, ct, ft = io.gmshio.read_from_msh(file, MPI.COMM_WORLD, 0, gdim=GEO_DIM)
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
    u0_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(LAMBDA)))
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
    
    # += Identify the subdomains of interest  
    str_myo = ct.find(MYO_STRAIGHT)
    upa_myo = ct.find(MYO_UPANGLED)
    dwa_myo = ct.find(MYO_DNANGLED)
    cytosol = ct.find(CYTOSOL)

    # +==+==+
    # Variational Problem Setup
    # += Test and Trial Parameters
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    # += Curvilinear Coordinates
    x = Function(V)
    x.interpolate(lambda x: (x[0], x[1]))

    Q = FunctionSpace(domain, ("DG", 0))
    r = Function(Q)
    gc1 = Function(Q)

    if len(str_myo):
        r.x.array[str_myo] = np.full_like(str_myo, 0.0, dtype=default_scalar_type)
        gc1.x.array[str_myo] = np.full_like(str_myo, GCC_CONS[1], dtype=default_scalar_type)
    if len(upa_myo):
        r.x.array[upa_myo] = np.full_like(upa_myo, 0.29, dtype=default_scalar_type)
        gc1.x.array[upa_myo] = np.full_like(upa_myo, GCC_CONS[1], dtype=default_scalar_type)
    if len(dwa_myo):
        r.x.array[dwa_myo] = np.full_like(dwa_myo, -0.29, dtype=default_scalar_type)
        gc1.x.array[dwa_myo] = np.full_like(dwa_myo, GCC_CONS[1], dtype=default_scalar_type)
    if len(cytosol):
        r.x.array[cytosol] = np.full_like(cytosol, 0.0, dtype=default_scalar_type)
        gc1.x.array[cytosol] = np.full_like(cytosol, 1.0, dtype=default_scalar_type)

    # += Tensor Indices
    i, j, k, a, b = ufl.indices(5)
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
        gc1 * E[0,0]**2 + 
        GCC_CONS[2] * (E[1,1]**2) + 
        GCC_CONS[3] * (2*E[0,1]*E[1,0])
    )
    piola = GCC_CONS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*gc1*E[0,0], 2*GCC_CONS[3]*(E[1,0] + E[0,1])],
        [2*GCC_CONS[3]*(E[0,1] + E[1,0]), 4*GCC_CONS[2]*E[1,1]],
    ]) - p * Z_un
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    metadata = {"quadrature_degree": quad_order}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)

    R = term * dx + q * (J - 1) * dx

    Vu_sol, up_to_u_sol = W.sub(0).collapse() 
    u_sol = Function(Vu_sol) 

    Vp_sol, up_to_p_sol = W.sub(1).collapse() 
    p_sol = Function(Vp_sol) 

    u_sol.name = "disp"
    p_sol.name = "pressure"
    
    # += Nonlinear Solver
    problem = NonlinearProblem(R, w, bc)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-5
    solver.rtol = 1e-5
    solver.convergence_criterion = "incremental"

    # +==+==+
    # Solution and Output
    # += Solve
    num_its, converged = solver.solve(w)
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged after {num_its} iterations.")

    # # +==+==+
    # # Interpolate Stress
    # def cauchy(u, p):
    #     i = ufl.Identity(MESH_DIM)
    #     f = i + ufl.grad(u)
    #     c = f.T * f
    #     e = 0.5 * (c - i)
    #     j = ufl.det(f)
    #     q = (
    #         GCC_CONS[1] * e[0,0]**2 + 
    #         GCC_CONS[2] * (e[1,1]**2) + 
    #         GCC_CONS[3] * (2*e[0,1]*e[1,0])
    #     )
    #     s = GCC_CONS[0]/4 * ufl.exp(q) * ufl.as_matrix([
    #         [4*GCC_CONS[1]*e[0,0], 2*GCC_CONS[3]*(e[1,0] + e[0,1])],
    #         [2*GCC_CONS[3]*(e[0,1] + e[1,0]), 4*GCC_CONS[2]*e[1,1]],
    #     ]) - p * ufl.inv(c)
    #     sig = 1/j * f*s*f.T
    #     return sig
    # #    (1): Create Tensor Function Space
    # #    (2): Define expression over points
    # #    (3): Create function variable
    # #    (4): Interpolate
    # TS = FunctionSpace(domain, ("CG", elem_order, (2,2)))
    # piola_expr = Expression(cauchy(u_sol, p_sol), TS.element.interpolation_points())
    # sig = Function(TS)
    # sig.interpolate(piola_expr)

    # += Export
    strains = w.sub(0).collapse()
    strains.name = "eps"
    # stresses = sig
    # stresses.name = "sig"
    with io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ID + "_eps.bp", strains, engine="BP4") as vtx:
        vtx.write(0)
        vtx.close()
    # with io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + "_sig.bp", stresses, engine="BP4") as vtx:
    #     vtx.write(0)
    #     vtx.close()
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "SUB_REDQUAD_XX_TRANSFER_DOUBLE"
    # += Element order
    elem_order = 2
    # += Quadature Degree
    quad_order = 4
    # += Feed Main()
    main(test_name, elem_order, quad_order, ID="_5PCT")