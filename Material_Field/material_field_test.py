# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 15/12/2023
# +==+==+==+==+==+

# +==+==+==+==+
# Code: material_field_test.py
#   Testing contraction over a material field. 
#   Obseving impacts of changes in alignment of fibres on contraction.
# +==+==+==+==+

# +==+==+==+
# Setup
# += Imports
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import basix
import ufl
# += Parameters
MESH_DIM = 2
X, Y = 0, 1
X_ELS = 1
Y_ELS = 1
LAMBDA = 0.05 # 5% Contraction
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "area": 5}

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, elem_order):
    # +==+==+
    # Setup problem space
    #    (0): Mesh, unit square
    #    (1): Element definition
    #    (2): Pressure space definition
    #    (2): Mixed function space for interpolation
    #    (3): Function for solving within space | displacement and hydrostatic pressure
    domain = mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=X_ELS, ny=Y_ELS, cell_type=mesh.CellType.quadrilateral)
    Ve = ufl.VectorElement(family="CG", cell=domain.ufl_cell(), degree=elem_order)
    Vp = ufl.FiniteElement("CG", domain.ufl_cell(), degree=elem_order-1)  
    W = fem.FunctionSpace(domain, ufl.MixedElement([Ve, Vp]))
    w = fem.Function(W)

    # +==+==+ 
    # Extract subdomains for dofs
    #    (0): Element domain for displacement
    #    (1): Subdomains of element for each component
    V, _ = W.sub(0).collapse()
    Vx, _ = V.sub(X).collapse()
    Vy, _ = V.sub(Y).collapse()

    # +==+==+
    # Facet assignment
    fdim = MESH_DIM - 1
    # += Locate Facets
    #    (0): Over boundary, set boolean on if marker is at location
    x0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 0))
    x1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 1))
    y0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 0))
    y1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 1))
    # += Collate facets into stack
    mfacets = np.hstack([x0_facets, x1_facets, y0_facets, y1_facets])
    # += Assign boundaries IDs in stack
    mvalues = np.hstack([
        np.full_like(x0_facets, FACET_TAGS["x0"]), 
        np.full_like(x1_facets, FACET_TAGS["x1"]),
        np.full_like(y0_facets, FACET_TAGS["y0"]), 
        np.full_like(y1_facets, FACET_TAGS["y1"])
    ])
    # += Sort and assign all tags
    sfacets = np.argsort(mfacets)
    ft = mesh.meshtags(mesh=domain, dim=fdim, entities=mfacets[sfacets], values=mvalues[sfacets])

    # +==+==+
    # BC: Base [x0]
    # += Locate subdomain dofs
    x0_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x0_facets)
    x0_dofs_y = fem.locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, x0_facets)
    # += Interpolate 
    u0_bc_x = fem.Function(Vx)
    u0_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(LAMBDA)))
    u0_bc_y = fem.Function(Vy)
    u0_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    # += Create Dirichlet over subdomains
    bc_z0_x = fem.dirichletbc(u0_bc_x, x0_dofs_x, W.sub(0).sub(X))
    bc_z0_y = fem.dirichletbc(u0_bc_y, x0_dofs_y, W.sub(0).sub(Y))

    # +==+==+
    # BC: Base [x1]
    # += Locate subdomain dofs
    x1_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x1_facets)
    x1_dofs_y = fem.locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, x1_facets)
    # += Interpolate 
    u1_bc_x = fem.Function(Vx)
    u1_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-LAMBDA)))
    u1_bc_y = fem.Function(Vy)
    u1_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    # += Create Dirichlet over subdomains
    bc_z1_x = fem.dirichletbc(u1_bc_x, x1_dofs_x, W.sub(0).sub(X))
    bc_z1_y = fem.dirichletbc(u1_bc_y, x1_dofs_y, W.sub(0).sub(Y))

    # +==+ BC Concatenate
    bc = [bc_z0_x, bc_z0_y, bc_z1_x, bc_z1_y]

    # +==+==+
    # Variational Problem Setup
    # += Test and Trial Parameters
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    # += Tensors
    #    (0): Identity Matrix
    #    (1): Deformation Gradient Tensor
    #    (2): Right Cauchy-Green Tensor
    #         (0): Invariants, Ic, IIc, J
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
    J = ufl.variable(ufl.det(F))
    # += Material Setup | Mooney-Rivlin
    #    (0): Constants
    #    (1): Strain Density Function
    #    (2): Chain Rule Differentiation Terms
    #    (3): First Piola-Kirchoff Stress
    c1 = 2
    c2 = 6
    psi = c1 * (Ic - 3) + c2 *(IIc - 3) 
    term1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
    term2 = -ufl.diff(psi, IIc)
    fPK = 2 * F * (term1*I + term2*C) + p * J * ufl.inv(F).T
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    #    (0): Quadrature order
    #    (1): Integration domains
    #    (2): Residual equation
    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    R = ufl.inner(ufl.grad(v), fPK) * dx + q * (J - 1) * dx 
    # += Nonlinear Solver
    #    (0): Definition
    #    (1): Newton Iterator
    #         (0): Tolerances
    #         (1): Convergence
    problem = NonlinearProblem(R, w, bc)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    # +==+==+
    # Solution and Output
    # += Solve
    #    (0): Iteration and convergence
    #    (1): Return data
    num_its, converged = solver.solve(w)
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged after {num_its} iterations.")
    u_sol, p_sol = w.split()
    # += Export
    with io.VTXWriter(MPI.COMM_WORLD, test_name + ".bp", w.sub(0).collapse(), engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "simple_contraction"
    # += Element order
    elem_order = 2
    # += Feed Main()
    main(test_name, elem_order)