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
import math
import basix
import ufl
# += Parameters
MESH_DIM = 2
X, Y = 0, 1
X_ELS = 1
Y_ELS = 1
LAMBDA = 0.05 # extension
ROT = np.pi/4
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
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()

    # +==+==+
    # Create Material Coordinates
    class MaterialCoordinates():
        # (0): Initiate a class defining rotation
        def __init__(self, rot=ROT):
            self.rot = ROT
        # (1): Rotate coordinates into new field
        def eval(self, x):
            self.xMat = np.cos(self.rot) * x[0] + np.sin(self.rot) * x[1]
            self.yMat = -np.sin(self.rot) * x[0] + np.sin(self.rot) * x[1]
            return (self.xMat, self.yMat)
    # += Define class
    matCoords = MaterialCoordinates()
    # += Interpolate
    x_mat = fem.Function(V)
    x_mat.interpolate(matCoords.eval)
    # print(x_mat.x.array[dofsX])
    # print(x_mat.x.array[dofsY])

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

    # +==+==+
    # Metric Tensors
    CART = ufl.SpatialCoordinate(domain)

    Push = ufl.as_matrix([
        [math.cos(ROT), -math.sin(ROT)], 
        [math.sin(ROT), math.cos(ROT)]
    ])
    Pull = ufl.inv(Push)
    CURV = ufl.variable(Pull * CART)
    G = ufl.as_tensor([[1, 0], [0, 1]])

    x_til = fem.Function(Vx)
    y_til = fem.Function(Vx)

    class affine:
        def tranx(x):
            x_tild = np.full(x.shape[1], default_scalar_type(math.cos(ROT)*x[0] + math.sin(ROT)*x[1]))
            return x_tild
        def trany(x):
            y_tild = np.full(x.shape[1], default_scalar_type(-math.sin(ROT)*x[0] + math.cos(ROT)*x[1]))
            return y_tild

    x_til.interpolate(affine.tranx)
    y_til.interpolate(affine.trany)

    x_cart = ufl.variable(CART[0] + u[0])
    y_cart = ufl.variable(CART[1] + u[1])
    x_curv = ufl.variable(CURV[0] + u[0])
    y_curv = ufl.variable(CURV[1] + u[1])
    g = ufl.as_tensor([
        [
            x_cart*math.cos(ROT)**2 + x_cart*math.sin(ROT)**2,
            -x_cart*math.cos(ROT)*y_cart*math.sin(ROT) + x_cart*math.sin(ROT)*y_cart*math.cos(ROT)
        ],
        [
            -x_cart*math.cos(ROT)*y_cart*math.sin(ROT) + x_cart*math.sin(ROT)*y_cart*math.cos(ROT),
            y_cart*math.sin(ROT)**2 + y_cart*math.cos(ROT)**2
        ]
    ])
    E = ufl.variable(0.5*(g-G))

    # += Tensors
    #    (0): Identity Matrix
    #    (1): Deformation Gradient Tensor
    #    (2): Right Cauchy-Green Tensor
    #         (0): Invariants, Ic, IIc, J
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.variable(ufl.inv(Push).T * (I + ufl.grad(u)) * ufl.inv(Push))
    
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
    J = ufl.variable(ufl.det(F))

    # c = 0.876
    # bf = 18.48
    # bt = 3.58
    # bfs = 1.627

    c = 0.5
    bf = 6
    bt = 2
    bfs = 0

    Q = bf * E[0,0]**2 + bt * E[1,1]**2 + bfs * (E[0,1]**2 + E[1,0]**2)
    psi = c/2 * (ufl.exp(Q) - 1)
    T = ufl.as_tensor([
        [
            c/2 * ufl.exp(Q) * bf*E[0,0],
            c/2 * ufl.exp(Q) * (bfs*E[1,0] + bfs*E[0,1])
        ],
        [
            c/2 * ufl.exp(Q) * (bfs*E[0,1] + bfs*E[1,0]),
            c/2 * ufl.exp(Q) * bt*E[1,1]
        ]
    ])
    fPK = F * T + p * J * ufl.inv(F).T
    # += Material Setup | Mooney-Rivlin
    #    (0): Constants
    #    (1): Strain Density Function
    #    (2): Chain Rule Differentiation Terms
    #    (3): First Piola-Kirchoff Stress
    # c1 = 2
    # c2 = 6
    # psi = c1 * (Ic - 3) + c2 *(IIc - 3) 
    # term1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
    # term2 = -ufl.diff(psi, IIc)
    # fPK = 2 * F * (term1*I + term2*C) + p * J * ufl.inv(F).T
    n = ufl.FacetNormal(domain)
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    #    (0): Quadrature order
    #    (1): Integration domains
    #    (2): Residual equation
    metadata = {"quadrature_degree": 2}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    R = ufl.inner(ufl.grad(v), F * fPK) * dx + q * (J - 1) * dx 
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

    print(u_sol.x.array[dofsX])
    
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