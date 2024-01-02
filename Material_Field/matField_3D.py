# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 29/12/2023
# +==+==+==+==+==+

# +==+==+==+==+
# Code: matField_3D.py
#   Updating earlier test to move to 3D for more successful implementation
# +==+==+==+==+

# +==+==+==+
# Setup
# += Imports
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import math
import basix
import ufl
# += Parameters
MESH_DIM = 3
X, Y, Z = 0, 1, 2
X_ELS = 1
Y_ELS = 1
Z_ELS = 1
LAMBDA = 0.01 # 10% extension
ROT = 0#np.pi/4
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "area": 7}

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
    domain = mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=X_ELS, ny=Y_ELS, nz=Z_ELS, cell_type=mesh.CellType.hexahedron)
    Ve = ufl.VectorElement(family="CG", cell=domain.ufl_cell(), degree=elem_order)
    Vp = ufl.FiniteElement("CG", domain.ufl_cell(), degree=elem_order-1)  
    W = fem.FunctionSpace(domain, ufl.MixedElement([Ve, Vp]))
    w = fem.Function(W)

    # +==+==+ 
    # Extract subdomains for dofs
    #    (0): Element domain for displacement
    #    (1): Subdomains of element for each component
    V, _ = W.sub(0).collapse()
    P, _ = W.sub(1).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()
    Vz, dofsZ = V.sub(Z).collapse()

    # +==+==+
    # Facet assignment
    fdim = MESH_DIM - 1
    # += Locate Facets
    #    (0): Over boundary, set boolean on if marker is at location
    x0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 0))
    x1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 1))
    y0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 0))
    y1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 1))
    z0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], 0))
    z1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], 1))
    # += Collate facets into stack
    mfacets = np.hstack([x0_facets, x1_facets, y0_facets, y1_facets, z0_facets, z1_facets])
    # += Assign boundaries IDs in stack
    mvalues = np.hstack([
        np.full_like(x0_facets, FACET_TAGS["x0"]), 
        np.full_like(x1_facets, FACET_TAGS["x1"]),
        np.full_like(y0_facets, FACET_TAGS["y0"]), 
        np.full_like(y1_facets, FACET_TAGS["y1"]),
        np.full_like(z0_facets, FACET_TAGS["z0"]), 
        np.full_like(z1_facets, FACET_TAGS["z1"])
    ])
    # += Sort and assign all tags
    sfacets = np.argsort(mfacets)
    ft = mesh.meshtags(mesh=domain, dim=fdim, entities=mfacets[sfacets], values=mvalues[sfacets])

    # +==+==+ 
    # Boundary Conditions
    # +==+ [x0]
    x0_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x0_facets)
    u0_bc_x = fem.Function(Vx)
    u0_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_x0_x = fem.dirichletbc(u0_bc_x, x0_dofs_x, W.sub(0).sub(X))
    # +==+ [x1]
    x1_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x1_facets)
    u1_bc_x = fem.Function(Vx)
    u1_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(LAMBDA)))
    bc_x1_x = fem.dirichletbc(u1_bc_x, x1_dofs_x, W.sub(0).sub(X))
    # +==+ [y0]
    y0_dofs_y = fem.locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, y0_facets)
    u0_bc_y = fem.Function(Vy)
    u0_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_y0_y = fem.dirichletbc(u0_bc_y, y0_dofs_y, W.sub(0).sub(Y))
    # +==+ [z0]
    z0_dofs_z = fem.locate_dofs_topological((W.sub(0).sub(Z), Vz), ft.dim, z0_facets)
    u0_bc_z = fem.Function(Vz)
    u0_bc_z.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_z0_z = fem.dirichletbc(u0_bc_z, z0_dofs_z, W.sub(0).sub(Z))
    # +==+ BC Concatenate
    bc = [bc_x0_x, bc_x1_x, bc_y0_y, bc_z0_z]

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
    # F = ufl.variable(I + ufl.grad(u))

    # # +==+==+
    # # Create Material Coordinates
    # class MaterialCoordinates():
    #     # (0): Initiate a class defining rotation
    #     def __init__(self, rot=ROT):
    #         self.rot = ROT
    #     # (1): Rotate coordinates into new field
    #     def eval(self, x):
    #         self.xMat = np.cos(self.rot) * x[0] + np.sin(self.rot) * x[1]
    #         self.yMat = -np.sin(self.rot) * x[0] + np.sin(self.rot) * x[1]
    #         self.zMat = x[2]
    #         return (self.xMat, self.yMat, self.zMat)
    # # += Define class
    # matCoords = MaterialCoordinates()
    # # += Interpolate
    # x_mat = fem.Function(V)
    
    # x_mat.interpolate(matCoords.eval)
    # expr = ufl.as_vector(("x[0]", "x[1]", "x[2]"))
    x = ufl.SpatialCoordinate(domain)
    x_nu = ufl.as_vector((
        ufl.cos(ROT)*x[0] + ufl.sin(ROT)*x[1], 
        -ufl.sin(ROT)*x[0] + ufl.cos(ROT)*x[1], 
        x[2]
    ))
    # x_nu_exp = fem.Expression(x_nu, V.element.interpolation_points())

    # Projection function inspired by @michalhabera 
    # https://github.com/michalhabera/dolfiny/blob/master/dolfiny/projection.py
    def project(v, V):
        dx = ufl.dx(V.mesh)
        w = ufl.TestFunction(V)
        Pv = ufl.TrialFunction(V)
        a = ufl.inner(Pv, w) * dx
        L = ufl.inner(v, w) * dx
        problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        nu = problem.solve()     
        return nu
        
    nu = project(x_nu, V)

    # +==+ Covariant Metric Tensor
    Z_ij = ufl.grad(nu).T * ufl.grad(nu)
    # +==+ Covariant Basis
    z0, z1, z2 = nu.dx(0), nu.dx(1), nu.dx(2)
    # +==+ Tensor Indices
    i, j, k, m = ufl.indices(4)
    # +==+ Christoffel Symbol
    gamma = ufl.as_tensor((
        0.5 * ufl.inv(Z_ij)[k, m] * (
            ufl.grad(Z_ij)[m, i, j] + ufl.grad(Z_ij)[m, j, i] - ufl.grad(Z_ij)[i, j, m]
        )
    ), [k, i, j])
    # +==+ Covariant Derivative
    covDev = ufl.as_tensor(v[j].dx(k) - gamma[i, j, k] * v[i], [j, k])
    # print(covDev)

    ux_nu = ufl.cos(ROT) * u[0]
    uy_nu = ufl.sin(ROT) * u[1]
    uz_nu = u[2]
    u_nu = ufl.as_vector((ux_nu, uy_nu, uz_nu))
    
    # undef = ufl.SpatialCoordinate(domain)
    # x_nu = ufl.cos(ROT) * undef[0] - ux_nu
    # y_nu = ufl.sin(ROT) * undef[1] - uy_nu
    # z_nu = undef[2] - uz_nu
    # nu = ufl.variable(ufl.as_vector((x_nu, y_nu, z_nu)))

    F = ufl.variable(I + ufl.grad(u))

    # # g_ij = ei•ej = ∂R/∂xi * ∂R/∂xj
    # cov_met = ufl.grad(x_mat).T * ufl.grad(x_mat)
    # print(cov_met)
    # print(okay)

    # # g^ij
    # con_met, j_met = ufl.inv(cov_met), ufl.det(cov_met)
    # # Directional Derivatives to get base vectors
    # e_i, e_j, e_k = x_mat.dx(0), x_mat.dx(1), x_mat.dx(2)
    # e = ufl.as_vector((e_i, e_j, e_k))
    # # Contravariant base vectors
    # e_i_c, e_j_c, e_k_c = (
    #     con_met[0,0]*e_i + con_met[0,1]*e_j + con_met[0,2]*e_k, 
    #     con_met[1,0]*e_i + con_met[1,1]*e_j + con_met[1,2]*e_k, 
    #     con_met[2,0]*e_i + con_met[2,1]*e_j + con_met[2,2]*e_k
    # )
    # e_c = ufl.as_vector((e_i_c, e_j_c, e_k_c))
    # # Forward transform
    # Push = ufl.as_matrix([[np.cos(ROT), -np.sin(ROT)], [np.sin(ROT), np.cos(ROT)]])
    # Pull = ufl.as_matrix([[np.cos(ROT), np.sin(ROT)], [-np.sin(ROT), np.cos(ROT)]])
    
    # i, j, k, l = ufl.indices(4)
    
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
    J = ufl.variable(ufl.det(F))

    # E = ufl.variable(ufl.as_tensor((0.5*(cov_met[i,j]-I[i,j])), [i,j]))

    # c = 0.876
    # bf = 18.48
    # bt = 3.58
    # bs = 1.627

    # # c = 0.5
    # # bf = 6
    # # bt = 2
    # # bfs = 0

    # Q = (
    #     bf * E[0,0]**2 + 
    #     bt * (E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2) + 
    #     bs * (E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2)
    # )
    # psi = c/2 * (ufl.exp(Q) - 1)
    # # T = ufl.as_tensor((0.5*c/2 * ufl.exp(Q) * (ufl.diff(Q,E[i,j])+ufl.diff(Q,E.T[j,i]))), [i,j])
    # T = ufl.as_tensor([
    #     [
    #         c/2 * ufl.exp(Q) * bf*E[0,0],
    #         c/2 * ufl.exp(Q) * (bs*E[0,1] + bs*E[1,0]),
    #         c/2 * ufl.exp(Q) * (bs*E[0,2] + bs*E[2,0])
    #     ],
    #     [
    #         c/2 * ufl.exp(Q) * (bs*E[1,0] + bs*E[0,1]),
    #         c/2 * ufl.exp(Q) * bt*E[1,1],
    #         c/2 * ufl.exp(Q) * (bt*E[1,2] + bt*E[2,1])
    #     ],
    #     [
    #         c/2 * ufl.exp(Q) * (bs*E[2,0] + bs*E[0,2]),
    #         c/2 * ufl.exp(Q) * (bt*E[2,1] + bt*E[1,2]),
    #         c/2 * ufl.exp(Q) * bt*E[2,2],
    #     ]
    # ]) - p*I
    # fPK = F * T #+ p * J * ufl.inv(F).T
    # n = ufl.FacetNormal(domain)

    # Na = 1/J * F.T*n
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

    # first = ufl.as_tensor(fPK[k, m] * F[j, m] * covDev[j, k])
    # n = ufl.FacetNormal(domain)
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    #    (0): Quadrature order
    #    (1): Integration domains
    #    (2): Residual equation
    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    R = ufl.inner(covDev, fPK * F) * dx + q * (J - 1) * dx 
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
    test_name = "unitCube_extension"
    # += Element order
    elem_order = 2
    # += Feed Main()
    main(test_name, elem_order)