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
LAMBDA = 0.1 # 10% extension
ROT = np.pi/2
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "area": 7}

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, elem_order, constitutive):
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
    # +==+ Tensor Indices
    i, j, k, a, b, c, d = ufl.indices(7)
    # +==+ Curvilinear Mapping
    Push = ufl.as_tensor([
        [ufl.cos(ROT), -ufl.sin(ROT), 0],
        [ufl.sin(ROT), ufl.cos(ROT), 0],
        [0, 0, 1]
    ])
    # +==+ Get Deformed Coordinates in Curvilinear 
    x = fem.Function(V)
    x.interpolate(lambda x: x)
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # +==+ Metric Tensors
    Z_co = ufl.grad(nu).T * ufl.grad(nu)
    Z_un = ufl.grad(x_nu).T * ufl.grad(x_nu)
    Z_ct = ufl.inv(Z_co)
    # +==+ Covariant and Contravariant Basis
    z_co = ufl.as_tensor((nu.dx(0), nu.dx(1), nu.dx(2)))
    z_ct = ufl.as_tensor(
        (
            Z_ct[0,0]*z_co[0] + Z_ct[1,0]*z_co[1] + Z_ct[2,0]*z_co[2],
            Z_ct[0,1]*z_co[0] + Z_ct[1,1]*z_co[1] + Z_ct[2,1]*z_co[2],
            Z_ct[0,2]*z_co[0] + Z_ct[1,2]*z_co[1] + Z_ct[2,2]*z_co[2]
        )
    )
    # +==+ Christoffel Symbol
    # christoffel = ufl.as_tensor(z_co[j, b] * nu[k] * Push[b, k] * z_ct[i, :], [i, j, a])
    christoffel = ufl.as_tensor((
        0.5 * Z_ct[k, a] * (
            ufl.grad(Z_co)[a, i, j] + ufl.grad(Z_co)[a, j, i] - ufl.grad(Z_co)[i, j, a]
        )
    ), [k, i, j])
    # christoffel = ufl.as_tensor((
    #     0.5 * Z_ct[k, a] * (
    #         ufl.grad(Z_co)[a, i, b]*Push[b, j] + ufl.grad(Z_co)[a, j, c]*Push[c, i] - ufl.grad(Z_co)[i, j, d]*Push[d, a]
    #     )
    # ), [k, i, j])
    # +==+ Covariant Derivative
    covDev = ufl.grad(v) - ufl.as_tensor(v[k]*christoffel[k, i, j], [i, j])
    # print(okay)
    # covDev = ufl.as_tensor(ufl.grad(v)[j, b] * Push[b, a] - christoffel[i, j, a] * v[i], [j, a])

    # += Kinematics
    #    (0): Identity Matrix
    #    (1): Deformation Gradient Tensor
    #    (2): Right Cauchy-Green Tensor
    #    (3): Green-Strain Tensor
    #    (4): Jacobian of Deformation Gradient
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
    # F = ufl.as_tensor(I[i, j] + u[i].dx(j), [i, j])
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
    
    if constitutive == 0:
        # += Material Setup | Guccione
        #    (0): Constants
        #    (1): Strain Density Function
        #    (2): Chain Rule Differentiation Terms
        #    (3): First Piola-Kirchoff Stress
        c = 0.5 # 0.876
        b0 = 20  # 18.48
        b1 = 4  # 3.58
        b2 = 1  # 1.627
        Q = (
            b0 * E[0,0]**2 + 
            b1 * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
            b2 * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
        )
        # psi = c/2 * (ufl.exp(Q) - 1)
        piola = c/4 * ufl.exp(Q) * ufl.as_matrix([
            [4*b0*E[0,0], 2*b2*(E[1,0] + E[0,1]), 2*b2*(E[2,0] + E[0,2])],
            [2*b2*(E[0,1] + E[1,0]), 4*b1*E[1,1], 2*b1*(E[2,1] + E[1,2])],
            [2*b2*(E[0,2] + E[2,0]), 2*b1*(E[1,2] + E[2,1]), 4*b2*E[2,2]],
        ]) - p * Z_un
        # S = ufl.diff(psi, E)
        # piola = F * S + p * ufl.inv(Z_un) * J * ufl.inv(F).T
        # piola = F * T + p * J * ufl.inv(F).T

    elif constitutive == 1:  
        # += Material Setup | Mooney-Rivlin
        #    (0): Constants
        #    (1): Strain Density Function
        #    (2): Chain Rule Differentiation Terms
        #    (3): First Piola-Kirchoff Stress
        c1 = 2
        c2 = 6
        Ic = ufl.variable(ufl.tr(C))
        IIc = ufl.variable((Ic**2 - ufl.inner(C, C))/2)
        psi = c1 * (Ic - 3) + c2 *(IIc - 3) 
        # S = 2 * ufl.diff(psi, C) #- p * ufl.inv(Z_un)
        # piola = F * S + p * ufl.inv(Z_un) * J * ufl.inv(F).T
        # piola = F * S + p * J * ufl.inv(F).T #- p * ufl.inv(Z_un)
        # piola = ufl.as_tensor((0.5 * (ufl.diff(psi, E[a, b]) + ufl.diff(psi, E[b, a])) - p * ufl.inv(Z_un)[a, b]), [a, b])
        term1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
        term2 = -ufl.diff(psi, IIc)
        # piola = (term1*I + term2*C) - p * Z_un

        piola = 2 * F * (term1*I + term2*C) + p * ufl.inv(Z_un) * J * ufl.inv(F).T
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    #    (0): Quadrature order
    #    (1): Integration domains
    #    (2): Residual equation
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    R = term * dx + q * (J - 1) * dx 
    # R = ufl.inner(covDev[j], piola[a, b] * F[j, b]) * dx + q * (J - 1) * dx 
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

    print(u_sol.x.array[dofsY])
    # print(x_nu)
    # x = m + u_sol
    # print(x)
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "unitCube_Guccione90"
    # += Element order
    elem_order = 2
    # += Consitutive Equation
    constitutive = 0
    # += Feed Main()
    main(test_name, elem_order, constitutive)