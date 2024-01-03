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
import matplotlib.pyplot as plt
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
ROT = np.pi/4
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "area": 7}
PLOT = 0
# Guccione
# c, GCC_CONS[1], GCC_CONS[2], GCC_CONS[3]
GCC_CONS = [0.5, 10, 1, 1]
c = 0.5 # 0.876
GCC_CONS[1] = 20  # 18.48
GCC_CONS[2] = 4  # 3.58
GCC_CONS[3] = 1  # 1.627
# Mooney
# c1, c2
MNR_CONS = [2, 6]
c1 = 2
c2 = 6

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, elem_order, constitutive, quad_order):
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
    # += Tensor Indices
    i, j, k, a, b, c, d = ufl.indices(7)
    # += Curvilinear Mapping
    Push = ufl.as_tensor([
        [ufl.cos(ROT), -ufl.sin(ROT), 0],
        [ufl.sin(ROT), ufl.cos(ROT), 0],
        [0, 0, 1]
    ])
    # += Curvilinear Coordinates
    #    (0): Function Variable
    #    (1): Interpolation for x-values
    #    (2): Curvilinear undeformed coordinates | X_(θ)
    #    (3): Curvilinear displacement           | u_(θ)
    #    (4): Curvilinear deformed coordinates   | x_(θ)
    x = fem.Function(V)
    x.interpolate(lambda x: x)
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # += Metric Tensors
    #    (0): Covariant undeformed
    #    (1): Covariant deformed
    #    (2): Contravariant deformed
    Z_un = ufl.grad(x_nu).T * ufl.grad(x_nu)
    Z_co = ufl.grad(nu).T * ufl.grad(nu)
    Z_ct = ufl.inv(Z_co)
    # += Covariant and Contravariant Basis
    z_co = ufl.as_tensor((nu.dx(0), nu.dx(1), nu.dx(2)))
    z_ct = ufl.as_tensor(
        (
            Z_ct[0,0]*z_co[0] + Z_ct[1,0]*z_co[1] + Z_ct[2,0]*z_co[2],
            Z_ct[0,1]*z_co[0] + Z_ct[1,1]*z_co[1] + Z_ct[2,1]*z_co[2],
            Z_ct[0,2]*z_co[0] + Z_ct[1,2]*z_co[1] + Z_ct[2,2]*z_co[2]
        )
    )
    # += Christoffel Symbol | Γ^{i}_{j, a}
    gamma = ufl.as_tensor((
        0.5 * Z_ct[k, a] * (
            ufl.grad(Z_co)[a, i, j] + ufl.grad(Z_co)[a, j, i] - ufl.grad(Z_co)[i, j, a]
        )
    ), [k, i, j])
    # += Covariant Derivative
    covDev = ufl.grad(v) - ufl.as_tensor(v[k]*gamma[k, i, j], [i, j])
    # += Kinematics
    #    (0): Identity Matrix                   | I
    #    (1): Deformation Gradient Tensor       | F
    #    (2): Right Cauchy-Green Tensor         | C
    #    (3): Green-Strain Tensor               | E
    #    (4): Jacobian of Deformation Gradient  | J
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
    # += Constitutive Equations
    if constitutive == 0:
        # += Material Setup | Guccione
        Q = (
            GCC_CONS[1] * E[0,0]**2 + 
            GCC_CONS[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
            GCC_CONS[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
        )
        piola = GCC_CONS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
            [4*GCC_CONS[1]*E[0,0], 2*GCC_CONS[3]*(E[1,0] + E[0,1]), 2*GCC_CONS[3]*(E[2,0] + E[0,2])],
            [2*GCC_CONS[3]*(E[0,1] + E[1,0]), 4*GCC_CONS[2]*E[1,1], 2*GCC_CONS[2]*(E[2,1] + E[1,2])],
            [2*GCC_CONS[3]*(E[0,2] + E[2,0]), 2*GCC_CONS[2]*(E[1,2] + E[2,1]), 4*GCC_CONS[3]*E[2,2]],
        ]) - p * Z_un
    elif constitutive == 1:  
        # += Material Setup | Mooney-Rivlin
        Ic = ufl.variable(ufl.tr(C))
        IIc = ufl.variable((Ic**2 - ufl.inner(C, C))/2)
        psi = MNR_CONS[0] * (Ic - 3) + MNR_CONS[1] *(IIc - 3) 
        term1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
        term2 = -ufl.diff(psi, IIc)
        piola = 2 * F * (term1*I + term2*C) + p * ufl.inv(Z_un) * J * ufl.inv(F).T
    
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

    # +==+==+
    # Interpolate Stress
    #    (0): Function for calculating cauchy stress
    def cauchy(u, p):
        i = ufl.Identity(MESH_DIM)
        f = i + ufl.grad(u)
        c = f.T * f
        e = 0.5 * (c - i)
        j = ufl.det(f)
        q = (
            GCC_CONS[1] * e[0,0]**2 + 
            GCC_CONS[2] * (e[1,1]**2 + e[2,2]**2 + 2*(e[1,2] + e[2,1])) + 
            GCC_CONS[3] * (2*e[0,1]*e[1,0] + 2*e[0,2]*e[2,0])
        )
        s = GCC_CONS[0]/4 * ufl.exp(q) * ufl.as_matrix([
            [4*GCC_CONS[1]*e[0,0], 2*GCC_CONS[3]*(e[1,0] + e[0,1]), 2*GCC_CONS[3]*(e[2,0] + e[0,2])],
            [2*GCC_CONS[3]*(e[0,1] + e[1,0]), 4*GCC_CONS[2]*e[1,1], 2*GCC_CONS[2]*(e[2,1] + e[1,2])],
            [2*GCC_CONS[3]*(e[0,2] + e[2,0]), 2*GCC_CONS[2]*(e[1,2] + e[2,1]), 4*GCC_CONS[3]*e[2,2]],
        ]) - p * ufl.inv(c)
        sig = 1/j * f*s*f.T
        return sig
    #    (1): Create Tensor Function Space
    #    (2): Define expression over points
    #    (3): Create function variable
    #    (4): Interpolate
    TS = fem.FunctionSpace(domain, ("CG", elem_order, (3, 3)))
    piola_expr = fem.Expression(cauchy(u_sol, p_sol), TS.element.interpolation_points())
    sig = fem.Function(TS)
    sig.interpolate(piola_expr)
    # print(ufl.shape(sig))
    # print(sig.eval([1,1,1], [0]))
    # print(sig.x.array[dofsX])
    # exit

    # +==+==+
    # Scatter plot of x-displacement values at quadrature points
    if PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        cm = plt.cm.get_cmap('coolwarm')
        quad_points, _ = basix.make_quadrature(basix.cell.string_to_type(domain.topology.cell_name()), quad_order)
        quad_x = ufl.SpatialCoordinate(domain)
        quad_expr = fem.Expression(quad_x, quad_points)
        cells_n = domain.topology.index_map(domain.topology.dim).size_local
        u_x = list()
        q_pts = list()
        for i in range(0, cells_n, 1):
            pts = quad_expr.eval(domain, [i])
            for j in range(0, len(pts[0]), 3):
                eval_at = pts[0, j:j+3]
                q_pts.append(eval_at)
                u_x.append(u_sol.eval(eval_at, [i])[0])
        ps = np.array(q_pts)
        sc = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=u_x, vmin=min(u_x), vmax=max(u_x), s=50)

    # += Export
    strains = w.sub(0).collapse()
    strains.name = "strains"
    stresses = sig
    stresses.name = "stresses"
    with io.VTXWriter(MPI.COMM_WORLD, "Material_Field/paraview_bp/" + test_name + "_strains.bp", strains, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    with io.VTXWriter(MPI.COMM_WORLD, "Material_Field/paraview_bp/" + test_name + "_stresses.bp", stresses, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()

    # += Plot
    if PLOT:
        plt.colorbar(sc)
        plt.show()

    # print(u_sol.x.array[dofsY])
    # print(x_nu)
    # x = m + u_sol
    # print(x)
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "folder_test"
    # += Element order
    elem_order = 2
    # += Consitutive Equation
    constitutive = 0
    # += Quadature Degree
    quad_order = 4
    # += Feed Main()
    main(test_name, elem_order, constitutive, quad_order)