# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 17/01/2024
# +==+==+==+==+==+

# +==+==+==+==+
# Code: iterAnis.py
#   Itererative Aniostropic solver of simple extension test
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
from dolfinx.mesh import (create_unit_cube, CellType, locate_entities_boundary, meshtags)
from petsc4py import PETSc
from mpi4py import MPI
#
import matplotlib.pyplot as plt
import numpy as np
import ufl
# += Parameters
MESH_DIM = 3
X, Y, Z = 0, 1, 2
X_ELS = 1
Y_ELS = 1
Z_ELS = 1
ROT = 0 #np.pi/4
LAMBDA = 0.1 # 10% extension
MAX_ITS = 50
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "area": 7}
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
    domain = create_unit_cube(comm=MPI.COMM_WORLD, nx=X_ELS, ny=Y_ELS, nz=Z_ELS, cell_type=CellType.hexahedron)
    Ve = ufl.VectorElement(family="CG", cell=domain.ufl_cell(), degree=elem_order)
    Vp = ufl.FiniteElement("CG", domain.ufl_cell(), degree=elem_order-1)  
    W = FunctionSpace(domain, ufl.MixedElement([Ve, Vp]))
    w = Function(W)

    # +==+==+ 
    # Extract subdomains for dofs
    V, _ = W.sub(0).collapse()
    P, _ = W.sub(1).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()
    Vz, dofsZ = V.sub(Z).collapse()

    # +==+==+
    # Facet assignment
    fdim = MESH_DIM - 1
    # += Locate Facets
    x0_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 0))
    x1_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 1))
    y0_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 0))
    y1_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 1))
    z0_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], 0))
    z1_facets = locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], 1))
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
    ft = meshtags(mesh=domain, dim=fdim, entities=mfacets[sfacets], values=mvalues[sfacets])

    # +==+==+ 
    # Boundary Conditions
    # +==+ [x0]
    x0_dofs_x = locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x0_facets)
    u0_bc_x = Function(Vx)
    u0_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_x0_x = dirichletbc(u0_bc_x, x0_dofs_x, W.sub(0).sub(X))
    # +==+ [x1]
    x1_dofs_x = locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, x1_facets)
    u1_bc_x = Function(Vx)
    u1_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(LAMBDA)))
    bc_x1_x = dirichletbc(u1_bc_x, x1_dofs_x, W.sub(0).sub(X))
    # +==+ [y0]
    y0_dofs_y = locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, y0_facets)
    u0_bc_y = Function(Vy)
    u0_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_y0_y = dirichletbc(u0_bc_y, y0_dofs_y, W.sub(0).sub(Y))
    # +==+ [z0]
    z0_dofs_z = locate_dofs_topological((W.sub(0).sub(Z), Vz), ft.dim, z0_facets)
    u0_bc_z = Function(Vz)
    u0_bc_z.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_z0_z = dirichletbc(u0_bc_z, z0_dofs_z, W.sub(0).sub(Z))
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
    x = Function(V)
    x.interpolate(lambda x: x)
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # += Metric Tensors
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
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
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
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    metadata = {"quadrature_degree": quad_order}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    R = term * dx + q * (J - 1) * dx 

    class NonlinearPDE_SNESProblem:
        def __init__(self, F, u, bc):
            V = u.function_space
            du = ufl.TrialFunction(V)
            self.L = form(F)
            self.a = form(ufl.derivative(F, u, du))
            self.bc = bc
            self._F, self._J = None, None
            self.u = u

        def F(self, snes, x, F):
            """Assemble residual vector."""
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            x.copy(self.u.vector)
            self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            with F.localForm() as f_local:
                f_local.set(0.0)
            assemble_vector(F, self.L)
            apply_lifting(F, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
            F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(F, [self.bc], x, -1.0)

        def J(self, snes, x, J, P):
            """Assemble Jacobian matrix."""
            J.zeroEntries()
            assemble_matrix(J, self.a, bcs=[self.bc])
            J.assemble()
    
    w0 = Function(W)
    problem = NonlinearPDE_SNESProblem(R, w, bc)
    b = la.create_petsc_vector(W.dofmap.index_map, W.dofmap.index_map_bs)
    newtJac = create_matrix(problem.a)
    snes = PETSc.SNES().create()
    opts = PETSc.Options()
    opts['snes_monitor'] = None
    snes.setFromOptions()
    snes.setType('vinewtonrsls')  # Changed even to vinewtonssls
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, newtJac)
    snes.setTolerances(rtol=1.0e-8, max_it=50)
    snes.getKSP().setType("preonly")
    snes.getKSP().setTolerances(rtol=1.0e-8)
    snes.getKSP().getPC().setType("lu")
    file = io.XDMFFile(MPI.COMM_WORLD, "SNES.xdmf", "w")
    file.write_mesh(domain)
    t = 0
    w.x.array[:] = 0.0
    w0.x.array[:] = w.x.array
    while (t < 5):
        t += 1
        r = snes.solve(None, w.vector)
        # print(f"Step {int(t/dt)}: num iterations: {r[0]}")
        w0.x.array[:] = w.x.array
        file.write_function(w.sub(0), t)
    file.close()

    # R = ufl.inner(covDev[j], piola[a, b] * F[j, b]) * dx + q * (J - 1) * dx 
    # += Nonlinear Solver
    #    (0): Definition
    #    (1): Newton Iterator
    #         (0): Tolerances
    #         (1): Convergence
    # problem = NonlinearProblem(R, w, bc)
    # solver = NewtonSolver(domain.comm, problem)
    # solver.atol = 1e-8
    # solver.rtol = 1e-8
    # solver.convergence_criterion = "incremental"

    # +==+==+
    # Solution and Output
    # += Solve
    #    (0): Iteration and convergence
    #    (1): Return data
    # num_its, converged = solver.solve(w)
    # if converged:
    #     print(f"Converged in {num_its} iterations.")
    # else:
    #     print(f"Not converged after {num_its} iterations.")
    # u_sol, p_sol = w.split()

    # +==+==+
    # Interpolate Stress
    #    (0): Function for calculating cauchy stress
    def cauchy(u, p):
        print(u.x.array)
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
    TS = FunctionSpace(domain, ("CG", elem_order, (3, 3)))
    piola_expr = Expression(cauchy(u_sol, p_sol), TS.element.interpolation_points())
    sig = Function(TS)
    sig.interpolate(piola_expr)

    # += Export
    strains = w.sub(0).collapse()
    strains.name = "eps"
    stresses = sig
    stresses.name = "sig"
    with io.VTXWriter(MPI.COMM_WORLD, "Iterative_Anisotropic/paraview_bp/" + test_name + "_eps.bp", strains, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    with io.VTXWriter(MPI.COMM_WORLD, "Iterative_Anisotropic/paraview_bp/" + test_name + "_sig.bp", stresses, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "GC45_111"
    # += Element order
    elem_order = 2
    # += Quadature Degree
    quad_order = 4
    # += Feed Main()
    main(test_name, elem_order, quad_order)