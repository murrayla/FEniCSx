"""
    Author: Liam Murray
    Contact: murrayla@student.unimelb.edu.au
    Initial Commit: 06/02/2024
    Code: simple_geo_test.py
        Tranverse Isotropic contraction over simple generated .msh branch geometry
"""

# +==+==+==+
# Setup
# += Imports
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import gmsh
import ufl
# += Parameters
MESH_DIM = 3
ORDER = 2
X, Y, Z = 0, 1, 2
LAMBDA = -900 #-0.02*65000
ROT = 0
Z_DISCS = 14
SARC_N = 7
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "area": 7}
SARC_L = 5000
MESH_R = 10000
PTs = list(range(100001, 100001+(Z_DISCS*20), 5))
CRs = list(range(10001, 10001+(Z_DISCS*20), 5))
CVs = list(range(20001, 20001+(Z_DISCS*20), 5))
SFs = list(range(1001, 1001+(Z_DISCS*20), 5))
PYs = list(range(101, 101+(Z_DISCS*20), 5))
VOs = list(range(11, 11+(Z_DISCS*20), 5))
# Guccione
GCC_CONS = [0.5, 1, 1, 1]

# +==+==+==+
# Gmsh Function for constructing idealised geometry
# +==+==+==+
def create_gmsh_cylinder(test):
    
    if test == 0:
        GEOMS = {
            0: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            1: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            2: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            3: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            4: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            5: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            6: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            7: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            8: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            9: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            10: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            11: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            12: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            13: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
        }
    elif test == 1:
        GEOMS = {
            0: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            1: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            2: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            3: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            4: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            5: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            6: {"z": [[0, 0, 1800]], "a": [[0, 0, 2566]]},
            7: {"z": [[0, 0, 1900]], "a": [[0, 0, 2608]]},
            8: {"z": [[-2326, 0, 1100], [1772, 0, 800]], "a": [[-2326, 0, 2325], [1772, 0, 1771]]},
            9: {"z": [[-2326, 0, 1100], [1772, 0, 800]], "a": [[-2326, 0, 2325], [1772, 0, 1771]]},
            10: {"z": [[-2326, 0, 1100], [1772, 0, 800]], "a": [[-2326, 0, 2325], [1772, 0, 1771]]},
            11: {"z": [[-2326, 0, 1100], [1772, 0, 800]], "a": [[-2326, 0, 2325], [1772, 0, 1771]]},
            12: {"z": [[-2326, 0, 1100], [1772, 0, 800]], "a": [[-2326, 0, 2325], [1772, 0, 1771]]},
            13: {"z": [[-2326, 0, 1100], [1772, 0, 800]], "a": [[-2326, 0, 2325], [1772, 0, 1771]]}
        }

    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(test_name)
    z_pts = []
    a_pts = []
    z_crv_ts = []
    a_crv_ts = []
    z_loo_ts = []
    a_loo_ts = []
    all_crv_ts = []
    ide_curves = []
    left_curves = []
    right_curves = []
    
    for i, d in enumerate(GEOMS.keys()):
        for j in range(0, len(GEOMS[d]["z"]), 1):
            pt = gmsh.model.occ.addPoint(
                x=GEOMS[d]["z"][j][0], y=0, z=SARC_L*d, 
                 meshSize=MESH_R
            )
            z_pts.append(pt)
            z_disc = gmsh.model.occ.addCircle(
                x=GEOMS[d]["z"][j][0], y=0, z=SARC_L*d,
                r=GEOMS[d]["z"][j][2], tag=CRs.pop(0)
            )
            z_crv_ts.append(z_disc)
            z_loop = gmsh.model.occ.addCurveLoop(curveTags=[z_disc], tag=CVs.pop(0))
            z_loo_ts.append(z_loop)
            all_crv_ts.append(z_loop)

            if test == 0:
                ide_curves.append(z_loop)
            elif test == 1:
                if i < 8:
                    ide_curves.append(z_loop)
                elif j == 0:
                    left_curves.append(z_loop)
                elif j == 1:
                    right_curves.append(z_loop)

        if i < Z_DISCS-1:

            for k in range(0, len(GEOMS[d]["a"]), 1):
                pt = gmsh.model.occ.addPoint(
                    x=GEOMS[d]["a"][k][0], y=0, z=SARC_L*d + SARC_L//2, 
                        meshSize=MESH_R
                )
                a_pts.append(pt)
                a_band = gmsh.model.occ.addCircle(
                    x=GEOMS[d]["a"][k][0], y=0, z=SARC_L*d + SARC_L//2,
                    r=GEOMS[d]["a"][k][2], tag=CRs.pop(0)
                )
                a_crv_ts.append(a_band)
                a_loop = gmsh.model.occ.addCurveLoop(curveTags=[a_band], tag=CVs.pop(0))
                a_loo_ts.append(a_loop)
                all_crv_ts.append(a_loop)

                if test == 0: 
                    ide_curves.append(a_loop)
                elif test == 1:
                    if i < 8:
                        ide_curves.append(a_loop)
                    elif k == 0:
                        left_curves.append(a_loop)
                    elif k == 1:
                        right_curves.append(a_loop)
    
    
    a = gmsh.model.occ.addThruSections(wireTags=ide_curves, tag=11111, makeSolid=True, makeRuled=True)

    if test == 1:
        b = gmsh.model.occ.addThruSections(wireTags=[ide_curves[-1]] + left_curves, tag=22222, makeSolid=True, makeRuled=True)
        c = gmsh.model.occ.addThruSections(wireTags=[ide_curves[-1]] + right_curves, tag=33333, makeSolid=True, makeRuled=True)
        gmsh.model.occ.remove(a, recursive=False)
        gmsh.model.occ.remove(b, recursive=False)
        gmsh.model.occ.remove(c, recursive=False)
        d = gmsh.model.occ.fuse([(2, 18)], [(2, 31)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(2, 29)], recursive=False) 
        gmsh.model.occ.remove([(2, 48)], recursive=False) 
        gmsh.model.occ.remove([(2, 45)], recursive=False) 
        gmsh.model.occ.remove([(2, 44)], recursive=False) 
        gmsh.model.occ.remove([(2, 42)], recursive=False) 
        ideal = list(range(1, 17, 1))

    gmsh.model.occ.synchronize()

    if test == 1:
        gmsh.model.occ.fuse([(2, 15)], [(2, 46), (2, 47)], removeObject=True, removeTool=True)
        branch = list(range(19, 29, 1)) + [30] + list(range(32, 42, 1)) + [43, 48, 47]
        branch_sl = gmsh.model.occ.addSurfaceLoop(ideal+branch, sewing=True)
        branch_vol = gmsh.model.occ.addVolume([branch_sl])
        gmsh.model.occ.synchronize()

    if test == 0:
        gmsh.model.addPhysicalGroup(3, [11111], name="Myo_Vol")
        gmsh.model.addPhysicalGroup(2, [27], name="Myo_Base")
        gmsh.model.addPhysicalGroup(2, [28], name="Sarc_Top")

    if test == 1:
        gmsh.model.addPhysicalGroup(3, [1], name="Myo_Vol")
        gmsh.model.addPhysicalGroup(2, [16], name="Myo_Base")
        gmsh.model.addPhysicalGroup(2, [30, 43 ], name="Sarc_Top")

    gmsh.model.occ.synchronize()
    
    # += Create Mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(2)
    # += Write File
    gmsh.write("P_Branch_Contraction/gmsh_msh/" + test_name + ".msh")
    gmsh.finalize()

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, quad_order, test_type):
    # +==+==+
    # Setup problem space
    create_gmsh_cylinder(test_type)
    
    file = "P_Branch_Contraction/gmsh_msh/" + test_name + ".msh"
    domain, _, ft = io.gmshio.read_from_msh(file, MPI.COMM_WORLD, 0, gdim=MESH_DIM)
    # ft.name = "Facet markers"
    # domain = mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=X_ELS, ny=Y_ELS, nz=Z_ELS, cell_type=mesh.CellType.hexahedron)
    Ve = ufl.VectorElement(family="CG", cell=domain.ufl_cell(), degree=2)
    Vp = ufl.FiniteElement("CG", domain.ufl_cell(), degree=2-1)  
    W = fem.FunctionSpace(domain, ufl.MixedElement([Ve, Vp]))
    w = fem.Function(W)

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
    x0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 0))
    x1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], 1))
    y0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 0))
    y1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], 1))
    z0_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], 0))
    z1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], 65000))
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
    x0_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, z0_facets)
    u0_bc_x = fem.Function(Vx)
    u0_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_x0_x = fem.dirichletbc(u0_bc_x, x0_dofs_x, W.sub(0).sub(X))
    # +==+ [y0]
    y0_dofs_y = fem.locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, z0_facets)
    u0_bc_y = fem.Function(Vy)
    u0_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_y0_y = fem.dirichletbc(u0_bc_y, y0_dofs_y, W.sub(0).sub(Y))
    # +==+ [z0]
    z0_dofs_z = fem.locate_dofs_topological((W.sub(0).sub(Z), Vz), ft.dim, z0_facets)
    u0_bc_z = fem.Function(Vz)
    u0_bc_z.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-LAMBDA)))
    bc_z0_z = fem.dirichletbc(u0_bc_z, z0_dofs_z, W.sub(0).sub(Z))
    # +==+ [x1]
    x1_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), ft.dim, z1_facets)
    u1_bc_x = fem.Function(Vx)
    u1_bc_x.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_x1_x = fem.dirichletbc(u1_bc_x, x1_dofs_x, W.sub(0).sub(X))
    # +==+ [y1]
    y1_dofs_y = fem.locate_dofs_topological((W.sub(0).sub(Y), Vy), ft.dim, z1_facets)
    u1_bc_y = fem.Function(Vy)
    u1_bc_y.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_y1_y = fem.dirichletbc(u1_bc_y, y1_dofs_y, W.sub(0).sub(Y))
    # +==+ [z1]
    z1_dofs_z = fem.locate_dofs_topological((W.sub(0).sub(Z), Vz), ft.dim, z1_facets)
    u1_bc_z = fem.Function(Vz)
    u1_bc_z.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(LAMBDA)))
    bc_z1_z = fem.dirichletbc(u1_bc_z, z1_dofs_z, W.sub(0).sub(Z))
    # +==+ BC Concatenate
    bc = [bc_x0_x, bc_y0_y, bc_z0_z, bc_x1_x, bc_y1_y, bc_z1_z]
    # bc = [bc_z0_z, bc_z1_z]

    # +==+==+
    # Setup Parameteres for Variational Equation
    # += Test and Trial Functions
    v, q = ufl.TestFunctions(W)
    u, p = ufl.split(w)
    # # += Identity Tensor
    I = ufl.variable(ufl.Identity(MESH_DIM))
    # += Deformation Gradient Tensor
    #    F = ∂u/∂X + I
    F = ufl.variable(I + ufl.grad(u))
    # += Right Cauchy-Green Tensor
    C = ufl.variable(F.T * F)
    # += Invariants
    #    (1): λ1^2 + λ2^2 + λ3^2; tr(C)
    Ic = ufl.variable(ufl.tr(C))
    #    (2): λ1^2*λ2^2 + λ2^2*λ3^2 + λ3^2*λ1^2; 0.5*[(tr(C)^2 - tr(C^2)]
    IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
    #    (3): λ1^2*λ2^2*λ3^2; det(C) = J^2
    J = ufl.variable(ufl.det(F))
    # IIIc = ufl.variable(ufl.det(C))
    # += Material Parameters
    c1 = 2
    c2 = 6
    # += Mooney-Rivlin Strain Energy Density Function
    psi = c1 * (Ic - 3) + c2 *(IIc - 3) 
    # Terms
    gamma1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
    gamma2 = -ufl.diff(psi, IIc)
    # += First Piola Stress
    firstPK = 2 * F * (gamma1*I + gamma2*C) + p * J * ufl.inv(F).T
    cau = (1/J * firstPK * F).T

    # +==+==+
    # Setup Variational Problem Solver
    # += Gaussian Quadrature
    metadata = {"quadrature_degree": 4}
    # += Domains of integration
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    # += Residual Equation (Variational, for solving)
    R = ufl.inner(ufl.grad(v), firstPK) * dx + q * (J - 1) * dx
    problem = NonlinearProblem(R, w, bc)

    solver = NewtonSolver(domain.comm, problem)
    # += Tolerances for convergence
    solver.atol = 1e-4
    solver.rtol = 1e-4
    # += Convergence criteria
    solver.convergence_criterion = "incremental"

    num_its, converged = solver.solve(w)
    u_sol, p_sol = w.split()
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged after {num_its} iterations.")

    # print(u_sol.x.array[x0_dofs_x])

    # # +==+==+
    # # ParaView export
    # with io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ".bp", w.sub(0).collapse(), engine="BP4") as vtx:
    #     vtx.write(0.0)
    #     vtx.close()

    # # +==+==+
    # # Variational Problem Setup
    # # += Test and Trial Parameters
    # u, p = ufl.split(w)
    # v, q = ufl.TestFunctions(W)
    # # += Tensor Indices
    # i, j, k, a, b, c, d = ufl.indices(7)
    # # += Curvilinear Mapping
    # Push = ufl.as_tensor([
    #     [ufl.cos(ROT), -ufl.sin(ROT), 0],
    #     [ufl.sin(ROT), ufl.cos(ROT), 0],
    #     [0, 0, 1]
    # ])
    # # += Curvilinear Coordinates
    # x = fem.Function(V)
    # x.interpolate(lambda x: x)
    # x_nu = ufl.inv(Push) * x
    # u_nu = ufl.inv(Push) * u
    # nu = ufl.inv(Push) * (x + u_nu)
    # # += Metric Tensors
    # Z_un = ufl.grad(x_nu).T * ufl.grad(x_nu)
    # Z_co = ufl.grad(nu).T * ufl.grad(nu)
    # Z_ct = ufl.inv(Z_co)
    # # += Covariant and Contravariant Basis
    # z_co = ufl.as_tensor((nu.dx(0), nu.dx(1), nu.dx(2)))
    # # += Christoffel Symbol | Γ^{i}_{j, a}
    # gamma = ufl.as_tensor((
    #     0.5 * Z_ct[k, a] * (
    #         ufl.grad(Z_co)[a, i, j] + ufl.grad(Z_co)[a, j, i] - ufl.grad(Z_co)[i, j, a]
    #     )
    # ), [k, i, j])
    # # += Covariant Derivative
    # covDev = ufl.grad(v) - ufl.as_tensor(v[k]*gamma[k, i, j], [i, j])
    # # += Kinematics
    # I = ufl.variable(ufl.Identity(MESH_DIM))
    # F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
    # C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    # E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
    # J = ufl.variable(ufl.det(F))
    # aTen = ufl.as_tensor([[0.05, 0, 0],[0, 0, 0],[0, 0, 0]])
    # # += Constitutive Equations
    # # += Material Setup | Guccione
    # Q = (
    #     GCC_CONS[1] * E[0,0]**2 + 
    #     GCC_CONS[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
    #     GCC_CONS[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
    # )
    # piola = GCC_CONS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
    #     [4*GCC_CONS[1]*E[0,0], 2*GCC_CONS[3]*(E[1,0] + E[0,1]), 2*GCC_CONS[3]*(E[2,0] + E[0,2])],
    #     [2*GCC_CONS[3]*(E[0,1] + E[1,0]), 4*GCC_CONS[2]*E[1,1], 2*GCC_CONS[2]*(E[2,1] + E[1,2])],
    #     [2*GCC_CONS[3]*(E[0,2] + E[2,0]), 2*GCC_CONS[2]*(E[1,2] + E[2,1]), 4*GCC_CONS[3]*E[2,2]],
    # ]) - p * Z_un + aTen
    
    # # +==+==+
    # # Problem Solver
    # # += Residual Equation Integral
    # metadata = {"quadrature_degree": quad_order}
    # ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    # dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    # term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    # R = term * dx + q * (J - 1) * dx 
    # # += Nonlinear Solver
    # problem = NonlinearProblem(R, w, bc)
    # solver = NewtonSolver(domain.comm, problem)
    # solver.atol = 1e-8
    # solver.rtol = 1e-8
    # solver.convergence_criterion = "incremental"

    # # +==+==+
    # # Solution and Output
    # # += Solve
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
    
    TS = fem.FunctionSpace(domain, ("CG", 2, (3, 3)))
    piola_expr = fem.Expression(cauchy(u_sol, p_sol), TS.element.interpolation_points())
    sig = fem.Function(TS)
    sig.interpolate(piola_expr)

    # += Export
    strains = w.sub(0).collapse()
    strains.name = "strains"
    stresses = sig
    stresses.name = "stresses"
    with io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + "_strains.bp", strains, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    with io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + "_stresses.bp", stresses, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "TEST_QUICK_BRANCH"
    # += Quadature Degree
    quad_order = 4
    # += Test Type
    test_type = 1
    # += Feed Main()
    main(test_name, quad_order, test_type)