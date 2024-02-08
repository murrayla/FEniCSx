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
LAMBDA = 0
ROT = 0
Z_DISCS = 14
SARC_N = 7
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "area": 7}
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
SARC_L = 5000
MESH_R = 10000
PTs = list(range(100001, 100001+(Z_DISCS*10), 2))
CRs = list(range(10001, 10001+(Z_DISCS*10), 2))
CVs = list(range(20001, 20001+(Z_DISCS*10), 2))
SFs = list(range(1001, 1001+(Z_DISCS*10), 2))
PYs = list(range(101, 101+(Z_DISCS*10), 2))
VOs = list(range(11, 11+(Z_DISCS*10), 2))
# Guccione
GCC_CONS = [0.5, 1, 1, 1]

# +==+==+==+
# Gmsh Function for constructing idealised geometry
# +==+==+==+
def create_gmsh_cylinder():
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
    z_srf_ts = []
    a_srf_ts = []
    all_crv_ts = []
    all_srf_ts = []

    ide_curves = []
    left_curves = []
    right_curves = []
    
    for i, d in enumerate(GEOMS.keys()):

        for j in range(0, len(GEOMS[d]["z"]), 1):
            pt = gmsh.model.occ.addPoint(
                x=GEOMS[d]["z"][j][0], y=0, z=SARC_L*d, 
                tag=PTs.pop(0), meshSize=MESH_R
            )
            z_pts.append(pt)
            z_disc = gmsh.model.occ.addCircle(
                x=GEOMS[d]["z"][j][0], y=0, z=SARC_L*d,
                r=GEOMS[d]["z"][j][2], tag=CRs.pop(0)
            )
            z_crv_ts.append(z_disc)
            z_loop = gmsh.model.occ.addCurveLoop(curveTags=[z_disc], tag=CVs.pop(0))
            # z_surf = gmsh.model.occ.addPlaneSurface(wireTags=[z_loop], tag=SFs.pop(0))
            # z_srf_ts.append(z_surf)
            # gmsh.model.occ.synchronize()
            # gmsh.model.addPhysicalGroup(
            #     dim=MESH_DIM-1, tags=[z_surf], tag=PYs.pop(0), name="z_disc_" + str((i+1)*1+(j+1))
            # )
            z_loo_ts.append(z_loop)
            all_crv_ts.append(z_loop)
            if i < 8:
                ide_curves.append(z_loop)
            elif j == 0:
                left_curves.append(z_loop)
            elif j == 1:
                right_curves.append(z_loop)


        if i < Z_DISCS-1:

            if d == 7:
                # break
                cen_pt = gmsh.model.occ.addPoint(
                    x=GEOMS[d]["a"][0][0], y=0, z=SARC_L*d + SARC_L//2, 
                    tag=PTs.pop(0), meshSize=MESH_R
                )
                a_pts.append(cen_pt)
                top_pt = gmsh.model.occ.addPoint(
                    x=GEOMS[d]["a"][0][0], y=GEOMS[d]["a"][0][2], z=SARC_L*d + SARC_L//2, 
                    tag=PTs.pop(0), meshSize=MESH_R
                )
                a_pts.append(top_pt)
                bot_pt = gmsh.model.occ.addPoint(
                    x=GEOMS[d]["a"][0][0], y=-GEOMS[d]["a"][0][2], z=SARC_L*d + SARC_L//2, 
                    tag=PTs.pop(0), meshSize=MESH_R
                )
                a_pts.append(bot_pt)
                a_band_left = gmsh.model.occ.addCircleArc(startTag=bot_pt, centerTag=cen_pt, endTag=top_pt, tag=CRs.pop(0))
                a_band_right = gmsh.model.occ.addCircleArc(startTag=top_pt, centerTag=cen_pt, endTag=bot_pt, tag=CRs.pop(0))
                a_band_line = gmsh.model.occ.addLine(startTag=bot_pt, endTag=top_pt, tag=CRs.pop(0))
                a_loop_left = gmsh.model.occ.addCurveLoop(curveTags=[a_band_left, a_band_line], tag=CVs.pop(0))
                a_loop_right = gmsh.model.occ.addCurveLoop(curveTags=[a_band_right, a_band_line], tag=CVs.pop(0))

                a_crv_ts.append(a_band_left)
                a_crv_ts.append(a_band_right)
                a_crv_ts.append(a_band_line)

                a_loo_ts.append(a_loop_left)
                a_loo_ts.append(a_loop_right)
                all_crv_ts.append(a_loop_left)
                all_crv_ts.append(a_loop_right)

                left_curves.append(a_loop_left)
                right_curves.append(a_loop_right)

            else:

                for k in range(0, len(GEOMS[d]["a"]), 1):
                    pt = gmsh.model.occ.addPoint(
                        x=GEOMS[d]["a"][k][0], y=0, z=SARC_L*d + SARC_L//2, 
                        tag=PTs.pop(0), meshSize=MESH_R
                    )
                    a_pts.append(pt)
                    a_band = gmsh.model.occ.addCircle(
                        x=GEOMS[d]["a"][k][0], y=0, z=SARC_L*d + SARC_L//2,
                        r=GEOMS[d]["a"][k][2], tag=CRs.pop(0)
                    )
                    a_crv_ts.append(a_band)
                    a_loop = gmsh.model.occ.addCurveLoop(curveTags=[a_band], tag=CVs.pop(0))
                    # a_surf = gmsh.model.occ.addPlaneSurface(wireTags=[a_loop], tag=SFs.pop(0))
                    # a_srf_ts.append(a_surf)
                    # gmsh.model.occ.synchronize()
                    # gmsh.model.addPhysicalGroup(
                    #     dim=MESH_DIM-1, tags=[a_surf], tag=PYs.pop(0), name="a_band_" + str((i+1)*1+(j+k+2))
                    # )
                    a_loo_ts.append(a_loop)
                    all_crv_ts.append(a_loop)

                    if i < 8:
                        ide_curves.append(a_loop)
                    elif k == 0:
                        left_curves.append(a_loop)
                    elif k == 1:
                        right_curves.append(a_loop)
    
    # sarcomere = []
    # for i in range(0, len(a_loo_ts), 1):
    #     sarcomere.append([z_loo_ts[i], a_loo_ts[i], z_loo_ts[i+1]])
    # extrusion = []
    # # for gs in sarcomere:
    # ext = gmsh.model.occ.addThruSections(wireTags=all_crv_ts, makeSolid=True, makeRuled=False)
    # extrusion.append(ext)
    # gmsh.model.occ.synchronize()

    # volume = gmsh.model.addPhysicalGroup(3, [extrusion[1][1]], name="volume")
    # for k in range(0, 2, 1):
    #     ide = k + 1
    #     # gmsh.model.addPhysicalGroup(3, [ide], name="Sarc_" + str(k) + "_Vol")
    #     gmsh.model.addPhysicalGroup(2, [ide], name="Sarc_" + str(k) + "_b")
    #     gmsh.model.addPhysicalGroup(2, [ide+2], name="Sarc_" + str(k+5) + "_c")
    #     gmsh.model.addPhysicalGroup(2, [ide+4], name="Sarc_" + str(k+10) + "_d")
    # gmsh.model.occ.synchronize()

    # gmsh.model.occ.remove([(3, 1)])
    # gmsh.model.occ.remove([(3, 2)])
    # sl = gmsh.model.occ.addSurfaceLoop([1, 2, 4, 6])
    # gmsh.model.occ.addVolume([sl])
    # gmsh.model.occ.synchronize()
    

    # += Create Mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(2)
    # += Write File
    gmsh.write("Branch_Contraction/gmsh_msh/" + test_name + ".msh")
    gmsh.finalize()
    exit()

# +==+==+==+
# main()
#   Inputs: 
#       (0): test_name  | str
#       (1): elem_order | int | Order of elements to generate
#   Outputs:
#       (0): .bp folder of contracted unit square
def main(test_name, quad_order):
    # +==+==+
    # Setup problem space
    create_gmsh_cylinder()
    
    file = "Branch_Contraction/gmsh_msh/" + test_name + ".msh"
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
    z1_facets = mesh.locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], 25000))
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
    u0_bc_z.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(0.0)))
    bc_z0_z = fem.dirichletbc(u0_bc_z, z0_dofs_z, W.sub(0).sub(Z))
    # +==+ [z1]
    z1_dofs_z = fem.locate_dofs_topological((W.sub(0).sub(Z), Vz), ft.dim, z1_facets)
    u1_bc_z = fem.Function(Vz)
    u1_bc_z.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(LAMBDA)))
    bc_z1_z = fem.dirichletbc(u1_bc_z, z1_dofs_z, W.sub(0).sub(Z))
    # +==+ BC Concatenate
    bc = [bc_x0_x, bc_y0_y, bc_z0_z, bc_z1_z]

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
    x = fem.Function(V)
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
    aTen = ufl.as_tensor([[0.05, 0, 0],[0, 0, 0],[0, 0, 0]])
    # += Constitutive Equations
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
    ]) - p * Z_un + aTen
    
    # +==+==+
    # Problem Solver
    # += Residual Equation Integral
    metadata = {"quadrature_degree": quad_order}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    R = term * dx + q * (J - 1) * dx 
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
    u_sol, p_sol = w.split()

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
    with io.VTXWriter(MPI.COMM_WORLD, "Branch_Contraction/paraview_bp/" + test_name + "_strains.bp", strains, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    with io.VTXWriter(MPI.COMM_WORLD, "Branch_Contraction/paraview_bp/" + test_name + "_stresses.bp", stresses, engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()
    
# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "SIMPLE_GC0_NEG5_TEST"
    # += Quadature Degree
    quad_order = 4
    # += Feed Main()
    main(test_name, quad_order)