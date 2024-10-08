"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: fx_3DPasCon.py
        3D Passive Contraction model of scalable sarcomere number in FEniCSx
"""

# +==+==+==+
# Setup
# += Imports
from dolfinx import io,  default_scalar_type, geometry
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological, Expression, form
from dolfinx.mesh import locate_entities_boundary, meshtags
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gmsh
import ufl
import csv
import os

# += Parameters
DIM = 3
NULL = 0
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1024, "y": 1024, "z": 80}
ORDER = 2
X, Y, Z = 0, 1, 2
QUADRATURE = 4
EL_TAGS = {0: 1e0, 1: 1e2, 2: 1e3, 3: 5e3, 4: 1e4, 5: 2e4}
PY_TAGS = {0: 1e2, 1: 1e3, 2: 5e3, 3: 1e4, 4: 2e4, 5: 3e4, "x0": 1, "x1": 2}

# +==+==+==+
# fx_
#   Inputs: 
#       tnm  | str | test name
#   Outputs:
#       .bp folder of deformation
def fx_(tnm, file, depth):
    depth += 1

    # +==+ Domain Setup
    print("\t" * depth + "+= Generate Mesh")
    # += Load mesh data
    domain, ct, _ = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    # += Create vector spaces
    V2 = ufl.VectorElement(family="CG", cell=domain.ufl_cell(), degree=ORDER)
    V1 = ufl.FiniteElement(family="CG", cell=domain.ufl_cell(), degree=ORDER-1)  
    # += Mixed Space
    Mxs = FunctionSpace(mesh=domain, element=ufl.MixedElement([V2, V1]))
    # += Vector Spaces and Tensor Space
    Sos = FunctionSpace(mesh=domain, element=V2)
    Fos = FunctionSpace(mesh=domain, element=V1)
    Tes = FunctionSpace(mesh=domain, element=("CG", ORDER, (DIM, DIM)))

    # +==+ Function Variables
    # += Current
    mx = Function(Mxs)
    mv, mp = mx.split()
    (dv, dm) = ufl.TestFunctions(Mxs)
    # += Old
    mx_i = Function(Mxs)
    mv_i, mp_i = mx_i.split()

    # +==+ Extract subdomains for dofs
    V, _ = Mxs.sub(0).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()

    # +==+ Boundary Condition Setup
    def boundary_conditions(domain, W, Vx, Vy, Vz, du):
        # += Facet assignment
        fdim = DIM - 1
        xx0_ft, xx1_ft, yx0_ft, yx1_ft, zx0_ft, zx1_ft = (
            locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], min(x[0]))),
            locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[0], max(x[0]))),
            locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], min(x[0]))),
            locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[1], max(x[0]))),
            locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], min(x[0]))),
            locate_entities_boundary(mesh=domain, dim=fdim, marker=lambda x: np.isclose(x[2], max(x[0]))),
        )
        mfacets = np.hstack([xx0_ft, xx1_ft, yx0_ft, yx1_ft, zx0_ft, zx1_ft])
        # += Assign boundaries IDs and sort
        mvalues = np.hstack([
            np.full_like(xx0_ft, PY_TAGS["x0"]), 
            np.full_like(xx1_ft, PY_TAGS["x1"]), 
            np.full_like(yx0_ft, PY_TAGS["x0"]),
            np.full_like(yx1_ft, PY_TAGS["x1"]), 
            np.full_like(zx0_ft, PY_TAGS["x0"]),
            np.full_like(zx1_ft, PY_TAGS["x1"]), 
        ])
        sfacets = np.argsort(mfacets)
        ft = meshtags(mesh=domain, dim=fdim, entities=mfacets[sfacets], values=mvalues[sfacets])
        # += Locate subdomain dofs
        xx0_dofs, xx1_dofs, yx0_dofs, yx1_dofs, zx0_dofs, zx1_dofs = (
            locate_dofs_topological(V=(W.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=xx0_ft),
            locate_dofs_topological(V=(W.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=xx1_ft),
            locate_dofs_topological(V=(W.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=yx0_ft),
            locate_dofs_topological(V=(W.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=yx1_ft),
            locate_dofs_topological(V=(W.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=zx0_ft),
            locate_dofs_topological(V=(W.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=zx1_ft)
        )
        # += Interpolate 
        uxx0, uxx1, uyx0, uyx1, uzx0, uzx1 = (
            Function(Vx), Function(Vx), Function(Vy), Function(Vy), Function(Vz), Function(Vz)
        )
        uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du)))
        uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-du)))
        uyx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(NULL)))
        uyx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(NULL)))
        uzx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(NULL)))
        uzx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(NULL)))
        # += Create Dirichlet over subdomains
        bc_UxX0 = dirichletbc(value=uxx0, dofs=xx0_dofs, V=W.sub(0).sub(X))
        bc_UxX1 = dirichletbc(value=uxx1, dofs=xx1_dofs, V=W.sub(0).sub(X))
        bc_UyX0 = dirichletbc(value=uyx0, dofs=yx0_dofs, V=W.sub(0).sub(Y))
        bc_UyX1 = dirichletbc(value=uyx1, dofs=yx1_dofs, V=W.sub(0).sub(Y))
        bc_UzX0 = dirichletbc(value=uzx0, dofs=zx0_dofs, V=W.sub(0).sub(Z))
        bc_UzX1 = dirichletbc(value=uzx1, dofs=zx1_dofs, V=W.sub(0).sub(Z))
        # += Assign
        return [bc_UxX0, bc_UyX0, bc_UxX1, bc_UyX1], ft
    
    # += Extract boundary conditions and facet tags
    bc, ft = boundary_conditions(domain, Mxs, Vx, Vy, DISPLACEMENT)

    # +==+ Subdomain Setup
    V_1DG = FunctionSpace(domain, ("DG", 0))
    def subdomain_assignments(domain, ct, V_1DG):
        # += Locate cells of interest
        str_myo = ct.find(GROUP_IDS["Straight"])
        inc_myo = ct.find(GROUP_IDS["Incline"])
        dec_myo = ct.find(GROUP_IDS["Decline"])
        cytosol = ct.find(GROUP_IDS["Cytosol"])
        # += Create 1-DOF space for assignment
        fibre_rot, fibre_val = Function(V_1DG), Function(V_1DG)
        # += Conditionally assign rotation and material value
        if len(str_myo):
            fibre_rot.x.array[str_myo] = np.full_like(str_myo, NULL, dtype=default_scalar_type)
            fibre_val.x.array[str_myo] = np.full_like(str_myo, MATERIAL_CONSTANTS[1], dtype=default_scalar_type)
        if len(inc_myo):
            fibre_rot.x.array[inc_myo] = np.full_like(inc_myo, ROT, dtype=default_scalar_type)
            fibre_val.x.array[inc_myo] = np.full_like(inc_myo, MATERIAL_CONSTANTS[1], dtype=default_scalar_type)
        if len(dec_myo):
            fibre_rot.x.array[dec_myo] = np.full_like(dec_myo, -ROT, dtype=default_scalar_type)
            fibre_val.x.array[dec_myo] = np.full_like(dec_myo, MATERIAL_CONSTANTS[1], dtype=default_scalar_type)
        if len(cytosol):
            fibre_rot.x.array[cytosol] = np.full_like(cytosol, NULL, dtype=default_scalar_type)
            fibre_val.x.array[cytosol] = np.full_like(cytosol, 0.5, dtype=default_scalar_type)
        return fibre_rot, fibre_val
    
    # += Define rotation and material value
    fibre_rot, fibre_val = subdomain_assignments(domain, ct, V_1DG)

    # +==+ Variational Problem Setup
    # += Test and Trial Parameters
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    # += Coordinate values
    x = Function(V)
    x.interpolate(lambda x: (x[0], x[1]))
    # += Tensor Indices
    i, j, k, a, b = ufl.indices(5)
    # += Curvilinear mapping dependent on subdomain values
    Push = ufl.as_matrix([
        [ufl.cos(fibre_rot), -ufl.sin(fibre_rot)],
        [ufl.sin(fibre_rot), ufl.cos(fibre_rot)]
    ])
    # += Subdomain dependent rotations of displacement and coordinates
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)
    # += Metric tensors
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
    # += Kinematics variables
    I = ufl.variable(ufl.Identity(MESH_DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))
    # += Material Setup | Guccione
    Q = (
        fibre_val * E[0,0]**2 + 
        MATERIAL_CONSTANTS[2] * (E[1,1]**2) + 
        MATERIAL_CONSTANTS[3] * (2*E[0,1]*E[1,0])
    )
    piola = MATERIAL_CONSTANTS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*fibre_val*E[0,0], 2*MATERIAL_CONSTANTS[3]*(E[1,0] + E[0,1])],
        [2*MATERIAL_CONSTANTS[3]*(E[0,1] + E[1,0]), 4*MATERIAL_CONSTANTS[2]*E[1,1]],
    ]) - p * Z_un
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])
    
    # +==+ Solver Setup
    # += Measure assignment
    metadata = {"quadrature_degree": quad_order}
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata=metadata)
    # += Residual equation
    R = term * dx + q * (J - 1) * dx

    # +==+ Setup stress output
    def cauchy_tensor(u, p, x, r):
        # += Tensor Indices
        i, j = ufl.indices(2)
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
        # += Kinematics
        I = ufl.variable(ufl.Identity(MESH_DIM))
        F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
        E = ufl.variable(ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j]))
        J = ufl.variable(ufl.det(F))
        # += Material Setup | Guccione
        Q = (
            fibre_val * E[0,0]**2 + 
            MATERIAL_CONSTANTS[2] * (E[1,1]**2) + 
            MATERIAL_CONSTANTS[3] * (2*E[0,1]*E[1,0])
        )
        piola = MATERIAL_CONSTANTS[0]/4 * ufl.exp(Q) * ufl.as_matrix([
            [4*fibre_val*E[0,0], 2*MATERIAL_CONSTANTS[3]*(E[1,0] + E[0,1])],
            [2*MATERIAL_CONSTANTS[3]*(E[0,1] + E[1,0]), 4*MATERIAL_CONSTANTS[2]*E[1,1]],
        ]) - p * Z_un
        sig = 1/J * F*piola*F.T
        return sig
    
    # +==+ Setup strain output
    def green_tensor(u, p, x, r):
        # += Tensor Indices
        i, j = ufl.indices(2)
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
        # += Kinematics
        I = ufl.variable(ufl.Identity(MESH_DIM))
        F = ufl.as_tensor(I[i, j] + ufl.grad(u)[i, j], [i, j]) * Push
        E = ufl.as_tensor((0.5*(Z_co[i,j] - Z_un[i,j])), [i, j])
        return ufl.as_tensor([[E[0, 0], E[0, 1]], [E[0, 1], E[1, 1]]])

    # +==+ Setup Iteration Loop variables
    # += Preset displacements per iteration
    phase_ext = np.linspace(DISPLACEMENT/ITS, DISPLACEMENT, ITS)
    phase_con = np.linspace(DISPLACEMENT, -DISPLACEMENT, ITS*2)
    phase_exp = np.linspace(-DISPLACEMENT, 0, ITS)
    bc_u_iter = np.concatenate([phase_ext, phase_con, phase_exp])
    bc_u_iter = [DISPLACEMENT]
    # += Export data
    Vu_sol, _ = W.sub(0).collapse() 
    disp = Function(Vu_sol) 
    disp.name = "U - Displacement"
    TS = FunctionSpace(mesh=domain, element=("CG", elem_order, (2,2)))
    sig = Function(TS)
    sig.name = "S - Cauchy Stress"
    eps = Function(TS)
    eps.name = "E - Green Strain"
    # test_name = "TESTING_ITERATION"
    # += Setup file writers
    # dis_file = io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ID + "_DISP.bp", disp, engine="BP4")
    # eps_file = io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ID + "_EPS.bp", eps, engine="BP4")
    # sig_file = io.VTXWriter(MPI.COMM_WORLD, "P_Branch_Contraction/paraview_bp/" + test_name + ID + "_SIG.bp", sig, engine="BP4")

    # += Bounding box for cell searching
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    # +==+ Loop deformations
    for i, d in enumerate(bc_u_iter):
        # += Attain BC
        print("     += Iteration: {}".format(i))
        bc, _ = boundary_conditions(domain, W, Vx, Vy, d)
        
        # += Nonlinear Solver
        problem = NonlinearProblem(R, w, bc)
        solver = NewtonSolver(domain.comm, problem)
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"

        # += Solve
        num_its, _ = solver.solve(w)
        print(f"Converged in {num_its} iterations.")

        # += Store displacement results
        u_eval = w.sub(0).collapse()
        disp.interpolate(u_eval)
        # += Setup tensor space for stress tensor interpolation
        cauchy = Expression(
            e=cauchy_tensor(u_eval, w.sub(1).collapse(), x, fibre_rot), 
            X=TS.element.interpolation_points()
        )
        epsilon = Expression(
            e=green_tensor(u_eval, None, x, fibre_rot), 
            X=TS.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        eps.interpolate(epsilon)

        TEST_POINTS = (7, 5)
        x_locs = np.round([(x+1) * 0.125 for x in range(0, TEST_POINTS[0], 1)], 3)
        y_locs = np.round([x * 0.2 + 0.1 for x in range(0, TEST_POINTS[1], 1)], 3)
        x_u_data = np.zeros(TEST_POINTS)
        y_u_data = np.zeros(TEST_POINTS)
        xy_e_data = np.zeros(TEST_POINTS)
        xy_s_data = np.zeros(TEST_POINTS)

        for m, ylo in enumerate(y_locs):
            for n, xlo in enumerate(x_locs):
                # x0 = locate_entities_boundary(mesh=domain, dim=MESH_DIM-1, marker=lambda x: np.isclose(x[0], xlo, rtol=0.1, atol=0.1))
                # y0 = locate_entities_boundary(mesh=domain, dim=MESH_DIM-1, marker=lambda x: np.isclose(x[1], ylo))
                cells = []
                points_on_proc = []
                cell_candidates = geometry.compute_collisions_points(bb_tree, [xlo, ylo, 0])
                colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates,[xlo, ylo, 0])
                disp_xy = disp.eval([xlo, ylo, 0], colliding_cells[0])
                x_u_data[n, m] = disp_xy[0]
                y_u_data[n, m] = disp_xy[1]
                sig_xy = sig.eval([xlo, ylo, 0], colliding_cells[0])
                xy_s_data[n, m] = sig_xy[1]
                eps_xy = eps.eval([xlo, ylo, 0], colliding_cells[0])
                xy_e_data[n, m] = eps_xy[1]

        np.save("P_Branch_Contraction/numpy_data/disp_x_" + test_name + "_" + ID + ".npy", x_u_data)
        np.save("P_Branch_Contraction/numpy_data/disp_y_" + test_name + "_" + ID + ".npy", y_u_data)
        np.save("P_Branch_Contraction/numpy_data/sig_xy_" + test_name + "_" + ID + ".npy", xy_s_data)
        np.save("P_Branch_Contraction/numpy_data/eps_xy_" + test_name + "_" + ID + ".npy", xy_e_data)

        # += Write files 
        # for j in range(1, 10, 1): dis_file.write(i*10 + j)
        # for j in range(1, 10, 1): eps_file.write(i*10 + j)
        # for j in range(1, 10, 1): sig_file.write(i*10 + j)

    # += Close files
    # dis_file.close()
    # eps_file.close()
    # sig_file.close()

    return None

# +==+==+==+
# gmsh_cube:
#   Input of test_name and required graph network.
#   Output cube with refined generations.
def gmsh_cube(tnm, msh, depth):
    depth += 1
    print("\t" * depth + "+= Generate Mesh: {}.msh".format(tnm))

    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(tnm)

    # += Check for auto generation
    if msh:
        # += Create cube
        box = gmsh.model.occ.addBox(
            x=0, y=0, z=0, 
            dx=CUBE["x"]*PXLS["x"], dy=CUBE["y"]*PXLS["y"], dz=CUBE["z"]*PXLS["z"], 
            tag=int(EL_TAGS[5])
        )
        EL_TAGS[5] += 1
        gmsh.model.occ.synchronize()
        # += Generate physical groups and label contraction sites
        for i in range(0, 4, 1):
            for _, j in gmsh.model.occ.get_entities(dim=i):
                (x, y, z) = gmsh.model.occ.get_center_of_mass(dim=i, tag=j)
                if np.all(np.isclose([x, y, z], [0, CUBE["y"]*PXLS["y"]/2, CUBE["z"]*PXLS["z"]/2])):
                    gmsh.model.add_physical_group(dim=i, tags=[j], tag=int(PY_TAGS["x0"]), name="X_MIN")
                elif np.all(np.isclose([x, y, z], [CUBE["x"]*PXLS["x"], CUBE["y"]*PXLS["y"]/2, CUBE["z"]*PXLS["z"]/2])):
                    gmsh.model.add_physical_group(dim=i, tags=[j], tag=int(PY_TAGS["x1"]), name="X_MAX")
                else:
                    gmsh.model.add_physical_group(dim=i, tags=[j], tag=int(PY_TAGS[i]))
                PY_TAGS[i] += 1
        gmsh.model.occ.synchronize()

    # += Generate Mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(order=ORDER)

    # += Write File
    file = os.path.dirname(os.path.abspath(__file__)) + "/_msh/" + tnm + ".msh"
    gmsh.write(file)
    gmsh.finalize()
    return file

# +==+==+==+
# main
#   Inputs: 
#       tnm  | str | test name
#       msh  | bool | indicator of mesh generation style
#   Outputs:
#       .bp folder of deformation
def main(tnm, msh, depth):
    depth += 1
    # += Mesh generation 
    file = gmsh_cube(tnm, msh, depth)
    # += Enter FEniCSx
    fx_(tnm, file, depth)

# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    # +==+ Argument Parser
    # += Create parser for taking in input values
    # parser = argparse.ArgumentParser(description="3D FEniCSx Passive Contractio Model")
    # parser.add_argument('test_name', type=str, help="Test name")
    # parser.add_argument('auto_mesh', type=bool, help="True/False for Mesh auto generation")
    # args = parser.parse_args()
    # tnm = args.test_name
    # msh = args.auto_mesh
    tnm = "TEST"
    msh = True
    # += Run
    print("\t" * depth + "!! BEGIN TEST: " + tnm + " !!")
    main(tnm, msh, depth)
    print("\t" * depth + "!! END TEST: " + tnm + " !!")