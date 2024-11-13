"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: fx_VolContraction.py
        Contraction over volume from EM data informed anisotropy
"""

# +==+==+==+
# Setup
# += Imports
from dolfinx import io,  default_scalar_type
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological, Constant, Expression
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import operator as op
import pandas as pd
import numpy as np
import argparse
import basix
import gmsh
import ufl
import csv
import os

# += Parameters
DIM = 3
INC = 10
I_0 = 0
F_0 = 0.0 
ZDISC = 5 
ORDER = 2
QUADRATURE = 4
H_ROT = 0.91755
X, Y, Z = 0, 1, 2
# CONSTIT_CYT = [0.5]
CONSTIT_CYT = [1]
# CONSTIT_MYO = [1, 1, 0.5, 0.5]
CONSTIT_MYO = [1, 1, 1, 1]
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1024, "y": 1024, "z": 80}
SURF_NAMES = ["x0", "x1", "y0", "y1", "z0", "z1"]
SCREW_AXIS = {0: ["x0-", 161], 1: ["-x-", 162], 2: ["-x1", 163]}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
SUR_OBJ_TAGS = dict(zip(SURF_NAMES, list(range(1, 7, 1))))
EL_TAGS = {0: 1e0, 1: 1e2, 2: 1e3, 3: 5e3, 4: 1e4, 5: 2e4}
PY_TAGS = {0: 1e2, 1: 1e3, 2: 5e3, 3: 1e4, 4: 2e4, 5: 3e4}
LOCS_COM = [
    (F_0, EDGE[1]/2, EDGE[2]/2), (EDGE[0], EDGE[1]/2, EDGE[2]/2), 
    (EDGE[0]/2, F_0, EDGE[2]/2), (EDGE[0]/2, EDGE[1], EDGE[2]/2), 
    (EDGE[0]/2, EDGE[1]/2, F_0), (EDGE[0]/2, EDGE[1]/2, EDGE[2]) 
]
SUR_OBJ_ASSIGN = dict(zip(LOCS_COM, [[n+1, name] for n, name in enumerate(SURF_NAMES)]))
EMF_PATH = "/Users/murrayla/Documents/main_PhD/P_Segmentations/myofibril_segmentation/Segmented_Data/"

# +==+==+==+
# dir_bc
#   Inputs: 
#       mix_vs  | obj | mixed vector space
#       Vx, Vy, Vz | obj | collapsed vector spaces
#       ft | np_array | facet tag data
#       du | float | displacement value
#   Outputs:
#       numpy array of boudnary condition assignment data
def dir_bc(mix_vs, Vx, Vy, Vz, ft, du):
    # += Locate subdomain dofs
    # xx0_dofs, xx1_dofs, yx0_dofs, yx1_dofs, zx0_dofs, zx1_dofs = (
    #     locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
    #     locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
    #     locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
    #     locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
    #     locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
    #     locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
    # )
    # += Locate subdomain dofs
    xx0_dofs, xx1_dofs = (
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
    )
    # += Interpolate 
    # uxx0, uxx1, uyx0, uyx1, uzx0, uzx1 = (
    #     Function(Vx), Function(Vx), Function(Vy), Function(Vy), Function(Vz), Function(Vz)
    # )
    uxx0, uxx1 = Function(Vx), Function(Vx)
    # uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du)))
    # uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # uyx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # uyx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # uzx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # uzx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du)))
    uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # += Dirichlet Boundary Conditions
    # bc_UxX0 = dirichletbc(value=uxx0, dofs=xx0_dofs, V=mix_vs.sub(0).sub(X))
    # bc_UxX1 = dirichletbc(value=uxx1, dofs=xx1_dofs, V=mix_vs.sub(0).sub(X))
    # bc_UyX0 = dirichletbc(value=uyx0, dofs=yx0_dofs, V=mix_vs.sub(0).sub(Y))
    # bc_UyX1 = dirichletbc(value=uyx1, dofs=yx1_dofs, V=mix_vs.sub(0).sub(Y))
    # bc_UzX0 = dirichletbc(value=uzx0, dofs=zx0_dofs, V=mix_vs.sub(0).sub(Z))
    # bc_UzX1 = dirichletbc(value=uzx1, dofs=zx1_dofs, V=mix_vs.sub(0).sub(Z))
    bc_UxX0 = dirichletbc(value=uxx0, dofs=xx0_dofs, V=mix_vs.sub(0).sub(X))
    bc_UxX1 = dirichletbc(value=uxx1, dofs=xx1_dofs, V=mix_vs.sub(0).sub(X))
    # += Assign
    # bc = [bc_UxX0, bc_UxX1, bc_UyX0, bc_UyX1, bc_UzX0, bc_UzX1]
    bc = [bc_UxX0, bc_UxX1]

    return bc

# +==+==+==+
# prop_csv:
#   Inputs: 
#       tnm  | str | test name
#   Outputs:
#       dataframe with orientation data
def prop_csv(tnm, depth):
    depth += 1
    # += Determine file location
    print("\t" * depth + "+= Load z-disc property data...")
    file_path = "/Users/murrayla/Documents/main_PhD/P_BranchingPaper/A_Scripts/Vec_files/"
    files = []
    cdt = tnm.split("_")[0]
    nmb = tnm.split("_")[1] + "_"
    # += Splice fields
    if cdt == "raw":
        file_path += "health/"
    else:
        file_path += "infarct/"
    # += Read appropriate file into dataframe
    for file in os.listdir(file_path):
        if ("_props" in file) and (nmb in file):
            if "_t_" in file: 
                ele_df = pd.read_csv(file_path + file)
            else:
                azi_df = pd.read_csv(file_path + file)
    # += Create new dataframe
    ang_df = pd.concat(
        [azi_df.iloc[:, [1, 4, 7]], ele_df.iloc[:, 2]], axis=1
    )
    return ang_df

# +==+==+==+
# fx_
#   Inputs: 
#       tnm  | str | test name
#       file | str | file name
#       tg_c | dict | cytosol physical element data
#       tg_S | dict | sarcomere physical element data
#   Outputs:
#       .bp folder of deformation
def fx_(tnm, file, tg_c, tg_s, depth):
    depth += 1

    # +==+ Domain Setup
    print("\t" * depth + "+= Generate Mesh")
    # += Load mesh data
    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    print("\t" * depth + "+= Setup Vector Space ")
    # += Second and First order Continuous
    V2 = ufl.VectorElement(family="Lagrange", cell=domain.ufl_cell(), degree=ORDER, quad_scheme="default")
    V1 = ufl.FiniteElement(family="Lagrange", cell=domain.ufl_cell(), degree=ORDER-1, quad_scheme="default")
    # += Quadrature Element
    Q = ufl.FiniteElement(family="Quadrature", cell=domain.ufl_cell(), degree=ORDER, quad_scheme="default")
    # += Mixed Space
    Mxs = FunctionSpace(mesh=domain, element=ufl.MixedElement([V2, V1]))
    # += Vector Spaces and Tensor Space
    Tes = FunctionSpace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))
    Dcs = FunctionSpace(mesh=domain, element=("Discontinuous Lagrange", 0))

    # +==+ Extract subdomains for dofs
    V, _ = Mxs.sub(0).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()
    Vz, dofsZ = V.sub(Z).collapse()
    
    # +==+ Boundary Conditions
    print("\t" * depth + "+= Assign Boundary Conditions")
    bc = dir_bc(Mxs, Vx, Vy, Vz, ft, EDGE[0] * int(tnm.split("_")[3])/100)
    
    # +==+ Create rotation and material property arrays
    print("\t" * depth + "+= Assign Fibre Directions")
    # += 1-DF Spaces
    fbr_azi, fbr_elv, fbr_con = Function(Dcs), Function(Dcs), Function(Dcs)
    cyt_tgs = ct.find(tg_c[DIM][0])
    # += Cytosol [DEFAULT]
    # fbr_azi.x.array[:] = np.full_like(fbr_azi.x.array[:], I_0, dtype=default_scalar_type)
    # fbr_con.x.array[:] = np.full_like(fbr_azi.x.array[:], CONSTIT_CYT[0], dtype=default_scalar_type)
    fbr_azi.x.array[cyt_tgs] = np.full_like(cyt_tgs, I_0, dtype=default_scalar_type)
    fbr_elv.x.array[cyt_tgs] = np.full_like(cyt_tgs, I_0, dtype=default_scalar_type)
    fbr_con.x.array[cyt_tgs] = np.full_like(cyt_tgs, CONSTIT_CYT[0], dtype=default_scalar_type)
    # += For each other orientation assign arrays
    # for i in range(0, len(tg_s[DIM]["tag"]), 1):
    #     myo_tgs = ct.find(tg_s[DIM]["tag"][i])
    #     fbr_azi.x.array[myo_tgs] = np.full_like(myo_tgs, tg_s[DIM]["angles"][i][0], dtype=default_scalar_type)
    #     fbr_elv.x.array[myo_tgs] = np.full_like(myo_tgs, tg_s[DIM]["angles"][i][1], dtype=default_scalar_type)
    #     fbr_con.x.array[myo_tgs] = np.full_like(myo_tgs, CONSTIT_MYO[1], dtype=default_scalar_type)

    # +==+ Variables
    v, q = ufl.TestFunctions(Mxs)
    # += [CURRENT]
    mx = Function(Mxs)
    u, p = ufl.split(mx)
    
    # += Initial
    # +==+ Curvilinear setup
    x = ufl.SpatialCoordinate(domain)
    x_n = Function(V)
    # k = x_n.function_space.tabulate_dof_coordinates()
    # x_n_vals = x_n.interpolate(lambda x: x)
    # print(x_n_vals)
    azi, ele = Function(Vx), Function(Vy)

    def wei_azi(pos, df):
        mass_ang, mass = 0, 0
        for _, row in df.iterrows():
            # += Get position and angle
            cur = np.fromstring(row["Centroid"].strip('()'), sep=',')
            ang = row["Orientation [RAD]"]
            # += Euclidean distance
            dis = np.linalg.norm(pos - cur)
            # += Weight
            wei = 1 / (dis + 1e-10)
            mass_ang += ang * wei
            mass += wei
        # += Compute weighted value
        wei_ang = mass_ang / mass if mass != 0 else 0
        return wei_ang - H_ROT
    
    def wei_ele(pos, df):
        mass_ang, mass = 0, 0
        for _, row in df.iterrows():
            # += Get position and angle
            cur = np.fromstring(row["Centroid"].strip('()'), sep=',')
            ang = row["Elevation Orientation [RAD]"]
            # += Euclidean distance
            dis = np.linalg.norm(pos - cur)
            # += Weight
            wei = 1 / (dis + 1e-10)
            mass_ang += ang * wei
            mass += wei
        # += Compute weighted value
        wei_ang = mass_ang / mass if mass != 0 else 0
        return wei_ang

    ang_df = prop_csv(tnm, depth)
    for i in range(len(azi.x.array[:])):
        if tnm.split("_")[1] == "test":
            azi.x.array[i] = F_0
            ele.x.array[i] = F_0
            continue
        else:
            pos = np.array(x_n.function_space.tabulate_dof_coordinates()[i])
            azi.x.array[i] = wei_azi(pos, ang_df)
            ele.x.array[i] = wei_ele(pos, ang_df)

    # azi.x.array[:] = np.full_like(azi.x.array[:], I_0, dtype=default_scalar_type)
    # ele.x.array[:] = np.full_like(ele.x.array[:], I_0, dtype=default_scalar_type)

    # x = Function(Sos)
    # Push = ufl.as_matrix([
    #     [ufl.cos(fbr_azi), -ufl.sin(fbr_azi), 0],
    #     [ufl.sin(fbr_azi), ufl.cos(fbr_azi), 0],
    #     [0, 0, 1]
    # ])
    # x.interpolate(lambda x: (x[0], x[1], x[2]))
    i, j, k, a, b = ufl.indices(5)
    # # += Curvilinear mapping dependent on subdomain values
    Push = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi), ufl.cos(azi), 0],
        [0, 0, 1]
    ])
    # Push = ufl.as_matrix([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])
    Push = Push * ufl.as_matrix([
        [1, 0, 0],
        [0, ufl.cos(ele), -ufl.sin(ele)],
        [0, ufl.sin(ele), ufl.cos(ele)]
    ])

    # += Subdomain dependent rotations of displacement and coordinates
    x_nu = ufl.inv(Push) * x
    u_nu = ufl.inv(Push) * u
    nu = ufl.inv(Push) * (x + u_nu)

    # += Metric tensors
    Z_co = ufl.grad(x_nu).T * ufl.grad(x_nu)
    Z_ct = ufl.inv(Z_co)
    z_co = ufl.grad(nu).T * ufl.grad(nu)
    z_ct = ufl.inv(z_co)

    # += Christoffel Symbol | Γ^{i}_{j, a}
    gamma = ufl.as_tensor((
        0.5 * Z_ct[k, a] * (
            ufl.grad(Z_co)[a, i, j] + ufl.grad(Z_co)[a, j, i] - ufl.grad(Z_co)[i, j, a]
        )
    ), [k, i, j])
    # += Covariant Derivative
    covDev = ufl.grad(v) - ufl.as_tensor(v[k]*gamma[k, i, j], [i, j])

    # += Kinematics variables
    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.as_tensor(I[i, j] + ufl.grad(u_nu)[i, j], [i, j]) * Push
    C = ufl.variable(ufl.as_tensor(F[k, i]*F[k, j], [i, j]))
    E = ufl.variable(ufl.as_tensor((0.5*(z_co[i,j] - Z_co[i,j])), [i, j]))
    J = ufl.variable(ufl.det(F))

    # += Material Setup | Guccione
    Q = (
        fbr_con * E[0,0]**2 + 
        CONSTIT_MYO[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
        CONSTIT_MYO[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
    )
    piola = CONSTIT_MYO[0]/2 * ufl.exp(Q) * ufl.as_matrix([
        [4*fbr_con*E[0,0], 2*CONSTIT_MYO[3]*(E[1,0] + E[0,1]), 2*CONSTIT_MYO[3]*(E[2,0] + E[0,2])],
        [2*CONSTIT_MYO[3]*(E[0,1] + E[1,0]), 4*CONSTIT_MYO[2]*E[1,1], 2*CONSTIT_MYO[2]*(E[2,1] + E[1,2])],
        [2*CONSTIT_MYO[3]*(E[0,2] + E[2,0]), 2*CONSTIT_MYO[2]*(E[1,2] + E[2,1]), 4*CONSTIT_MYO[3]*E[2,2]],
    ]) - p * Z_co
    term = ufl.as_tensor(piola[a, b] * F[j, b] * covDev[j, a])

    # +==+ Solver Setup
    # += Measure assignment
    metadata = {"quadrature_degree": QUADRATURE}
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata=metadata)
    # += Residual equation
    R = term * dx + q * (J - 1) * dx

    Vu_sol, up_to_u_sol = Mxs.sub(0).collapse() 
    u_sol = Function(Vu_sol) 

    Vp_sol, up_to_p_sol = Mxs.sub(1).collapse() 
    p_sol = Function(Vp_sol) 

    u_sol.name = "disp"
    p_sol.name = "pressure"

    file = os.path.dirname(os.path.abspath(__file__)) + "/_bp/" + tnm + ".bp"
    eps_file = io.VTXWriter(MPI.COMM_WORLD, file, u_sol, engine="BP4")
    
    # += Nonlinear Solver
    problem = NonlinearProblem(R, mx, bc)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    # +==+==+
    # Solution and Output
    # += Solve
    num_its, converged = solver.solve(mx)
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged after {num_its} iterations.")

    u_eval = mx.sub(0).collapse()
    u_sol.interpolate(u_eval)

    eps_file.write(0)

    eps_file.close()

    return None

# +==+==+==+
# msh_:
#   Inputs: 
#       tnm  | str | test name
#       msh  | bool | automation check
#   Outputs:
#       .msh file of mesh
#       tg_c | dict | cytosol physical element data
#       tg_S | dict | sarcomere physical element data
def msh_(tnm, msh, depth):
    depth += 1
    print("\t" * depth + "+= Generate Mesh: {}.msh".format(tnm))

    # += Tag Storage
    TG_S = {DIM-1: [], DIM: {"tag": [], "angles": []}}
    TG_C = {DIM-1: [], DIM: []}

    # +==+ Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(tnm)

    # +==+ Create cube
    box = gmsh.model.occ.addBox(
        x=0, y=0, z=0, 
        dx=CUBE["x"]*PXLS["x"], dy=CUBE["y"]*PXLS["y"], dz=CUBE["z"]*PXLS["z"], 
        tag=int(EL_TAGS[5])
    )
    EL_TAGS[5] += 1
    gmsh.model.occ.synchronize()

    # += Add points of interest
    pts = [[*LOCS_COM[0]], [EDGE[0]/2, LOCS_COM[0][1], LOCS_COM[0][2]], [*LOCS_COM[1]]]
    for i, (x, y, z) in enumerate(pts):
        pt = gmsh.model.occ.add_point(x=x, y=y, z=z, tag=int(PY_TAGS[0]))
        PY_TAGS[0] += 1
        gmsh.model.add_physical_group(dim=1, tags=[pt], tag=SCREW_AXIS[i][1], name=SCREW_AXIS[i][0])
        gmsh.model.occ.synchronize()

    # +==+ Generate physical groups
    for i in range(0, DIM+1, 1):
        # += Generate mass, com and tag data
        _, tgs = zip(*gmsh.model.occ.get_entities(dim=i))
        data = {
            tgs[x]: [
                gmsh.model.occ.get_mass(dim=i, tag=tgs[x]),
                gmsh.model.occ.get_center_of_mass(dim=i, tag=tgs[x])
            ] for x in range(0, len(tgs), 1)
        }
        # += Dataframe for iteration
        df = pd.DataFrame(data).transpose().sort_values(by=[0], ascending=False)
        # += Generate physical groups 
        for n, (j, row) in enumerate(df.iterrows()):
            if i == 1:
                print(df)
                print(row[1])
            # += If of Dimension 2 find the Z-Discs from centroid data
            if i == DIM - 1:
                gmsh.model.add_physical_group(
                    dim=i, tags=[j], tag=SUR_OBJ_ASSIGN[row[1]][0], name=SUR_OBJ_ASSIGN[row[1]][1]
                )
                TG_C[DIM-1].append(SUR_OBJ_ASSIGN[row[1]][0])
            # += For Dimension 3, determine if sarcomere or cytosol
            if i == DIM:
                gmsh.model.add_physical_group(dim=i, tags=[j], tag=int(PY_TAGS[i]), name="Cytosol")
                TG_C[DIM].append(int(PY_TAGS[i]))
            # += Any other region can be arbitrarily labeled 
            if i == DIM-2: 
                com = [*row[i]]
                if [*row[i]] in pts:
                    l = np.where(np.fromstring(row[i].strip('()'), sep=',') in pts)
                    gmsh.model.add_physical_group(dim=1, tags=[pt], tag=SCREW_AXIS[l][1], name=SCREW_AXIS[l][0])
                    continue
                gmsh.model.add_physical_group(dim=i, tags=[j], tag=int(PY_TAGS[i]))
            PY_TAGS[i] += 1
        gmsh.model.occ.synchronize()

    # +==+ Generate Mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.setOrder(order=ORDER) 

    # +==+ Write File
    file = os.path.dirname(os.path.abspath(__file__)) + "/_msh/" + "EMSimCube_ScrewAxis" + ".msh"
    gmsh.write(file)
    gmsh.finalize()
    return file, TG_S, TG_C

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
    if msh:
        file, tg_s, tg_c = msh_(tnm, msh, depth)
    else:
        tg_s = {2: [], 3: {'tag': [], 'angles': []}}
        tg_c = {2: [5, 6, 1, 2, 3, 4], 3: [10000]}
        file = os.path.dirname(os.path.abspath(__file__)) + "/_msh/" + "EMSimCube" + ".msh"
    # += Enter FEniCSx
    fx_(tnm, file, tg_c, tg_s, depth)

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
    msh = True
    emf = "raw_0"
    ceq = "GC"
    dsp = "10"
    etp = "dir_ani"
    tnm = "_".join([emf, ceq, dsp, etp])
    # += Run
    print("\t" * depth + "!! BEGIN TEST: " + tnm + " !!")
    main(tnm, msh, depth)
    print("\t" * depth + "!! END TEST: " + tnm + " !!")