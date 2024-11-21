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
from dolfinx import log, io,  default_scalar_type 
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological, Constant, Expression, DirichletBC
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from scipy.spatial import cKDTree
from mpi4py import MPI
import operator as op
import pandas as pd
import numpy as np
import argparse
import basix
import gmsh
import math
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
E_ROT = -1.08994
X, Y, Z = 0, 1, 2
# CONSTIT_CYT = [0.5]
# CONSTIT_CYT = [1]
CONSTIT_MYO = [1, 1, 1, 1]
# CONSTIT_MYO = [1, 1, 1, 1]
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
def dir_bc(mix_vs, Vx, Vy, Vz, ft, uni, du):
    # += Locate subdomain dofs
    xx0_dofs, xx1_dofs, yx0_dofs, yx1_dofs, zx0_dofs, zx1_dofs = (
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
    )
    # += Interpolate 
    uxx0, uxx1, uyx0, uyx1, uzx0, uzx1 = Function(Vx), Function(Vx), Function(Vy), Function(Vy), Function(Vz), Function(Vz)
    if uni:
        uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du)))
        uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    else:
        uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du//2)))
        uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-du//2)))
    uyx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uyx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uzx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uzx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # += Dirichlet Boundary Conditions
    bc_UxX0 = dirichletbc(value=uxx0, dofs=xx0_dofs, V=mix_vs.sub(0).sub(X))
    bc_UxX1 = dirichletbc(value=uxx1, dofs=xx1_dofs, V=mix_vs.sub(0).sub(X))
    bc_UyX0 = dirichletbc(value=uyx0, dofs=yx0_dofs, V=mix_vs.sub(0).sub(Y))
    bc_UyX1 = dirichletbc(value=uyx1, dofs=yx1_dofs, V=mix_vs.sub(0).sub(Y))
    bc_UzX0 = dirichletbc(value=uzx0, dofs=zx0_dofs, V=mix_vs.sub(0).sub(Z))
    bc_UzX1 = dirichletbc(value=uzx1, dofs=zx1_dofs, V=mix_vs.sub(0).sub(Z))
    # += Assign
    bc = [bc_UxX0, bc_UxX1, bc_UyX0, bc_UyX1, bc_UzX0, bc_UzX1]
    # bc = [bc_UxX1 ,bc_UyX0, bc_UyX1, bc_UzX0, bc_UzX1]
    return bc

def dir_corner_prescribe_bc(mix_vs, Vx, Vy, Vz, ft, uni, du):
    # += Locate subdomain dofs
    xx0_dofs, xx1_dofs = (
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"])),
    )
    yx0y0z0_dofs, yx1y0z0_dofs, yx0y1z0_dofs, yx1y1z0_dofs, yx0y0z1_dofs, yx1y0z1_dofs, yx0y1z1_dofs, yx1y1z1_dofs = (
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(10)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(11)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(12)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(13)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(14)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(15)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(16)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=0, entities=ft.find(17)),
    )
    zx0y0z0_dofs, zx1y0z0_dofs, zx0y1z0_dofs, zx1y1z0_dofs, zx0y0z1_dofs, zx1y0z1_dofs, zx0y1z1_dofs, zx1y1z1_dofs = (
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(10)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(11)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(12)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(13)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(14)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(15)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(16)),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=0, entities=ft.find(17)),
    )
    # += Interpolate 
    uxx0, uxx1 = Function(Vx), Function(Vx)
    uy, uz =  Function(Vy), Function(Vz)
    if uni:
        uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du)))
        uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    else:
        uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du//2)))
        uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-du//2)))
    uy.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uz.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    # += Dirichlet Boundary Conditions
    bc_UxX0 = dirichletbc(value=uxx0, dofs=xx0_dofs, V=mix_vs.sub(0).sub(X))
    bc_UxX1 = dirichletbc(value=uxx1, dofs=xx1_dofs, V=mix_vs.sub(0).sub(X))
    
    (
        bc_yx0y0z0_dofs, bc_yx1y0z0_dofs, bc_yx0y1z0_dofs, bc_yx1y1z0_dofs, 
        bc_yx0y0z1_dofs, bc_yx1y0z1_dofs, bc_yx0y1z1_dofs, bc_yx1y1z1_dofs
    ) = (
        dirichletbc(value=uy, dofs=yx0y0z0_dofs, V=mix_vs.sub(0).sub(Y)),
        dirichletbc(value=uy, dofs=yx1y0z0_dofs, V=mix_vs.sub(0).sub(Y)),
        dirichletbc(value=uy, dofs=yx0y1z0_dofs, V=mix_vs.sub(0).sub(Y)),
        dirichletbc(value=uy, dofs=yx1y1z0_dofs, V=mix_vs.sub(0).sub(Y)),
        dirichletbc(value=uy, dofs=yx0y0z1_dofs, V=mix_vs.sub(0).sub(Y)),
        dirichletbc(value=uy, dofs=yx1y0z1_dofs, V=mix_vs.sub(0).sub(Y)),
        dirichletbc(value=uy, dofs=yx0y1z1_dofs, V=mix_vs.sub(0).sub(Y)),
        dirichletbc(value=uy, dofs=yx1y1z1_dofs, V=mix_vs.sub(0).sub(Y))
    )
    (
        bc_zx0y0z0_dofs, bc_zx1y0z0_dofs, bc_zx0y1z0_dofs, bc_zx1y1z0_dofs, 
        bc_zx0y0z1_dofs, bc_zx1y0z1_dofs, bc_zx0y1z1_dofs, bc_zx1y1z1_dofs
    ) = (
        dirichletbc(value=uz, dofs=zx0y0z0_dofs, V=mix_vs.sub(0).sub(Z)),
        dirichletbc(value=uz, dofs=zx1y0z0_dofs, V=mix_vs.sub(0).sub(Z)),
        dirichletbc(value=uz, dofs=zx0y1z0_dofs, V=mix_vs.sub(0).sub(Z)),
        dirichletbc(value=uz, dofs=zx1y1z0_dofs, V=mix_vs.sub(0).sub(Z)),
        dirichletbc(value=uz, dofs=zx0y0z1_dofs, V=mix_vs.sub(0).sub(Z)),
        dirichletbc(value=uz, dofs=zx1y0z1_dofs, V=mix_vs.sub(0).sub(Z)),
        dirichletbc(value=uz, dofs=zx0y1z1_dofs, V=mix_vs.sub(0).sub(Z)),
        dirichletbc(value=uz, dofs=zx1y1z1_dofs, V=mix_vs.sub(0).sub(Z))
    )
    # += Assign
    bc = [
        bc_UxX0, bc_UxX1,
        bc_yx0y0z0_dofs, bc_yx1y0z0_dofs, bc_yx0y1z0_dofs, bc_yx1y1z0_dofs, 
        bc_yx0y0z1_dofs, bc_yx1y0z1_dofs, bc_yx0y1z1_dofs, bc_yx1y1z1_dofs,
        bc_zx0y0z0_dofs, bc_zx1y0z0_dofs, bc_zx0y1z0_dofs, bc_zx1y1z0_dofs, 
        bc_zx0y0z1_dofs, bc_zx1y0z1_dofs, bc_zx0y1z1_dofs, bc_zx1y1z1_dofs]
    # bc = [bc_UxX0, bc_UxX1]
    return bc

def dir_corner_bc(domain, mix_vs, Vx, Vy, Vz, ft, uni, du):
    # += Locate subdomain dofs
    xx0_dofs, xx1_dofs = (
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x0"])),
        locate_dofs_topological(V=(mix_vs.sub(0).sub(X), Vx), entity_dim=ft.dim, entities=ft.find(SUR_OBJ_TAGS["x1"]))
    )
    def corner(x):
        return (
            np.isclose(x[0], min(x[0])) & 
            (
                np.isclose(x[1], min(x[1])) & np.isclose(x[2], min(x[2])) | 
                np.isclose(x[1], max(x[1])) & np.isclose(x[2], min(x[2])) |
                np.isclose(x[1], min(x[1])) & np.isclose(x[2], max(x[2])) |
                np.isclose(x[1], max(x[1])) & np.isclose(x[2], max(x[2]))
            ) |
            np.isclose(x[0], max(x[0])) & 
            (
                np.isclose(x[1], min(x[1])) & np.isclose(x[2], min(x[2])) | 
                np.isclose(x[1], max(x[1])) & np.isclose(x[2], min(x[2])) |
                np.isclose(x[1], min(x[1])) & np.isclose(x[2], max(x[2])) |
                np.isclose(x[1], max(x[1])) & np.isclose(x[2], max(x[2]))
            )
        )
    
    def left(x):
        print(np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0)))
        return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

    fdim = DIM - 1
    corner_ft = locate_entities(mesh=domain, dim=0, marker=corner)
    # print(np.full_like(corner_ft, 10))
    # exit()
    mesh_corner_ft = meshtags(mesh=domain, dim=fdim, entities=corner_ft, values=np.full_like(corner_ft, 10))
    # += Locate subdomain dofs
    y_dofs_corner = locate_dofs_topological(V=(mix_vs.sub(0).sub(Y), Vy), entity_dim=mesh_corner_ft.dim, entities=corner_ft)
    z_dofs_corner = locate_dofs_topological(V=(mix_vs.sub(0).sub(Z), Vz), entity_dim=mesh_corner_ft.dim, entities=corner_ft)
    
    # += Interpolate 
    uxx0, uxx1, uycorner, uzcorner = Function(Vx), Function(Vx), Function(Vy), Function(Vz)
    if uni:
        uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du)))
        uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    else:
        uxx0.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(du//2)))
        uxx1.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(-du//2)))
    uycorner.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    uzcorner.interpolate(lambda x: np.full(x.shape[1], default_scalar_type(F_0)))
    print(len(uzcorner.x.array[:]))
    # += Dirichlet Boundary Conditions
    bc_UxX0 = dirichletbc(value=uxx0, dofs=xx0_dofs, V=mix_vs.sub(0).sub(X))
    bc_UxX1 = dirichletbc(value=uxx1, dofs=xx1_dofs, V=mix_vs.sub(0).sub(X))
    bc_UyCorner = dirichletbc(value=uycorner, dofs=y_dofs_corner, V=mix_vs.sub(0).sub(Y))
    bc_UzCorner = dirichletbc(value=uzcorner, dofs=z_dofs_corner, V=mix_vs.sub(0).sub(Z))
    # += Assign
    bc = [bc_UxX0, bc_UxX1, bc_UyCorner, bc_UzCorner]
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
    
    # +==+ Create rotation and material property arrays
    print("\t" * depth + "+= Assign Fibre Directions")
    # += 1-DF Spaces
    myo = Function(Dcs)
    cyt_tgs = ct.find(tg_c[DIM][0])
    # += Cytosol [DEFAULT]
    myo.x.array[cyt_tgs] = np.full_like(cyt_tgs, CONSTIT_MYO[0], dtype=default_scalar_type)

    # +==+ Variables
    v, q = ufl.TestFunctions(Mxs)
    # += [CURRENT]
    mx = Function(Mxs)
    u, p = ufl.split(mx)
    
    # += Initial
    # +==+ Curvilinear setup
    x = ufl.SpatialCoordinate(domain)
    x_n = Function(V)
    azi, ele, con = Function(Vx), Function(Vy), Function(Vz)

    def wei_ang(pos, ang_type, df):
        cur = np.array([np.fromstring(row["Centroid"].strip('()'), sep=',') for _, row in df.iterrows()])
        for i, x in enumerate(["x", "y", "z"]):
            cur[:, i] = cur[:, i] * PXLS[x]
        
        ang = np.array(
            [row["Orientation [RAD]"] - H_ROT if not ang_type else row["Elevation Orientation [RAD]"] - E_ROT for _, row in df.iterrows()]
        )
        if not ang_type:
            ang[ang > 0.5] = 0
            ang[ang < -0.5] = 0
        else:
            ang[ang > 0.5] = 0
            ang[ang < -0.5] = 0
        dis = np.linalg.norm(cur - pos, axis=1)

        mask = dis <= 2000
        filt_dis = dis[mask]
        filt_ang = ang[mask] 

        if len(filt_dis) == 0:
            return 0
        
        wei = 1 / (filt_dis + 1e-10)
        wei_ang = np.sum(filt_ang * wei) / np.sum(wei)

        # if math.isnan(wei_ang):
        #     print("here")

        return wei_ang

    def gaus_smooth(crds, vals, thrs):
        tree = cKDTree(crds)
        s_vals = np.zeros_like(vals)
        
        # Iterate through each node
        for i, xyz in enumerate(crds):
            idx = tree.query_ball_point(xyz, thrs)
            nei_xyz = crds[idx]
            nei_val = vals[idx]
            dits = np.linalg.norm(nei_xyz - xyz, axis=1)
            weis = np.exp(-0.5 * (dits / thrs) ** 2)
            s_vals[i] = np.sum(weis * nei_val) / np.sum(weis)
        
        return s_vals
            
    ang_df = prop_csv(tnm, depth)
    for i in range(len(azi.x.array[:])):
        if tnm.split("_")[1] == "test":
            azi.x.array[i] = F_0
            ele.x.array[i] = F_0
            continue
        else:
            pos = np.array(x_n.function_space.tabulate_dof_coordinates()[i])
            azi.x.array[i] = wei_ang(pos, 0, ang_df)
            ele.x.array[i] = wei_ang(pos, 1, ang_df)    

    azi.x.array[:] = gaus_smooth(np.array(x_n.function_space.tabulate_dof_coordinates()[:]), azi.x.array[:], 4000)
    ele.x.array[:] = gaus_smooth(np.array(x_n.function_space.tabulate_dof_coordinates()[:]), ele.x.array[:], 4000)

    # azi_file = io.VTXWriter(MPI.COMM_WORLD, file + "_AZI" + ".bp", azi, engine="BP4")
    # ele_file = io.VTXWriter(MPI.COMM_WORLD, file + "_ELE" + ".bp", ele, engine="BP4")
    # azi_file.write(0)
    # azi_file.close()
    # ele_file.write(0)
    # ele_file.close()

    # print(azi.x.array[:])
    # print(ele.x.array[:])

    # exit(0)

    i, j, k, a, b = ufl.indices(5)
    # # += Curvilinear mapping dependent on subdomain values
    Push = ufl.as_matrix([
        [ufl.cos(azi), -ufl.sin(azi), 0],
        [ufl.sin(azi), ufl.cos(azi), 0],
        [0, 0, 1]
    ]) * ufl.as_matrix([
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
        CONSTIT_MYO[1] * E[0,0]**2 + 
        CONSTIT_MYO[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
        CONSTIT_MYO[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
    )
    piola = CONSTIT_MYO[0]/2 * ufl.exp(Q) * ufl.as_matrix([
        [4*CONSTIT_MYO[1]*E[0,0], 2*CONSTIT_MYO[3]*(E[1,0] + E[0,1]), 2*CONSTIT_MYO[3]*(E[2,0] + E[0,2])],
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

    # += Data functions for exporting with setup
    Vu_sol, up_to_u_sol = Mxs.sub(0).collapse() 
    dis = Function(Vu_sol) 
    sig = Function(Tes)
    eps = Function(Tes)
    # += Label Functions
    dis.name = "U - Displacement"
    eps.name = "E - Green Strain"
    sig.name = "S - Cauchy Stress"
    # += File setup
    file = os.path.dirname(os.path.abspath(__file__)) + "/_bp/" + tnm
    dis_file = io.VTXWriter(MPI.COMM_WORLD, file + "_DIS" + ".bp", dis, engine="BP4")
    azi_file = io.VTXWriter(MPI.COMM_WORLD, file + "_AZI" + ".bp", azi, engine="BP4")
    ele_file = io.VTXWriter(MPI.COMM_WORLD, file + "_ELE" + ".bp", ele, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, file + "_SIG" + ".bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, file + "_EPS" + ".bp", eps, engine="BP4")

    # +==+ Setup stress output
    def cauchy_tensor(u, p, x, azi, ele, myo):
        i, j, k, a, b = ufl.indices(5)
        # # += Curvilinear mapping dependent on subdomain values
        Push = ufl.as_matrix([
            [ufl.cos(azi), -ufl.sin(azi), 0],
            [ufl.sin(azi), ufl.cos(azi), 0],
            [0, 0, 1]
        ]) * ufl.as_matrix([
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
        # += Kinematics variables
        I = ufl.variable(ufl.Identity(DIM))
        F = ufl.as_tensor(I[i, j] + ufl.grad(u_nu)[i, j], [i, j]) * Push
        E = ufl.variable(ufl.as_tensor((0.5*(z_co[i,j] - Z_co[i,j])), [i, j]))
        J = ufl.variable(ufl.det(F))
        # += Material Setup | Guccione
        Q = (
            myo * E[0,0]**2 + 
            CONSTIT_MYO[2] * (E[1,1]**2 + E[2,2]**2 + 2*(E[1,2] + E[2,1])) + 
            CONSTIT_MYO[3] * (2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
        )
        piola = CONSTIT_MYO[0]/2 * ufl.exp(Q) * ufl.as_matrix([
            [4*myo*E[0,0], 2*CONSTIT_MYO[3]*(E[1,0] + E[0,1]), 2*CONSTIT_MYO[3]*(E[2,0] + E[0,2])],
            [2*CONSTIT_MYO[3]*(E[0,1] + E[1,0]), 4*CONSTIT_MYO[2]*E[1,1], 2*CONSTIT_MYO[2]*(E[2,1] + E[1,2])],
            [2*CONSTIT_MYO[3]*(E[0,2] + E[2,0]), 2*CONSTIT_MYO[2]*(E[1,2] + E[2,1]), 4*CONSTIT_MYO[3]*E[2,2]],
        ]) - p * Z_co
        sig = 1/J * F*piola*F.T
        return sig
    
    # +==+ Setup strain output
    def green_tensor(u, p, x, azi, ele):
        I = ufl.variable(ufl.Identity(DIM))
        F = ufl.variable(I + ufl.grad(u))
        C = ufl.variable(F.T * F)
        E = ufl.variable(0.5*(C-I))
        eps = ufl.as_tensor([
            [E[0, 0], E[0, 1], E[0, 2]], 
            [E[1, 0], E[1, 1], E[1, 2]], 
            [E[2, 0], E[2, 1], E[2, 2]]
        ])
        return eps
    
    # log.set_log_level(log.LogLevel.INFO)

    # +==+ Boundary Conditions
    print("\t" * depth + "+= Assign Boundary Conditions")
    disp = EDGE[0] * int(tnm.split("_")[3])/100
    if len(tnm.split("_")) == 7:
        bc = dir_bc(Mxs, Vx, Vy, Vz, ft, tnm.split("_")[6] == "x1c", disp)
    else: 
        its = int(tnm.split("_")[-1])
        phase_ext = np.linspace(0, disp, its)
        phase_con = np.linspace(disp, -disp, its*2)
        phase_exp = np.linspace(-disp, 0, its)
        bc_u_iter = np.concatenate([phase_ext, phase_con, phase_exp]).astype(np.int32)

    if len(tnm.split("_")) == 7:
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

        # += Evaluate values
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        # += Setup tensor space for stress tensor interpolation
        cauchy = Expression(
            e=cauchy_tensor(u_eval, p_eval, x, azi, ele, myo), 
            X=Tes.element.interpolation_points()
        )
        epsilon = Expression(
            e=green_tensor(u_eval, p_eval, x, azi, ele), 
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        eps.interpolate(epsilon)

        n_comps = 9
        sig_arr = sig.x.array
        eps_arr = eps.x.array
        n_nodes = len(sig_arr) // n_comps
        r_sig = sig_arr.reshape((n_nodes, n_comps))
        r_eps = eps_arr.reshape((n_nodes, n_comps))

        data_dict = {
            "sig_xx": r_sig[:, 0],
            "sig_yy": r_sig[:, 4],
            "sig_zz": r_sig[:, 8],
            "sig_xy": r_sig[:, 1],
            "sig_xz": r_sig[:, 2],
            "sig_yz": r_sig[:, 5],
            "eps_xx": r_eps[:, 0],
            "eps_yy": r_eps[:, 4],
            "eps_zz": r_eps[:, 8],
            "eps_xy": r_eps[:, 1],
            "eps_xz": r_eps[:, 2],
            "eps_yz": r_eps[:, 5],
            "Azimuth": azi.x.array[:],
            "Elevation": ele.x.array[:],
        }

        pd.DataFrame(data_dict).to_csv(os.path.dirname(os.path.abspath(__file__)) + "/_csv/" + tnm + ".csv")

        # += Write files
        dis_file.write(0)
        azi_file.write(0)
        ele_file.write(0)
        sig_file.write(0)
        eps_file.write(0)

    else:
        for i, d in enumerate(bc_u_iter):
            bc = dir_bc(Mxs, Vx, Vy, Vz, ft, tnm.split("_")[6] == "x1c", d)
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

            # += Evaluate values
            u_eval = mx.sub(0).collapse()
            p_eval = mx.sub(1).collapse()
            dis.interpolate(u_eval)
            # += Setup tensor space for stress tensor interpolation
            cauchy = Expression(
                e=cauchy_tensor(u_eval, p_eval, x, azi, ele, myo), 
                X=Tes.element.interpolation_points()
            )
            epsilon = Expression(
                e=green_tensor(u_eval, p_eval, x, azi, ele), 
                X=Tes.element.interpolation_points()
            )
            sig.interpolate(cauchy)
            eps.interpolate(epsilon)

            # += Write files
            dis_file.write(i)
            sig_file.write(i)
            eps_file.write(i)

    azi_file.write(0)
    ele_file.write(0)

    # += Close files
    dis_file.close()
    azi_file.close()
    ele_file.close()
    sig_file.close()
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

    def id_corner(coord):
        corners = {
            (0, 0, 0): 10,
            (EDGE[0], 0, 0): 11,
            (0, EDGE[1], 0): 12,
            (0, 0, EDGE[2]): 13,
            (EDGE[0], EDGE[1], 0): 14,
            (EDGE[0], 0, EDGE[2]): 15,
            (0, EDGE[1], EDGE[2]): 16,
            (EDGE[0], EDGE[1], EDGE[2]): 17
        }
        return corners.get(coord, False)

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
            # += If of Dimension 2 find the Z-Discs from centroid data
            if i == DIM - 1:
                gmsh.model.add_physical_group(
                    dim=i, tags=[j], tag=SUR_OBJ_ASSIGN[row[1]][0], name=SUR_OBJ_ASSIGN[row[1]][1]
                )
                TG_C[DIM-1].append(SUR_OBJ_ASSIGN[row[1]][0])
                continue
            # += For Dimension 3, determine if sarcomere or cytosol
            if i == DIM:
                gmsh.model.add_physical_group(dim=i, tags=[j], tag=int(PY_TAGS[i]), name="Cytosol")
                TG_C[DIM].append(int(PY_TAGS[i]))
                continue
            # += Any other region can be arbitrarily labeled 
            else: 
                if i == DIM - 3:
                    if not(id_corner(tuple(gmsh.model.getValue(dim=0, tag=j, parametricCoord=[])))):
                        continue
                    else:
                        gmsh.model.add_physical_group(dim=i, tags=[j], tag=id_corner(tuple(gmsh.model.getValue(dim=0, tag=j, parametricCoord=[]))), name="cx_" + str(j))
                        continue
                gmsh.model.add_physical_group(dim=i, tags=[j], tag=int(PY_TAGS[i]))
                PY_TAGS[i] += 1
        gmsh.model.occ.synchronize()

    # +==+ Generate Mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.setOrder(order=ORDER) 

    # +==+ Write File
    file = os.path.dirname(os.path.abspath(__file__)) + "/_msh/" + "EMSimCube" + ".msh"
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
    # += Arguments
    msh = False
    emfs = ["raw_" + x for x in ["test", "0", "1", "6", "8"]]
    # emfs = ["raw_1"]
    success = []
    failed = []
    for emf in emfs:
        ceq = "GC"
        dsp = "15"
        etp = "dir_ani_x1x0c"
        tnm = "_".join([emf, ceq, dsp, etp])
        # += Run
        print("\t" * depth + "!! BEGIN TEST: " + tnm + " !!")
        # main(tnm, msh, depth) 
        try:
            main(tnm, msh, depth)
            success.append(emf)
            print("\t" * depth + "!! TEST PASS: " + tnm + " !!")
        except:
            print("\t" * depth + "!! TEST FAIL: " + tnm + " !!")
            failed.append(emf)
            continue
    print("\t" * depth + "!! END !!") 
    print("\t" * depth + " ~> Pass: {}".format(success))
    print("\t" * depth + " ~> Fail: {}".format(failed))