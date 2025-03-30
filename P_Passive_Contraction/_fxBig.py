"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: hpc_fx.py
        Contraction over volume from EM data informed anisotropy
"""

# +==+==+==+
# Setup
# += Imports
from dolfinx import log, io,  default_scalar_type
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, Expression, element
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element, mixed_element
from mpi4py import MPI
import pandas as pd
import numpy as np
import argparse
import basix
import ufl
import os
# += Parameters
DIM = 3
INC = 10
I_0 = 0
F_0 = 0.0 
TOL = 1e-6
ZDISC = 5 
ZLINE = 14
ORDER = 2 
QUADRATURE = 4
X, Y, Z = 0, 1, 2
CONSTIT_MYO = [1, 1, 1, 1]
MNR_CONS = [1, 1]
HLZ_CONS = [x for x in [0.059, 8.023, 18.472, 16.026, 2.481, 11.120, 0.216, 11.436]]
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
constitutive = 1

"""
laplacian
  Inputs: 
      mix_vs  | obj | mixed vector space
      Vx, Vy, Vz | obj | collapsed vector spaces
      ft | np_array | facet tag data
      du | float | displacement value
  Outputs:
      numpy array of boudnary condition assignment data
"""
def gauss_smooth(coords, angles, depth):
    depth += 1
    print("\t" * depth + "~> Apply Gaussian smoothing with anisotropy")

    # += create new values
    s_data = np.zeros_like(angles)

    # += Standard deviations
    sigma_x = 2000 / 2.0 
    sigma_y = 500 / 2.0  
    sigma_z = 500 / 2.0

    # += Iterate through coordinates
    for i in range(len(coords)):
        dist_x = coords[:, 0] - coords[i, 0]
        dist_y = coords[:, 1] - coords[i, 1]
        dist_z = coords[:, 2] - coords[i, 2]
        weights = np.exp(-0.5 * ((dist_x / sigma_x) ** 2 +
                                 (dist_y / sigma_y) ** 2 +
                                 (dist_z / sigma_z) ** 2))
        weights /= weights.sum()
        s_data[i] = np.sum(weights * angles)

    return s_data


def anistropic(tnm, msh_ref, azi_vals, ele_vals, x_n, r, depth):
    depth += 1
    print("\t" * depth + "~> Load and apply anistropic fibre orientations")

    # += Load
    nmb = tnm.split("_")[0]
    ang_df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/_csv/vals_EMGEO_BIG.csv")
    n_list = []
    f = os.path.dirname(os.path.abspath(__file__)) + "/_msh/" + "EMGEO_BIG_" + str(r) + "_mesh.nodes"

    # += Generate node coordinates
    for line in open(f, 'r'):
        n_list.append(line.strip().replace('\t', ' ').split(' '))
    node = np.array(n_list[1:]).astype(np.float64)
    cart = node[:, 1:]
    new_id = []

    # += Capture coordinates
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    a = ang_df["a"].to_numpy()
    e = ang_df["e"].to_numpy()

    # += Reorganise data to match appropriate node order
    for i in range(len(coords)):
        pos = coords[i]
        dis = np.linalg.norm(pos - cart, axis=1)
        idx = np.argmin(dis)
        azi_vals[i] = a[idx]
        ele_vals[i] = e[idx]

    # += Smooth
    azi_vals = gauss_smooth(coords, azi_vals, depth)
    ele_vals = gauss_smooth(coords, ele_vals, depth)

    return azi_vals, ele_vals

# +==+==+==+
# fx_
#   Inputs: 
#       tnm  | str | test name
#       file | str | file name
#       tg_c | dict | cytosol physical element data
#       tg_S | dict | sarcomere physical element data
#   Outputs:
#       .bp folder of deformation
def fx_(tnm, r, depth):
    depth += 1
    print("\t" * depth + "+= Begin FE")
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_msh/EMGEO_BIG_" + str(r) + ".msh")

    # += Domain Setup
    print("\t" * depth + "+= Load Mesh and Setup Vector Spaces")

    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    V2 = element(family="Lagrange", cell=domain.basix_cell(), degree=ORDER, shape=(domain.geometry.dim,))
    V1 = element(family="Lagrange", cell=domain.basix_cell(), degree=ORDER-1)
    Mxs = functionspace(mesh=domain, element=mixed_element([V2, V1]))

    # += Extract subdomains for dofs
    print("\t" * depth + "+= Extract Subdomains")
    V, _ = Mxs.sub(0).collapse()
    Vx, dofsX = V.sub(X).collapse()
    Vy, dofsY = V.sub(Y).collapse()
    Vz, dofsZ = V.sub(Z).collapse()
    
    # += Anistropic setup
    print("\t" * depth + "+= Setup Anistropy")
    x_n = Function(V)
    ori = Function(V)
    ori_arr = ori.x.array.reshape(-1, 3)
    azi, ele = Function(Vx), Function(Vy)
    azi_vals = np.full_like(azi.x.array[:], F_0, dtype=default_scalar_type)
    ele_vals = np.full_like(ele.x.array[:], F_0, dtype=default_scalar_type)
    azi.x.array[:], ele.x.array[:] = anistropic(tnm, n, azi_vals, ele_vals, x_n, r, depth)
    
    # += Compute unit vector from azi/ele
    ori_arr[:, 0] = np.cos(ele.x.array) * np.cos(azi.x.array) 
    ori_arr[:, 1] = np.cos(ele.x.array) * np.sin(azi.x.array) 
    ori_arr[:, 2] = np.sin(ele.x.array) 

    ori.x.array[:] = ori_arr.reshape(-1)

    # += Write the vector field to file
    ori_file = io.VTXWriter(MPI.COMM_WORLD, os.path.dirname(os.path.abspath(__file__)) + "/_bp/BIG.bp", ori, engine="BP4")
    ori_file.write(0)
    ori_file.close()
    