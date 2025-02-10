"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: fx.py
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
import ast
import os

# += Parameters
DIM = 3
I_0 = 0
F_0 = 0.0
ORDER = 2
X, Y, Z = 0, 1, 2
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]

# += Centre of Masses
VOL = [(EDGE[0]/2, EDGE[1]/2, EDGE[2]/2)]
SUR = [
    (F_0, EDGE[1]/2, EDGE[2]/2), (EDGE[0], EDGE[1]/2, EDGE[2]/2), 
    (EDGE[0]/2, F_0, EDGE[2]/2), (EDGE[0]/2, EDGE[1], EDGE[2]/2), 
    (EDGE[0]/2, EDGE[1]/2, F_0), (EDGE[0]/2, EDGE[1]/2, EDGE[2]) 
]
LIN = [
    (EDGE[0]/2, F_0, F_0), (EDGE[0]/2, F_0, EDGE[2]), 
    (F_0, EDGE[1]/2, F_0), (EDGE[0], EDGE[1]/2, F_0),
    (EDGE[0]/2, EDGE[1], F_0), (EDGE[0]/2, EDGE[1], EDGE[2]), 
    (F_0, EDGE[1]/2, EDGE[2]), (EDGE[0], EDGE[1]/2, EDGE[2]),
    (F_0, F_0, EDGE[2]/2), (F_0, EDGE[1], EDGE[2]/2), 
    (EDGE[0], F_0, EDGE[2]/2), (EDGE[0], EDGE[1], EDGE[2]/2),
]
PNT = [
    (F_0, F_0, F_0), (EDGE[0], F_0, F_0), 
    (F_0, EDGE[1], F_0), (F_0, F_0, EDGE[2]), 
    (EDGE[0], EDGE[1], F_0), (EDGE[0], F_0, EDGE[2]), 
    (F_0, EDGE[1], EDGE[2]), (EDGE[0], EDGE[1], EDGE[2])
]
# += Labels
VOL_NAM = ["Volume"]
SUR_NAM = [
    "Surface_x0", "Surface_x1", 
    "Surface_y0", "Surface_y1", 
    "Surface_z0", "Surface_z1"
]
LIN_NAM = [
    "Line_xy0z0", "Line_xy0z1", 
    "Line_x0yz0", "Line_x1yz0",
    "Line_xy1z0", "Line_xy1z1", 
    "Line_x0yz1", "Line_x1yz1",
    "Line_x0y0z", "Line_x0y1z", 
    "Line_x1y0z", "Line_x1y1z",
]
PNT_NAM = [
    "Point_x0y0z0", "Point_x1y0z0", 
    "Point_x0y1z0", "Point_x0y0z1",
    "Point_x1y1z0", "Point_x1y0z1", 
    "Point_x0y1z1", "Point_x1y1z1"
]
# += Group Dictionaries 
# VOL_PHY = dict(zip(VOL_NAM, list(range(PHY_TGS[DIM][0], PHY_TGS[DIM][0]+len(VOL), 1))))
# SUR_PHY = dict(zip(SUR_NAM, list(range(PHY_TGS[DIM-1][0], PHY_TGS[DIM-1][0]+len(SUR), 1))))
# LIN_PHY = dict(zip(LIN_NAM, list(range(PHY_TGS[DIM-2][0], PHY_TGS[DIM-2][0]+len(LIN), 1))))
# PNT_PHY = dict(zip(PNT_NAM, list(range(PHY_TGS[DIM-3][0], PHY_TGS[DIM-3][0]+len(PNT), 1))))

# +==+==+==+
# msh_:
#   Inputs: 
#       tnm  | str | test name
#   Outputs:
#       .msh file of mesh
#       tg_c | dict | cytosol physical element data
#       tg_S | dict | sarcomere physical element data
def msh_(tnm, s, e_tg, p_tg, l_tg, depth):
    depth += 1
    print("\t" * depth + "+= Generate Mesh: {}.msh".format(tnm + str(s)))

    # +==+ Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(tnm)

    # # += Physical Tags
    # p_tg = PHY_TGS.copy()
    # e_tg = ELM_TGS.copy()
    # l_tg = LAB_TGS.copy()

    # +==+ Create cube
    box = gmsh.model.occ.addBox(
        x=0, y=0, z=0, 
        dx=CUBE["x"]*PXLS["x"], dy=CUBE["y"]*PXLS["y"], dz=CUBE["z"]*PXLS["z"], 
        tag=e_tg[DIM][-1]
    )
    e_tg[DIM].append(e_tg[DIM][-1]+1)
    gmsh.model.occ.synchronize()

    # +==+ Generate physical groups
    for i in range(0, DIM+1, 1):
        # += Generate mass, com and tag data
        _, tgs = zip(*gmsh.model.occ.get_entities(dim=i))
        # += Generate physical groups 
        for __, j in enumerate(tgs):
            tag = p_tg[i][-1]
            com = gmsh.model.occ.get_center_of_mass(dim=i, tag=j)
            # += Volumes
            if i == DIM:
                idx = np.where([x==com for x in VOL])[0][0]
                name = VOL_NAM[idx]
            # += Surfaces
            if i == DIM - 1:
                idx = np.where([x==com for x in SUR])[0][0]
                name = SUR_NAM[idx]
            # += Curves
            if i == DIM - 2:
                idx = np.where([x==com for x in LIN])[0][0]
                name = LIN_NAM[idx]
            # += Points
            if i == DIM - 3:
                com = tuple(np.round(gmsh.model.occ.get_bounding_box(dim=i, tag=j)[:3],1))
                idx = np.where([x==com for x in PNT])[0][0]
                name = PNT_NAM[idx]
            gmsh.model.add_physical_group(dim=i, tags=[j], tag=tag, name=name)
            p_tg[i].append(p_tg[i][-1]+1)
            l_tg[i].append(name)
        gmsh.model.occ.synchronize()


    points = gmsh.model.getEntities(dim=0)  # dim=0 for points
    for point in points:
        gmsh.model.mesh.setSize([point], s)  # Apply mesh size `s` to the point

    # +==+ Generate Mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.setOrder(order=ORDER) 
    # gmsh.model.mesh.refine()

    # +==+ Write File
    file = os.path.dirname(os.path.abspath(__file__)) + "/_msh/" + tnm + str(s).replace(".", "") + ".msh"
    gmsh.write(file)
    gmsh.finalize()
    return file, p_tg, l_tg

# +==+==+==+
# main
#   Inputs: 
#       tnm  | str | test name
#   Outputs:
#       .bp folder of deformation
def main(tnm, depth):
    depth += 1
    # += Iterate Mesh Sizes
    msize = np.round(np.arange(200, 3400, 400), 2)

    for s in msize:
        # += Tag Values
        ELM_TGS = {0: [1000], 1: [100], 2: [10], 3: [1]}
        PHY_TGS = {0: [5000], 1: [500], 2: [50], 3: [5]}
        LAB_TGS = {0: [], 1: [], 2: [], 3: []}
        # += Mesh generation
        f, tg, l_tg = msh_(tnm, s, ELM_TGS, PHY_TGS, LAB_TGS, depth)
    
# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    print("\t" * depth + "!! MESHING !!") 
    tnm = "EMGEO_"
    main(tnm, depth) 
    