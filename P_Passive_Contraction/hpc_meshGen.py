"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: hpc_meshGen.py
        HPC version of meshGen for operation in commandline
"""

# +==+==+==+
# Setup
# += Imports
import numpy as np
import argparse
import gmsh
import os

# += Parameters
DIM = 3
I_0 = 0
F_0 = 0.0
ORDER = 2

# += Labels
VOL_NAM = ["Volume"]
SUR_NAM = [
    "Surface_x0", "Surface_x1", 
    "Surface_y0", "Surface_y1", 
    "Surface_z0", "Surface_z1"
]
SUR_NAM_VAL = [
    1110, 1112, 
    1101, 1121, 
    1011, 1211
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

# +==+==+==+
# msh_:
#   Inputs: 
#       tnm  | str | test name
#   Outputs:
#       .msh file of mesh
#       tg_c | dict | cytosol physical element data
#       tg_S | dict | sarcomere physical element data
def msh_(tnm, s, b, e_tg, p_tg, l_tg, depth):
    depth += 1
    print("\t" * depth + "+= Generate Mesh: {}.msh".format(tnm + str(s)))

    # +==+ Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(tnm)
    
    # += Define Points
    pnt = []
    p_coords = [
        (0, 0, 0), (EDGE[0], 0, 0), 
        (EDGE[0], EDGE[1], 0), (0, EDGE[1], 0),
        (0, 0, EDGE[2]), (EDGE[0], 0, EDGE[2]), 
        (EDGE[0], EDGE[1], EDGE[2]), (0, EDGE[1], EDGE[2])
    ]
    # += Add Points
    for i, (x, y, z) in enumerate(p_coords):
        pnt.append(gmsh.model.occ.addPoint(x=x, y=y, z=z))
    
    # += Define lines
    lns = []
    pnt_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    # += Add Lines
    for i, (p1, p2) in enumerate(pnt_pairs):
        lns.append(gmsh.model.occ.addLine(startTag=pnt[p1], endTag=pnt[p2]))
    
    # += Define surfaces
    cvs = [
        [lns[i] for i in [0, 1, 2, 3]],
        [lns[i] for i in [4, 5, 6, 7]],
        [lns[i] for i in [0, 9, 4, 8]],
        [lns[i] for i in [1, 10, 5, 9]],
        [lns[i] for i in [2, 11, 6, 10]],
        [lns[i] for i in [3, 8, 7, 11]]
    ]
    # += Add Surfaces
    srf = [
        gmsh.model.occ.addPlaneSurface(
            wireTags=[gmsh.model.occ.addCurveLoop(curveTags=loop)]
        ) for loop in cvs
    ]
    
    # += Add volume
    vol = gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop(srf)])
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
                try:
                    idx = np.where([x==com for x in VOL])[0][0]
                except:
                    idx = 0
                name = VOL_NAM[idx]
            # += Surfaces
            if i == DIM - 1:
                try:
                    idx = np.where([x==com for x in SUR])[0][0]
                    name = SUR_NAM[idx]
                    tag = SUR_NAM_VAL[idx]
                    gmsh.model.add_physical_group(dim=i, tags=[j], tag=tag, name=name)
                    continue
                except:
                    name = f"Surface_{com}"
            # += Curves
            if i == DIM - 2:
                try:
                    idx = np.where([x==com for x in LIN])[0][0]
                    name = LIN_NAM[idx]
                except:
                    name = f"Curve_{com}"
            # += Points
            if i == DIM - 3:
                try:
                    com = tuple(np.round(gmsh.model.occ.get_bounding_box(dim=i, tag=j)[:3],1))
                    idx = np.where([x==com for x in PNT])[0][0]
                    name = PNT_NAM[idx]
                except:
                    name = f"Point_{com}"
            gmsh.model.add_physical_group(dim=i, tags=[j], tag=tag, name=name)
            p_tg[i].append(p_tg[i][-1]+1)
            l_tg[i].append(name)
        gmsh.model.occ.synchronize()

    # += Set mesh size
    pnt = gmsh.model.getEntities(dim=0)  # dim=0 for points
    for point in pnt:
        gmsh.model.mesh.setSize([point], s)  # Apply mesh size s to the point

    # # +==+ Generate Mesh
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.setOrder(order=ORDER) 

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
def main(tnm, s, b, depth):
    depth += 1
    # += Tag Values
    ELM_TGS = {0: [1000], 1: [100], 2: [10], 3: [1]}
    PHY_TGS = {0: [5000], 1: [500], 2: [50], 3: [5]}
    LAB_TGS = {0: [], 1: [], 2: [], 3: []}
    # += Mesh generation
    f, tg, l_tg = msh_(tnm, s, b, ELM_TGS, PHY_TGS, LAB_TGS, depth)
    
# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    print("\t" * depth + "!! MESHING !!") 
    tnm = "EMGEO_"
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--mesh_size",type=int)
    parser.add_argument("-b", "--test_type",type=int)
    args = parser.parse_args()
    s = args.mesh_size
    b = args.test_type
    # += Dimension data
    # += Per test
    if b:
        CUBE = {"x": 3900, "y": 1100, "z": 200}
        tnm += "BIG_"
    else:
        CUBE = {"x": 1000, "y": 1000, "z": 100}
    # += Constant
    PXLS = {"x": 11, "y": 11, "z": 50}
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
    main(tnm, s, b, depth) 
    