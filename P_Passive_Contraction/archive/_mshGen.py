"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _mshGen.py
"""

# +==+==+==+
# Setup
# += Imports
import numpy as np
import gmsh
import sys
import os

# += Parameters
DIM = 3
I_0 = 0
F_0 = 0.0
ORDER = 2
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
X, Y, Z = EDGE[0], EDGE[1], EDGE[2]

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
def msh_(tnm, s, ELM_TGS, PHY_TGS, LAB_TGS, depth):
    depth += 1

    # += Initialise and begin geometry
    print("\t" * depth + "~> Initialise") 
    gmsh.initialize()
    gmsh.model.add(tnm)
    
    # += Create slice [chatgpt]
    def generate_layer(w, h, r, z):
        p1 = gmsh.model.occ.addPoint(r, 0, z, s)
        p2 = gmsh.model.occ.addPoint(w - r, 0, z, s)
        p3 = gmsh.model.occ.addPoint(w, r, z, s)
        p4 = gmsh.model.occ.addPoint(w, h - r, z, s)
        p5 = gmsh.model.occ.addPoint(w - r, h, z, s)
        p6 = gmsh.model.occ.addPoint(r, h, z, s)
        p7 = gmsh.model.occ.addPoint(0, h - r, z, s)
        p8 = gmsh.model.occ.addPoint(0, r, z, s)
        c1_center = gmsh.model.occ.addPoint(w - r, r, z, s)
        c2_center = gmsh.model.occ.addPoint(w - r, h - r, z, s)
        c3_center = gmsh.model.occ.addPoint(r, h - r, z, s)
        c4_center = gmsh.model.occ.addPoint(r, r, z, s)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p5, p6)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p7, p8)
        c1 = gmsh.model.occ.addCircleArc(p3, c1_center, p2)
        c2 = gmsh.model.occ.addCircleArc(p5, c2_center, p4)
        c3 = gmsh.model.occ.addCircleArc(p7, c3_center, p6)
        c4 = gmsh.model.occ.addCircleArc(p1, c4_center, p8)
        loop = gmsh.model.occ.addCurveLoop([l1, c1, l3, c2, l2, c3, l4, c4])
        # surface = gmsh.model.occ.addPlaneSurface([loop])
        return loop
    
    # += Produce layers
    print("\t" * depth + "~> Create layers")
    rad = 1000
    z0 = generate_layer(X, Y, rad, 0)
    z3 = generate_layer(X, Y, rad, Z)
    
    # Create a smooth transition using ThruSections
    print("\t" * depth + "~> Through section for 3D") 
    vol = gmsh.model.occ.addThruSections([z0, z3], 1, True)

    gmsh.model.occ.fillet([1], [1,2,3,4,9,10,11,12], [500], removeVolume=True)
    gmsh.model.occ.synchronize()
    # gmsh.model.occ.remove([(2,1), (2,2)], recursive=True)
    # gmsh.model.occ.synchronize()

    # +==+ Generate physical groups
    for i in range(0, DIM+1, 1):
        # += Generate mass, com and tag data
        _, tgs = zip(*gmsh.model.occ.get_entities(dim=i))
        # += Generate physical groups 
        for __, j in enumerate(tgs):
            tag = PHY_TGS[i][-1]
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
                continue
                # try:
                #     com = tuple(np.round(gmsh.model.occ.get_bounding_box(dim=i, tag=j)[:3],1))
                #     idx = np.where([x==com for x in PNT])[0][0]
                #     name = PNT_NAM[idx]
                # except:
                #     name = f"Point_{com}"
            gmsh.model.add_physical_group(dim=i, tags=[j], tag=tag, name=name)
            PHY_TGS[i].append(PHY_TGS[i][-1]+1)
            LAB_TGS[i].append(name)
        gmsh.model.occ.synchronize()

    points = gmsh.model.getEntities(dim=0)  # dim=0 for points
    for point in points:
        gmsh.model.mesh.setSize([point], s)  # Apply mesh size s to the point
    
    # += Mesh data
    print("\t" * depth + "~> Mesh") 
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.setOrder(order=ORDER) 

    # +==+ Write File
    file = os.path.dirname(os.path.abspath(__file__)) + "/_msh/" + tnm + str(s).replace(".", "") + ".msh"
    print("\t" * depth + "~> Save") 
    gmsh.write(file)
    print(gmsh.model.getPhysicalGroups())
    gmsh.finalize()

    return

# +==+==+==+
# main
#   Inputs: 
#       tnm  | str | test name
#   Outputs:
#       .bp folder of deformation
def main(tnm, depth):
    depth += 1
    # += Ref level
    s = 1500
    # += Tag Values
    ELM_TGS = {0: [1000], 1: [100], 2: [10], 3: [1]}
    PHY_TGS = {0: [5000], 1: [500], 2: [50], 3: [5]}
    LAB_TGS = {0: [], 1: [], 2: [], 3: []}
    # += Mesh generation
    print("\t" * depth + "+= Generate Mesh") 
    msh_(tnm, s, ELM_TGS, PHY_TGS, LAB_TGS, depth)
    
# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    print("\t" * depth + "!! MESHING !!") 
    tnm = "EMGEO_"
    main(tnm, depth) 
