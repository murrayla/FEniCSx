# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 30/04/2024
# Code: QuadSubdomainGeneration.py
#   Generation of .msh cubes with appropriate refinements.
# +==+==+==+==+

# +==+==+ Setup
# += Dependencies
import csv
import gmsh
import numpy as np
# += Constants
DIM = 2
RAW = 1
REF = 0.5
ORDER = 2
DIST_TAG = 1
THRE_TAG = 2
MINF_TAG = 3
DELAUNAY = 5
SARC_RADIUS = 0.05
# += Group IDs
CYTOSOL = 1
MYO_STRAIGHT = 2
MYO_UPANGLED = 3
MYO_DNANGLED = 4
LEFT_SIDE = 5
RIGHT_SIDE = 6

# +==+==+==+
# CSV Network Loading:
#   Input of label value
#   Output connectivity matrix and positions of relevant centroids
def csvnet(test_name):
    print("     += loading {} data".format(test_name))
    # += Load centroid positions
    with open("P_TestNets/CENN_" + test_name + ".csv", newline='') as f:
        reader = csv.reader(f)
        cent_data = list(reader)
        cent_data = [[float(y) for y in x] for x in cent_data]
    # += Load connectivity matrix 
    with open("P_TestNets/CMAT_" + test_name + ".csv", newline='') as f:
        reader = csv.reader(f)
        cmat_data = list(reader)
        cmat_data = [[int(float(y)) for y in x] for x in cmat_data]
    return np.array(cmat_data), np.array(cent_data)

# +==+==+==+
# Gmsh Cube:
#   Input of test_name and required graph network.
#   Output cube with refined generations.
def gmsh_cube(test_name, test_case):
    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(test_name)

    # += Store entitu tags and assign physical group labels.
    pt_tgs = []
    cv_tgs = []
    sf_tgs = []
    zs_tgs = []
    br_tgs = []
    cent_tgs = []
    area_tgs = []
    below_tgs = []
    above_tgs = []
    branch_tgs = []
    even_cents = []
    upper_cents = []
    lower_cents = []

    # += Create Points 
    p_coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
    for i, (x, y) in enumerate(p_coords):
        pt = gmsh.model.occ.addPoint(x=x, y=y, z=0, meshSize=0.5, tag=i+1)
        pt_tgs.append(pt)
    # += Create Lines 
    c_edge = [[1, 2], [2, 3], [3, 4], [4, 1]]
    for i, (x, y) in enumerate(c_edge):
        cv = gmsh.model.occ.addLine(startTag=x, endTag=y, tag=i+1)
        gmsh.model.occ.synchronize()
        # gmsh.model.addPhysicalGroup(dim=1, tags=[cv], name="SIDE_" + str(i))
        cv_tgs.append(cv)
    # += Create Surface
    lo = gmsh.model.occ.addCurveLoop([1, 2, 3, 4], 5)
    sf_tgs.append(lo)
    sf = gmsh.model.occ.addPlaneSurface([lo], tag=max(sf_tgs)+1)
    sf_tgs.append(sf)
    area_tgs.append(sf)
    gmsh.model.occ.synchronize()
    # gmsh.model.addPhysicalGroup(dim=2, tags=[6], name="AREA")

    # += Load Connectivity and Centroids
    cmat, cenn = csvnet(test_name)
    # += Create points from centroids
    for i, (x, y, z) in enumerate(cenn):
        # += Create Point above each Centroid at sarc radius
        pt = gmsh.model.occ.addPoint(x=x, y=y-SARC_RADIUS, z=z, meshSize=0.5, tag=max(pt_tgs)+1)
        pt_tgs.append(pt)
        below_tgs.append(pt)
        # += Create Point below each Centroid at sarc radius
        pt = gmsh.model.occ.addPoint(x=x, y=y+SARC_RADIUS, z=z, meshSize=0.5, tag=max(pt_tgs)+1)
        pt_tgs.append(pt)
        above_tgs.append(pt)
        # += Create Line between each of these new points
        cv = gmsh.model.occ.addLine(
                    startTag=below_tgs[-1], endTag=above_tgs[-1], 
                    tag=max(max(sf_tgs), max(cv_tgs)) + 1
                )
        cv_tgs.append(cv)
        zs_tgs.append(cv)
    gmsh.model.occ.synchronize()

    # += Create lines from connection matrix
    for j, row in enumerate(cmat):
        row = row[j::]
        for l, k in enumerate(row):
            if k:
                # += Connect below centroid points
                cv = gmsh.model.occ.addLine(
                    startTag=below_tgs[j], endTag=below_tgs[l+j], 
                    tag=max(max(sf_tgs), max(cv_tgs)) + 1
                )
                cv_tgs.append(cv)
                branch_tgs.append(cv)
                # += Connect above centroid points
                cv = gmsh.model.occ.addLine(
                    startTag=above_tgs[j], endTag=above_tgs[l+j], 
                    tag=max(max(sf_tgs), max(cv_tgs)) + 1
                )
                cv_tgs.append(cv)
                branch_tgs.append(cv)
                # += Create surfaces from lines
                lo = gmsh.model.occ.addCurveLoop([zs_tgs[j], branch_tgs[-1], zs_tgs[j+l], branch_tgs[-2]], tag=max(sf_tgs)+1)
                sf_tgs.append(lo)
                sf = gmsh.model.occ.addPlaneSurface([lo], max(sf_tgs)+1)
                sf_tgs.append(sf)
                area_tgs.append(sf)
                gmsh.model.occ.synchronize()

    # += Fragment the existing surfaces into new surfaces that are joined at nodes
    gmsh.model.occ.fragment([(2, area_tgs[0])], [(2, i) for i in area_tgs[1:]])
    gmsh.model.occ.synchronize()

    # += Create points from centroids
    for i, (x, y, z) in enumerate(cenn):
        pt = gmsh.model.occ.addPoint(x=x, y=y, z=z, meshSize=0.5, tag=max(pt_tgs)+100)
        pt_tgs.append(pt)
        cent_tgs.append(pt)
    gmsh.model.occ.synchronize()
    # += Create lines from connection matrix
    for j, row in enumerate(cmat):
        row = row[j::]
        for l, k in enumerate(row):
            if k:
                cv = gmsh.model.occ.addLine(
                    startTag=cent_tgs[j], endTag=cent_tgs[l+j], 
                    tag=max(max(sf_tgs), max(cv_tgs)) + 100
                )
                cv_tgs.append(cv)
                br_tgs.append(cv)
                centroids = np.array([cenn[j], cenn[j+l]])
                if cenn[j][1] == cenn[j+l][1]:
                    av = np.mean(centroids, axis=0)
                    even_cents.append(av)
                if cenn[j][1] > cenn[j+l][1]:
                    av = np.mean(centroids, axis=0)
                    lower_cents.append(av)
                if cenn[j][1] < cenn[j+l][1]:
                    av = np.mean(centroids, axis=0)
                    upper_cents.append(av)
    gmsh.model.occ.synchronize()

    # += Mark Physical groups
    even, upper, lower, remain = [], [], [], []
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        ev_check = [np.allclose(com, x) for x in even_cents]
        lw_check = [np.allclose(com, x) for x in lower_cents]
        up_check = [np.allclose(com, x) for x in upper_cents]
        # += MANUALLY SUPPLY LABELS
        print("     += DECLARE SUFRACE FOR {} @ COM: {}".format(test_name, com))
        dec = int(input(" ~+ Enter: "))
        # if np.any(ev_check):
        #     even.append(surface[1])
        #     continue
        # if np.any(lw_check):
        #     lower.append(surface[1])
        #     continue
        # if np.any(up_check):
        #     upper.append(surface[1])
        #     continue
        # else:
        #     remain.append(surface[1])
        if dec == 0:
            even.append(surface[1])
            continue
        if dec == 1:
            lower.append(surface[1])
            continue
        if dec == 2:
            upper.append(surface[1])
            continue
        else:
            remain.append(surface[1])
    gmsh.model.addPhysicalGroup(2, even, MYO_STRAIGHT, "Straight Sarcomere")
    gmsh.model.addPhysicalGroup(2, lower, MYO_DNANGLED, "Sarcomere Angled Down")
    gmsh.model.addPhysicalGroup(2, upper, MYO_UPANGLED, "Sarcomere Angled Up")
    gmsh.model.addPhysicalGroup(2, remain, CYTOSOL, "Cytosol")
    # += Mark right and left
    left, right = [], []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0):
            left.append(line[1])
            continue
        if np.isclose(com[0], 1):
            right.append(line[1])
    gmsh.model.addPhysicalGroup(1, left, LEFT_SIDE, "X0")
    gmsh.model.addPhysicalGroup(1, right, RIGHT_SIDE, "X1")

    # +==+==+
    # Create Mesh fields
    # += Crete distance field to begin mesh generation
    gmsh.model.mesh.field.add("Distance", DIST_TAG)
    # gmsh.model.mesh.field.setNumbers(DIST_TAG, "PointsList", cent_tgs)
    gmsh.model.mesh.field.setNumbers(DIST_TAG, "CurvesList", br_tgs)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    # += Create threshold field
    gmsh.model.mesh.field.add("Threshold", THRE_TAG)
    # gmsh.model.mesh.field.setNumber(THRE_TAG, "Sigmoid", True)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "InField", DIST_TAG)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "SizeMin", 0.02)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "SizeMax", 1)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "DistMax", 1)
    # += Create minimum field
    gmsh.model.mesh.field.add("Min", MINF_TAG)
    gmsh.model.mesh.field.setNumbers(MINF_TAG, "FieldsList", [THRE_TAG])
    # += Set min field as background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(MINF_TAG)
    # += Set and Stabalise meshing
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


    # += set "Delaunay" algorithm for meshing due to complex gradients
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", value=DELAUNAY)

    # += Generate Mesh
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.setOrder(order=ORDER)
    # += Write File
    gmsh.write("P_Branch_Contraction/gmsh_msh/SUB_" + test_name + ".msh")
    gmsh.finalize()

# +==+==+==+
# Main
def main(test_name, test_case):
    for i in test_case:
        print(" += Create Mesh: {}".format(test_name[i]))
        gmsh_cube(test_name[i], i)

# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+==+
    # Test Parameters
    # += Test name
    test = [
        "SINGLE_MIDDLE", "SINGLE_DOUBLE", "SINGLE_ACROSS", 
        "DOUBLE_ACROSS", "BRANCH_ACROSS", "BRANCH_MIDDLE", 
        "TRANSFER_DOUBLE"
    ]
    test_name = ["QUAD_XX_" + x for x in test]
    # += Cases
    test_case = list(range(0, 7, 1))
    # += Feed Main()
    main(test_name, test_case)