# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 15/05/2024
# Code: CubeSubdomainGeneration.py
#   Generation of .msh cubes with appropriate refinements.
# +==+==+==+==+

# +==+==+ Setup
# += Dependencies
import csv
import gmsh
import numpy as np
from scipy.spatial import distance_matrix
# += Constants
DIM = 3
RAW = 1
REF = 0.5
ORDER = 2
DIST_TAG = 1
THRE_TAG = 2
MINF_TAG = 3
DELAUNAY = 5
SARC_RADIUS = 10
NAP_X, NAP_Y, NAP_Z = 110, 110, 40
EXT = 0.0
MAX_X, MAX_Y, MAX_Z = NAP_X+NAP_X*EXT, NAP_Y+NAP_Y*EXT, NAP_Z+NAP_Z*EXT
MIN_X, MIN_Y, MIN_Z = -NAP_X*EXT, -NAP_Y*EXT, -NAP_Z*EXT
DIS_X, DIS_Y, DIS_Z = MAX_X+MIN_X, MAX_Y+MIN_Y, MAX_Z+MIN_Z
# += Group IDs
XX = [1, 2]
YY = [3, 4]
ZZ = [5, 6]
CYT = 10
MYO = [x + 11 for x in range(0, 50, 1)]
VOLUME = 100

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

    # += API Setup
    mdl = gmsh.model
    occ = mdl.occ
    msh = mdl.mesh

    # += Store entitu tags and assign physical group labels.
    pt_tgs = []
    cv_tgs = []
    lo_tgs = []
    sf_tgs = []
    vo_tgs = []
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
    volume_tgs = []

    # += Create cube
    occ.addBox(
        x=MIN_X, y=MIN_Y, z=MIN_Z, 
        dx=MAX_X+abs(MIN_X), dy=MAX_Y+abs(MIN_Y), dz=MAX_Z+abs(MIN_Z), 
        tag=VOLUME
    )
    occ.synchronize()
    ent = occ.getEntities()
    curr_ents = []
    # += Assign physical groups to entitiies 
    for (d, e) in ent:
        curr_ents.append(e)
        if d == 0:
            pt_tgs.append(e)
        if d == 1: 
            cv_tgs.append(e)
        if d == 2: 
            sf_tgs.append(e)
            com = occ.getCenterOfMass(d, e)
            if np.allclose(com, [MIN_X, DIS_Y/2, DIS_Z/2]):
                mdl.addPhysicalGroup(dim=d, tags=[XX[0]], name="SURF_" + "xx-")
            if np.allclose(com, [MAX_X, DIS_Y/2, DIS_Z/2]):
                mdl.addPhysicalGroup(dim=d, tags=[XX[1]], name="SURF_" + "xx+")
            if np.allclose(com, [DIS_X/2, MIN_Y, DIS_Z/2]):
                mdl.addPhysicalGroup(dim=d, tags=[YY[0]], name="SURF_" + "yy-")
            if np.allclose(com, [DIS_X/2,MAX_Y, DIS_Z/2]):
                mdl.addPhysicalGroup(dim=d, tags=[YY[1]], name="SURF_" + "yy+")
            if np.allclose(com, [DIS_X/2, DIS_Y/2, MIN_Z]):
                mdl.addPhysicalGroup(dim=d, tags=[ZZ[0]], name="SURF_" + "zz-")
            if np.allclose(com, [DIS_X/2, DIS_Y/2, MAX_Z]):
                mdl.addPhysicalGroup(dim=d, tags=[ZZ[1]], name="SURF_" + "zz+")
            occ.synchronize()
        if d == 3: 
            vo_tgs.append(e)

    # += Load Connectivity and Centroids
    cmat, cenn = csvnet(test_name.split("_")[2])
    x_hat = np.average(cenn[:, 0])
    y_hat = np.average(cenn[:, 1])
    z_hat = np.average(cenn[:, 2])
    cenn_nu = np.zeros_like(cenn)
    cenn_al = np.zeros_like(cenn)
    for i, (x, y, z) in enumerate(cenn):
        x_nu = np.matmul(
            np.array([(x-x_hat)/max(cenn[:, 0]), (y-y_hat)/max(cenn[:, 1]), (z-z_hat)/max(cenn[:, 2])]).T, 
            np.array([[np.cos(np.pi/4), np.sin(np.pi/4), 0], [-np.sin(np.pi/4), np.cos(np.pi/4), 0], [0, 0, 1]])
        )
        cenn_nu[i, :] = [x_nu[0]+NAP_X/2, x_nu[1]+NAP_Y/2, x_nu[2]+NAP_Z/2]
    for i, (x, y, z) in enumerate(cenn_nu):
        cenn_al[i, :] = [
            (x-min(cenn_nu[:, 0]))/(max(cenn_nu[:, 0]) - min(cenn_nu[:, 0])) * NAP_X, 
            (y-min(cenn_nu[:, 1]))/(max(cenn_nu[:, 1]) - min(cenn_nu[:, 1])) * NAP_Y, 
            (z-min(cenn_nu[:, 2]))/(max(cenn_nu[:, 2]) - min(cenn_nu[:, 2])) * NAP_Z
        ]

    # += Create Circles and Loops for each Z-Disc
    cv_tgs.append(10)
    lo_tgs.append(max(cv_tgs)+1)
    for i, (x, y, z) in enumerate(cenn_al):
        # += Create circle for Z-Disc
        cv = occ.addCircle(x=x, y=y, z=z, r=SARC_RADIUS, zAxis=[1, 0, 0])
        cv_tgs.append(cv)
        # += Create curve loop
        zl = occ.addCurveLoop(curveTags=[cv], tag=max(lo_tgs)+1)
        lo_tgs.append(zl)
        occ.synchronize()
        # += Assign Physical groups
        # mdl.addPhysicalGroup(dim=1, tags=[cv], name="SURF_" + "Disc_" + str(zl))                  
        zs_tgs.append(zl)

    # += Create lines from connection matrix
    e = 1
    for j, row in enumerate(cmat):
        row = row[j::]
        for l, k in enumerate(row):
            if k:
                # += Connect Z-Discs
                c = occ.addThruSections(wireTags=[zs_tgs[j], zs_tgs[l+j]], tag=e, makeSolid=True, makeRuled=False)
                volume_tgs.append(c[0][1])
                occ.synchronize()
                e += 1

    # += Check on surface entitites to seperate on an overlap of z-disc
    ents = occ.get_entities(dim=DIM-1)
    coms = np.array(
        [
            [occ.getCenterOfMass(d, e)[0], occ.getCenterOfMass(d, e)[1], occ.getCenterOfMass(d, e)[2]] 
            for (d, e) in ents
        ]
    )
    dmat = distance_matrix(coms, coms)
    ovlp = []
    for j, row in enumerate(dmat):
        curr = [j]
        row = row[j::]
        for l, k in enumerate(row):
            # += Identify Overlap
            if not(k):
                continue
            if np.isclose(coms[j][0], coms[j+l][0]) or np.isclose(coms[j][1], coms[j+l][1]) or np.isclose(coms[j][2], coms[j+l][2]):
                if k <= SARC_RADIUS*2:
                    curr.append(l)
        if len(curr) > 1:
            ovlp.append(curr)
            occ.fragment([(DIM-1, curr[0])], [(DIM-1, i) for i in curr[1::]], removeObject=True, removeTool=True)
            occ.synchronize()

    # += Check on volume entitites to determine if there are overlapping components

    def check_intersect(obj, too):
        try:
            intscpt = occ.intersect(obj, too, tag=-1, removeObject=False, removeTool=False)
            occ.synchronize()
        except:
            return False, None
        if intscpt[0]:
            return True, intscpt
        else:
            return False, None
        
    def remove_overlaps(intscpt, dim):
        occ.synchronize()
        try:
            occ.fragment([(dim, intscpt[0][0][1])], [(dim, i[0][1]) for i in intscpt[1::]], removeObject=True, removeTool=True)
        except:
            return
        occ.synchronize()

    for d in range(1, DIM, 1):
        ents = occ.get_entities(dim=d)
        ovlp_srf = []
        for j, obj in enumerate(ents):
            curr = [[obj]]
            next = ents[j+1::]
            for l, too in enumerate(next):
                check_intscpt, intscpt = check_intersect([obj], [too])
                if check_intscpt:
                    curr.append([too])
            if len(curr) > 1:
                remove_overlaps(curr, d)
                ovlp_srf.append(curr)
        
        for row in ovlp_srf:
            for col in row:
                occ.remove(dimTags=col, recursive=True)
                occ.synchronize()

    ent = occ.get_entities()
    centre = np.array([DIS_X/2, DIS_Y/2, DIS_Z/2])
    for (d, e) in ent:
        if d == DIM-DIM:
            msh.set_size([(d, e)], 10)
        if d == DIM-1: 
            sf_tgs.append(e)
            com = np.array([x for x in occ.getCenterOfMass(d, e)])
            for row in cenn_al:
                if np.allclose(row, com):
                    mdl.addPhysicalGroup(dim=d, tags=[e], name="SURF_" + "ZDisc" + str(e))
                    occ.synchronize()
                    break
            if e in [XX[0], XX[1], YY[0], YY[1], ZZ[0], ZZ[1]]:
                continue
            mdl.addPhysicalGroup(dim=d, tags=[e], name="SURF_" + "ABand" + str(e))
            occ.synchronize()
        if d == DIM: 
            vo_tgs.append(e)
            com = np.array([x for x in occ.getCenterOfMass(d, e)])
            if np.allclose(centre, com):
                mdl.addPhysicalGroup(dim=d, tags=[e], name="VOLUME_" + "Cytosol")
            else: 
                mdl.addPhysicalGroup(dim=d, tags=[e], name="VOLUME_" + "Myofibril_" + str(e))
            occ.synchronize()

    # +z
    # mdl.addPhysicalGroup(dim=3, tags=[c[0][1]], name="VOLUME_" + "Myofibril_" + str(e))
 
    # # += Fragment the existing surfaces into new surfaces that are joined at nodes
    # occ.fragment([(3, VOLUME)], [(3, i) for i in volume_tgs], removeObject=True, removeTool=True)
    # occ.synchronize()
    # surfs = occ.get_entities(dim=DIM-1)
    # occ.fragment([(2, 101)], [(2, i[1]) for i in surfs], removeObject=True, removeTool=False)
    # occ.synchronize()

    # # += Create points from centroids
    # for i, (x, y, z) in enumerate(cenn_al):
    #     pt = occ.addPoint(x=x, y=y, z=z, meshSize=1, tag=max(pt_tgs)+100)
    #     pt_tgs.append(pt)
    #     cent_tgs.append(pt)
    # occ.synchronize()
    # # += Create lines from connection matrix
    # for j, row in enumerate(cmat):
    #     row = row[j::]
    #     for l, k in enumerate(row):
    #         if k:
    #             cv = occ.addLine(
    #                 startTag=cent_tgs[j], endTag=cent_tgs[l+j], 
    #                 tag=max(max(sf_tgs), max(cv_tgs)) + 100
    #             )
    #             cv_tgs.append(cv)
    #             br_tgs.append(cv)
    #             centroids = np.array([cenn[j], cenn[j+l]])
    #             if cenn[j][1] == cenn[j+l][1]:
    #                 av = np.mean(centroids, axis=0)
    #                 even_cents.append(av)
    #             if cenn[j][1] > cenn[j+l][1]:
    #                 av = np.mean(centroids, axis=0)
    #                 lower_cents.append(av)
    #             if cenn[j][1] < cenn[j+l][1]:
    #                 av = np.mean(centroids, axis=0)
    #                 upper_cents.append(av)
    # occ.synchronize()

    # # += Mark Physical groups
    # even, upper, lower, remain = [], [], [], []
    # for surface in mdl.getEntities(dim=2):
    #     com = occ.getCenterOfMass(surface[0], surface[1])
    #     ev_check = [np.allclose(com, x) for x in even_cents]
    #     lw_check = [np.allclose(com, x) for x in lower_cents]
    #     up_check = [np.allclose(com, x) for x in upper_cents]
    #     # += MANUALLY SUPPLY LABELS
    #     print("     += DECLARE SUFRACE FOR {} @ COM: {}".format(test_name, com))
    #     dec = int(input(" ~+ Enter: "))
    #     if dec == 0:
    #         even.append(surface[1])
    #         continue
    #     if dec == 1:
    #         lower.append(surface[1])
    #         continue
    #     if dec == 2:
    #         upper.append(surface[1])
    #         continue
    #     else:
    #         remain.append(surface[1])
    # mdl.addPhysicalGroup(2, even, MYO_STRAIGHT, "Straight Sarcomere")
    # mdl.addPhysicalGroup(2, lower, MYO_DNANGLED, "Sarcomere Angled Down")
    # mdl.addPhysicalGroup(2, upper, MYO_UPANGLED, "Sarcomere Angled Up")
    # mdl.addPhysicalGroup(2, remain, CYTOSOL, "Cytosol")
    # # += Mark right and left
    # left, right = [], []
    # for line in mdl.getEntities(dim=1):
    #     com = occ.getCenterOfMass(line[0], line[1])
    #     if np.isclose(com[0], 0):
    #         left.append(line[1])
    #         continue
    #     if np.isclose(com[0], 1):
    #         right.append(line[1])
    # mdl.addPhysicalGroup(1, left, LEFT_SIDE, "X0")
    # mdl.addPhysicalGroup(1, right, RIGHT_SIDE, "X1")

    # # +==+==+
    # # Create Mesh fields
    # # += Crete distance field to begin mesh generation
    # msh.field.add("Distance", DIST_TAG)
    # # msh.field.setNumbers(DIST_TAG, "PointsList", cent_tgs)
    # msh.field.setNumbers(DIST_TAG, "CurvesList", cv_tgs)
    # msh.field.setNumber(1, "Sampling", 100)
    # # += Create threshold field
    # msh.field.add("Threshold", THRE_TAG)
    # # msh.field.setNumber(THRE_TAG, "Sigmoid", True)
    # msh.field.setNumber(THRE_TAG, "InField", DIST_TAG)
    # msh.field.setNumber(THRE_TAG, "SizeMin", 0.5)
    # msh.field.setNumber(THRE_TAG, "SizeMax", 1)
    # msh.field.setNumber(THRE_TAG, "DistMin", 0.05)
    # msh.field.setNumber(THRE_TAG, "DistMax", 1)
    # # += Create minimum field
    # msh.field.add("Min", MINF_TAG)
    # msh.field.setNumbers(MINF_TAG, "FieldsList", [THRE_TAG])
    # # += Set min field as background mesh
    # msh.field.setAsBackgroundMesh(MINF_TAG)
    # # += Set and Stabalise meshing
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


    # += set "Delaunay" algorithm for meshing due to complex gradients
    occ.synchronize()
    # gmsh.option.setNumber("Mesh.Algorithm", value=DELAUNAY)

    # += Generate Mesh
    msh.generate(dim=DIM)
    msh.setOrder(order=ORDER)
    # += Write File
    gmsh.write("P_Branch_Contraction/gmsh_msh/SUB_RED_LESS" + test_name + ".msh")
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
    # test = [
    #     "SINGLE_MIDDLE", "SINGLE_DOUBLE", "SINGLE_ACROSS", 
    #     "DOUBLE_ACROSS", "BRANCH_ACROSS", "BRANCH_MIDDLE", 
    #     "TRANSFER_DOUBLE"
    # ]
    test = [
        "28"
    ]
    test_name = ["HEX_XX_" + x for x in test]
    # += Cases
    test_case = list(range(0, len(test_name), 1))
    # test_case = list(range(0, 1, 1))
    # += Feed Main()
    main(test_name, test_case)