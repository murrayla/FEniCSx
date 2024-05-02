# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 30/04/2024
# Code: testCubeGeneration.py
#   Generation of .msh cubes with appropriate refinements.
# +==+==+==+==+

# +==+==+ Setup
# += Dependencies
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

# +==+==+==+
# Gmsh Cube:
#   Input of test_name and required graph network.
#   Output cube with refined generations.
def gmsh_cube(test_name, data):
    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(test_name)

    # += Store entitu tags and assign physical group labels.
    pt_tgs = []
    cv_tgs = []
    sf_tgs = []
    vo_tgs = []

    # += Create Points 
    p_coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
    for i, (x, y) in enumerate(p_coords):
        pt = gmsh.model.occ.addPoint(x=x, y=y, z=0, meshSize=0.5, tag=i+1)
        pt_tgs.append(pt)
    # += Create Lines 
    c_edge = [[1, 2], [2, 3], [3, 4], [4, 1]]
    for i, (x, y) in enumerate(c_edge):
        cv = gmsh.model.occ.addLine(startTag=x, endTag=y, tag=i+1)
        cv_tgs.append(cv)
    # += Create Surface
    gmsh.model.occ.addCurveLoop([1, 2, 3, 4], 5)
    gmsh.model.occ.addPlaneSurface([5], 6)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(dim=2, tags=[6], name="N_ZZ")

    # gmsh.model.occ.addPoint(0, 0.35, 0, tag=5)
    # gmsh.model.occ.addPoint(1, 0.35, 0, tag=6)
    # gmsh.model.occ.addLine(5, 6, 100)

    # gmsh.model.occ.addPoint(0, 0.65, 0, tag=7)
    # gmsh.model.occ.addPoint(1, 0.65, 0, tag=8)
    # gmsh.model.occ.addLine(7, 8, 101)

    # # gmsh.model.occ.addPoint(0.25, 0.35, 0, tag=9)
    # # gmsh.model.occ.addPoint(0.75, 0.65, 0, tag=10)
    # # gmsh.model.occ.addLine(9, 10, 102)
    # gmsh.model.occ.synchronize()

    # +==+==+
    # Create Mesh fields
    # += Crete distance field to begin mesh generation
    # gmsh.model.mesh.field.add("Distance", DIST_TAG)
    # gmsh.model.mesh.field.setNumbers(DIST_TAG, "PointsList", [5, 6, 7, 8])
    # gmsh.model.mesh.field.setNumbers(DIST_TAG, "CurvesList", [100, 101])
    # gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    # gmsh.model.occ.addPoint(0, 0.5, 0, tag=9)
    # gmsh.model.occ.addPoint(0.5, 0.5, 0, tag=10)
    # gmsh.model.occ.addLine(9, 10, 100)

    # gmsh.model.occ.addPoint(0, 0.5, 0, tag=9)
    # gmsh.model.occ.addPoint(0.5, 0.5, 0, tag=10)
    # gmsh.model.occ.addLine(9, 10, 100)

    # gmsh.model.occ.addPoint(1, 0.25, 0, tag=11)
    # gmsh.model.occ.addLine(10, 11, 101)

    # gmsh.model.occ.addPoint(1, 0.75, 0, tag=12)
    # gmsh.model.occ.addLine(10, 12, 102)

    gmsh.model.occ.addPoint(0, 0.35, 0, tag=5)
    gmsh.model.occ.addPoint(1, 0.65, 0, tag=6)
    gmsh.model.occ.addLine(5, 6, 100)

    gmsh.model.occ.addPoint(1, 0.35, 0, tag=7)
    gmsh.model.occ.addPoint(0, 0.35, 0, tag=8)
    gmsh.model.occ.addLine(7, 8, 101)
    gmsh.model.occ.synchronize()

    # +==+==+
    # Create Mesh fields
    # += Crete distance field to begin mesh generation
    gmsh.model.mesh.field.add("Distance", DIST_TAG)
    gmsh.model.mesh.field.setNumbers(DIST_TAG, "PointsList", [5, 6, 7, 8])
    gmsh.model.mesh.field.setNumbers(DIST_TAG, "CurvesList", [100, 101])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    # gmsh.model.occ.addPoint(0, 0.35, 0, tag=5)
    # gmsh.model.occ.addPoint(1, 0.65, 0, tag=6)
    # gmsh.model.occ.addLine(5, 6, 100)

    # # +==+==+
    # # Create Mesh fields
    # # += Crete distance field to begin mesh generation
    # gmsh.model.mesh.field.add("Distance", DIST_TAG)
    # gmsh.model.mesh.field.setNumbers(DIST_TAG, "PointsList", [9, 10])
    # gmsh.model.mesh.field.setNumbers(DIST_TAG, "CurvesList", [100])
    # gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

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
    # # += Set and Stabalise meshing
    # # def meshSizeCallback(dim, tag, x, y, z, lc):
    # #     return max(lc, 0.02 * x + 0.01)
    # # gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # # += set "Delaunay" algorithm for meshing due to complex gradients
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", value=DELAUNAY)

    # += Generate Mesh
    gmsh.model.mesh.generate(dim=DIM)
    gmsh.model.mesh.setOrder(order=ORDER)
    # += Write File
    gmsh.write("P_Branch_Contraction/gmsh_msh/" + test_name + ".msh")
    gmsh.finalize()

# +==+==+==+
# Main
def main(test_name):
    gmsh_cube(test_name, None)

# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+==+
    # Test Parameters
    # += Test name
    test_name = "QUAD_XX_BRANCH_ACROSS"
    # += Feed Main()
    main(test_name)