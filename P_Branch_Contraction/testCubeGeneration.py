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
DIM = 3
RAW = 1
REF = 0.01
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

    # += Create cube
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 1)
    # += Identify entities to develop physical regions
    entitites = gmsh.model.occ.get_entities()
    # += Store entitu tags and assign physical group labels.
    pt_tgs = []
    cv_tgs = []
    sf_tgs = []
    vo_tgs = []
    side = ["-N_XX", "N_XX", "-N_YY", "N_YY", "-N_ZZ", "N_ZZ"]

    # +==+==+
    # Loop through each entity and assign tags dependent on value.
    # In case of surfaces and volumes provide physical grouping.
    gmsh.model.occ.synchronize()
    for i, (d, t) in enumerate(entitites):
        if not(d):
            pt_tgs.append(t)
        if d == 1:
            cv_tgs.append(t)
        if d == 2:
            sf_tgs.append(t)
            gmsh.model.addPhysicalGroup(dim=d, tags=[t], name=side.pop(0))
            gmsh.model.occ.synchronize()
        if d == 3:
            vo_tgs.append(t)
            gmsh.model.addPhysicalGroup(dim=d, tags=[t], name="Volume")

    gmsh.model.occ.addPoint(0.5, 1, 0.5, tag=9)
    gmsh.model.occ.addPoint(0.5, 0.5, 0.5, tag=10)
    gmsh.model.occ.addLine(9, 10, 100)

    gmsh.model.occ.addPoint(0.25, 0, 0.5, tag=11)
    gmsh.model.occ.addLine(10, 11, 101)

    gmsh.model.occ.addPoint(0.75, 0, 0.5, tag=12)
    gmsh.model.occ.addLine(10, 12, 102)
    gmsh.model.occ.synchronize()

    # +==+==+
    # Create Mesh fields
    # += Crete distance field to begin mesh generation
    gmsh.model.mesh.field.add("Distance", DIST_TAG)
    gmsh.model.mesh.field.setNumbers(DIST_TAG, "PointsList", [9, 10, 11, 12])
    gmsh.model.mesh.field.setNumbers(DIST_TAG, "CurvesList", [100, 101, 102])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    # += Create threshold field
    gmsh.model.mesh.field.add("Threshold", THRE_TAG)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "InField", DIST_TAG)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "SizeMin", REF)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "SizeMax", RAW)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(THRE_TAG, "DistMax", 0.6)
    # += Create minimum field
    gmsh.model.mesh.field.add("Min", MINF_TAG)
    gmsh.model.mesh.field.setNumbers(MINF_TAG, "FieldsList", [THRE_TAG])
    # += Set min field as background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(MINF_TAG)
    # += Set and Stabalise meshing
    # def meshSizeCallback(dim, tag, x, y, z, lc):
    #     return max(lc, 0.02 * x + 0.01)
    # gmsh.model.mesh.setSizeCallback(meshSizeCallback)
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
    test_name = "CUBE_YY_BRANCH_MIDDLE"
    # += Feed Main()
    main(test_name)