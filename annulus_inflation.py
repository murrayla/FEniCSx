# +==+===+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 01/12/2023
# +==+===+==+==+==+

# +==+===+==+==+
# Annulus Inflation
#   Testing inflation of annulus for comparison with Nash Thesis data and
#   comparison with OpenCMISS. 
# +==+===+==+==+

# +==+==+
# Setup
# += Imports
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import argparse
import meshio
import gmsh
import ufl
# += Parameters
MESH_DIM = 3
LEN_Z = 1
R0 = 1
R1 = 1.5
P = 1.5

# +==+==+==+
# gen_annulus Function 
# Description:
#   Applies gmsh API to generate an annulus
# Inputs:
#   test_name: title of output files
#   test_type: indicator for tetrahedral or hexagonal generation
#   test_order: order of generated mesh
#   refine_check: check for refinement
# Outputs:
#   "test_name".msh file for reading into main script
# +==+==+==+
def gen_annulus(test_name, test_type, test_order, refine_check):
    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(test_name)
    # += Setup base of annulus, surface
    innerCircle = gmsh.model.occ.addCircle(x=0, y=0, z=0, r=R0, tag=1)
    outerCircle = gmsh.model.occ.addCircle(x=0, y=0, z=0, r=R1, tag=2)
    innerCircleCurve = gmsh.model.occ.addCurveLoop([innerCircle], 1)
    outerCircleCurve = gmsh.model.occ.addCurveLoop([outerCircle], 2)
    baseSurface = gmsh.model.occ.addPlaneSurface(wireTags=[outerCircle, innerCircle], tag=1)
    # += Synchronize and add physical group
    basePhysicalGroup = gmsh.model.addPhysicalGroup(dim=2, tags=[baseSurface], tag=100, name="Annulus Base Surface")
    gmsh.model.occ.synchronize()
    # += Extrude from geometry
    baseExtrusion = gmsh.model.occ.extrude(dimTags=[(2, baseSurface)], dx=0, dy=0, dz=LEN_Z)
    gmsh.model.occ.synchronize()
    # += Create Physical Group on volume
    volumePhysicalGroup = gmsh.model.addPhysicalGroup(3, [baseExtrusion[1][1]], tag=1000, name="Internal Volume")
    innerPhysicalGroup = gmsh.model.addPhysicalGroup(2, [baseExtrusion[3][1]], tag =101, name="Inner Surface")
    outerPhysicalGroup = gmsh.model.addPhysicalGroup(2, [baseExtrusion[2][1]], tag =102, name="Outer Surface")
    topPhysicalGroup = gmsh.model.addPhysicalGroup(2, [baseExtrusion[0][1]], tag =103, name="Annulus Top Surface")
    # += Create Mesh
    gmsh.model.mesh.generate(MESH_DIM)
    if refine_check == "True":
        gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(test_order)
    # += Write File
    gmsh.write("gmsh_msh/" + test_name + ".msh")
    gmsh.finalize()

# +==+==+==+
# main Function 
# Description:
#   Runs main computation 
# Inputs:
#   test_name: title of output files
#   test_type: indicator for tetrahedral or hexagonal generation
#   test_order: order of generated mesh
#   refine_check: check for refinement
# Outputs:
#   test_name.bp folder for visualisation in paraview.
# +==+==+==+
def main(test_name, test_type, test_order, refine_check):
    # +==+==+
    # Mesh Generation
    gen_annulus(test_name, test_type, test_order, refine_check)

    # +==+==+
    # Load Domain & Interpolation
    # += Read .msh into domain for FEniCSx
    #    (1): File name .msh
    #    (2): Multiprocessing assignment
    #    (3): Rank of multiprocessing
    #    (4): Dimension of mesh
    domain, _, _ = io.gmshio.read_from_msh("gmsh_msh/testCone.msh", MPI.COMM_WORLD, 0, gdim=MESH_DIM)
    # += Create Vector Element
    #    (1): Interpolation style
    #    (2): Cell from mesh domain
    #    (3): Degree of interpolation style
    fe_interp = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree=MESH_DIM-1)  
    # += Create Function Space
    #    (1): Mesh domain space
    #    (2): Finite element setup
    # += Geometry Space
    func_space = fem.FunctionSpace(domain, fe_interp)

    # +==+==+
    # Boundary Conditions
    # += Base Dirichlet BCs
    #    (1): Interpolation space
    #    (2): Internal function to determine if on z = 0
    base_dofs = fem.locate_dofs_geometrical(func_space, lambda x: np.isclose(x[2], 0))
    # += Set values at the base boundary
    u_base = fem.Function(func_space)
    u_base.interpolate(lambda x: 0)
    # += Set Dirichlet BCs of (1) on (2)
    bc_base = fem.dirichletbc(u_base, base_dofs)
    # += Top Dirichlet BCs
    #    (1): Interpolation space
    #    (2): Internal function to determine if on z = 0
    top_dofs = fem.locate_dofs_geometrical(func_space, lambda x: np.isclose(x[2], LEN_Z))
    # += Set values at the top boundary
    u_top = fem.Function(func_space)
    u_top.interpolate(lambda x: 0)
    # += Set Dirichlet BCs of (1) on (2)
    bc_top = fem.dirichletbc(u_top, top_dofs)
    # += Concatenate boundaries
    bc = [bc_base, bc_top]

    # # +==+==+ 
    # # Pressure Setup
    # # += Pressure expression, contribution of pressure at internal radius
    # p = ufl.exp(('p*x[0]/R', 'p*x[1]/R'), R = R0)
    # # += Pressure Space (decreasing order of interpolation by 1) 
    # pre_interp = fem.FunctionSpace(domain, ("Lagrange", test_order-1))
    # pressure = fem.Function(pre_interp)
    # # +=  Interpolate expression over points
    # pre_expr = fem.Expression(p, pre_interp.element.interpolation_points())
    # pressure.interpolate(pre_expr)

# +==+==+
# Main check for script operation.
#   Will operate argparse to take values from terminal. 
#   Then runs main() program for script execution 
if __name__ == '__main__':
    
    # +==+ Intake Arguments
    argparser = argparse.ArgumentParser("FEniCSz Program for Passive Annulus Inflation")
    # += Name for file writing
    argparser.add_argument("test_ID")
    # += Type of generation, i.e. Tetrahedral or Hexagonal
    argparser.add_argument("gen_type")
    # += Order of generated mesh
    argparser.add_argument("gen_order", type=int)
    # += Refinement level
    argparser.add_argument("refinement", type=bool)
    # += Capture arguments and store accordingly
    args = argparser.parse_args()
    test_name = args.test_ID
    test_type = args.gen_type
    test_order = args.gen_order
    refine_check = args.refinement

    # +==+ Feed Main()
    main(test_name, test_type, test_order, refine_check)
