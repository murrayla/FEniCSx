# +==+===+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 01/12/2023
# +==+===+==+==+==+

# +==+===+==+==+
# Linear elasticity
#   Testing beam structures under cantilever deformation
# +==+===+==+==+

# +==+==+==+
# Setup
# += Imports
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import meshio
import gmsh
import ufl
# += Parameters
# Test type
TEST_CASE = 0
# Geometry
L = 1
W = 0.2
MESH_DIM = 3
FACE_DIM = 2
# Material
LAMBDA = 1.25
MU = 1
# Problem
RHO = 1

def create_mesh(msh, cell_type, prune_z=False):
    cells = msh.get_cells_type(cell_type)
    cell_data = msh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=msh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh

# +==+==+==+
# Gmsh Function for constructing simple converging cone mesh
#   testing functionality of physical groups to understand loading into 
#   FEniCSx
# +==+==+==+
def create_gmsh_cone():
    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add("testCone")
    # += Setup base of cylinder
    baseCircle = gmsh.model.occ.addCircle(x=0, y=0, z=0, r=0.5, tag=1)
    baseCircleCurve = gmsh.model.occ.addCurveLoop([baseCircle], 1)
    # += Setup middle of cylinder
    midCircle = gmsh.model.occ.addCircle(x=0.1, y=0.05, z=1, r=0.1, tag=2)
    midCircleCurve = gmsh.model.occ.addCurveLoop([midCircle], 2)
    # += Setup top of cylinder
    topCircle = gmsh.model.occ.addCircle(x=-0.1, y=-0.1, z=2, r=0.3, tag=3)
    topCircleCurve = gmsh.model.occ.addCurveLoop([topCircle], 3)
    # += Setup volume by extruding through sections and syunchronize
    thruVolume = gmsh.model.occ.addThruSections([baseCircle, midCircle, topCircle], tag=1)
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    # += Create Physical Group on volume
    physicalVolume = gmsh.model.addPhysicalGroup(dim=3, tags=[1], tag=2, name="Volume")
    # += Setup boundary surfaces
    base_surf, wall_surf, top_surf = [], [], []
    boundaries = gmsh.model.getBoundary(dimTags=volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, 0, 0]):
            base_surf.append(boundary[1])
        elif np.allclose(center_of_mass, [0, 0, 0.9166666663886719]):
            wall_surf.append(boundary[1])
        elif np.allclose(center_of_mass, [0, 0, 2]):
            top_surf.append(boundary[1])
    physicalCylinderSurface = gmsh.model.addPhysicalGroup(dim=2, tags=wall_surf, tag=2, name="Cylinder Surface")
    physicalBaseSurface = gmsh.model.addPhysicalGroup(dim=2, tags=base_surf, tag=3, name="Base")
    PhysicalTopSurface = gmsh.model.addPhysicalGroup(dim=2, tags=top_surf, tag=4, name="Top")
    # += Create Mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(2)
    # += Write File
    gmsh.write("gmsh_msh/testCone.msh")
    gmsh.finalize()

def create_gmsh_cylinder():
    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add("testCylinder")
    # += Setup base of cylinder, surface
    baseCircle = gmsh.model.occ.addCircle(x=0, y=0, z=0, r=0.5, tag=1)
    baseCircleCurve = gmsh.model.occ.addCurveLoop([baseCircle], 1)
    baseCircleSurface = gmsh.model.occ.addPlaneSurface(wireTags=[baseCircle], tag=1)
    # += Synchronize and add physical group
    gmsh.model.addPhysicalGroup(dim=2, tags=[baseCircleSurface], tag=100, name="lower_surface")
    gmsh.model.occ.synchronize()
    # += Extrude from geometry
    extrusion = gmsh.model.occ.extrude(dimTags=[(2, baseCircleSurface)], dx=0, dy=0, dz=4)
    gmsh.model.occ.synchronize()
    # += Create Physical Group on volume
    volume = gmsh.model.addPhysicalGroup(3, [extrusion[1][1]], name="volume")
    lateral_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[2][1]], tag = 101, name="lateral_surface")
    upper_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[0][1]], tag = 102, name="upper_surface")
    # += Create Mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(2)
    # += Write File
    gmsh.write("gmsh_msh/testCylinder.msh")
    gmsh.finalize()

# +==+==+==+
# Main Function for running computation
# +==+==+==+
def main():
    # +==+==+
    # Allocate meshes and load them for analysis
    # += Simple Cylinder
    if TEST_CASE == 0:
        # += Create mesh
        # create_gmsh_cylinder()
        # += Read .msh into domain for FEniCSx
        #    (1): File name .msh
        #    (2): Multiprocessing assignment
        #    (3): Rank of multiprocessing
        #    (4): Dimension of mesh
        # domain, _, ft = io.gmshio.read_from_msh("gmsh_msh/testCylinder.msh", MPI.COMM_WORLD, 0, gdim=MESH_DIM)
        # ft.name = "Facet markers"
        file = "/Users/murrayla/Documents/main_PhD/P_BranchingPaper/A_Scripts/_meshdata/sarc.msh"
        domain, _, ft = io.gmshio.read_from_msh(file, MPI.COMM_WORLD, 0, gdim=MESH_DIM)
        ft.name = "Facet markers"
    # += Cone shape
    if TEST_CASE == 1:
        # += Create mesh
        create_gmsh_cone()
        # += Read .msh into domain for FEniCSx
        #    (1): File name .msh
        #    (2): Multiprocessing assignment
        #    (3): Rank of multiprocessing
        #    (4): Dimension of mesh
        domain, _, ft = io.gmshio.read_from_msh("gmsh_msh/testCone.msh", MPI.COMM_WORLD, 0, gdim=MESH_DIM)
        ft.name = "Facet markers"

    # +==+==+
    # Interpolation of mesh 
    # += Create Vector Element
    #    (1): Interpolation style
    #    (2): Cell from mesh domain
    #    (3): Degree of interpolation style
    fe_interpolation = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree=2)  
    # += Create Function Space
    #    (1): Mesh domain space
    #    (2): Finite element setup
    V = fem.FunctionSpace(domain, fe_interpolation)    
    
    # +==+==+ 
    # Setup boundary conditions for cantilever under gravity
    # += Function for identifying the correct nodes
    #    Setup here for a boundary condition at 0
    #    (1): marker
    def clamped_boundary(x):
        # += Test what coordinate of the marker is close to desired value
        #    (1): Coordinate value {2 here for Z position}
        #    (2): Desired boundary value
        return np.isclose(x[2], 0)
    # += Determine the relevant nodes
    #    (1): Domain of nodes
    #    (2): Dimension of face
    #    (3): Marker for facets
    boundary_facets = mesh.locate_entities_boundary(mesh=domain, dim=FACE_DIM, marker=clamped_boundary)
    # += Set the conditions at the boundary
    #    0s indicating a locked boundary
    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    # += Implement Dirichlet 
    #    (1): Values for condtition
    #    (2): DOF identifier 
    #         (1): Interpolation scheme
    #         (2): Dimensions of DOFs relevant
    #         (3): Indices of the nodes that fit the requirement
    #    (3): Interpolation scheme
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, FACE_DIM, boundary_facets), V)
    # += Traction
    #    (1): Mesh
    #    (2): Values for the force, here traction
    traction = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    # += Integration measure, here a boundary integratio, requires Unified Form Language
    #    (1): Required measurement, here defining surface
    #    (2): Geometry of interest
    ds = ufl.Measure("ds", domain=domain)

    # +==+==+ 
    # Setup weak form for solving over domain
    # += Strain definition
    def epsilon(u):
        # ε = 0.5 * (∇u + ∇u.T)
        eng_strain = 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        return eng_strain
    # += Stress definition
    def sigma(u):
        # σ = λ * ∇u * I + 2 * μ * ε
        cauchy_stress = LAMBDA * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * MU * epsilon(u)
        return cauchy_stress
    # += Trial and Test functions for weak form
    #    (1): Finite Element interpolation
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # += Force term, volumetric gravity over non-bound side
    #    (1): Mesh domain
    #    (2): Force terms in (x, y, z)
    #         Here using -RHO * K where K is a scaled version of gravity for the geometry
    f = fem.Constant(domain, default_scalar_type((-RHO * 0.01, 0, 0)))
    # += Setup of integral term for inner product of cauchy stress and derivative of displacement
    variational_bilinear = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    # += Setup of force term over volume and surface term of traction
    variational_linear = ufl.dot(f, v) * ufl.dx + ufl.dot(traction, v) * ds

    # +==+==+ 
    # Setup problem solver
    # += Define problem parameters
    #    (1): Bilinear (LHS) term
    #    (2): Linear (RHS) term
    #    (3): Boundary Conditions
    #    (4): Linear Solver parameters:
    #         (1): KSP_TYPE, Krylov solver (iteractive method) - here just preconditioning
    #         (2): PC_TYPE - LU Decomposition
    problem = LinearProblem(
        variational_bilinear, 
        variational_linear, 
        bcs=[bc], 
        petsc_options={
            "ksp_type": "preonly", "pc_type": "lu"
        }
    )
    # += Solve
    uh = problem.solve()

    # +==+==+
    # ParaView export
    with io.VTXWriter(MPI.COMM_WORLD, "deformation.bp", [uh], engine="BP4") as vtx:
        vtx.write(0.0)

if __name__ == "__main__":
    main()