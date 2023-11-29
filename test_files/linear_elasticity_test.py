# FEniCSx Cantilever Test

# +==+===+==+==+
# Linear elasticity
# Left hand of beam will be clamped with homgenous Dirchlet boundary conditions
# +==+===+==+==+

# +==+==+==+
# Setup
# += Imports
from mpi4py import MPI
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
import meshio
import gmsh

# += Parameters
L = 1
W = 0.2
MU = 1
RHO = 1
DELTA = W / L
GAMMA = 0.4 * DELTA**2
BETA = 1.25
LAMBDA = BETA
G = GAMMA

def create_gmsh_structure():
    gmsh.initialize()
    gmsh.model.add("testCylinder")
    # += Setup base of cylinder
    baseCircle = gmsh.model.occ.addCircle(x=0, y=0, z=0, r=0.5, tag=1)
    baseCircleCurve = gmsh.model.occ.addCurveLoop([baseCircle], 1)
    gmsh.model.occ.synchronize()
    # += Setup middle of cylinder
    midCircle = gmsh.model.occ.addCircle(x=0, y=0, z=1, r=0.4, tag=2)
    midCircleCurve = gmsh.model.occ.addCurveLoop([midCircle], 2)
    gmsh.model.occ.synchronize()
    # += Setup top of cylinder
    topCircle = gmsh.model.occ.addCircle(x=0, y=0, z=2, r=0.3, tag=3)
    topCircleCurve = gmsh.model.occ.addCurveLoop([topCircle], 3)
    gmsh.model.occ.synchronize()
    # += Setup wire between circles
    thruVolume = gmsh.model.occ.addThruSections([baseCircle, midCircle, topCircle], tag=1)
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], tag=1)
    gmsh.model.setPhysicalName(volumes[0][0], 1, "Obj Vol")
    base, wall, top = 2, 3, 4
    base_surf, wall_surf,top_surf = [], [], []
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, 0, 0]):
            base_surf.append(boundary[1])
        elif np.allclose(center_of_mass, [0, 0, 1]):
            wall_surf.append(boundary[1])
        elif np.allclose(center_of_mass, [0, 0, 2]):
            top_surf.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, wall_surf, wall)
    gmsh.model.setPhysicalName(1, wall, "Cylinder Surface")
    gmsh.model.addPhysicalGroup(1, base_surf, base)
    gmsh.model.setPhysicalName(1, base, "Base")
    gmsh.model.addPhysicalGroup(1, top_surf, top)
    gmsh.model.setPhysicalName(1, top, "Top")
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(2)
    gmsh.write("testCylinder.msh")
    gmsh.finalize()

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh

# +==+==+==+
# Main Function for running computation
# +==+==+==+
def main():
    create_gmsh_structure()
    gmsh_mesh, cell_markers, facet_markers = io.gmshio.read_from_msh("testCylinder.msh", MPI.COMM_WORLD, gdim=3)
    print("It Loaded?")
    # gmsh_mesh = meshio.read("gmsh_files/myo_SEM0.msh")
    # tetra_mesh = create_mesh(gmsh_mesh, "tetrahedron10", False)
    # meshio.write("mesh.xdmf", tetra_mesh)
    # +==+==+ 
    # Setup geometry for solution
    # += Setup generated mesh for problem
    #   (1): allowing for the whole mesh to be treated as a unit for parallel processing
    #   (2): array which contains the bottom left and top right of geometry for bounding
    #   (3): length, width, and height broken into number of elements
    #   (4): structure type
    domain = mesh.create_box(
        MPI.COMM_WORLD, 
        [np.array([0, 0, 0]), np.array([L, W, W])],
        [20, 6, 6], 
        cell_type=mesh.CellType.tetrahedron
    )
    # += Interpolation of mesh 
    #   (1): mesh to interpolate
    #   (2): type of interpolation i.e. (equation, order)
    # V = fem.FunctionSpace(domain, ("CG", 2))      
    vV = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree=2)  
    # vu = fem.Function(V, name="u")
    V = fem.FunctionSpace(domain, vV)    
    
    # +==+==+ 
    # Setup boundary conditions for cantilever under gravity
    # += Function for identifying the correct nodes
    #   Setup here for a boundary condition at 0
    #   (1): marker
    def clamped_boundary(x):
        return np.isclose(x[0], 0)
    # += Face dimension
    fdim = domain.topology.dim - 1
    # += Determine the relevant nodes
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
    # += Set the conditions at the boundary
    #   0s indicating a locted boundary
    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    # += Implement Dirichlet 
    #   (1): values for condtition
    #   (2): DOF identifier 
    #       (1): interpolation scheme
    #       (2): dimensions of DOFs relevant
    #       (3): indices of the nodes that fit the requirement
    #   (3): interpolation scheme
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    # += Traction
    #   (1): mesh
    #   (2): values for the force, here traction
    traction = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    # += Integration measure, here a boundary integratio, requires Unified Form Language
    #   (1): required measurement, here defining surface
    #   (2): geometry of interest
    ds = ufl.Measure("ds", domain=domain)

    # +==+==+ 
    # Setup weak form for solving over domain
    # += Strain definition
    def epsilon(u):
        eng_strain = 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        return eng_strain
    # += Stress definition
    def sigma(u):
        cauchy_stress = LAMBDA * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * MU * epsilon(u)
        return cauchy_stress
    # += Trial functions for weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # += Force term, volumetric gravity over non-bound side
    f = fem.Constant(domain, default_scalar_type((0, 0, -RHO * G)))
    # += Setup of integral term for inner product of cauchy stress and derivative of displacement
    lhs = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    # += Setup of force term over volume and surface term of traction
    rhs = ufl.dot(f, v) * ufl.dx + ufl.dot(traction, v) * ds

    # +==+==+ 
    # Setup problem solver
    # += Define problem parameters
    problem = LinearProblem(lhs, rhs, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    # += Solve
    uh = problem.solve()

    # +==+==+
    # ParaView export
    # with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(domain)
    #     uh.name = "Deformation"
    #     xdmf.write_function(uh)
    with io.VTXWriter(MPI.COMM_WORLD, "deformation.bp", [uh], engine="BP4") as vtx:
        vtx.write(0.0)

    s = sigma(uh) - 1. / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
    von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))  

    V_von_mises = fem.FunctionSpace(domain, ("DG", 0))
    stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
    stresses = fem.Function(V_von_mises)
    stresses.interpolate(stress_expr)

if __name__ == "__main__":
    main()