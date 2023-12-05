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
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from numpy import linalg as LNG
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
    domain, _, facet_markers = io.gmshio.read_from_msh("gmsh_msh/testCone.msh", MPI.COMM_WORLD, 0, gdim=MESH_DIM)
    # += Create Vector Element
    #    (1): Interpolation style
    #    (2): Cell from mesh domain
    #    (3): Degree of interpolation style
    element = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree=MESH_DIM-1)  
    # += Create Function Space
    #    (1): Mesh domain space
    #    (2): Finite element setup
    # += Geometry Space
    V = fem.FunctionSpace(domain, element)

    # +==+==+
    # Determine Boundaries
    #    (1): Base
    base_facets = mesh.locate_entities_boundary(domain, MESH_DIM-1, lambda x: np.isclose(x[2], 0))
    #    (2): Top
    top_facets = mesh.locate_entities_boundary(domain, MESH_DIM-1, lambda x: np.isclose(x[2], LEN_Z))
    #    (3): Inner Cylinder Surface
    def inner_bound(x):
        return np.isclose(np.sqrt(x[0]**2+x[1]**2), R0)
    inner_facets = mesh.locate_entities_boundary(domain, MESH_DIM-1, inner_bound)
    #    (4): Outer Cylinder Surface
    def outer_bound(x):
        return np.isclose(np.sqrt(x[0]**2+x[1]**2), R1)
    outer_facets = mesh.locate_entities_boundary(domain, MESH_DIM-1, outer_bound)
    # += Collate Marked boundaries
    marked_facets = np.hstack([base_facets, top_facets, inner_facets, outer_facets])
    # += Assign boundaries IDs
    marked_values = np.hstack([
        np.full_like(base_facets, 1), 
        np.full_like(top_facets, 2),
        np.full_like(inner_facets, 3), 
        np.full_like(outer_facets, 4)
    ])
    # += Sort and assign
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(domain, MESH_DIM-1, marked_facets[sorted_facets], marked_values[sorted_facets])

    # +==+==+
    # Boundary Conditions
    # += Base Dirichlet BCs but broken per subspace
    #    (1): Interpolation space
    #    (2): Internal function to determine if on z = 0
    # base_dofs_x = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(1))
    # base_dofs_y = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(1))
    base_dofs_z = fem.locate_dofs_topological(V.sub(2), facet_tag.dim, facet_tag.find(1))
    # += Set Dirichlet BCs of (1) on (2)
    bc_base = fem.dirichletbc(default_scalar_type(0), base_dofs_z, V.sub(2))
    # += Top Dirichlet BCs
    #    (1): Interpolation space
    #    (2): Internal function to determine if on z = 0
    top_dofs_z = fem.locate_dofs_topological(V.sub(2), facet_tag.dim, facet_tag.find(2))
    # += Set Dirichlet BCs of (1) on (2)
    bc_top = fem.dirichletbc(default_scalar_type(0.1), top_dofs_z, V.sub(2))
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
    # Setup Parameteres for Variational Equation
    # += Body Forces
    B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    # += Traction Forces (nominal Piola-Kirchoff)
    T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    # += Test and Trial Functions
    v = ufl.TestFunction(V)
    u = fem.Function(V)
    # += Identity Tensor
    I = ufl.variable(ufl.Identity(MESH_DIM))
    # += Deformation Gradient Tensor
    #    F = ∂u/∂X + I
    F = ufl.variable(I + ufl.grad(u))
    # += Right Cauchy-Green Tensor
    C = ufl.variable(F.T * F)
    # += Invariants
    #    (1): λ1^2 + λ2^2 + λ3^2; tr(C)
    Ic = ufl.variable(ufl.tr(C))
    #    (2): λ1^2*λ2^2 + λ2^2*λ3^2 + λ3^2*λ1^2; 0.5*[(tr(C)^2 - tr(C^2)]
    # IIc = ufl.variable(0.5 * [(ufl.tr(C)**2 - ufl.tr(C**2))])
    #    (3): λ1^2*λ2^2*λ3^2; det(C) = J^2
    J = ufl.variable(ufl.det(F))
    # += Material Parameters
    E = default_scalar_type(1.0e4)
    nu = default_scalar_type(0.3)
    #    (1): μ = E/(2*(1+v))
    mu = fem.Constant(domain, E / (2 * (1 + nu)))
    #    (2): λ = E * v/((1+v)(1-2v))
    lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
    # += Strain Energy Density (compressible neo-Hookean model)
    #    (1): ψ = μ/2 * (Ic - 2) - μ * log(J) + λ/2 * log(J)^2
    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
    # += First Piola Stress
    piola = ufl.diff(psi, F)

    # +==+==+
    # Setup Variational Problem Solver
    # += Gaussian Quadrature
    metadata = {"quadrature_degree": 4}
    # += Domains of integration
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    # += Residual Equation (Variational, for solving)
    R = ufl.inner(ufl.grad(v), piola) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)
    # += Problem Setup and Solver
    problem = NonlinearProblem(R, u, bc)
    solver = NewtonSolver(domain.comm, problem)
    # += Tolerances for convergence
    solver.atol = 1e-8
    solver.rtol = 1e-8
    # += Convergence criteria
    solver.convergence_criterion = "incremental"

    num_its, converged = solver.solve(u)
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged.")

    # +==+==+
    # ParaView export
    with io.VTXWriter(MPI.COMM_WORLD, "deformation.bp", [u], engine="BP4") as vtx:
        vtx.write(0.0)

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
