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

# +==+==+==+
# Main Function for running computation
# +==+==+==+
def main():
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
    V = fem.VectorFunctionSpace(domain, ("Lagrange", 2))            
    
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
    with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        uh.name = "Deformation"
        xdmf.write_function(uh)

    s = sigma(uh) - 1. / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
    von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))  

    V_von_mises = fem.FunctionSpace(domain, ("DG", 0))
    stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
    stresses = fem.Function(V_von_mises)
    stresses.interpolate(stress_expr)

if __name__ == "__main__":
    main()