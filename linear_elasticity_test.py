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
    #   MPI.COMM_WORLD: allowing for the whole mesh to be treated as a unit for parallel processing
    #   (2): array which contains the bottom left and top right of geometry for bounding
    #   (3): length, width, and height broken into number of elements
    #   (4): structure type
    box_mesh = mesh.create_box(
        MPI.COMM_WORLD, 
        [
            np.array([0, 0, 0]), 
            np.array([L, W, W])
        ],
        [20, 6, 6], 
        cell_type=mesh.CellType.hexahedron
    )
    # += Interpolation of mesh 
    #   (1): mesh to interpolate
    #   (2): type of interpolation i.e. (equation, order)
    lagrange_interpolation = fem.FunctionSpace(
        box_mesh, 
        ("Lagrange", 1) 
    )

    # +==+==+ 
    # Setup boundary conditions for cantilever under gravity
    # += Function for identifying the correct nodes
    #   Setup here for a boundary condition at 0
    #   (1): marker
    def clamped_boundary(x):
        return np.isclose(x[0], 0)
    # += Face dimension
    fdim = box_mesh.topology.dim - 1
    # += Determine the relevant nodes
    boundary_facets = mesh.locate_entities_boundary(box_mesh, fdim, clamped_boundary)
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
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(lagrange_interpolation, fdim, boundary_facets), lagrange_interpolation)
    # += Traction
    #   (1): mesh
    #   (2): values for the force, here traction
    traction = fem.Constant(box_mesh, default_scalar_type((0, 0, 0)))
    # += Integration measure, here a boundary integratio, requires Unified Form Language
    #   (1): required measurement, here defining surface
    #   (2): geometry of interest
    ds = ufl.Measure("ds", domain=box_mesh)

    
    # +==+==+ 
    # Setup weak form for solving over domain
    # += Strain definition
    def epsilon(u):
        engineering_Strain = 0.5 * (fem.nabla_grad(u) + fem.nabla_grad(u.T))
        return engineering_Strain
    
    def sigma(u):
        cauchy_stress = (
            LAME_LAMBDA * dfx.fem.tr(epsilon(u)) * dfx.fem.Identity(3)
            + 
            2 * LAME_MU * epsilon(u)
        )
        return cauchy_stress
    
    u_trial = fe.TrialFunction(lagrange_interpolation)
    v_test  = fe.TestFunction(lagrange_interpolation)
    forcing = fe.Constant((0.0, 0.0, - DENSITY * ACCELERATION_DUE_TO_GRAVITY))
    traction = fe.Constant((0.0, 0.0, 0.0))

    weak_form_lhs = fe.inner(sigma(u_trial), epsilon(v_test)) * fe.dx
    weak_form_rhs = (
        fe.dot(forcing, v_test) * fe.dx
        +
        fe.dot(traction, v_test) * fe.ds
    )

    u_solution = fe.Function(lagrange_interpolation)
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        dircichlet_clamped_boundary
    )

    deviatoric_stress_tensor = (
        sigma(u_solution)
        -
        1/3 * fe.tr(sigma(u_solution)) * fe.Identity(3)
    )
    von_Mises_stress = fe.sqrt(3/2 * fe.inner(deviatoric_stress_tensor, deviatoric_stress_tensor))
    lagrange_scalar_space_first_order = fe.FunctionSpace(
        box_mesh,
        "Lagrange",
        1
    )
    von_Mises_stress = fe.project(von_Mises_stress, lagrange_scalar_space_first_order)

    u_solution.rename("Displacement Vector", "")
    von_Mises_stress.rename("von Mises stress", "")

    beam_deflection_file = fe.XDMFFile("beam_deflection.xdmf")
    beam_deflection_file.parameters["flush_output"] = True
    beam_deflection_file.parameters["functions_share_mesh"] = True
    beam_deflection_file.write(u_solution, 0.0)
    beam_deflection_file.write(von_Mises_stress, 0.0)

if __name__ == "__main__":
    main()