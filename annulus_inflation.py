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
from generateMesh import annulus
from numpy import linalg as LNG
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import gmsh
import ufl
# += Parameters
MESH_DIM = 3
LEN_Z = 1
R0 = 1
R1 = 1.5
P_INNER = -150
P_OUTER = 0
LAMBDA = 0.2 * LEN_Z
ROT_RAD = math.pi/6
FT = {"z0": 1, "z1": 2, "r0": 3, "r1": 4, "volume": 5}
X, Y, Z = 0, 1, 2

# Gmsh Numbering for Hexa-27
#   *  z = 0           z = 0.5         z = 1    
#   *  3--13--2     * 15--24--14    *  7--19--6      
#   *  |      |     *  |      |     *  |      |       
#   *  9  20  11    * 22  26  23    * 17  25  18     
#   *  |      |     *  |      |     *  |      |     
#   *  0-- 8--1     * 10--21--12    *  4--16--5   
# 
# Vijay Numbering for Hexa-27
#   *  z = 0           z = 0.5         z = 1    
#   *  0-- 9--18    *  1--10--19    *  2--11--20     
#   *  |      |     *  |      |     *  |      |       
#   *  3  12  21    *  4  13  22    *  5  14  23     
#   *  |      |     *  |      |     *  |      |     
#   *  6--15--24    *  7--16--25    *  8--17--26     
# 
GEN2MSH = [
    6, 24, 18, 0, 8, 26, 20, 2, 
    15, 3, 7, 21, 25, 9, 19, 1,
    17, 5, 23, 11, 12, 16, 4, 22, 10,
    14, 13
]  

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
    if test_type == "Hexa":
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.add(test_name)
    # += Setup base of annulus, surface
    innerCircle = gmsh.model.occ.addCircle(x=0, y=0, z=0, r=R0, tag=FT["r0"])
    outerCircle = gmsh.model.occ.addCircle(x=0, y=0, z=0, r=R1, tag=FT["r1"])
    innerCircleCurve = gmsh.model.occ.addCurveLoop(curveTags=[innerCircle], tag=FT["r0"])
    outerCircleCurve = gmsh.model.occ.addCurveLoop(curveTags=[outerCircle], tag=FT["r1"])
    baseSurface = gmsh.model.occ.addPlaneSurface(wireTags=[outerCircle, innerCircle], tag=FT["z0"])
    gmsh.model.occ.synchronize()
    # += Extrude from geometry
    baseExtrusion = gmsh.model.occ.extrude(dimTags=[(2, baseSurface)], dx=0, dy=0, dz=LEN_Z)
    gmsh.model.occ.synchronize()
    # # += Create Physical Group on volume
    basePhysicalGroup = gmsh.model.addPhysicalGroup(dim=2, tags=[baseSurface], tag=FT["z0"], name="Base Surface(1)")
    topPhysicalGroup = gmsh.model.addPhysicalGroup(dim=2, tags=[baseExtrusion[0][1]], tag=FT["z1"], name="Top Surface (2)")
    innerPhysicalGroup = gmsh.model.addPhysicalGroup(dim=2, tags=[baseExtrusion[3][1]], tag=FT["r0"], name="Inner Surface (3)")
    outerPhysicalGroup = gmsh.model.addPhysicalGroup(dim=2, tags=[baseExtrusion[2][1]], tag=FT["r1"], name="Outer Surface (4)")
    volumePhysicalGroup = gmsh.model.addPhysicalGroup(dim=3, tags=[baseExtrusion[1][1]], tag=FT["volume"], name="Volume (5)")
    # += Create Mesh
    gmsh.model.mesh.generate(MESH_DIM)
    if refine_check == "True":
        gmsh.model.mesh.refine()
    
    gmsh.model.mesh.setOrder(test_order)
    # += Write File
    gmsh.write("gmsh_msh/" + test_name + ".msh")
    gmsh.finalize()

def plot_vals():
    fig, ax = plt.subplots(3, 2)
    # += Plot 1
    ax[0, 0].set_title('(a) $\\sigma^{(rr)}$')
    ax[0, 0].set(xlabel='Underformed Radial Position \n (\% of wall thickness)', ylabel='Stress (kPA)')
    # += Plot 2
    ax[0, 1].set_title('(b) $\\sigma^{(\\theta \\theta)}$')
    ax[0, 1].set(xlabel='Underformed Radial Position \n (\% of wall thickness)', ylabel='Stress (kPA)')
    # += Plot 3
    ax[1, 0].set_title('(c) $\\sigma^{(\\theta r)}$')
    ax[1, 0].set(xlabel='Underformed Radial Position \n (\% of wall thickness)', ylabel='Stress (kPA)')
    # += Plot 4
    ax[1, 1].set_title('(d) $\\sigma^{(zz)}$')
    ax[1, 1].set(xlabel='Underformed Radial Position \n (\% of wall thickness)', ylabel='Stress (kPA)')
    # += Plot 5
    ax[2, 0].set_title('(e) Hydrostatic Pressure')
    ax[2, 0].set(xlabel='Underformed Radial Position \n (\% of wall thickness)', ylabel='(kPA)')
    # += Plot 6
    ax[2, 1].set_title('(f) $I_{3}$')
    ax[2, 1].set(xlabel='Underformed Radial Position \n (\% of wall thickness)')
    
    plt.show()

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
    if test_type == 0 or test_type == 1:
        gen_annulus(test_name, test_type, test_order, refine_check)
    elif test_type == 2:
        ref_l = [[1,4,1],[1,8,1],[2,4,1],[2,8,1],[2,8,2],[4,16,2],[4,16,8],[4,32,16],[8,64,16]]
        l = 0
        node_list, e_assign = annulus(R0, R1, LEN_Z, ref_l[l][0], ref_l[l][1], ref_l[l][2], test_order)
        hexa_points = np.array(node_list, dtype=np.float64)
        hexahedrals = np.array(e_assign[:, GEN2MSH], dtype=np.int64)
        

    # +==+==+
    # Load Domain & Interpolation
    # += Read .msh into domain for FEniCSx
    if test_type == 0 or test_type == 1:
        domain, _, facet_markers = io.gmshio.read_from_msh("gmsh_msh/" + test_name + ".msh", MPI.COMM_WORLD, 0, gdim=MESH_DIM)
    elif test_type == 2:
        domain = ufl.Mesh(ufl.VectorElement("CG", ufl.hexahedron, 2))
        quad_mesh = mesh.create_mesh(
            MPI.COMM_WORLD, hexahedrals, hexa_points, domain
        )

    fdim = MESH_DIM-1
    # += Assign facets for key surfaces
    z0_facets = facet_markers.find(FT["z0"])
    z1_facets = facet_markers.find(FT["z1"])
    r0_facets = facet_markers.find(FT["r0"])
    r1_facets = facet_markers.find(FT["r1"])
    
    # += Create Vector Element
    #    (1): Continuous lagrange interpolation of order 2 for geometry
    VE2 = ufl.VectorElement("CG", domain.ufl_cell(), degree=2)  
    # += Create Finite Element
    #    (1): Continuous lagrange interpolation of order 1 for pressure
    FE1 = ufl.FiniteElement("CG", domain.ufl_cell(), degree=1)  
    # += Create Mixed Function Space
    #    (1): Mesh domain space
    #    (2): Mixed element space
    W = fem.FunctionSpace(domain, ufl.MixedElement([VE2, FE1]))
    # += Create finite element function
    w = fem.Function(W)
    # += Extract geometry function space and corresponding subdomains
    V, _ = W.sub(0).collapse()
    Vx, _ = V.sub(X).collapse()
    Vy, _ = V.sub(Y).collapse()
    Vz, _ = V.sub(Z).collapse()

    # += Collate Marked boundaries
    marked_facets = np.hstack([z0_facets, z1_facets, r0_facets, r1_facets])
    # += Assign boundaries IDs in stack
    marked_values = np.hstack([
        np.full_like(z0_facets, FT["z0"]), 
        np.full_like(z1_facets, FT["z1"]),
        np.full_like(r0_facets, FT["r0"]), 
        np.full_like(r1_facets, FT["r1"])
    ])
    # += Sort and assign all tags
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

    def rot_x(x):
        rot = x[0]*math.cos(ROT_RAD) - x[1]*math.sin(ROT_RAD)
        return rot - x[0]
    
    def rot_y(x):
        rot = x[0]*math.sin(ROT_RAD) + x[1]*math.cos(ROT_RAD) 
        return rot - x[1]

    # +==+==+
    # BC: Base [Z0]
    # += Locate subdomain dofs
    z0_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), facet_tag.dim, z0_facets)
    z0_dofs_y = fem.locate_dofs_topological((W.sub(0).sub(Y), Vy), facet_tag.dim, z0_facets)
    z0_dofs_z = fem.locate_dofs_topological((W.sub(0).sub(Z), Vz), facet_tag.dim, z0_facets)
    # += Set rules for interpolation
    u0_x = lambda x: np.full(x.shape[1], default_scalar_type(0.0))
    u0_y = lambda x: np.full(x.shape[1], default_scalar_type(0.0))
    u0_z = lambda x: np.full(x.shape[1], default_scalar_type(0.0))
    # += Interpolate 
    u0_bc_x = fem.Function(Vx)
    u0_bc_x.interpolate(u0_x)
    u0_bc_y = fem.Function(Vy)
    u0_bc_y.interpolate(u0_y)
    u0_bc_z = fem.Function(Vz)
    u0_bc_z.interpolate(u0_z)
    # += Create Dirichlet over subdomains
    bc_z0_x = fem.dirichletbc(u0_bc_x, z0_dofs_x, W.sub(0).sub(X))
    bc_z0_y = fem.dirichletbc(u0_bc_y, z0_dofs_y, W.sub(0).sub(Y))
    bc_z0_z = fem.dirichletbc(u0_bc_z, z0_dofs_z, W.sub(0).sub(Z))

    # +==+==+
    # BC: Top [Z1]
    # += Locate subdomain dofs
    # z1_dof = fem.locate_dofs_topological((W.sub(0), V), facet_tag.dim, z1_facets)
    z1_dofs_x = fem.locate_dofs_topological((W.sub(0).sub(X), Vx), facet_tag.dim, z1_facets)
    z1_dofs_y = fem.locate_dofs_topological((W.sub(0).sub(Y), Vy), facet_tag.dim, z1_facets)
    z1_dofs_z = fem.locate_dofs_topological((W.sub(0).sub(Z), Vz), facet_tag.dim, z1_facets)
    # += Set rules for interpolation
    u1_x = rot_x 
    u1_y = rot_y 
    u1_z = lambda x: np.full(x.shape[1], default_scalar_type(LAMBDA))
    # += Interpolate 
    u1_bc_x = fem.Function(Vx)
    u1_bc_x.interpolate(u1_x)
    u1_bc_y = fem.Function(Vy)
    u1_bc_y.interpolate(u1_y)
    u1_bc_z = fem.Function(Vz)
    u1_bc_z.interpolate(u1_z)
    # += Create Dirichlet over subdomains
    bc_z1_x = fem.dirichletbc(u1_bc_x, z1_dofs_x, W.sub(0).sub(X))
    bc_z1_y = fem.dirichletbc(u1_bc_y, z1_dofs_y, W.sub(0).sub(Y))
    bc_z1_z = fem.dirichletbc(u1_bc_z, z1_dofs_z, W.sub(0).sub(Z))

    # +==+==+
    # Concatenate BCs
    bc = [bc_z0_x, bc_z0_y, bc_z0_z, bc_z1_x, bc_z1_y, bc_z1_z]

    # # +==+==+ 
    # # Pressure Setup
    p_inner = fem.Constant(domain, default_scalar_type(P_INNER))
    p_outer = fem.Constant(domain, default_scalar_type(P_OUTER))
    n = ufl.FacetNormal(domain)

    # +==+==+
    # Setup Parameteres for Variational Equation
    # += Test and Trial Functions
    v, q = ufl.TestFunctions(W)
    u, p = ufl.split(w)
    # # += Identity Tensor
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
    IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
    #    (3): λ1^2*λ2^2*λ3^2; det(C) = J^2
    J = ufl.variable(ufl.det(F))
    # IIIc = ufl.variable(ufl.det(C))
    # += Material Parameters
    c1 = 2
    c2 = 6
    # += Mooney-Rivlin Strain Energy Density Function
    psi = c1 * (Ic - 3) + c2 *(IIc - 3) 
    # Terms
    gamma1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
    gamma2 = -ufl.diff(psi, IIc)
    # += First Piola Stress
    firstPK = 2 * F * (gamma1*I + gamma2*C) + p * J * ufl.inv(F).T

    # +==+==+
    # Setup Variational Problem Solver
    # += Gaussian Quadrature
    metadata = {"quadrature_degree": 4}
    # += Domains of integration
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    # += Residual Equation (Variational, for solving)
    R = ufl.inner(ufl.grad(v), firstPK) * dx + q * (J - 1) * dx \
        - p_inner * ufl.inner(n, v) * ds(3) \
        - p_outer * ufl.inner(n, v) * ds(4) 
    problem = NonlinearProblem(R, w, bc)
    solver = NewtonSolver(domain.comm, problem)
    # += Tolerances for convergence
    solver.atol = 1e-8
    solver.rtol = 1e-8
    # += Convergence criteria
    solver.convergence_criterion = "incremental"

    num_its, converged = solver.solve(w)
    u_sol, p_sol = w.split()
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged after {num_its} iterations.")

    # +==+==+
    # ParaView export
    with io.VTXWriter(MPI.COMM_WORLD, test_name + ".bp", w.sub(0).collapse(), engine="BP4") as vtx:
        vtx.write(0.0)
        vtx.close()

    r = ufl.SpatialCoordinate(domain)
    theta = ufl.SpatialCoordinate(domain)
    sigma_rr = ufl.dot(r, p_sol*r) / ufl.dot(r,r)
    print(sigma_rr)
    # plot_vals()

# +==+==+
# Main check for script operation.
#   Will operate argparse to take values from terminal. 
#   Then runs main() program for script execution 
if __name__ == '__main__':
    # +==+ Input Parameters
    # += Test name
    test_name = "genAnnulus"
    # += Type of element generation
    #    0: gmsh Tetrahedron
    #    1: gmsh Hexahedron
    #    2: manual Hexahedron 
    #   >2: No generation
    test_type = 2
    # += Element Order (interpolation)
    test_order = 2
    # += Refinement style
    refine_check = True

    # +==+ Feed Main()
    main(test_name, test_type, test_order, refine_check)
