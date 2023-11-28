from dolfinx.plot import vtk_mesh
import pyvista
import numpy as np
from mpi4py import MPI
from ufl import Identity, Measure, TestFunction, TrialFunction, VectorElement, dot, dx, inner, grad, nabla_div, sym
from dolfinx import default_scalar_type
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, locate_dofs_geometrical,
                         locate_dofs_topological)
L = 1
H = 1.3
lambda_ = 1.25
mu = 1
rho = 1
g = 1

mesh = create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [L, H]]), [30, 30], cell_type=CellType.triangle)
element = VectorElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, element)

def clamped_boundary(x):
    return np.isclose(x[1], 0)


u_zero = np.array((0,) * mesh.geometry.dim, dtype=default_scalar_type)
bc = dirichletbc(u_zero, locate_dofs_geometrical(V, clamped_boundary), V)

def right(x):
    return np.logical_and(np.isclose(x[0], L), x[1] < H)

boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
boundary_dofs_x = locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, boundary_facets)

bcx = dirichletbc(default_scalar_type(0), boundary_dofs_x, V.sub(0))
bcs = [bc, bcx]

T = Constant(mesh, default_scalar_type((0, 0)))

ds = Measure("ds", domain=mesh)

def epsilon(u):
    return sym(grad(u))


def sigma(u):
    return lambda_ * nabla_div(u) * Identity(len(u)) + 2 * mu * epsilon(u)


u = TrialFunction(V)
v = TestFunction(V)
f = Constant(mesh, default_scalar_type((0, -rho * g)))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds

problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, x = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

# Attach vector values to grid and warp grid by vector

vals = np.zeros((x.shape[0], 3))
vals[:, :len(uh)] = uh.x.array.reshape((x.shape[0], len(uh)))
grid["u"] = vals
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, opacity=0.8)
p.view_xy()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    fig_array = p.screenshot(f"component.png")