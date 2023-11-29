
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, locate_entities
import numpy as np

from mpi4py import MPI
import gmsh
import meshio

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

# mesh, cell_markers, facet_markers = gmshio.read_from_msh("testCylinder.msh", MPI.COMM_WORLD, gdim=3)

# # create mesh function
# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
#     points = mesh.points[:,:3] if prune_z else mesh.points
#     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
#     return out_mesh

# # read in mesh
# msh = meshio.read("testCylinder.msh")
# triangle_mesh = create_mesh(msh, "tetra", prune_z=False)
# meshio.write("testCylinder.xdmf", triangle_mesh)