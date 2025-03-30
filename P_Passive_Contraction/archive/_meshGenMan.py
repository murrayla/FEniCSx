import gmsh
import os

gmsh.initialize()
gmsh.model.add("test")

# Define box dimensions
x_len, y_len, z_len = 1000 * 11, 1000 * 11, 100 * 50
radius = 2000  # Fillet radius

# Define corner points
points = []
p_coords = [
    (0, 0, 0), (x_len, 0, 0), (x_len, y_len, 0), (0, y_len, 0),
    (0, 0, z_len), (x_len, 0, z_len), (x_len, y_len, z_len), (0, y_len, z_len)
]

for x, y, z in p_coords:
    points.append(gmsh.model.occ.addPoint(x, y, z))

# Define arc for rounded vertical edge
arc_center = gmsh.model.occ.addPoint(-radius, 0, z_len / 2)
arc = gmsh.model.occ.addCircleArc(points[0], arc_center, points[4])

# Define straight edges
lines = {
    "bottom_front": gmsh.model.occ.addLine(points[1], points[2]),
    "bottom_right": gmsh.model.occ.addLine(points[2], points[3]),
    "bottom_back": gmsh.model.occ.addLine(points[3], points[0]),

    "top_front": gmsh.model.occ.addLine(points[5], points[6]),
    "top_right": gmsh.model.occ.addLine(points[6], points[7]),
    "top_back": gmsh.model.occ.addLine(points[7], points[4]),

    "front_right": gmsh.model.occ.addLine(points[1], points[5]),
    "back_right": gmsh.model.occ.addLine(points[2], points[6]),
    "back_left": gmsh.model.occ.addLine(points[3], points[7]),
}

# Surface loops (corrected for the arc)
loops = [
    # Bottom face
    [lines["bottom_front"], lines["bottom_right"], lines["bottom_back"], arc],
    
    # Top face
    [lines["top_front"], lines["top_right"], lines["top_back"], -arc],

    # Front face
    [lines["bottom_front"], lines["front_right"], lines["top_front"], -lines["back_right"]],

    # Right face
    [lines["bottom_right"], lines["back_right"], lines["top_right"], -lines["back_left"]],

    # Left face (includes arc)
    [lines["bottom_back"], lines["back_left"], lines["top_back"], -arc]
]

# Create surfaces
surfaces = []
for loop in loops:
    curve_loop = gmsh.model.occ.addCurveLoop(loop)
    surfaces.append(gmsh.model.occ.addPlaneSurface([curve_loop]))

# Create volume
volume = gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop(surfaces)])
gmsh.model.occ.synchronize()

# Generate mesh and save
gmsh.model.mesh.generate(3)
file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.msh")
gmsh.write(file)
gmsh.finalize()
