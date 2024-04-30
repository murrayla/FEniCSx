"""
    Author: Liam Murray
    Contact: murrayla@student.unimelb.edu.au
    Initial Commit: 16/02/2024
    Code: geo_gen.py
        Generate Geometry from DataFrame
"""

# +==+==+==+
# Setup
# += Imports
from scipy import ndimage, spatial, interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import gmsh
# += Parameters
MESH_DIM = 3
X, Y, Z = 0, 1, 2
FACET_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "area": 7}
FILE = "/Users/murrayla/Documents/main_PhD/P_BranchingPaper/A_Scripts/geo_data.csv"

# +==+==+==+
# Laplacian function to move ellipse to slice values
# +==+==+==+
def laplacian_smooth(xy, contour):

    # +==+==+
    # Setup values
    z = len(xy)
    y = len(contour)
    xy_lap = []
    # += Mean conditions
    n = 3
    we = [1] + [0.5, 0.5, 0.5]
    
    # += Loop values to average
    for i in range(0, z, 1): 
        # += Store distance data
        hold = np.zeros((y, 2))
        hold[:, 0] = np.arange(0, y, 1).T
        # += 'cityblock' to increase punishment for different x values with Manhattan distance
        hold[:, 1] = spatial.distance.cdist([xy[i]], contour, metric='cityblock')
        hold = list(map(int, hold[hold[:,1].argsort()][:n, 0]))
        # += Store laplacian value
        xy_lap.append([
            np.average([xy[i, 0]] + [contour[x, 0] for x in hold], weights=we),
            np.average([xy[i, 1]] + [contour[x, 1] for x in hold], weights=we)
        ])
        
    # += Return as a numpy array
    return np.array(xy_lap)

# +==+==+==+
# Ellipse function to build ellipse from contour data
# +==+==+==+
def ellipse_contour(contour):

    # += Isolate slice contour
    curr = np.array(contour[0])
    c = np.vstack((curr[:, 0, 0], curr[:, 0, 1])).T
    # += Contour parameters
    xmin, xmax = (np.min(c[:, 0])), (np.max(c[:, 0]))
    ymin, ymax = (np.min(c[:, 1])), (np.max(c[:, 1]))
    cenx, ceny = (xmax + xmin)/2, (ymax + ymin)/2
    a, b = xmax - cenx, ymax - ceny
    # += Check for undefined 
    if not a or not b:
        return None

    # +==+==+
    # Build ellipse
    xy = []
    xy_pos = []
    xy_neg = []
    rht = cenx + a
    xpts = np.linspace(rht-2*cenx, rht, 2000)
    # += Calculate x and y positions
    for j, x in enumerate(xpts):
        inner = (b**2 * (1-((x-cenx)**2/a**2)))
        # += Check if root is real
        if inner > 0:
            xy_pos.append([x, inner**0.5 + ceny])
            xy_neg.append([x, ceny - inner**0.5])
    # += Collate and return
    xy_pos = np.array(xy_pos)
    xy_neg = np.flipud(np.array(xy_neg))[1:-1]
    xy = np.vstack((xy_pos, xy_neg))
    return xy

# +==+==+==+
# Gmsh Function for constructing idealised geometry
# +==+==+==+
def create_gmsh_cylinder(test_name):

    # +==+==+
    # Initialise and begin geometry
    gmsh.initialize()
    gmsh.model.add(test_name)

    df = pd.read_csv(FILE, usecols=range(1,25))
    # df = df.loc[(df!=0).any(axis=1)]
    for i in range(0, len(df), 1):
        if (df.iloc[i].loc["c0RX"] == 0) or (df.iloc[i].loc["c0RY"] == 0):
            continue
        z_disc = gmsh.model.occ.addEllipse(
            x=df.iloc[i].loc["c0CX"], y=df.iloc[i].loc["c0CY"], z=i*50,
            r1=df.iloc[i].loc["c0RY"], r2=df.iloc[i].loc["c0RX"]
        )
        z_loop = gmsh.model.occ.addCurveLoop(curveTags=[z_disc])
        gmsh.model.occ.addSurfaceFilling(z_loop)
        gmsh.model.occ.synchronize()
    
    # += Create Mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.refine()
    gmsh.model.mesh.setOrder(2)
    # += Write File
    gmsh.write("P_Branch_Contraction/gmsh_msh/" + test_name + ".msh")
    gmsh.finalize()

# +==+==+==+
# main()
def main(test_name, c_list):
    # +==+==+
    # Produce Ellipse from Contours
    xy_ellipse = []
    for contour in c_list:
        xy = ellipse_contour(contour)
        if xy is not None:
            xy_ellipse.append(xy)
    np_xy = np.array(xy_ellipse)
    # +==+==+
    # Create mesh
    # create_gmsh_cylinder(test_name)
    
# +==+==+
# Main check for script operation.
if __name__ == '__main__':
    # +==+ Test Parameters
    # += Test name
    test_name = "csvGeom"
    # += Contour list
    with open("P_Branch_Contraction/contour_list", "rb") as f:
        c_list = pickle.load(f)
    # += Feed Main()
    main(test_name, c_list)