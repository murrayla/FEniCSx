#A module listing functions to generate simple meshes.

import math
import numpy as np
import matplotlib.pyplot as plt

# +==+==+==+
# Print winding for created element (test)
# +==+==+==+
def print_winding():
    print("+== ... DISPLAY")
    # += Create test structure
    node_list, node_idx_list, _, _, _, _, _, _, elem = annulus(1,2,1,2,4,2,2)
    np_nodes = np.array(node_list)
    np_elems = np.array(elem)
    np_idx = np.array(node_idx_list)
    # np_idx -= 1

    # += Extract node positions
    x = np_nodes[:, 0].flatten()
    y = np_nodes[:, 1].flatten()
    z = np_nodes[:, 2].flatten()

    # += Setup plot structure and scatter visualisation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=50)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # += Add text to itedntify winding
    for i, txt in enumerate(np_idx):  
        el_assign = np.where(np_elems == txt[0])[0]
        ax.text(x[i], y[i], z[i], str(txt) + "-" + str(el_assign), color='red')

    # += Plot
    plt.show()
    return

# +==+==+==+
# Inputs:
    # r_inner = radius of internal circle nodes
    # r_outer = radius of outer circle nodes
    # z_length = screw axis length of cylinder
    # numberOfRadialElements = number of elements in the radial direction (how thick it is)
    # numberOfCircumferentialElements = number of elements in the annulus ring
    # numberOfZElements = number of elements in screw axis (how tall)
    # InterpolationType = interpolation function type, i.e. quadratic
# 
# +==+==+==+

# +==+==+==+
# Outputs:
    # node_list = 2D list storing positions of each node
    # node_idx_list = 1D list of node indexes
    # top_node_list = 1D list of final node for each element
    # bottom_node_list = 1D list of first node for each element
    # yfix_node_list = 1D list with reference y node per element
    # xfix_node_list = 1D list with reference x node per element
    # internal_node_list = 1D list of inner radius elements nodes
    # outer_node_list = 1D list of outer radius element nodes
    # elems = 2D list that holds node assingment per element
# 
# +==+==+==+

def annulus(r_inner,r_outer,z_length,numberOfRadialElements,numberOfCircumferentialElements,numberOfZElements,InterpolationType):
    # +==+==+
    # Parameter Setup
    # += Preallocation
    node_list = []
    node_idx = 0

    elem_n = numberOfRadialElements * numberOfCircumferentialElements * numberOfZElements
    # += Element base centre
    xorig = 0.0 # x origin
    yorig = 0.0 # y origin
    zorig = 0.0 # z origin
    theta_orig = 2 * math.pi # theta origin

    # +==+==+
    # Element and Node position parameters
    # += Determine size of elements
    elem_r_thickness = (r_outer - r_inner)/numberOfRadialElements
    elem_theta_rad = theta_orig / numberOfCircumferentialElements
    elem_z_height = z_length / numberOfZElements
    # += Determine seperation distance between nodes for each interpolation
    # += Linear
    if (InterpolationType == 1):
        r_delta = elem_r_thickness
        ridx_end = numberOfRadialElements+1
        theta_delta = elem_theta_rad
        thetaidx_end = numberOfCircumferentialElements
        z_delta = elem_z_height
        zidx_end = numberOfZElements+1
    # += Quadratic
    elif (InterpolationType == 2):
        r_delta = elem_r_thickness/2.0
        ridx_end = numberOfRadialElements*2 + 1
        theta_delta = elem_theta_rad / 2.0
        thetaidx_end = numberOfCircumferentialElements*2
        z_delta = elem_z_height/2.0
        zidx_end = numberOfZElements*2 + 1

    node_map = np.zeros((ridx_end, thetaidx_end, zidx_end))
    # +==+==+
    # Node set for radius values
    # += Iterate through radius indexes
    for ridx in range(0, ridx_end, 1):
        r = r_inner + ridx * r_delta
        # print(r)
        # += Iterate through circumference indexes
        for thetaidx in range(0, thetaidx_end, 1):
            theta = theta_orig - thetaidx * theta_delta
            # += Iterate through height indexes
            for zidx in range(0, zidx_end, 1):
                z = zorig + zidx * z_delta
                # += Node coordinate per idx
                node_list.append(
                    [
                        xorig + r * math.cos(theta),
                        yorig + r * math.sin(theta),
                        zorig + z
                    ]
                )
                node_idx = node_idx + 1 
                node_map[ridx, thetaidx, zidx] = node_idx

    if numberOfCircumferentialElements > 1:
        end_points = np.array([[node_map[x, 0, :]] for x in range(0, ridx_end, 1)])
        node_map = np.concatenate((node_map, end_points), axis=1)
        thetaidx_end += 1

    e_assign = np.zeros((elem_n, 27),dtype=np.int32)
    e = 0
    for i in range(0, ridx_end, 2):
        for j in range(0, thetaidx_end, 2):
            for k in range(0, zidx_end, 2):
                if i+3 > ridx_end or j+3 > thetaidx_end or k+3 > zidx_end:
                    continue
                e_assign[e, :] = node_map[i:(i+3), j:(j+3), k:(k+3)].flatten() - 1
                e += 1

    print('Total elements in mesh=',len(e_assign))
    return node_list, e_assign

if __name__ == '__main__':
    print_winding()