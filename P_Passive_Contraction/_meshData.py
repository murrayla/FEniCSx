"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _meshData.py
        Read in node data from mesh and then create allocated node anistropic values
"""

# +==+==+==+
# Setup
# += Imports
import pandas as pd
import numpy as np
import gmsh
import math
import csv
import ast
import os
from tqdm import tqdm
import multiprocessing as mp

# += Parameters
PXLS = {"x": 11, "y": 11, "z": 50}

def average_angle(n_id, cart, azi, ele, i, depth):
    depth += 1

    print("\t" * depth + ".. load region {}".format(i))
    s_df = pd.read_csv("/Users/murrayla/Documents/main_PhD/P_BranchingPaper/A_Scripts/P_SegData/_csv/snode_{}.csv".format(i))
    a, e = s_df["Azi_[RAD]"].to_numpy(), s_df["Ele_[RAD]"].to_numpy()
    # += Isolate a node
    for j, pos in enumerate(cart):
        cur = np.array([ast.literal_eval(row) for row in s_df["Node"]])
        cur = cur * np.array([PXLS[x] for x in ["x", "y", "z"]])
        dis = np.linalg.norm(cur - pos, axis=1)
        idx = np.where(dis < 500)
        if not(len(dis[idx])):
            continue
        wi = [1/d for d in dis[idx]]
        a_xi, e_xi = [x for x in a[idx]], [x for x in e[idx]]
        a_xiwi, e_xiwi = [x * y for x, y in zip(a_xi, wi)], [x * y for x, y in zip(e_xi, wi)]
        azi[j] = sum(a_xiwi) / sum(wi)
        ele[j] = sum(e_xiwi) / sum(wi)
    # += Save to dataframe
    pd.DataFrame(
        data={
            "n": n_id,
            "a": azi,
            "e": ele
        }
    ).to_csv("P_Passive_Contraction/_csv/vals_{}.csv".format(i), index=False)


def angle_assign(depth):
    depth += 1
    print("\t" * depth + "~> Begin anistropic node assignment")

    # += Load mesh node data
    print("\t" * depth + "~> Load mesh node data")
    # reg = "/Users/murrayla/Documents/main_PhD/FEniCSx/FEniCSx/P_Passive_Contraction/_msh/_mesh.nodes"
    # nodes = open(reg, 'r')
    # 

    # += Store node data
    n_list = []
    f = "/Users/murrayla/Documents/main_PhD/FEniCSx/FEniCSx/P_Passive_Contraction/_msh/_mesh.nodes"
    for line in open(f, 'r'):
        n_list.append(line.strip().replace('\t', ' ').split(' '))
    node = np.array(n_list[1:]).astype(np.float64)
    n_id = node[:, 0]
    cart = node[:, 1:]
    azi = np.full_like(n_id, 0)
    ele = np.full_like(n_id, 0)

    # += Iterate each region
    d_df = pd.read_csv("/Users/murrayla/Documents/main_PhD/P_BranchingPaper/A_Scripts/BranchingContraction/DiaRegions.csv")
    r_num = len(d_df["x"])

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(average_angle, [(n_id, cart, azi, ele, i, depth) for i, _ in d_df.iterrows()])


# +==+==+==+
# mesh_nodes
#   Input(s): 
#   Output(s): 
def mesh_nodes(file, enum, depth):
    depth += 1
    print("\t" * depth + "~> Load .msh file and output nodes")

    # += Setup
    elems = {
        2: "3-node-triangle", 3: "4-node-quadrangle", 4: "4-node-tetrahedron",  
        5: "8-node-hexahedron",  9: "6-node-second-order-triangle", 
        10: "9-node-second-order-quadrangle",  11: "10-node-second-order-tetrahedron \n \n"
    }    
    msh_file = open(file, 'r')
    brks = dict()
    n_list, n_stop = list(), 0
    e_list, e_stop = list(), 0

    # += Iterate over mesh file
    print("\t" * depth + "~> Iterate .msh")
    for i, line in enumerate(msh_file):
        if line[0][0] == '$':
            brks[line[1:-1]] = i
        if n_stop:
            n_list.append(line[:-1])
        if e_stop:
            e_list.append(line[:-1])
        if line[1:-1] == 'Nodes':
            n_stop = 1
            continue
        elif line[1:-1] == 'EndNodes':
            n_stop = 0
            continue
        elif line[1:-1] == 'Elements':
            e_stop = 1 
            continue
        elif line[1:-1] == 'EndElements':
            e_stop = 0
            continue

    n_list.pop()
    e_list.pop()

    # += Store node data
    print("\t" * depth + "~> Write node data")
    path = "P_Passive_Contraction/_msh/"
    n_file = open(path + '_mesh.nodes', 'w')
    n_file.write(n_list[0] + "\n")
    # += Iterate values
    n_pos = dict()
    count = 1
    for i, block in enumerate(n_list):
        if count:
            count -= 1
            continue
        for j in range(int(block.split(" ")[3])):
            n_pos[int(n_list[i+j+1])] = (
                float(n_list[int(block.split(" ")[3])+i+j+1].split(" ")[0]), 
                float(n_list[int(block.split(" ")[3])+i+j+1].split(" ")[1]), 
                float(n_list[int(block.split(" ")[3])+i+j+1].split(" ")[2])
            )
            n_file.write(n_list[i+j+1]+"\t")
            n_file.write(
                n_list[int(block.split(" ")[3])+i+j+1].split(" ")[0] + "\t" +
                n_list[int(block.split(" ")[3])+i+j+1].split(" ")[1] + "\t" +
                n_list[int(block.split(" ")[3])+i+j+1].split(" ")[2] + "\n"
            )
            count +=2

    # += Store element data
    print("\t" * depth + "~> Write element data")
    element_file = open(path + '_mesh.ele', 'w')
    element_file.write(e_list[0] + "\n")
    types = {enum: elems[enum].strip()}
    # += Iterate values
    count = 1
    for i, block in enumerate(e_list):
        if count:
            count -= 1
            continue
        for j in range(int(block.split(" ")[3])):
            count +=1
            if int(block.split(" ")[2]) in types.keys():
                element_file.write(types[int(block.split(" ")[2])] + "\t")
                element_file.write(block.split(" ")[1] + "\t")
                for value in e_list[i+j+1].split():
                    element_file.write(value + "\t")
                element_file.write("\n")
            else:
                continue

# +==+==+==+
# main
#   Input(s): 
#   Output(s): 
def main(depth):
    depth += 1

    # += Output node positional data
    print("\t" * depth + "+= Output node data")
    # mesh_nodes("P_Passive_Contraction/_msh/RotCube.msh", 11, depth)
    angle_assign(depth)

# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    depth = 1
    # += Run on each dataset in that condition
    print("\t" * depth + "!! BEGIN DATA GENERATION !!")
    # += Feed Main()
    main(depth)