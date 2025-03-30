"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: refinementTest.py
        Mesh refinement test
"""

# +==+==+==+
# Setup
# += Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, csv, ast
# += Parameters


def main(depth):
    depth += 1
    tests = ["test"]# + [str(x) for x in [0, 2, 5, 6, 7, 9, 10, 18, 26, 29, 31]]
    refs = np.flip(np.array([500, 600, 700, 800, 900, 1000, 1100, 1300, 1400, 1800, 2200, 2600, 3000, 3500, 4000, 5000, 6000]))
    pos = [341, 342, 343, 344, 345, 346, 347, 348, 349, 3410, 3411, 3412]
    nodes = np.array([161, 267, 331, 495, 508, 783, 856, 1641, 2365, 3103, 4202, 5110, 8140, 9978, 14026, 23025, 34267])

    # += Load data
    print("\t" * depth + " += Load Data")

    # += Setup figure
    plt.figure(figsize=(25, 10))
    plt.xlabel("Mesh Refinement Level")
    plt.ylabel("Maximum Stress [Pa]")
    plt.title("Mesh Refinement Impact on Stress")
    stat = ["sig_xx","sig_yy","sig_zz","sig_xy","sig_xz","sig_yz","eps_xx","eps_yy","eps_zz","eps_xy","eps_xz","eps_yz"]

    # Example usage
    c = ["r", "b", "k"]

    # += Loop Data
    for i, t in enumerate(stat):
        ax = plt.subplot(3, 4, 1+i)
        ax.title.set_text(f"{t}")  
        ax.set_xscale('log') 
        for j, s in enumerate(["0", "2", "5"]):
            vals = []
            for r in refs:
                df = pd.read_csv("P_Passive_Contraction/_csv/" + f"{s}_{str(r)}.csv")
                vals.append(df[t][0])
            ax.plot(nodes, vals, marker="o", linestyle="--", c=c[j])

    # += Save data
    plt.savefig("P_Passive_Contraction/_png/mref.png")

# +==+==+ Main Check
if __name__ == '__main__':
    depth = 0
    print("\t" * depth + "!! END !!")
    main(depth)