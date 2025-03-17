"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: hpc_fx.py
        Contraction over volume from EM data informed anisotropy
"""

# +==+==+ Setup
# += Dependencies
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# += Constants
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]


def stress_strain(test, r, depth):
    depth += 1
    print("\t" * depth + "~> Create Stress Strain plots for Discrete Contraction")

    # += Set theme
    sns.set_style("whitegrid") 

    # += Isolate data files
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv/")
    files = [file for file in os.listdir(path) if ((str(test) + "_" + str(int(r)) + "_") in file)]

    # += Store data
    data_l = []

    # += Iterate files
    for file in files:
        file_df = pd.read_csv(path + file)

        # += Filter
        f_df = file_df[(
            (file_df["X"] >= 2000) & (file_df["X"] <= EDGE[0]//2) & 
            (file_df["Y"] >= 2000) & (file_df["Y"] <= EDGE[1] - 2000) & 
            (file_df["Z"] >= 1000) & (file_df["Z"] <= EDGE[2] - 1000) 
        )].copy()

        # += Sample points
        s_df = f_df.sample(n=min(2000, len(f_df)), random_state=42)
        eps = int(file.split("_")[-1].split(".")[0])

        # += Append data 
        for _, row in s_df.iterrows():
            data_l.append([eps, "σ_xx", row["sig_xx"]])
            data_l.append([eps, "σ_yy", row["sig_yy"]])
            data_l.append([eps, "σ_zz", row["sig_zz"]])

    # += Create dataframe
    df = pd.DataFrame(data_l, columns=["Strain", "Stress Type", "Stress"])

    # += Computer Stats
    sum_df = df.groupby(["Strain", "Stress Type"]).agg(
        mean_stress=("Stress", "mean"),
        std_stress=("Stress", "std")
    ).reset_index()

    # += Figure format
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True)
    fig.suptitle("Stress-Strain Relationships")
    sigs = ["σ_xx", "σ_yy", "σ_zz"]
    titles = ["Normal XX", "Normal YY", "Normal ZZ",]

    # += Create distributions
    for ax, sig, title in zip(axes.flatten(), sigs, titles):
        subset = sum_df[sum_df["Stress Type"] == sig]

        # += Plot trend
        ax.plot(subset["Strain"], subset["mean_stress"], marker="o", label=sig, color="tab:blue")

        # += Plot error cloud
        ax.fill_between(
            subset["Strain"],
            subset["mean_stress"] - subset["std_stress"],
            subset["mean_stress"] + subset["std_stress"],
            alpha=0.3,
            color="tab:blue"
        )

        # += Format subplot
        ax.set_title(title)
        ax.set_xlabel("Strain, ε [%]")
        ax.set_ylabel("Stress [μPa]" if "Normal" in title else "Shear Stress [μPa]")

    plt.tight_layout()
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/_png/" + str(test) + "_" + str(int(r)) + ".png")
    plt.close()

# +==+==+==+
# main
#   Input(s):
#   Output(s): data plots
def main(test, r, depth):
    depth += 1
    # += Stress Strain
    print("\t" * depth + "+= Plot Stress-Strain")
    stress_strain(test, r, depth)

# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    depth = 1
    # += Run on each dataset in that condition
    print("\t" * depth + "!! BEGIN IMAGE GENERATION !!")
    # += Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--test_name",type=str)
    parser.add_argument("-r", "--ref_level",type=float)
    args = parser.parse_args()
    test = args.test_name
    r = args.ref_level
    # += Feed Main()
    main(test, r, depth)