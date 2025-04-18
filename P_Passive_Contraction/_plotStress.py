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

def stretch(test, r, depth):
    # += Set theme
    sns.set_style("whitegrid")

    # += Isolate data files
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv/")
    files = [file for file in os.listdir(path) if (((str(test) + "_" + str(int(r)) + "_") in file)) and ("stretch" in file)]

    # += Store data
    data_l = []

    # += Iterate files
    for file in files:
        file_df = pd.read_csv(path + file)
        
        # += Filter
        # f_df = file_df[(
        #     (file_df["X"] >= 50) & (file_df["X"] <= EDGE[0] - 50) & 
        #     (file_df["Y"] >= 50) & (file_df["Y"] <= EDGE[1] - 50) & 
        #     (file_df["Z"] >= 50) & (file_df["Z"] <= EDGE[2] - 50) 
        # )].copy()
        f_df = file_df.copy()
        
        # += Sample points
        s_df = f_df.sample(n=len(f_df), random_state=42)
        eps = int(file.split("_")[-2])
        
        # += Append data
        for _, row in s_df.iterrows():
            data_l.append([eps, "σ_xx", row["sig_xx"]])

    # += Create dataframe
    df = pd.DataFrame(data_l, columns=["Strain", "Stress Type", "Stress"])

    # += Compute Stats
    sum_df = df.groupby(["Strain", "Stress Type"]).agg(
        mean_stress=("Stress", "max"),
        std_stress=("Stress", "std")
    ).reset_index()

    # += Extract SIG_XX subset
    sig_xx_df = sum_df[sum_df["Stress Type"] == "σ_xx"]

    # += Experimental Data (from Table S10A)
    exp_strain = [0, 5, 10, 15, 20]
    exp_stress = [0, 0.3857, 1.1048, 1.8023, 2.6942] 
    exp_sem = [0, 0.0715, 0.2257, 0.3251, 0.3999]    

    # += Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Normal Stress in X Direction [σ] - Stretch")
    ax.set_xlabel("Strain, ΔL/L [%]")
    ax.set_ylabel("Cauchy Stress [kPa]")

    # += Plot simulated data
    ax.plot(sig_xx_df["Strain"], sig_xx_df["mean_stress"], marker="o", label="Simulation", color="tab:blue")
    ax.fill_between(
        sig_xx_df["Strain"],
        sig_xx_df["mean_stress"] - sig_xx_df["std_stress"],
        sig_xx_df["mean_stress"] + sig_xx_df["std_stress"],
        alpha=0.3,
        color="tab:blue"
    )

    # += Plot experimental data
    ax.errorbar(exp_strain, exp_stress, yerr=exp_sem, fmt="--o", label="Li et al. (2023)", color="tab:red")

    exp_strain = [0, 2.5, 5, 7.5, 10, 11]
    exp_stress = [0, 0.24, 0.375, 0.625, 1.1, 1.27] 
    ax.plot(exp_strain, exp_stress, "--g", marker="o", label="Caporizzo et al. (2018)")

    # exp_strain = [0, (2-1.6)//1.6 * 100, (2.1-1.6)//1.6 * 100, (2.2-1.6)//1.6 * 100, (2.3-1.6)//1.6 * 100] 
    # exp_stress = [0, 1.25//2, 1.4//2, 1.25, 1.5] 
    # ax.plot(exp_strain, exp_stress, "--o", marker="o", label="Fish et al. (1984) [Cardiac]")

    # exp_strain = [0, (2-1.9)//1.6 * 100, (2.1-1.6)//1.6 * 100, (2.2-1.6)//1.6 * 100, (2.3-1.6)//1.6 * 100] 
    # exp_stress = [0, 1.25//2, 1.4//2, 1.25, 1.5] 
    # ax.plot(exp_strain, exp_stress, "--o", marker="o", label="Fish et al. (1984) [Skeletal]")

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/_png/" + str(test) + "_" + str(int(r)) + "_sig_xx.png")
    plt.close()

    # # += Store data array
    # data_array = {
    #     "simulated": list(zip(sig_xx_df["Strain"], sig_xx_df["mean_stress"], sig_xx_df["std_stress"])),
    #     "experimental": list(zip(exp_strain, exp_stress, exp_sem))
    # }

def compression(test, r, depth):
    # += Set theme
    sns.set_style("whitegrid")

    # += Isolate data files
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_csv/")
    files = [file for file in os.listdir(path) if (((str(test) + "_" + str(int(r)) + "_") in file) and ("stretch" not in file))]

    # += Store data
    data_l = []

    # += Iterate files
    for file in files:
        file_df = pd.read_csv(path + file)
        
        # += Filter
        # f_df = file_df[(
        #     (file_df["X"] >= 50) & (file_df["X"] <= EDGE[0] - 50) & 
        #     (file_df["Y"] >= 50) & (file_df["Y"] <= EDGE[1] - 50) & 
        #     (file_df["Z"] >= 50) & (file_df["Z"] <= EDGE[2] - 50) 
        # )].copy()
        f_df = file_df.copy()
        
        # += Sample points
        s_df = f_df.sample(n=len(f_df), random_state=42)
        eps = int(file.split("_")[-2])
        
        # += Append data
        for _, row in s_df.iterrows():
            data_l.append([eps, "σ_xx", row["sig_xx"]])

    # += Create dataframe
    df = pd.DataFrame(data_l, columns=["Strain", "Stress Type", "Stress"])

    # += Compute Stats
    sum_df = df.groupby(["Strain", "Stress Type"]).agg(
        mean_stress=("Stress", "max"),
        std_stress=("Stress", "std")
    ).reset_index()

    # += Extract SIG_XX subset
    sig_xx_df = sum_df[sum_df["Stress Type"] == "σ_xx"]

    # += Experimental Data (from Table S10A)
    exp_strain = [0, 5, 10, 15, 20]
    exp_stress = [0, 0.3857, 1.1048, 1.8023, 2.6942] 
    exp_sem = [0, 0.0715, 0.2257, 0.3251, 0.3999]    

    # += Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Normal Stress in X Direction [σ] - Compression")
    ax.set_xlabel("Strain, ΔL/L [%]")
    ax.set_ylabel("Cauchy Stress [kPa]")

    # += Plot simulated data
    ax.plot(sig_xx_df["Strain"], sig_xx_df["mean_stress"], marker="o", label="Simulation", color="tab:blue")
    ax.fill_between(
        sig_xx_df["Strain"],
        sig_xx_df["mean_stress"] - sig_xx_df["std_stress"],
        sig_xx_df["mean_stress"] + sig_xx_df["std_stress"],
        alpha=0.3,
        color="tab:blue"
    )

    # += Plot experimental data
    ax.errorbar(exp_strain, exp_stress, yerr=exp_sem, fmt="--o", label="Li et al. (2023)", color="tab:red")

    exp_strain = [0, 2.5, 5, 7.5, 10, 11]
    exp_stress = [0, 0.24, 0.375, 0.625, 1.1, 1.27] 
    ax.plot(exp_strain, exp_stress, "--g", marker="o", label="Caporizzo et al. (2018)")

    # exp_strain = [0, (2-1.6)//1.6 * 100, (2.1-1.6)//1.6 * 100, (2.2-1.6)//1.6 * 100, (2.3-1.6)//1.6 * 100] 
    # exp_stress = [0, 1.25//2, 1.4//2, 1.25, 1.5] 
    # ax.plot(exp_strain, exp_stress, "--o", marker="o", label="Fish et al. (1984) [Cardiac]")

    # exp_strain = [0, (2-1.9)//1.6 * 100, (2.1-1.6)//1.6 * 100, (2.2-1.6)//1.6 * 100, (2.3-1.6)//1.6 * 100] 
    # exp_stress = [0, 1.25//2, 1.4//2, 1.25, 1.5] 
    # ax.plot(exp_strain, exp_stress, "--o", marker="o", label="Fish et al. (1984) [Skeletal]")

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/_png/" + str(test) + "_" + str(int(r)) + "_sig_xx.png")
    plt.close()

    # # += Store data array
    # data_array = {
    #     "simulated": list(zip(sig_xx_df["Strain"], sig_xx_df["mean_stress"], sig_xx_df["std_stress"])),
    #     "experimental": list(zip(exp_strain, exp_stress, exp_sem))
    # }


# +==+==+==+
# main
#   Input(s):
#   Output(s): data plots
def plot_(n, r, s, depth):
    depth += 1
    # += Stress Strain
    print("\t" * depth + "+= Plot Stress-Strain")
    if s:
        stretch(n, r, depth)
    else:
        compression(n, r, depth)