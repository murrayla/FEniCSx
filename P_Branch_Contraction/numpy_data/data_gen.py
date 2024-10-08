# +==+==+==+==+==+
# Author: Liam Murray
# Contact: murrayla@student.unimelb.edu.au
# Date: 27/05/2024
# Code: data_Ggen.py
#   Generation data trends from FEniCSx simulations
# +==+==+==+==+

# +==+==+ Setup
# += Dependencies
import csv
import numpy as np
import matplotlib.pyplot as plt
# += Constants
ITERATIONS = 19
DATA_TYPE = ["disp_x", "eps_xy", "sig_xy"]
TEST_POINTS = (7, 5)

# +==+==+==+
# Process numpy data files:
#   Input of label name
#   Output processed data
def process_data(test_name, d_t):
    arg_array = np.zeros((TEST_POINTS[0], TEST_POINTS[1], len(test_name)))

    for i, test in enumerate(test_name):
        file = "P_Branch_Contraction/numpy_data/" + d_t + "_SUB_REDQUAD_XX_" + test + "__10PctIter.npy"
        cur = np.load(file)
        arg_array[:, :, i] = cur

    test_col = [[name] for name in test_name]
    for row in range(0, TEST_POINTS[1], 1):
        with open("P_Branch_Contraction/csv_data/" + d_t + "_row_" + str(row) + "_.csv", 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(test_col)
            writer.writerows(arg_array[:, row, :]) 
    return None

# +==+==+==+
# Main
def main(test_name, test_case):
    for d_t in DATA_TYPE:
        process_data(test_name, d_t)
    return

# +==+==+
# Main check for script operation.
#   runs main()
if __name__ == '__main__':
    # +==+==+
    # Test Parameters
    # += Test name
    test = [
        "SINGLE_MIDDLE", "SINGLE_DOUBLE", "SINGLE_ACROSS", 
        "DOUBLE_ACROSS", "BRANCH_ACROSS", "BRANCH_MIDDLE", 
        "TRANSFER_DOUBLE", "CYTOSOL"
    ]
    test_name = [x for x in test]
    # += Cases
    test_case = list(range(0, len(test_name), 1))
    # += Feed Main()
    main(test_name, test_case)