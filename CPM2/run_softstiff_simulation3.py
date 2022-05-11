import numpy as np
from cpm import CPM
import matplotlib.pyplot as plt
import time
import os
import sys


if __name__ == "__main__":

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results/variable_soft"):
        os.mkdir("results/variable_soft")


    iter_i = int(sys.argv[1])
    lambda_P_mult_range = np.flip(np.linspace(0.2,1,9))
    for i, lpm in enumerate(lambda_P_mult_range):
        dir_name = "results/variable_soft/%.2f"%lpm
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    A0 = 30
    P0 = 0
    lambda_A = 1
    lambda_P = 0.2
    b_e = -0.5

    W = np.array([[b_e,b_e,b_e,b_e],
                  [b_e,1.911305,0.494644,0.505116],
                  [b_e,0.494644,2.161360,0.420959],
                  [b_e,0.505116,0.420959,0.529589]])*6.02

    lpm = lambda_P_mult_range[0]

    params = {"A0":[A0,A0,A0],
              "P0":[P0,P0,P0],
              "lambda_A":[lambda_A,lambda_A,lambda_A],
              "lambda_P":[lambda_P,lambda_P,lambda_P*lpm],
              "W":W,
              "T":15}
    cpm = CPM(params)
    cpm.make_grid(100,100)
    cpm.generate_cells(N_cell_dict={"E": 8, "T": 8,"X":6})
    cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)
    # adhesion_vals_full = np.load("../adhesion_matrices/%i.npz" % iter_i).get("adhesion_vals")
    adhesion_vals_full = np.load("../adhesion_matrices/%i.npz" % iter_i).get("adhesion_vals")
    # adhesion_vals_full = np.load("adhesion_matrices/%i.npz" % iter_i).get("adhesion_vals")

    adhesion_vals_full[0] = b_e*cpm.lambda_P
    adhesion_vals_full[:,0] = b_e*cpm.lambda_P
    adhesion_vals_full[0,0] = 0
    cpm.J = -adhesion_vals_full * 6
    cpm.get_J_diff()
    t0 = time.time()
    cpm.simulate(int(1e7), int(1000), initialize=True, J0=-8)

    # t1 = time.time()
    cpm.save_simulation("results/variable_soft/%.2f"%lpm,str(iter_i))


    for i in range(len(lambda_P_mult_range)):
        lpm = lambda_P_mult_range[i+1]

        params2 = {"A0":[A0,A0,A0],
                  "P0":[P0,P0,P0],
                  "lambda_A":[lambda_A,lambda_A,lambda_A],
                  "lambda_P":[lambda_P,lambda_P,lambda_P*lpm],
                  "W":W,
                  "T":15}
        cpm2 = CPM(params2)
        cpm2.make_grid(100, 100)
        cpm2.generate_cells(N_cell_dict={"E": 8, "T": 8,"X":6})
        cpm2.I0 = cpm.I_save[0]
        cpm2.I = cpm2.I0.copy()
        cpm2.n_cells = cpm.n_cells
        cpm2.assign_AP()
        cpm2.J = cpm.J.copy()
        cpm2.J[0] = -6*b_e*cpm2.lambda_P
        cpm2.J[:,0] = -6*b_e*cpm2.lambda_P
        cpm2.J[0,0] = 0
        cpm2.J_diff = cpm.J_diff.copy()
        cpm2.simulate(int(1e7), int(1000), initialize=False)
        # t1 = time.time()
        cpm2.save_simulation("results/variable_soft/%.2f"%lpm,str(iter_i))
