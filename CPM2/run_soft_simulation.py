import numpy as np
from cpm import CPM
import matplotlib.pyplot as plt
import time
import os
import sys


if __name__ == "__main__":

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results/soft"):
        os.mkdir("results/soft")

    iter_i = int(sys.argv[1])

    lambda_A = 1
    lambda_P = 0.2
    A0 = 30
    P0 = 0
    b_e = -0.2

    W = np.array([[b_e,b_e,b_e,b_e],
                  [b_e,1.911305,0.494644,0.505116],
                  [b_e,0.494644,2.161360,0.420959],
                  [b_e,0.505116,0.420959,0.529589]])*6.02


    params = {"A0":[A0,A0,A0],
              "P0":[P0,P0,P0],
              "lambda_A":[lambda_A,lambda_A,lambda_A],
              "lambda_P":[lambda_P,lambda_P,lambda_P*0.5],
              "W":W,
              "T":15}
    cpm = CPM(params)
    cpm.make_grid(100, 100)
    cpm.generate_cells(N_cell_dict={"E": 8, "T": 8,"X":6})
    cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)
    adhesion_vals_full = np.load("../adhesion_matrices/%i.npz" % iter_i).get("adhesion_vals")
    adhesion_vals_full[0] = b_e
    adhesion_vals_full[:,0] = b_e
    adhesion_vals_full[0,0] = 0
    cpm.J = -adhesion_vals_full * 90
    lambda_mult = np.zeros((len(cpm.lambda_P), len(cpm.lambda_P)))
    for i in range(len(cpm.lambda_P)):
        lambda_mult[i:] = cpm.lambda_P
    # lambda_mult[0] = lambda_Ps
    cpm.J *= lambda_mult

    cpm.get_J_diff()
    t0 = time.time()
    cpm.simulate(int(1e7),int(1000))
    # t1 = time.time()
    cpm.save_simulation("results/soft",str(iter_i))
    # print(t1-t0)
    # cpm.generate_image_t(res=4,col_dict={1:"red",2:"blue",3:"green"})
    # cpm.animate()
