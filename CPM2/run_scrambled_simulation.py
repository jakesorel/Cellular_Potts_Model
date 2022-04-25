import numpy as np
from cpm import CPM
import matplotlib.pyplot as plt
import time
import os
import sys


if __name__ == "__main__":

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results/scrambled"):
        os.mkdir("results/scrambled")

    iter_i = int(sys.argv[1])



    def get_normal_params(p0, r, beta, gamma,delta,epsilon,A0):
        """K = 1"""
        P0 = p0*np.sqrt(A0)
        lambda_P = A0/r
        J00 = -P0*lambda_P
        lambda_A = 1
        W = J00*np.array([[0, 0, 0,0],
                    [0, 1, (1-beta),(1-delta)],
                    [0, (1-beta), (1+gamma),(1-delta)],
                    [0, (1-delta),(1-delta),(1-epsilon)]])
        return lambda_A,lambda_P,W,P0,A0


    lambda_A, lambda_P, W, P0, A0 = get_normal_params(p0=10, r=100, beta=0.4, gamma=0, delta=0.7,epsilon=0.8, A0=30)

    b_e = -0.5

    W = np.array([[b_e,b_e,b_e,b_e],
                  [b_e,1.911305,0.494644,0.505116],
                  [b_e,0.494644,2.161360,0.420959],
                  [b_e,0.505116,0.420959,0.529589]])*6.02


    params = {"A0":[A0,A0,A0],
              "P0":[P0,P0,P0],
              "lambda_A":[lambda_A,lambda_A,lambda_A],
              "lambda_P":[lambda_P,lambda_P,lambda_P],
              "W":W,
              "T":15}
    cpm = CPM(params)
    cpm.make_grid(100, 100)
    cpm.generate_cells(N_cell_dict={"E": 8, "T": 8,"X":6})
    cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)
    adhesion_vals_full = np.load("../adhesion_matrices_scrambled/%i.npz" % iter_i).get("adhesion_vals")
    adhesion_vals_full[0] = b_e
    adhesion_vals_full[:,0] = b_e
    adhesion_vals_full[0,0] = 0
    cpm.J = -adhesion_vals_full*6.
    cpm.get_J_diff()
    t0 = time.time()
    cpm.simulate(int(1e7),int(1000))
    # t1 = time.time()
    cpm.save_simulation("results/scrambled",str(iter_i))
    # print(t1-t0)
    # cpm.generate_image_t(res=4,col_dict={1:"red",2:"blue",3:"green"})
    # cpm.animate()
