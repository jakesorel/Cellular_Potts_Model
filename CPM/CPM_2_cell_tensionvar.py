from CPM.CPM import CPM
import numpy as np
import os
import dask
from dask.distributed import Client
from scipy.sparse import csc_matrix, save_npz
import sys






def get_normal_params(p0, r, beta, gamma,delta,epsilon,A0,eta = 1):
    """K = 1"""
    P0 = p0*np.sqrt(A0)
    lambda_P = A0/r
    J00 = -P0*lambda_P*eta
    lambda_A = 1
    W = J00*np.array([[0, 0, 0,0],
                [0, 1, (1-beta),(1-delta)],
                [0, (1-beta), (1+gamma),(1-delta)],
                [0, (1-delta),(1-delta),(1-epsilon)]])
    return lambda_A,lambda_P,W,P0,A0




lPmult,T,Id = 1,15,0
cpm = CPM()
cpm.make_grid(100, 100)
lambda_A, lambda_P, W, P0, A0 = get_normal_params(p0=8, r=100, beta=0.9, gamma=0, delta=0.7, epsilon=0.8, A0=30,eta=1)
cpm.lambd_A = lambda_A
cpm.lambd_P = lambda_P*lPmult
cpm.P0 = P0
cpm.A0 = A0
cpm.generate_cells(N_cell_dict={"E": 10, "T": 10, "X": 0})
cpm.set_lambdP(np.array([0.0, cpm.lambd_P, cpm.lambd_P, cpm.lambd_P]))
cpm.make_J(W)  # ,sigma = np.ones_like(W)*0.2)
cpm.make_init("circle", np.sqrt(cpm.A0 / np.pi) * 0.8, np.sqrt(cpm.A0 / np.pi) * 0.2)
cpm.T = T
cpm.I0 = cpm.I
cpm.run_simulation(int(1e3), int(2e2), polarise=False)

cpm.generate_image_t(res=4,col_dict={"E":"red","T":"blue","X":"green"})
cpm.animate()
