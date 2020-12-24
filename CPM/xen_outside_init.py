from CPM import CPM
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


cpm = CPM()
cpm.make_grid(100, 100)
cpm.tau = 100
cpm.eta = 1
cpm.beta_start, cpm.beta_end = 0.4, 0.8
lambda_A, lambda_P, W_start, P0, A0 = get_normal_params(p0=8, r=100, beta=cpm.beta_start, gamma=0, delta=0.7,
                                                        epsilon=0.8, A0=30,
                                                        eta=cpm.eta)
cpm.lambd_A = lambda_A
cpm.lambd_P = lambda_P
cpm.P0 = P0
cpm.A0 = A0
cpm.generate_cells(N_cell_dict={"E": 34, "T": 0, "X": 0})
cpm.pol_amount = 1
# for cll in cpm.cells:
#     if cll.type is "X":
#         cll.P0 = P0*1
cpm.set_lambdP(np.array([0.0, cpm.lambd_P, cpm.lambd_P, cpm.lambd_P]))
cpm.make_J(W_start)  # ,sigma = np.ones_like(W)*0.2)
cpm.make_init("circle", np.sqrt(cpm.A0 / np.pi) * 0.8, np.sqrt(cpm.A0 / np.pi) * 0.2)
cpm.T = 5
cpm.I0 = cpm.I
cpm.run_simulation(int(3e2), int(1e1))

bound = np.zeros_like(cpm.I,dtype=np.bool)
shift = np.array((-1,1))
axes = np.array((0,1))
S,A = np.meshgrid(shift,axes,indexing="ij")
for i,j in zip(S.ravel(),A.ravel()):
    bound += (cpm.I!=0)*(np.roll(cpm.I,i,axis=j)==0)

X_ids = np.unique(cpm.I[bound])
nX = X_ids.size
nE = int((34-nX)/2)
nT = 34 - nX - nE
cpm.generate_cells(N_cell_dict={"E": nE, "T": nT, "X": nX})

I = cpm.I.copy()

nn = 10
Is = np.zeros((nn,cpm.I.shape[0],cpm.I.shape[1]),dtype=np.int64)
npop = np.zeros(10)

for mm in range(10):

    X_ids = np.unique(I[bound])
    nX = X_ids.size

    nX_ids = []
    for i in range(1,35):
        if i not in X_ids:
            nX_ids.append(i)
    nX_ids = np.array(nX_ids)
    np.random.shuffle(nX_ids)
    mapp = np.concatenate((nX_ids,X_ids))

    I = -I
    for i in range(1,35):
        I[I==-mapp[i-1]] = i
    npop[mm] = cpm.find_subpopulations(I).sum()
    Is[mm] = I




np.save("I0_xenout",Is[npop==npop.max()].reshape(I.shape[0],I.shape[1]))