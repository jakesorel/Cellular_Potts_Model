from CPM import CPM
import numpy as np
import os
import dask
from dask.distributed import Client
from scipy.sparse import csc_matrix, save_npz
import sys



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



def do_job(inputt):
    delta,epsilon,pol_amount,Id = inputt
    cpm = CPM()
    cpm.make_grid(100, 100)
    lambda_A, lambda_P, W, P0, A0 = get_normal_params(p0=8, r=100, beta=0.4, gamma=0, delta=delta, epsilon=epsilon, A0=30)
    cpm.lambd_A = lambda_A
    cpm.lambd_P = lambda_P
    cpm.pol_amount = pol_amount
    cpm.P0 = P0
    cpm.A0 = A0
    cpm.generate_cells(N_cell_dict={"E": 10, "T": 10, "X": 14})
    cpm.set_lambdP(np.array([0.0, lambda_P, lambda_P, lambda_P]))
    cpm.make_J(W)  # ,sigma = np.ones_like(W)*0.2)
    cpm.make_init("circle", np.sqrt(cpm.A0 / np.pi) * 0.8, np.sqrt(cpm.A0 / np.pi) * 0.2)
    cpm.T = 15
    cpm.I0 = cpm.I
    cpm.run_simulation_XEN_polarise(int(1e4), int(2e2), pol_skip=2)
    I_SAVE = csc_matrix(cpm.I_save.reshape((cpm.num_x, cpm.num_y * cpm.I_save.shape[0])))
    save_npz("pol_robust/%d_%d.npz"%(int(sys.argv[1]),int(Id)), I_SAVE)


if __name__ == "__main__":
    if not os.path.exists("pol_robust"):
        os.makedirs("pol_robust")
    n_param_step = int(sys.argv[2])
    n_rep = int(sys.argv[3])
    delta_space,eps_space,pol_amount_space = np.linspace(0,1,n_param_step),np.linspace(0,1,n_param_step),np.array([0.4,1])
    DD, EE,PP = np.meshgrid(delta_space,eps_space,pol_amount_space,indexing="ij")
    inputs = np.array([DD.ravel(),EE.ravel(),PP.ravel()]).T
    input = inputs[int(sys.argv[1])]
    inputs = np.array([np.repeat(input[0],n_rep),np.repeat(input[1],n_rep),np.repeat(input[2],n_rep),np.arange(n_rep)]).T
    n_slurm_tasks = int(os.environ["SLURM_NTASKS"])
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks,memory_limit="1GB")
    lazy_results = []
    for inputt in inputs:
        lazy_result = dask.delayed(do_job)(inputt)
        lazy_results.append(lazy_result)
    dask.compute(*lazy_results)


