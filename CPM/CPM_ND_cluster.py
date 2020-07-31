from CPM import CPM
import numpy as np
import os
import dask
from dask.distributed import Client
from scipy.sparse import csc_matrix, save_npz
import sys




def get_normal_params(p0, r, beta, gamma,A0):
    """K = 1"""
    P0 = p0*np.sqrt(A0)
    lambda_P = A0/r
    J00 = -P0*lambda_P
    lambda_A = 1
    W = J00*np.array([[0, 0, 0],
                [0, 1, (1-beta)],
                [0, (1-beta), (1+gamma)]])
    return lambda_A,lambda_P,W,P0,A0



def do_job(inputt):
    p0,r,beta,T,i = inputt
    cpm = CPM()
    cpm.make_grid(100, 100)
    lambda_A, lambda_P, W, P0, A0 = get_normal_params(p0=p0, r=r, beta=beta, gamma=0, A0=30)
    cpm.lambd_A = lambda_A
    cpm.lambd_P = lambda_P
    cpm.P0 = P0
    cpm.A0 = A0
    cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
    cpm.set_lambdP(np.array([0.0, lambda_P, lambda_P]))
    cpm.make_J(W)  # ,sigma = np.ones_like(W)*0.2)
    cpm.make_init("circle", np.sqrt(cpm.A0 / np.pi) * 0.8, np.sqrt(cpm.A0 / np.pi) * 0.2)
    cpm.T = T
    cpm.I0 = cpm.I
    cpm.run_simulation(int(1e4), int(2e2), polarise=False)
    I_SAVE = csc_matrix(cpm.I_save.reshape((cpm.num_x, cpm.num_y * cpm.I_save.shape[0])))
    save_npz("/central/scratch/jakecs/Cellular_Potts_Model/results/I_save_%d.npz"%i, I_SAVE)

if __name__ == "__main__":
    # if not os.path.exists("/central/scratch/jakecs/Cellular_Potts_Model/results"):
    #     os.makedirs("/central/scratch/jakecs/Cellular_Potts_Model/results")
    n_iter = int(sys.argv[1])
    n_param_step = int(sys.argv[2])
    N_job = int(sys.argv[3])
    i_job = int(sys.argv[4])
    p0_space, r0_space, beta_space, T_space = np.linspace(3,10,n_param_step),np.logspace(0,2,n_param_step),np.linspace(0,1,n_param_step),np.logspace(0,2,n_param_step)
    rep_space = np.arange(n_iter)
    PP,RR,BB,TT,NN = np.meshgrid(p0_space, r0_space, beta_space, T_space,rep_space,indexing="ij")
    inputs = np.array([PP.ravel()[i_job::N_job],RR.ravel()[i_job::N_job],BB.ravel()[i_job::N_job],TT.ravel()[i_job::N_job],np.arange(NN.size)[i_job::N_job]]).T
    n_slurm_tasks = int(os.environ["SLURM_NTASKS"])
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks)
    lazy_results = []
    for inputt in inputs:
        lazy_result = dask.delayed(do_job)(inputt)
        lazy_results.append(lazy_result)
    dask.compute(*lazy_results)


"""getting some strange errors. As a fail-safe, fix the dodgy LA.any() = None issue by catching the error"""