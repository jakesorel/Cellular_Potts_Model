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




def do_job(inputt):
    t0,Id = inputt

    cpm = CPM()
    cpm.make_grid(100, 100)
    cpm.tau = 100
    cpm.t0 = t0
    cpm.lPend = 5
    cpm.eta = 1
    cpm.beta_start,cpm.beta_end = 0.4,0.8
    lambda_A, lambda_P, W_start, P0, A0 = get_normal_params(p0=8, r=100, beta=cpm.beta_start, gamma=0, delta=0.7, epsilon=0.8, A0=30,
                                                      eta=cpm.eta)
    cpm.lambd_A = lambda_A
    cpm.lambd_P = lambda_P
    cpm.P0 = P0
    cpm.A0 = A0
    cpm.generate_cells(N_cell_dict={"E": 10, "T": 10, "X": 0})
    cpm.pol_amount = 1
    # for cll in cpm.cells:
    #     if cll.type is "X":
    #         cll.P0 = P0*1
    cpm.set_lambdP(np.array([0.0, cpm.lambd_P, cpm.lambd_P, cpm.lambd_P]))
    cpm.make_J(W_start)  # ,sigma = np.ones_like(W)*0.2)
    cpm.make_init("circle", np.sqrt(cpm.A0 / np.pi) * 0.8, np.sqrt(cpm.A0 / np.pi) * 0.2)
    cpm.T = 15
    cpm.I0 = cpm.I
    # plt.imshow(cpm.boundary_mask)
    # plt.show()
    cpm.run_simulation_dynamictension(int(1e4), int(2e2))

    cpm.generate_image_t(res=4, col_dict={"E": "red", "T": "blue", "X": "green"})
    cpm.animate()

    I_SAVE = csc_matrix(cpm.I_save.reshape((cpm.num_x, cpm.num_y * cpm.I_save.shape[0])))
    save_npz("dynamic_tension/%d_%d_1.npz"%(int(sys.argv[1]),int(Id)), I_SAVE)


if __name__ == "__main__":
    if not os.path.exists("dynamic_tension"):
        os.makedirs("dynamic_tension")
    n_param_step = int(sys.argv[2])
    n_rep = int(sys.argv[3])
    t0_space = np.linspace((1e4)/4,3*(1e4)/4,n_param_step)
    inputs = np.array([np.repeat(t0_space[int(sys.argv[1])],n_rep),np.arange(n_rep)]).T
    n_slurm_tasks = int(os.environ["SLURM_NTASKS"])
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks,memory_limit="1GB")
    lazy_results = []
    for inputt in inputs:
        lazy_result = dask.delayed(do_job)(inputt)
        lazy_results.append(lazy_result)
    dask.compute(*lazy_results)


