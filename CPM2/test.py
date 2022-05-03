import numpy as np
from cpm import CPM
import matplotlib.pyplot as plt
import time

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

"""
In the low lambda_P regime, for some unknown reason, structures are breaking down. 

It looks like XEN cells are "pulling" them apart. 

Try: increasing lambdaP and adhesion 


"""

lambda_A, lambda_P, W, P0, A0 = get_normal_params(p0=5, r=100, beta=0.4, gamma=0, delta=0.7, epsilon=0.8, A0=30)

b_e = 0
# W = np.array([[0,0,0,0],
#               [0,1.911305,0.494644,0.505116],
#               [0,0.494644,2.161360,0.420959],
#               [0,0.505116,0.420959,0.529589]])*6.02

W = np.array([[b_e,b_e,b_e,b_e],
              [b_e,1.911305,0.494644,0.505116],
              [b_e,0.494644,2.161360,0.420959],
              [b_e,0.505116,0.420959,0.529589]])*8

lambda_A = 1
lambda_P = 0.3
lambda_Ps = [0,lambda_P,lambda_P,lambda_P*0.1]
P0 = 30
# lambda_mult = np.zeros((4,4))
# for i in range(4):
#     lambda_mult[i:] = lambda_Ps[i]
# # lambda_mult[0] = lambda_Ps
# W *= lambda_mult

# W = W.T

params = {"A0":[A0,A0,A0],
          "P0":[P0,P0,P0],
          "lambda_A":[lambda_A,lambda_A,lambda_A],
          "lambda_P":[lambda_P,lambda_P,lambda_P*0.2],
          "W":W,
          "T":15}
# np.random.seed(2022)
cpm = CPM(params)
cpm.make_grid(150,150)
cpm.generate_cells(N_cell_dict={"E": 8, "T": 8,"X":6})
cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)
iter_i = 10
adhesion_vals_full = np.load("adhesion_matrices/%i.npz" % iter_i).get("adhesion_vals")
adhesion_vals_full[0] = b_e
adhesion_vals_full[:,0] = b_e
adhesion_vals_full[0,0] = 0
cpm.J = -adhesion_vals_full*5
# lambda_mult = np.sqrt(cpm.lambda_P * np.expand_dims(cpm.lambda_P,1))
# lambda_mult *= (1 - np.eye(cpm.lambda_P.size))
# lambda_mult /= np.max(lambda_mult)

# lambda_mult = np.zeros((len(cpm.lambda_P),len(cpm.lambda_P)))
# for i in range(len(cpm.lambda_P)):
#     lambda_mult[i:] = cpm.lambda_P
# lambda_mult[0] = lambda_Ps
# cpm.J *= lambda_mult

cpm.get_J_diff()
t0 = time.time()
cpm.simulate(int(6e4),int(20),initialize=True,J0 = -8)
t1 = time.time()
# cpm.save_simulation("results","test_sim_soft")
# print(t1-t0)


cpm.generate_image_t(res=4,col_dict={1:"red",2:"blue",3:"green"})
cpm.animate()




