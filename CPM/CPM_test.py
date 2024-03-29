from CPM.CPM import CPM
import numpy as np
import matplotlib.pyplot as plt
import time
from line_profiler import LineProfiler
import dask
from dask.distributed import Client
from joblib import Parallel, delayed
import multiprocessing
from numba import jit






cpm = CPM()

cpm.make_grid(100,100)
cpm.generate_cells(N_cell_dict={"E":10,"T":10,"X":0})
cpm.A0 = 20
# cpm.lambd_P = 20
# cpm.lambd_P = 0.05
cpm.set_lambdP(np.array([0,0.15,0.15,0.05]))
# cpm.set_div_thresh(A0_div=40)
# cpm.set_cll_growth(0.001)
W = -np.array([[0, 0, 0,0],
                [0, 4, 2,1],
                [0, 2, 4,1],
               [0,1,1,1]])*10
cpm.make_J(W)#,sigma = np.ones_like(W)*0.2)
cpm.make_init("circle",np.sqrt(cpm.A0/np.pi)*0.8,np.sqrt(cpm.A0/np.pi)*0.2)
cpm.T = 40
cpm.I0 = cpm.I
t0 = time.time()
I = cpm.run_simulation(int(5e2),50,polarise=False)
t1 = time.time()

cpm.generate_image_t(res=4,col_dict={"E":"red","T":"blue","X":"green"})
cpm.animate()


# plt.imshow(cpm.Pol)
# plt.show()
#
print(t1-t0)
cpm.get_centroids_t()
cpm.get_neighbourhood_index2()
fig, ax = plt.subplots(2,3)
ax = ax.ravel()
ax[0].scatter(np.arange(cpm.I_save.shape[0]),cpm.neighbourhood_percentage[:,0],color="r",s=4)
ax[0].scatter(np.arange(cpm.I_save.shape[0]),cpm.neighbourhood_percentage[:,1],color="b",s=4)
ax[0].set(ylabel="% self-self neighbours")
cpm.find_subpopulations_t()
ax[1].plot(cpm.subpopulations_t[:,0],color="r")
ax[1].plot(cpm.subpopulations_t[:,1],color="b")
ax[1].set(ylabel="num clusters")
cpm.get_polt()
ax[2].plot(cpm.polt)
ax[2].set(ylabel="f")
cpm.generate_image_t(res=4,col_dict={"E":"red","T":"blue","X":"green"})
ax[3].imshow(cpm.Im_save[-1])
ax[4].plot(1-cpm.shannon_entropy)
fig.show()
#

cpm.generate_image_t(res=4,col_dict={"E":"red","T":"blue","X":"green"})

cpm.animate()

#
# I = np.zeros([7,7])
# I[3:] = 2
# I[2:5,2:5] = 1
# plt.imshow(I)
# plt.show()
# s = 1
# bes = cpm.get_perimeter_elements_Moore(I==s)*(I!=s)
# plt.imshow(bes)
# plt.show()
#
#
# X,Y = np.mgrid[:I.shape[0],:I.shape[1]]
# xc,yc = np.sum(X*(I==s))/np.sum(I==s),np.sum(Y*(I==s))/np.sum(I==s)
# b_vals = ((I==0)*-1 + (I!=s)*(I!=0)*1)*bes
# polx,poly = np.sum(b_vals*(X*bes - xc)),np.sum(b_vals*(Y*bes - yc))
# pol = np.array([polx, poly]) #points away from medium
# div_plane = np.dot(np.array([[0,-1],[1,0]]),pol)
#
#
# @jit(nopython=True)
# def element_cross_abs(X,x):
#     Y = np.empty((X.shape[1],X.shape[2]),dtype=X.dtype)
#     for i in range(X.shape[1]):
#         for j in range(X.shape[2]):
#             Y[i,j] = np.sign(X[0,i,j]*x[1] - X[1,i,j]*x[0])
#     return Y
#
# div_mask = element_cross_abs(np.array([X-xc,Y-yc]),div_plane)>0
#
#




#
# P = np.linspace(0,1)
# plt.plot(P,(-P*np.log2(P) - (1-P)*np.log2(1-P)))
# plt.show()

# cpm.get_mu_sigd()
#
# plt.plot(cpm.mut)
# plt.show()
#
# plt.plot(cpm.sigt)
# plt.show()


#
# time = int(5e2)
# t0 = 10
#
# cpm = CPM()
# cpm.make_grid(100,100)
# cpm.generate_cells(N_cell_dict={"E":20,"T":20})
# cpm.A0 = 20
# cpm.lambd_P = 30
# cpm.make_J(W)
# cpm.make_init("circle",np.sqrt(cpm.A0/np.pi)*0.8,np.sqrt(cpm.A0/np.pi)*0.2)
# cpm.T = 10
# I = cpm.run_simulation(time,50)
# cpm.get_centroids_t()
# cpm.get_neighbourhood_index(2)
#
# MSD1 = np.array([np.mean([np.linalg.norm(cpm.centroids[t0,i] - cpm.centroids[t,i]) for i in range(cpm.n_cells+1)]) for t in range(t0,cpm.I_save.shape[0])])
#
#
# cpm = CPM()
# cpm.make_grid(100,100)
# cpm.generate_cells(N_cell_dict={"E":20,"T":20})
# cpm.A0 = 20
# cpm.lambd_P = 80
# cpm.make_J(W)
# cpm.make_init("circle",np.sqrt(cpm.A0/np.pi)*0.8,np.sqrt(cpm.A0/np.pi)*0.2)
# cpm.T = 10
# I = cpm.run_simulation(time,50)
# cpm.get_centroids_t()
# MSD2 = np.array([np.mean([np.linalg.norm(cpm.centroids[t0,i] - cpm.centroids[t,i]) for i in range(cpm.n_cells+1)]) for t in range(t0,cpm.I_save.shape[0])])
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax.plot(MSD1,label="Low tension")
# ax.plot(MSD2,label="High tension")
# ax.set(xlabel="Time",ylabel="MSD")
# ax.legend()
# fig.show()


# cpm.generate_image_t(res=4)
# plt.imshow(cpm.Im_save[-1])
# plt.show()
# cpm.animate()
#
# X,Y = np.mgrid[-50:51,-50:51]
# I = X**2 + Y**2 < 40**2
# cpm.I = I
# cpm.get_xy_clls(I)

# FX,FY = cpm.get_F_xy(cpm.I)
# FX,FY = cpm.smooth_Fxy(I,FX,FY)
# fig, ax = plt.subplots()
# ax.imshow(cpm.I.T)
# I,J = np.where(cpm.FX**2+cpm.FY**2 !=0)
# # ax.scatter(I,J)
# ax.quiver(I,J, cpm.FX[I,J],cpm.FY[I,J])
# # ax.set(xlim=(40,60),ylim=(40,60))
# fig.show()


"""Note: for division, can calculate lone pixels and force them to join the other cell. May need to account for doublets"""

#
#
#
# #
# # A,P = [],[]
# # SI = 5
# # a_range = np.linspace(0,200,500)
# # for a in a_range:
# #     X,Y = np.meshgrid(np.arange(-100,100),np.arange(-100,100))
# #     r = np.sqrt(a/np.pi)
# #     cll = ((X)**2 + (Y)**2 < r**2)
# #     PI = cpm.get_perimeter_elements(cll)
# #     A.append(np.sum(cll))
# #     P.append(np.sum(PI*cll))
# #
# # plt.plot(P,np.sqrt(A))
# # plt.plot(P,np.array(P)/SI)
# # plt.plot(P,np.sqrt(a_range))
# # plt.show()
#
#
# # np.polyfit(P, np.sqrt(A), 1)
#
#
# def raw_moment(data, i_order, j_order):
#   nrows, ncols = data.shape
#   y_indices, x_indicies = np.mgrid[:nrows, :ncols]
#   return (data * x_indicies**i_order * y_indices**j_order).sum()
#
# def moments_cov(data):
#     """https://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python"""
#     data_sum = data.sum()
#     m10 = raw_moment(data, 1, 0)
#     m01 = raw_moment(data, 0, 1)
#     x_centroid = m10 / data_sum
#     y_centroid = m01 / data_sum
#     u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
#     u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
#     u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
#     cov = np.array([[u20, u11], [u11, u02]])
#     return cov,np.array([x_centroid,y_centroid])
#
# def split_cell(X):
#     cov, centroid = moments_cov(X)
#     evals, evecs = np.linalg.eig(cov)
#     short_ax = evecs[:,evals.argmin()]
#     nrows, ncols = X.shape
#     x_indices, y_indices = np.mgrid[:nrows, :ncols]
#     grad = short_ax[1]/short_ax[0]
#     mask = grad*(x_indices - centroid[0]) > (y_indices - centroid[1])
#     return (mask.T)*X, (~mask.T)*X
#
# cA, cB = split_cell(X)

# X = cpm.I==2
# cov,centroid = moments_cov(X)
# evals, evecs = np.linalg.eig(cov)
#
# fig, ax = plt.subplots()
# ax.imshow((X*5 + mask).T)
# # ax.imshow(np.flip(X.T,axis=1))
# ax.quiver(centroid[0],centroid[1],evecs[0][0],evecs[0][1])
# ax.quiver(centroid[0],centroid[1],evecs[1][0],evecs[1][1],color="red")
# fig.show()
#
# t0 = time.time()
# for i in range(int(1e3)):
#     CA,CB = split_cell(cpm.I==2)
# t1 = time.time()
# print(t1-t0)
#
#
# fig, ax =plt.subplots()
# ax.imshow(CA*2 + CB)
# fig.show()
#
# #
# # t0 = time.time()
# # cpm.generate_image_t()
# # t1 = time.time()
# # print(t1-t0)
#
# #
# # Na = I[48:51,48:51]
# # s = 1
# #
# # z1_masks = cpm.z1_masks
# # z2_masks = cpm.z2_masks
# # z3_masks = cpm.z3_masks
# # z = 2
# # @njit
# # def _LA(z1_masks,z2_masks,z3_masks,s,Na,z):
# #     if z == 1:
# #         return sum_axis12((Na == s) == z1_masks)
# #     if z == 2:
# #         return sum_axis12((Na == s) == z2_masks) #or should be geq?
# #     if z == 3:
# #         return sum_axis12((Na == s) == z3_masks) #or should be geq?
# #
# # @njit
# # def sum_axis12(X):
# #     n = X.shape[0]
# #     X_out = np.zeros(n,dtype=np.bool_)
# #     for i in range(n):
# #         X_out[i] = np.sum(X[i])==9
# #     return X_out
# #
# # def LA(self,s,Na,z):
# #     if z == 0:
# #         return False
# #     if z == 4:
# #         return False
# #     if z == 1:
# #         return ((Na == s) == z1_masks).all(axis=1).all(axis=1)
# #     if z == 2:
# #         return ((Na == s) == z2_masks).all(axis=1).all(axis=1) #or should be geq?
# #
# #     if z == 3:
# #         return ((Na == s) == z3_masks).all(axis=1).all(axis=1) #or should be geq?
#
#
# #
# #
# # self = cpm
# # import time
# #
# #
# # Na = np.zeros([3,3])*np.nan
# # Na[0,1] = 1
# # s = 1
# #
# # t0 = time.time()
# # for i in range(int(1e5)):
# #     ((Na == s) == self.z1_masks).all(axis=1).all(axis=1).any()
# # t1 = time.time()
# # print(t1-t0)
# #
# # Na = np.zeros([3,3])
# # Na[0,1] = 1
# # s = 1
# #
# # t0 = time.time()
# # for i in range(int(1e5)):
# #     ((Na == s) == self.z1_masks).all(axis=1).all(axis=1).any()
# # t1 = time.time()
# # print(t1-t0)
# #
# # t0 = time.time()
# # for i in range(int(1e5)):
# #     Ni = Na * self.Ni_mask
# #     z = np.sum(Ni == s)
# # t1 = time.time()
# # print(t1-t0)
# #
# # t0 = time.time()
# # for i in range(int(1e5)):
# #     Ni = Na * self.Ni_mask
# #     z = sum(sum(Ni == s))
# # t1 = time.time()
# # print(t1-t0)
# #
# #
# # SX, SY = np.array([[0,1],[1,0],[1,2],[2,1]]).T
# #
# # t0 = time.time()
# # for i in range(int(1e5)):
# #     Na.ravel()[[1,3,5,7]]
# # t1 = time.time()
# # print(t1-t0)
# #
# # t0 = time.time()
# # for i in range(int(1e5)):
# #     Na[self.chooses2i, self.chooses2j]
# # t1 = time.time()
# # print(t1-t0)
# #
# #
# # f
# # cpm.get_s2(I,5,5)
# #
#
# #
# # i_change = []
# # for i in range(int(3e3)-1):
# #
# #     Sum = (cpm.I_save[i] != cpm.I_save[i+1]).sum()
# #     if Sum!=0:
# #         i_change.append(i)
# #
# #
# # i = i_change[2]
# # fig, ax  = plt.subplots(1,2)
# # ax[0].imshow(cpm.I_save[i])
# # ax[1].imshow(cpm.I_save[i+1])
# # plt.show()
# #
# # ii,jj = np.where(cpm.I_save[i]!=cpm.I_save[i+1])
#
#
# # cpm.animate()
# #
# #
# # fig, ax = plt.subplots(2,4)
# # ax = ax.ravel()
# # for i in range(8):
# #     ax[i].imshow(cpm.z2_masks[i])
# # fig.savefig("z2masks.pdf")
# # #
# # fig, ax = plt.subplots(2,4)
# # ax = ax.ravel()
# # for i in range(8):
# #     ax[i].imshow(cpm.z3_masks[i])
# # fig.savefig("z3masks.pdf")
# #
# # fig, ax = plt.subplots(8,4)
# # ax = ax.ravel()
# # for i in range(32):
# #     ax[i].imshow(cpm.z1_masks[i])
# # fig.savefig("z1masks.pdf")
# #
# #
#
# # plt.imshow(cpm.I_save[-2])
# # plt.show()
# #
# # for i in range(1000):
# #
# #     Sum = (cpm.I_save[i] != cpm.I_save[i+1]).sum()
# #     if Sum!=0:
# #         print(i,Sum)
# #
# #
# # i = 6
# # fig, ax  = plt.subplots(1,2)
# # ax[0].imshow(cpm.I_save[i])
# # ax[1].imshow(cpm.I_save[i+1])
# # plt.show()
# #
# #
# #
# # ii,jj = np.where(cpm.I_save[i]!=cpm.I_save[i+1])
# # i,j = 18,17
# #
# # I, I2 = cpm.I_save[i], cpm.I_save[i+1]
# # s, s2 = 0,1


n1 = np.array([2,2,2])
n2 = np.array([1,1,4])

def S(n):
    M = n.size
    N = n.sum()
    return (np.product(n)**(M**-1) - 1)/(N-1)
#

def S(n):
    M = n.size
    N = n.sum()
    return (np.sqrt(1/N * np.sum(n**2)) - np.sqrt(N))/(1-np.sqrt(N))
#

# def S(n):
#     M = n.size
#     N = n.sum()
#     return (np.mean(n) - 1)/(N-1)
# #
# def S(n):
#     M = n.size
#     N = n.sum()
#     hmean = N*np.sum(1/n)**-1
#     return (hmean - 1)/(N-1)
# #

# def S(n):
#     M = n.size
#     N = n.sum()
#     return ((np.product(n) - 1)/(N**M-1))**(M**-1)


def partition(number):
     answer = set()
     answer.add((number, ))
     for x in range(1, number):
         for y in partition(number - x):
             answer.add(tuple(sorted((x, ) + y)))
     return answer

def S_for_cell_N(N):
    part = list(partition(N))
    Ss = []
    for parT in part:
        Ss.append(S(np.array(parT)))
    return Ss,part

Ss, part = S_for_cell_N(15)


plt.plot((np.sort(Ss)))

#
plt.show()

from CPM import *

n_E,n_T = [],[]
for I in cpm.I_save:
    Im = cpm.cell_type_matrix(I)
    for i in np.arange(1, len(cpm.cell_type_list)):
        grid = 1 * (Im == i)
        g = Graph()
        A = np.array([g.assign_islands(grid.tolist())[1]]).reshape(cpm.num_x,cpm.num_y)
        cluster_ids = np.unique(A)
        cluster_ids = cluster_ids[cluster_ids!=0]
        n = np.zeros(cluster_ids.size,dtype=np.int32)
        for j, id in enumerate(cluster_ids):
            n[j] = np.unique(I[A == id]).size
        if i == 1:
            n_E.append(n)
        if i == 2:
            n_T.append(n)

def S_t(ns):
    Ss = []
    for n in ns:
        Ss.append(S(n))
    return Ss

plt.plot(S_t(n_E))
plt.plot(S_t(n_T))
plt.show()

plt.plot(cpm.subpopulations_t)
plt.show()