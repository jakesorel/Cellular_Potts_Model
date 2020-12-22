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
from scipy.sparse import csc_matrix, save_npz,load_npz




n_param_step = 25
repeat_id = 25
p0mult = np.linspace(1,2,n_param_step)
n_save = 200

cpm = CPM()
cpm.make_grid(100, 100)
cpm.generate_cells(N_cell_dict={"E": 10, "T": 10, "X": 14})

XEN_ids = []
for cll in cpm.cells:
    if cll.type is "X":
        XEN_ids.append(cll.id)
#
# def polarity_displacement(C):
#     x,y = np.where(C!=0)
#     centre = np.array([x.mean(),y.mean()])
#     polarity, displacement = np.zeros([2,2]),np.zeros(2)
#     for i, j in enumerate([1,2]):
#         x, y = np.where(C ==j)
#         displ = np.array([x,y]).T - centre
#         polarity[i] = np.mean(displ,axis=0)
#         displacement[i] = np.mean(np.linalg.norm(displ,axis=1))
#     x, y = np.where(C != 0)
#     displT = np.array([x, y]).T - centre
#     displacementT = np.mean(np.linalg.norm(displT, axis=1))
#     pol = np.abs(polarity[0]).sum()/displacementT
#     disp = np.diff(displacement)/displacementT
#     return pol, disp
#
#
# def get_radial_polarity(i,j,k,t):
#     I_save = load_npz("CPM/XEN_p0_results/%d_%d_%d.npz"%(i,j,k)).toarray()
#     I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
#     cpm.I_save = I_save
#     # cpm.generate_image_t(res=4, col_dict={"E": "red", "T": "blue", "X": "green"})
#     # cpm.animate()
#     C = np.zeros_like(I_save[t])
#     for idd in XEN_ids:
#         C += I_save[t] == idd
#     C+=(I_save[t]!=0)
#     # plt.imshow(C)
#     # plt.show()
#     __,disp = polarity_displacement(C)
#     return disp
#
#
# def radial_polarity_centroids(centroids,XEN_ids):
#     XEN_cells = np.zeros(centroids.shape[0]).astype(np.bool)
#     XEN_cells[XEN_ids] = 1
#     centre = centroids.mean(axis=0)
#     displT = centroids - centre
#     displacementT = np.mean(np.linalg.norm(displT, axis=1))
#
#     displX = centroids[XEN_cells] - centre
#     displacementX = np.mean(np.linalg.norm(displX, axis=1))
#     # polarityX = np.mean(displX, axis=0)
#
#     displnX = centroids[~XEN_cells] - centre
#     displacementnX = np.mean(np.linalg.norm(displnX, axis=1))
#     # polaritynX = np.mean(displnX, axis=0)
#
#     disp = (displacementX-displacementnX)/displacementT
#     return disp
#
#
# def get_radial_polarity2(i,j,k,t):
#     I_save = load_npz("CPM/XEN_p0_results2/%d_%d_%d.npz"%(i,j,k)).toarray()
#     I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
#     cpm.I_save = I_save
#     cents = cpm.get_centroids(I_save[t])
#     # cpm.generate_image_t(res=4, col_dict={"E": "red", "T": "blue", "X": "green"})
#     # cpm.animate()
#     # C = np.zeros_like(I_save[t])
#     # for idd in XEN_ids:
#     #     C += I_save[t] == idd
#     # C+=(I_save[t]!=0)
#     # plt.imshow(C)
#     # plt.show()
#     disp = radial_polarity_centroids(cents,XEN_ids)
#     return disp
#
#
# nT = 20
# n_slurm_tasks = 8
# client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
# Is,Js,Ks,Ts = np.arange(25),np.arange(5),np.arange(3),np.linspace(0,199,nT).astype(np.int64)
# II,JJ,KK = np.meshgrid(Is,Js,Ks,Ts,indexing="ij")
# inputs = np.array([II.ravel(),JJ.ravel(),KK.ravel(),Ts])
# lazy_results = []
# for i in np.arange(inputs.shape[0]):
# # for i in range(8):
#     lazy_result = dask.delayed(job)(i)
#     lazy_results.append(lazy_result)
# out = dask.compute(*lazy_results)
#
# RP = np.zeros((25,5,2))
# for i in range(25):
#     for j in range(5):
#         for l,k in enumerate([0,1,2]):
#              RP[i, j, l]= get_radial_polarity(i,j,k,40)
#
# RP = RP.reshape(25,-1)
#
# RPmean = np.mean(RP,axis=1)
#
# fig, ax = plt.subplots()
# for i in range(10):
#     ax.scatter(np.arange(25),RP[:,i],s=6)
# fig.show()
#
# RP2 = np.zeros((25,5,2))
# for i in range(25):
#     for j in range(5):
#         for l,k in enumerate([0,1]):
#              RP2[i, j, l]= get_radial_polarity2(i,j,k,-1)
#
# RP2 = RP2.reshape(25,-1)
#
# RP2mean = np.mean(RP2,axis=1)
#
#
# nT = 20
# RP2t = np.zeros((25,5,2,nT))
# for ti, t in enumerate(np.linspace(0,199,nT).astype(np.int64)):
#     for i in range(25):
#         for j in range(5):
#             for l,k in enumerate([0,1]):
#                  RP2t[i, j, l,ti]= get_radial_polarity2(i,j,k,t)
#
#
#
#
# RP2t = RP2t.reshape(25,10,nT)
# RP2tmean = np.mean(RP2t,axis=1)
#
# plt.imshow(RP2tmean,vmax=0.6)
# plt.show()
#
# """
# ^^ Plot of sorting vs time vs stiffness
# """
#
#
#
# t_span = np.linspace(0,1e4,nT)
#
# sort_vals = np.zeros((25,10))
# for i in range(25):
#     for j in range(10):
#         a,b,c =
#         sort_vals[i,j] = c
#
#
# fig, ax = plt.subplots()
# for i in range(10):
#     ax.scatter(np.arange(25),sort_vals[:,i],s=6)
# ax.set(ylim=(0,2000))
# fig.show()
#
# i,j = 3,0
# plt.scatter(t_span,RP2t[i,j])
# plt.plot(t_span,sort_curve(t_span,*curve_fit(sort_curve,t_span,RP2t[i,j],[0.6,0.6,1000])[0]))
# plt.show()
#
#
# plt.plot(np.median(1/sort_vals,axis=1))
# plt.show()
#
# plt.imshow(RP2tmean,vmax = 0.6)
# plt.show()
#
#
# plt.plot(RPmean)
# plt.plot(RP2mean)
# plt.show()
#
#
# disps1 = [get_radial_polarity(0,0,1,t) for t in range(n_save)]
# disps25 = [get_radial_polarity(24,0,1,t) for t in range(n_save)]
# plt.plot(disps1)
# plt.plot(disps25)
# plt.show()
#
#
# def plot_image(i,j,k,t):
#     I_save = load_npz("CPM/XEN_p0_results/%d_%d_%d.npz" % (i, j, k)).toarray()
#     I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
#     cpm.I_save = I_save
#
#     Im = cpm.generate_image(I_save[t], res=4, col_dict={"E": "red", "T": "blue", "X": "green"})
#     plt.imshow(Im)
#     plt.show()
#
#
#
#
#
#
#
#
#
#




def radial_polarity_centroids(centroids,XEN_ids):
    XEN_cells = np.zeros(centroids.shape[0]).astype(np.bool)
    XEN_cells[XEN_ids] = 1
    nXEN_cells = ~XEN_cells
    nXEN_cells[0] = False
    centre = centroids.mean(axis=0)
    displT = centroids - centre
    displacementT = np.mean(np.linalg.norm(displT, axis=1))

    displX = centroids[XEN_cells] - centre
    displacementX = np.mean(np.linalg.norm(displX, axis=1))
    # polarityX = np.mean(displX, axis=0)

    displnX = centroids[nXEN_cells] - centre
    displacementnX = np.mean(np.linalg.norm(displnX, axis=1))
    # polaritynX = np.mean(displnX, axis=0)

    disp = (displacementX-displacementnX)/displacementT
    return disp


def get_radial_polarity_t(i,j,k):
    nT = 40
    t_span = np.linspace(0,80,nT).astype(np.int64)
    try:
        I_save = load_npz("CPM/XEN_p0_results2/%d_%d_%d.npz"%(i,j,k)).toarray()
        I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
        cpm.I_save = I_save
        disp = np.array([radial_polarity_centroids(cpm.get_centroids(I_save[t]),XEN_ids) for t in t_span])
    except FileNotFoundError:
        disp = np.ones(nT)*np.nan

    return disp



n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
Is,Js,Ks = np.arange(13),np.arange(5),np.array([0,1,2])
II,JJ,KK = np.meshgrid(Is,Js,Ks,indexing="ij")
inputs = np.array([II.ravel(),JJ.ravel(),KK.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_radial_polarity_t)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)

RPt = np.array(out).reshape(II.shape[0],II.shape[1],II.shape[2],40)
nRPt = (RPt.T-RPt[:,:,:,0].T).T
RPtmean = np.nanmean(nRPt,axis=(1,2))
# RPtmean = np.nanmean(RPt[:,:,2],axis=(1))


# fig, ax = plt.subplots()
# ax.imshow(np.flip(RPtmean[:12,:30],axis=0),aspect=3,vmin = 0.15,vmax = 0.4,cmap = plt.cm.Greens)


"""
FINALISED FIGURE START
"""

fig, ax = plt.subplots(figsize=(3.1,3))
vmax = 0.4
ax.imshow(np.flip(RPtmean[:12,:30],axis=0),aspect=3600,vmin = 0.15,vmax = vmax,cmap = plt.cm.Greens,extent=[0,1e4 * 8/20 * (61.538/80),0.1,1.4291667])
ax.set(xlabel="Time (MCS)",ylabel="Relative circumferential \n elastic modulus "r"$(\lambda_P^{XEN} / \lambda_P^{ES})$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.Greens,norm=plt.Normalize(vmax=vmax,vmin=0.15))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.125, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Normalized XEN \n radial asymmetry")
fig.subplots_adjust(top=0.9, bottom=0.25, left=0.2, right=0.75,wspace=0.05)

fig.savefig("XEN radial asymmetry vs time.pdf",dpi=300)


"""
END
"""
#
# lambda_P_span = np.linspace(0.1,3,25)[:12]
# tt_span = np.linspace(0,80,40)[:30]
# TT,LL = np.meshgrid(tt_span,lambda_P_span,indexing="ij")
#
# levels = np.linspace(0.15,0.4,6)
# vals = (RPtmean[:12,:30].T).ravel()
# vals[vals>levels.max()] = levels.max() - 1e-4
# plt.tricontourf(TT.ravel(),LL.ravel(),vals,levels=levels,cmap=plt.cm.Greens)
# plt.show()
#
#
# from scipy.interpolate import bisplrep,bisplev
# xx_new, yy_new = np.mgrid[0:1.4:100j, 0:62:100j]
#
# z = bisplev(xx_new[:,0],yy_new[0], bisplrep(LL.ravel(),TT.ravel(),RPtmean[:12,:30].ravel(),s=5))
# plt.imshow(np.flip(z.T,axis=0))
# plt.show()
#
# from scipy.optimize import curve_fit
#
# def sort_curve(x,a,b,c):
#     return a - b*np.exp(-x/c)
#
# t_span = np.linspace(0,1e4,nT)
#
#
# RPt_smooth = np.zeros_like(RPt)
# for i in range(25):
#     for j in range(10):
#         for k in range(3):
#             RPt_smooth[i,j,k] = sort_curve(t_span, *curve_fit(sort_curve, t_span, RPt[i, j,k], [0.6, 0.6, 1000])[0])
#
#
# RPtmean_smooth = np.mean(RPt_smooth,axis=(1,2))
#
# plt.imshow(RPtmean_smooth,vmax=0.6)
# plt.show()


def plot_image(ax,i,j,k,t):
    I_save = load_npz("CPM/XEN_p0_results2/%d_%d_%d.npz" % (i, j, k)).toarray()
    I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
    cpm.I_save = I_save

    Im = cpm.generate_image(I_save[t], res=4, col_dict={"E": "red", "T": "blue", "X": "green"})

    ax.imshow(Im)

fig, ax = plt.subplots(2,5,sharex=True,sharey=True)
for j in range(5):
    for k in range(2):
        plot_image(ax[k,j],4,j,k,40)
fig.show()