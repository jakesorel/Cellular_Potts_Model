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



def radial_polarity_centroids(centroids,XEN_ids):
    XEN_cells = np.zeros(centroids.shape[0]).astype(np.bool)
    XEN_cells[XEN_ids] = 1
    centre = centroids.mean(axis=0)
    displT = centroids - centre
    displacementT = np.mean(np.linalg.norm(displT, axis=1))

    displX = centroids[XEN_cells] - centre
    displacementX = np.mean(np.linalg.norm(displX, axis=1))
    # polarityX = np.mean(displX, axis=0)

    displnX = centroids[~XEN_cells] - centre
    displacementnX = np.mean(np.linalg.norm(displnX, axis=1))
    # polaritynX = np.mean(displnX, axis=0)

    disp = (displacementX-displacementnX)/displacementT
    return disp


def get_radial_polarity_t(i,j):
    nT = 40
    t_span = np.linspace(0,80,nT).astype(np.int64)
    try:
        I_save = load_npz("CPM/polarisation_results_old/%d_%d.npz"%(i,j)).toarray()
        I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
        cpm.I_save = I_save
        disp = np.array([radial_polarity_centroids(cpm.get_centroids(I_save[t]),XEN_ids) for t in t_span])
    except FileNotFoundError:
        disp = np.ones(nT)*np.nan

    return disp



n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
Is,Js = np.arange(10),np.arange(10)
II,JJ = np.meshgrid(Is,Js,indexing="ij")
inputs = np.array([II.ravel(),JJ.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_radial_polarity_t)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)

RPt = np.array(out).reshape(II.shape[0],II.shape[1],40)
nRPt = (RPt.T-RPt[:,:,0].T).T
RPtmean = np.nanmean(nRPt,axis=(1))
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