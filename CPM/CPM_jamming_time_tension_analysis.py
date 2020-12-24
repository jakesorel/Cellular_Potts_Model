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




n_param_step = 15
repeat_id = 15
t0_space = np.linspace((1e4)/4,3*(1e4)/4,n_param_step)
n_save = 200

cpm = CPM()
cpm.make_grid(100, 100)
cpm.generate_cells(N_cell_dict={"E": 10, "T": 10, "X": 0})

def get_num_clusters(i,j):
    I_save = load_npz("CPM/dynamic_tension/%d_%d_1.npz" % (i, j)).toarray()
    I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
    return cpm.find_subpopulations(I_save[-1])


n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
Is,Js = np.arange(15),np.arange(30)
II,JJ = np.meshgrid(Is,Js,indexing="ij")
inputs = np.array([II.ravel(),JJ.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_num_clusters)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)


num_clusters = np.array(out).reshape(II.shape[0],II.shape[1],3)
tot_clusters = num_clusters.sum(axis=2)

for i in range(15):
    plt.scatter(np.arange(15),tot_clusters[:,i])
plt.show()
plt.plot(np.mean(num_clusters.sum(axis=2),axis=1))
plt.show()


t_range = (np.arange(200)[::10]).astype(int)

def get_num_clusters_t(i,j):
    I_save = load_npz("CPM/dynamic_tension/%d_%d_1.npz" % (i, j)).toarray()
    I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
    return np.array([cpm.find_subpopulations(I_save[i]) for i in t_range])


n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
Is,Js = np.arange(15),np.arange(30)
II,JJ = np.meshgrid(Is,Js,indexing="ij")
inputs = np.array([II.ravel(),JJ.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_num_clusters_t)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)


num_clusters = np.array(out).reshape(II.shape[0],II.shape[1],t_range.size,3)
tot_clusters = num_clusters.sum(axis=-1)

for i in range(15):
    plt.scatter(np.arange(15),tot_clusters[:,i])
plt.show()
plt.plot(np.mean(num_clusters.sum(axis=2),axis=1))
plt.show()



ES_ids = []
for i, cll in enumerate(cpm.cells):
    if cll.type == "E":
        ES_ids.append(i)


def polarity_centroids(centroids,ES_ids):
    ES_cells = np.zeros(centroids.shape[0]).astype(np.bool)
    ES_cells[ES_ids] = 1
    TS_cells = ~ES_cells
    TS_cells[0] = False
    centre = centroids[1:].mean(axis=0)
    displT = centroids[1:] - centre
    displacementT = np.mean(np.linalg.norm(displT, axis=1))

    displE = centroids[ES_cells] - centre
    # displacementX = np.mean(np.linalg.norm(displX, axis=1))
    polarityE = np.mean(displE, axis=0)

    displnE = centroids[TS_cells] - centre
    # displacementnX = np.mean(np.linalg.norm(displnX, axis=1))
    polaritynE = np.mean(displnE, axis=0)

    pol = (np.abs(polarityE).sum())/(2*displacementT)
    return pol


def get_polarity_cluster(i,j):
    I_save = load_npz("CPM/dynamic_tension/%d_%d_1.npz" % (i, j)).toarray()
    I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
    cpm.I_save = I_save
    cpm.get_centroids_t()
    pol = np.array([polarity_centroids(centroids, ES_ids) for centroids in cpm.centroids])
    return pol

n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
Is,Js = np.arange(15),np.arange(30)
II,JJ = np.meshgrid(Is,Js,indexing="ij")
inputs = np.array([II.ravel(),JJ.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_polarity_cluster)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)

APt = np.array(out).reshape(15,30,200)

nAPt = (APt.T - APt[:,:,0].T).T

fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = np.percentile(nAPt,10),np.percentile(nAPt,90)
ax.imshow(np.flip(nAPt.mean(axis=1),axis=0),aspect=2,cmap=plt.cm.plasma,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Mean axial \n asymmetry")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.show()
fig.savefig("tension time.pdf",dpi=300)


normAPt = ((APt.mean(axis=1).T - APt.mean(axis=1)[:,0].T)/(APt.mean(axis=1)[:,-1].T - APt.mean(axis=1)[:,0].T)).T
fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = np.percentile(normAPt,10),np.percentile(normAPt,90)
ax.imshow(np.flip(normAPt,axis=0),aspect=2,cmap=plt.cm.plasma,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Normalized mean \n axial asymmetry")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.savefig("tension _norm time.pdf",dpi=300)



dt = 1e4/200

from scipy import ndimage
def get_derivative(nAPtmean,dt):
    """https://www.semicolonworld.com/question/57970/python-finite-difference-functions"""
    f = nAPtmean
    df = np.diff(f) / dt
    cf = np.convolve(f, [1,-1]) / dt
    gf = ndimage.gaussian_filter1d(f, sigma=10, order=1) / dt
    return gf




dnAPt = np.array([get_derivative(APt.mean(axis=1)[i],dt) for i in range(15)])*10**5

fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = np.percentile(dnAPt,20),np.percentile(dnAPt,80)
ax.imshow(np.flip(dnAPt,axis=0),aspect=2,cmap=plt.cm.inferno,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Rate of increase of \n mean axial polarity \n "r"$(\times 10^{-5} \ MCS^{-1})$")
# cl.ticklabel_format(style="sci")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.savefig("tension diff time.pdf",dpi=300)


#
#
# def plot_image(ax,i,j,t):
#     I_save = load_npz("CPM/dynamic_jamming/%d_%d.npz" % (i, j)).toarray()
#     I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
#     cpm.I_save = I_save
#
#     Im = cpm.generate_image(I_save[t], res=4, col_dict={"E": "red", "T": "blue", "X": "green"})
#
#     ax.imshow(Im)
#
# fig, ax = plt.subplots(4,4,sharex=True,sharey=True)
# ax = ax.ravel()
# for j in range(15):
#     plot_image(ax[j],j,0,40)
# fig.show()
#
#
# fig, ax = plt.subplots()
# cols = plt.cm.plasma(np.arange(15)/15)
# for i in range(15):
#     plt.plot(nAPt[i, 1].T,color = cols[i])
# fig.show()
#
#
#
#
#
#
# def get_num_clusters(i,j):
#     I_save = load_npz("CPM/dynamic_jamming/%d_%d.npz" % (i, j)).toarray()
#     I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
#     cpm.I_save = I_save
#     return cpm.find_subpopulations_t()
#
# def get_num_clusters_repeat(i,j):
#     I_save = load_npz("CPM/dynamic_jamming_1/%d_%d.npz" % (i, j)).toarray()
#     I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
#     cpm.I_save = I_save
#
#     return cpm.find_subpopulations_t()
#
#
#
# n_slurm_tasks = 8
# client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
# Is,Js = np.arange(15),np.arange(15)
# II,JJ = np.meshgrid(Is,Js,indexing="ij")
# inputs = np.array([II.ravel(),JJ.ravel()]).T
# inputs = inputs.astype(np.int64)
# lazy_results = []
# for inputt in inputs:
#     lazy_result = dask.delayed(get_num_clusters)(*inputt)
#     lazy_results.append(lazy_result)
# nout = dask.compute(*lazy_results)
#
# lazy_results = []
# for inputt in inputs:
#     lazy_result = dask.delayed(get_num_clusters)(*inputt)
#     lazy_results.append(lazy_result)
# nout2 = dask.compute(*lazy_results)
#
# nout,nout2 = np.array(nout),np.array(nout2)
#
# cluster_index = np.zeros((15,30,200))
# cluster_index[:,:15] = 2/nout.reshape(15,15,200,3).sum(axis=-1)
# cluster_index[:,15:] = 2/nout2.reshape(15,15,200,3).sum(axis=-1)






n_param_step = 15
repeat_id = 15
t0_space = np.linspace((1e4)/4,3*(1e4)/4,n_param_step)
n_save = 200

cpm = CPM()
cpm.make_grid(100, 100)
cpm.generate_cells(N_cell_dict={"E": 8, "T": 8, "X": 0})

def get_num_clusters(i,j):
    I_save = load_npz("CPM/dynamic_jamming/%d_%d.npz" % (i, j)).toarray()
    I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
    return cpm.find_subpopulations(I_save[-1])


n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
Is,Js = np.arange(15),np.arange(15)
II,JJ = np.meshgrid(Is,Js,indexing="ij")
inputs = np.array([II.ravel(),JJ.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_num_clusters)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)


num_clusters = np.array(out).reshape(II.shape[0],II.shape[1],3)
tot_clusters = num_clusters.sum(axis=2)

for i in range(15):
    plt.scatter(np.arange(15),tot_clusters[:,i])
plt.show()
plt.plot(np.mean(num_clusters.sum(axis=2),axis=1))
plt.show()






"""
Reverse
"""



ES_ids = []
for i, cll in enumerate(cpm.cells):
    if cll.type == "E":
        ES_ids.append(i)


def polarity_centroids(centroids,ES_ids):
    ES_cells = np.zeros(centroids.shape[0]).astype(np.bool)
    ES_cells[ES_ids] = 1
    TS_cells = ~ES_cells
    TS_cells[0] = False
    centre = centroids[1:].mean(axis=0)
    displT = centroids[1:] - centre
    displacementT = np.mean(np.linalg.norm(displT, axis=1))

    displE = centroids[ES_cells] - centre
    # displacementX = np.mean(np.linalg.norm(displX, axis=1))
    polarityE = np.mean(displE, axis=0)

    displnE = centroids[TS_cells] - centre
    # displacementnX = np.mean(np.linalg.norm(displnX, axis=1))
    polaritynE = np.mean(displnE, axis=0)

    pol = (np.abs(polarityE).sum())/(2*displacementT)
    return pol


def get_polarity_cluster(i,j):
    I_save = load_npz("CPM/dynamic_jamming_reverse/%d_%d.npz" % (i, j)).toarray()
    I_save = I_save.reshape((n_save, cpm.num_x, cpm.num_y))
    cpm.I_save = I_save
    cpm.get_centroids_t()
    pol = np.array([polarity_centroids(centroids, ES_ids) for centroids in cpm.centroids])
    return pol


n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
Is,Js = np.arange(15),np.arange(15)
II,JJ = np.meshgrid(Is,Js,indexing="ij")
inputs = np.array([II.ravel(),JJ.ravel()]).T
inputs = inputs.astype(np.int64)
lazy_results = []
for inputt in inputs:
    lazy_result = dask.delayed(get_polarity_cluster)(*inputt)
    lazy_results.append(lazy_result)
out = dask.compute(*lazy_results)


APt = np.array(out).reshape(15,15,200)

nAPt = (APt.T - APt[:,:,0].T).T

# fig, ax = plt.subplots(figsize=(3.2,3))
# vmin,vmax = np.percentile(nAPt,10),np.percentile(nAPt,90)
# ax.imshow(np.flip(nAPt.mean(axis=1),axis=0),aspect=2,cmap=plt.cm.plasma,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
# ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
# sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(vmax=vmax,vmin=vmin))
# sm._A = []
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
# cl.set_label("Mean axial \n asymmetry")
# fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
# fig.savefig("jamming time.pdf",dpi=300)


# normAPt = nAPt.mean(axis=1).T/nAPt.mean(axis=1)[:,-1].T
normAPt = ((APt.mean(axis=1).T - APt.mean(axis=1)[:,0].T)/(APt.mean(axis=1)[:,-1].T - APt.mean(axis=1)[:,0].T)).T
fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = np.percentile(normAPt,10),np.percentile(normAPt,90)
ax.imshow(np.flip(normAPt,axis=0),aspect=2,cmap=plt.cm.plasma,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Normalized mean \n axial asymmetry")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.savefig("jamming time reverse.pdf",dpi=300)



nAPtmean = nAPt.mean(axis=1)
dnAPt = np.gradient(nAPtmean,10,axis=1)#nAPtmean[:,1:] - nAPtmean[:,:-1]
dt = 1e4/200


dnAPt = np.array([get_derivative(APt.mean(axis=1)[i],dt) for i in range(15)])*10**5

fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = np.percentile(dnAPt,20),np.percentile(dnAPt,80)
ax.imshow(np.flip(dnAPt,axis=0),aspect=2,cmap=plt.cm.inferno,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Rate of increase of \n mean axial polarity \n "r"$(\times 10^{-5} \ MCS^{-1})$")
# cl.ticklabel_format(style="sci")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.savefig("jamming diff time reverse.pdf",dpi=300)


N = 1000
t_span = np.linspace(0,1e4,N)
t0_span = np.linspace(1e4/4,3e4/4,N)
TT,T0T0 = np.meshgrid(t_span,t0_span,indexing="ij")

lambdP = np.zeros((N,N))
for i, t in enumerate(t_span):
    for j, t0 in enumerate(t0_span):
        lambdP[i,j] = cpm.dynamic_var(t,t0,5,1,100)*0.3



fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = 1,5
ax.imshow(np.flip(lambdP.T,axis=0),aspect=2,cmap=plt.cm.viridis,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Circumferential \n elastic modulus"r"$(\lambda_P)$")
# cl.ticklabel_format(style="sci")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.savefig("forward param.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = 1,5
ax.imshow(np.flip((5-(lambdP-1)).T,axis=0),aspect=2,cmap=plt.cm.viridis,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Circumferential \n elastic modulus"r"$(\lambda_P)$")
# cl.ticklabel_format(style="sci")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.savefig("reverse param.pdf",dpi=300)
