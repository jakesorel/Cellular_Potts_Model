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
    I_save = load_npz("CPM/dynamic_jamming/%d_%d.npz" % (i, j)).toarray()
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


fig, ax = plt.subplots(figsize=(3.2,3))
vmin,vmax = np.percentile(np.array(out),10),np.percentile(np.array(out),90)
ax.imshow(np.array(out).reshape(15,15,200).mean(axis=1),aspect=2,cmap=plt.cm.plasma,extent = [0,1e4,(1e4)/4,(3e4/4)],vmin=vmin,vmax=vmax)
ax.set(xlabel="Time (MCS)",ylabel=r"$t_0$"" (MCS)")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(vmax=vmax,vmin=vmin))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.085, aspect=10, orientation="vertical")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Mean axial \n asymmetry")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.25, right=0.75,wspace=0.05)
fig.savefig("unjamming time.pdf",dpi=300)

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
