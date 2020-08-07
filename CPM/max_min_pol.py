import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
from CPM.CPM import CPM
import dask
from dask.distributed import Client
from matplotlib.colors import ListedColormap


def norm(x,xmin,xmax):
    return (x-xmin)/(xmax-xmin)



def swap_I(I,n_cell):
    i,j = int(np.random.random()*n_cell)+1,int(np.random.random()*n_cell)+1
    J = I.copy()
    J[I==i],J[I==j] = j,i
    return J

def get_polmaxmin(I,n_iter,Tlim,n_cell,maxmin):
    val = cpm.get_pol_stat(I)[0]
    T = Tlim[1] - np.linspace(0,Tlim[0]-Tlim[1],n_iter)
    for i in range(n_iter):
        J = swap_I(I,n_cell)
        new_val = cpm.get_pol_stat(J)[0]
        if maxmin*new_val>maxmin*val:
            val = new_val
            I = J.copy()
        elif np.random.random()>np.exp((maxmin*new_val-maxmin*val)/T[i]):
            val = new_val
            I = J.copy()
    return val


def swap_ct(cell_type_mask):
    """This works only for 2 cell types"""
    A = np.nonzero(cell_type_mask[0])[0]
    B = np.nonzero(cell_type_mask[1])[0]
    i,j = A[int(np.random.random()*A.size)],B[int(np.random.random()*B.size)]
    cell_type_mask2 = cell_type_mask.copy()
    cell_type_mask2[:,i] = np.flip(cell_type_mask[:,i])
    cell_type_mask2[:,j] = np.flip(cell_type_mask[:,j])
    return cell_type_mask2



def get_polmaxmin(centroids,cell_type_mask,n_iter,Tlim,maxmin):
    val = get_pol_stat2(centroids,cell_type_mask)[0]
    T = Tlim[1] - np.linspace(0,Tlim[0]-Tlim[1],n_iter)
    for i in range(n_iter):
        new_cell_type_mask = swap_ct(cell_type_mask)
        new_val = get_pol_stat2(centroids,new_cell_type_mask)[0]
        if maxmin*new_val>maxmin*val:
            val = new_val
            cell_type_mask = new_cell_type_mask
        elif np.random.random()>np.exp((maxmin*new_val-maxmin*val)/T[i]):
            val = new_val
            cell_type_mask = new_cell_type_mask
    return val

get_polmaxmin(centroids,cell_type_mask,100,[10,1],1)

def get_norm_I(I):
    centroids = get_centroids(I)
    pol_max = get_polmaxmin(centroids,cell_type_mask,100,[10,1],1)
    pol_min = get_polmaxmin(centroids,cell_type_mask,100,[10,1],-1)
    pol = get_pol_stat2(centroids,cell_type_mask)[0]
    return (pol-pol_min)/(pol_max-pol_min)



n_param_step = 12
n_iter = 8
beta_space, gamma_space = np.linspace(-1, 1, n_param_step), np.linspace(-1, 1, n_param_step)
rep_space = np.arange(n_iter)
BB, GG, NN = np.meshgrid(beta_space, gamma_space, rep_space, indexing="ij")
inputs = np.array([BB.ravel(), GG.ravel(), np.arange(NN.size)]).T

numbers = []
with open('CPM/filenamesBG.txt', encoding="utf-8") as f:
    for line in f:
        __, __, filE = line.split("_")  # line.split("\t") if numbers are seperated by tab
        number, __ = filE.split(".")
        numbers.append(int(number))


inputs = inputs[numbers]
# inputs[:,:-1] = inputs[:,:-1] + np.random.normal(0,1e-1,inputs[:,:-1].shape)

pol = np.zeros((inputs.shape[0],2))



for i, number in enumerate(inputs[:,-1].astype(int)):
    I_save = load_npz("results_beta_gamma/I_save_%d.npz" % number).toarray()
    I_save = I_save.reshape(int(I_save.shape[1]/I_save.shape[0]),I_save.shape[0],I_save.shape[0])
    I = I_save[-1]
    pol[i] = get_norm_I(I)


BB, GG = np.meshgrid(beta_space, gamma_space, indexing="ij")
pol_Mat = np.zeros((n_param_step,n_param_step))
for i, beta in enumerate(beta_space):
    for j, gamma in enumerate(gamma_space):
        wheRe = np.nonzero((inputs[:,0] == beta)&(inputs[:,1]==gamma))
        print(wheRe[0].shape)
        pol_Mat[i,j] = np.mean(pol[wheRe].sum(axis=-1))


# plt.imshow(sub_P_end_Mat,cmap=plt.cm.plasma)
# plt.show()

fig,ax = plt.subplots()
b,g,val = BB.ravel(),GG.ravel(),pol_Mat.ravel()
b,g,val = b[~np.isnan(val)],g[~np.isnan(val)],val[~np.isnan(val)]
pol_lim = val.min(),val.max()

Levels = np.linspace(pol_lim[0],pol_lim[1],N)
ax.tricontourf(g,b,val,levels=Levels,cmap=plt.cm.plasma)
ax.set(xlabel="$\gamma$",ylabel=r"$\beta$")
my_cmap = ListedColormap(plt.cm.plasma(norm(Levels,pol_lim[0],pol_lim[1])))
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmax=pol_lim[1], vmin=pol_lim[0]))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
cl.set_label(r"$Pol$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
fig.savefig("polarity.pdf")

cell_type_mask = np.zeros((2,cpm.n_cells+1)).astype(bool)
for i in range(1,cpm.n_cells+1):
    for j, cell_type in enumerate(cpm.cell_type_list):
        if cpm.cells[i].type == cell_type:
            cell_type_mask[j-1,i] = True


fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(np.flip(pol_Mat,axis=0),cmap=plt.cm.plasma,vmax = np.percentile(pol_Mat,85),extent=[-1,1,-1,1])
ax.set(ylabel=r"$\beta$",xlabel=r"$\gamma$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=np.percentile(pol_Mat,85), vmin=pol_Mat.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.06, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$Polarity$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.7)
fig.savefig("polarity.pdf")

sub_P_end = np.zeros((inputs.shape[0], 2))

cpm = CPM()
cpm.make_grid(100, 100)
cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
n_save = int(2e2)
for i, number in enumerate(inputs[:, -1].astype(int)):
    I_save = load_npz("results_beta_gamma/I_save_%d.npz" % number).toarray()
    I_save = I_save.reshape(int(I_save.shape[1]/I_save.shape[0]),I_save.shape[0],I_save.shape[0])
    I = I_save[-1]
    sub_P_end[i] = cpm.find_subpopulations(I)


BB, GG = np.meshgrid(beta_space, gamma_space, indexing="ij")
sub_P_end_Mat = np.zeros((n_param_step,n_param_step))
for i, beta in enumerate(beta_space):
    for j, gamma in enumerate(gamma_space):
        wheRe = np.nonzero((inputs[:,0] == beta)&(inputs[:,1]==gamma))
        print(wheRe[0].shape)
        sub_P_end_Mat[i, j] = sub_P_end[wheRe].sum(axis=-1).mean()

sub_P_end_Mat = 2 / sub_P_end_Mat

# plt.imshow(sub_P_end_Mat,cmap=plt.cm.plasma)
# plt.show()

fig, ax = plt.subplots()
b,g,val = BB.ravel(),GG.ravel(),sub_P_end_Mat.ravel()
b,g,val = b[~np.isnan(val)],g[~np.isnan(val)],val[~np.isnan(val)]
subPlim = val.min(), 1

Levels = np.linspace(subPlim[0], subPlim[1], N)
ax.tricontourf(g,b, val, levels=Levels, cmap=plt.cm.plasma)
ax.set(xlabel=r"$\gamma$", ylabel=r"$\beta$")
my_cmap = ListedColormap(plt.cm.plasma(norm(Levels, subPlim[0], subPlim[1])))
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmax=subPlim[1], vmin=subPlim[0]))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
cl.set_label(r"$S$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
fig.savefig("sorting.pdf")



vmax = np.percentile(sub_P_end_Mat,80)
fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(np.flip(sub_P_end_Mat,axis=0),cmap=plt.cm.plasma,vmax = vmax,vmin=sub_P_end_Mat.min(),extent=[-1,1,-1,1])
ax.set(ylabel=r"$\beta$",xlabel=r"$\gamma$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=vmax,vmin=sub_P_end_Mat.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.06, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$S$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.7)
fig.savefig("sort.pdf")

