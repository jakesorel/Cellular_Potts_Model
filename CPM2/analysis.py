import _pickle as cPickle
import bz2
import os
import sys
import numpy as np
from scipy import sparse

bootstrap_files = os.listdir("results/bootstrap")


c_types = np.zeros(23,dtype=int)
c_types[1:] = 1
c_types[9:] = 2
c_types[17:] = 3

E_mask = c_types==1
T_mask = c_types == 2
X_mask = c_types == 3

i_, j_ = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
Moore = np.array([i_.ravel(), j_.ravel()]).T
i_2, j_2 = np.delete(i_.ravel(), 4), np.delete(j_.ravel(), 4)
perim_neighbour = np.array([i_2, j_2]).T

from scipy.sparse import hstack,vstack

def roll_sparse_ax1(A):
    return hstack((A[:, 1:], A[:, :1]), format='csr')

def roll_sparse_ax0(A):
    return vstack((A[1:], A[:1]), format='csr')

def roll_sparse_ax0and1(A):
    return roll_sparse_ax0(roll_sparse_ax1(A))


perim_neighbour_reduced = np.array([[0,1],[1,0],[1,1]])


def get_adj(I_sparse):
    I = I_sparse.toarray()
    vals = []
    neighs = []
    for i,j in perim_neighbour_reduced:
        rolled = np.roll(np.roll(I,i,axis=0),j,axis=1)
        mask = I!=rolled
        vals += list(I[mask])
        neighs += list(rolled[mask])
    adj = sparse.csr_matrix(sparse.coo_matrix(([True]*len(vals),(vals,neighs))))
    adj += adj.T
    return adj


def get_conn_comp(adj):
    conn_comp = np.zeros(3,dtype=int)
    for i, mask in enumerate([E_mask,T_mask,X_mask]):
        conn_comp[i] = sparse.csgraph.connected_components(adj[mask].T[mask].T)[0]
    return conn_comp



def get_n_external(adj):
    i,j = np.nonzero(adj)
    j_out = j[i==0]
    return np.bincount(c_types.take(j_out))[1:]





def get_conn_comp_t(I_save_sparse):
    conn_comp_t = np.zeros((len(I_save_sparse),3),dtype=int)
    for t, I_sparse in enumerate(I_save_sparse):
        adj = get_adj(I_sparse)
        conn_comp_t[t] = get_conn_comp(adj)
    return conn_comp_t



def get_top_values(I_sparse):
    adj = get_adj(I_sparse)
    conn_comp = get_conn_comp(adj)
    n_external = get_n_external(adj)
    return conn_comp,n_external


def get_top_values_t(I_save_sparse):
    conn_comp_t = np.zeros((len(I_save_sparse),3),dtype=int)
    n_external_t = np.zeros((len(I_save_sparse),3),dtype=int)
    for t, I_sparse in enumerate(I_save_sparse):
        adj = get_adj(I_sparse)
        conn_comp_t[t] = get_conn_comp(adj)
        n_external_t[t] = get_n_external(adj)
    return conn_comp_t,n_external_t

cct = np.zeros((50,40,3))

for i, fl in enumerate(bootstrap_files[:50]):
    index = int(fl.split(".")[0])
    I_save_sparse = cPickle.load(bz2.BZ2File("results/bootstrap/" + fl, 'rb'))
    cct[i] = get_conn_comp_t(I_save_sparse[::25])


plt.plot(cct.mean(axis=0))
plt.show()


plt.scatter(np.arange(40),np.mean((cct[:,:,0]==1)*(cct[:,:,1]==1),axis=0))
plt.show()


cc = np.zeros((500,3))
n_ext = np.zeros((500,3))

for i, fl in enumerate(bootstrap_files):
    index = int(fl.split(".")[0])
    I_save_sparse = cPickle.load(bz2.BZ2File("results/bootstrap/" + fl, 'rb'))
    cc[i],n_ext[i] = get_top_values(I_save_sparse[-1])
    print(i)


cc_scrambled = np.zeros((500,3))
n_ext_scrambled = np.zeros((500,3))

for i, fl in enumerate(bootstrap_files):
    index = int(fl.split(".")[0])
    I_save_sparse = cPickle.load(bz2.BZ2File("results/scrambled/" + fl, 'rb'))
    cc_scrambled[i],n_ext_scrambled[i] = get_top_values(I_save_sparse[-1])
    print(i)


def get_bs_distribution(val,fn,N=100,n=96):
    # n = len(val)
    out = np.zeros(N)
    for i in range(N):
        out[i] = fn(np.random.choice(val,n,replace=True))
    return out

def get_bs_ci(val,fn,N=1000,ci=5):
    distrib = get_bs_distribution(val, fn, N)
    return [np.percentile(distrib,ci),np.percentile(distrib,100-ci)]


def get_bs_ci_err(val,fn,N=1000,ci=5):
    distrib = get_bs_distribution(val, fn, N)
    return [fn(val) - np.percentile(distrib,ci),np.percentile(distrib,100-ci)-fn(val)]


get_bs_distribution((cc[:,0] == 1)*(cc[:,1] == 1),np.mean,N=1000)

get_bs_distribution((cc[:,0] == 1),np.mean,N=1000)


ET_sorted = (cc[:,0] == 1)*(cc[:,1] == 1)
ET_sorted_scrambled = (cc_scrambled[:,0] == 1)*(cc_scrambled[:,1] == 1)

E_sorted = (cc[:,0] == 1)
E_sorted_scrambled = (cc_scrambled[:,0] == 1)

T_sorted = (cc[:,1] == 1)
T_sorted_scrambled = (cc_scrambled[:,1] == 1)

X_sorted = (n_ext[:,2] == 6)
X_sorted_scrambled = (n_ext_scrambled[:,2] == 6)

EX_sorted = E_sorted*X_sorted
EX_sorted_scrambled = E_sorted_scrambled*X_sorted_scrambled

TX_sorted = T_sorted*X_sorted
TX_sorted_scrambled = T_sorted_scrambled*X_sorted_scrambled

ETX_sorted = E_sorted*X_sorted*T_sorted
ETX_sorted_scrambled = E_sorted_scrambled*T_sorted_scrambled*X_sorted_scrambled

fig, ax = plt.subplots(1,7,sharey=True)
cols = ["red","blue","green","black","black","black","black"]
scrambled_col = "grey"
for i, (val,val_scrambled) in enumerate([[E_sorted,E_sorted_scrambled],[T_sorted,T_sorted_scrambled],[X_sorted,X_sorted_scrambled],[ET_sorted,ET_sorted_scrambled],[EX_sorted,EX_sorted_scrambled],[TX_sorted,TX_sorted_scrambled],[ETX_sorted,ETX_sorted_scrambled]]):
    confint = list(zip(get_bs_ci_err(val_scrambled, np.mean), get_bs_ci_err(val, np.mean)))
    ax[i].errorbar(x=(0,1),y=(val_scrambled.mean(),val.mean()),yerr=confint,fmt='o',ecolor=[scrambled_col,cols[i]],zorder=-1)
    ax[i].scatter(x=(0),y=(val_scrambled.mean()),color=scrambled_col)
    ax[i].scatter(x=(1),y=(val.mean()),color=cols[i])
for axx in ax:
    axx.set_xticks([0,1])
    axx.set_xticklabels(["S","M"])
    axx.set(xlim=[-0.5,1.5])
for axx,nm in zip(ax,["ES","TS","X","ET","EX","TX","ETX"]):
    axx.set_title(nm)
ax[0].set(ylabel="Percentage Sorted")
fig.subplots_adjust(bottom=0.4)
fig.show()



fig, ax = plt.subplots(1,figsize=(4,4))
cols = ["red","blue","green","purple","brown","#40e0d0","black"]
scrambled_col = "grey"
for i, (val,val_scrambled) in enumerate([[E_sorted,E_sorted_scrambled],[T_sorted,T_sorted_scrambled],[X_sorted,X_sorted_scrambled],[ET_sorted,ET_sorted_scrambled],[EX_sorted,EX_sorted_scrambled],[TX_sorted,TX_sorted_scrambled],[ETX_sorted,ETX_sorted_scrambled]]):
    confint = list(zip(get_bs_ci_err(val_scrambled, np.mean), get_bs_ci_err(val, np.mean)))
    ax.errorbar(x=(i,i),y=(val_scrambled.mean(),val.mean()),yerr=confint,fmt='.',ecolor=[scrambled_col,cols[i]],zorder=-1)
    ax.scatter(x=(i),y=(val_scrambled.mean()),color=scrambled_col,s=10)
    ax.scatter(x=(i),y=(val.mean()),color=cols[i],s=10)
ax.set_xticks(np.arange(7))
ax.set_xticklabels(["ES","TS","XEN","ET","EX","TX","ETX"])
ax.set(xlim=[-0.5,6.5])
ax.set(ylabel="Percentage Sorted")
fig.subplots_adjust(bottom=0.4,left=0.3)
fig.savefig("plots/percentage_sorted.pdf",dpi=300)


