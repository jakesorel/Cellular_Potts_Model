import _pickle as cPickle
import bz2
import os
import sys
import numpy as np
from scipy import sparse

import pandas as pd

if __name__ == "__main__":

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

    iter_i = int(sys.argv[1])


    if not os.path.exists("results/compiled"):
        os.mkdir("results/compiled")

    if not os.path.exists("results/compiled/bootstrap"):
        os.mkdir("results/compiled/bootstrap")


    if not os.path.exists("results/compiled/scrambled"):
        os.mkdir("results/compiled/scrambled")


    I_save_sparse = cPickle.load(bz2.BZ2File("results/bootstrap/%d.pbz2"%iter_i, 'rb'))

    cc,next = get_top_values_t(I_save_sparse)

    df = pd.DataFrame({"t":np.arange(cc.shape[0])*1e4,"E_cc":cc[:,0],"T_cc":cc[:,1],"X_cc":cc[:,2],
                  "E_ex":next[:,0],"T_ex":next[:,1],"X_ex":next[:,2]})
    df.to_csv("results/compiled/bootstrap/%d.csv"%iter_i)



    I_save_sparse = cPickle.load(bz2.BZ2File("results/scrambled/%d.pbz2"%iter_i, 'rb'))

    cc,next = get_top_values_t(I_save_sparse)

    df = pd.DataFrame({"t":np.arange(cc.shape[0])*1e4,"E_cc":cc[:,0],"T_cc":cc[:,1],"X_cc":cc[:,2],
                  "E_ex":next[:,0],"T_ex":next[:,1],"X_ex":next[:,2]})
    df.to_csv("results/compiled/scrambled/%d.csv"%iter_i)