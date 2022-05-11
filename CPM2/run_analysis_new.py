import _pickle as cPickle
import bz2
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage.measurements import center_of_mass

if __name__ == "__main__":

    """
    Could determine how many XEN cells are outside by measuring the number of XEN clusters that are completely surrounded by other cell types
    
    Algo: 
    
    Find the connected components of XEN cells 
    For each of these, determine whether any of the neighbours are zero
    If so, the whole cluster is external.  
    """

    c_types = np.zeros(23, dtype=int)
    c_types[1:] = 1
    c_types[9:] = 2
    c_types[17:] = 3

    E_mask = c_types == 1
    T_mask = c_types == 2
    X_mask = c_types == 3

    i_, j_ = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    Moore = np.array([i_.ravel(), j_.ravel()]).T
    i_2, j_2 = np.delete(i_.ravel(), 4), np.delete(j_.ravel(), 4)
    perim_neighbour = np.array([i_2, j_2]).T

    perim_neighbour_reduced = np.array([[0, 1], [1, 0], [1, 1]])

    iter_i = int(sys.argv[1])

    if not os.path.exists("results/compiled"):
        os.mkdir("results/compiled")

    if not os.path.exists("results/compiled/bootstrap"):
        os.mkdir("results/compiled/bootstrap")

    if not os.path.exists("results/compiled/scrambled"):
        os.mkdir("results/compiled/scrambled")

    if not os.path.exists("results/compiled/soft"):
        os.mkdir("results/compiled/soft")

    if not os.path.exists("results/compiled/stiff"):
        os.mkdir("results/compiled/stiff")


    def get_adj(I_sparse):
        I = I_sparse.toarray()
        vals = []
        neighs = []
        for i, j in perim_neighbour_reduced:
            rolled = np.roll(np.roll(I, i, axis=0), j, axis=1)
            mask = I != rolled
            vals += list(I[mask])
            neighs += list(rolled[mask])
        adj = sparse.csr_matrix(
            sparse.coo_matrix(([True] * len(vals), (vals, neighs)), shape=(len(c_types), len(c_types))))
        adj += adj.T
        return adj


    def get_conn_comp(adj, c_types_i):
        ###NOTE THE UPDATE HERE!
        conn_comp = np.zeros(3, dtype=int)
        for i in range(1, 4):
            mask = c_types_i == i
            conn_comp[i-1] = sparse.csgraph.connected_components(adj[mask].T[mask].T)[0]
        return conn_comp


    def remove_non_attached(I_sparse_):
        I_sparse = I_sparse_.copy()
        adj = get_adj(I_sparse)
        n_cc, cc_id = sparse.csgraph.connected_components(adj[1:, 1:])
        c_types_i = c_types.copy()
        if n_cc > 1:
            dom_id = np.argmax(np.bincount(cc_id))
            mask = cc_id != dom_id
            cids_drop = np.nonzero(mask)[0]
            for cid in cids_drop:
                I_sparse[I_sparse == cid + 1] = 0
                c_types_i[cid + 1] = 0
        return I_sparse, c_types_i


    def get_n_external(adj, c_types_i):
        i, j = np.nonzero(adj)
        j_out = j[i == 0]
        return np.bincount(c_types_i.take(j_out), minlength=4)[1:]


    def get_n_external_2(adj, c_types_i):
        n_ext = np.zeros(3, dtype=int)
        b_mask = np.zeros_like(E_mask)
        b_mask[0] = True
        for i in range(3):
            msk = c_types_i == (i + 1)
            mask = msk + b_mask
            bc = np.bincount(sparse.csgraph.connected_components(adj[mask].T[mask].T)[1][1:])
            ne = sum(msk)
            if bc.size > 1:
                ne -= bc[1:].sum()
            n_ext[i] = ne
        return n_ext


    def get_most_external_from_unrolled(unrolled):
        nr = unrolled.shape[1]
        nt = unrolled.shape[0]
        most_external = np.zeros(nt, dtype=int)
        for j in range(nt):
            me = 0
            k = 0
            while (k < nr) * (me == 0):
                me = unrolled[j, -k - 1]
                if me == 0:
                    k += 1
            most_external[j] = me
        return most_external


    def get_n_external_3(I_sparse, c_types_i):
        I = I_sparse.toarray()
        mid_pt = np.array(center_of_mass(I != 0))
        r_max = (100 - mid_pt).min()
        r_max = np.min((r_max, np.min(mid_pt)))
        r_max *= 0.95
        r = np.linspace(0, r_max, 30)
        theta = np.expand_dims(np.linspace(0, np.pi * 2, 100), 1)
        x_i, y_i = mid_pt[0] + r * np.cos(theta), mid_pt[1] + r * np.sin(theta)
        x_i = np.round(x_i).astype(int)
        y_i = np.round(y_i).astype(int)
        unrolled = I[(x_i, y_i)]
        most_external = get_most_external_from_unrolled(unrolled)
        cll_in_theta = np.zeros((len(unrolled), len(c_types)), dtype=bool)
        for i, ct in enumerate(c_types_i):
            cll_in_theta[:, i] = (unrolled == i).any(axis=1)
        cll_same_ctype_as_external = (np.expand_dims(c_types_i.take(most_external), 1) == c_types_i)
        frac_external = (cll_in_theta * cll_same_ctype_as_external).sum(axis=0) / cll_in_theta.sum(axis=0)
        is_external = frac_external > 0.5
        ES_external = is_external[c_types_i == 1].sum()
        TS_external = is_external[c_types_i == 2].sum()
        XEN_external = is_external[c_types_i == 3].sum()
        return np.array([ES_external, TS_external, XEN_external])


    def get_top_values_t(I_save_sparse):
        conn_comp_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_external_1_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_external_2_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_external_3_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_ctypes = np.zeros((len(I_save_sparse), 3), dtype=int)
        external_direction_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        for t, I_sparse_full in enumerate(I_save_sparse):
            I_sparse, c_types_i = remove_non_attached(I_sparse_full)
            n_ctypes[t] = np.bincount(c_types_i, minlength=4)[1:]
            adj = get_adj(I_sparse)
            conn_comp_t[t] = get_conn_comp(adj, c_types_i)
            n_external_1_t[t] = get_n_external(adj, c_types_i)
            n_external_2_t[t] = get_n_external_2(adj, c_types_i)
            n_external_3_t[t] = get_n_external_3(I_sparse, c_types_i)
            external_direction_t[t] = get_external_direction(adj, c_types_i, n_ctypes[t])
        return conn_comp_t, n_ctypes, n_external_1_t, n_external_2_t, n_external_3_t,external_direction_t


    def types_xy_n_external(ctype_i, ctype_j, c_types_i, adj):
        b_mask = np.zeros_like(E_mask)
        b_mask[(c_types_i != ctype_i) * (c_types_i != ctype_j)] = True


        adj_mod = adj.copy()

        i_mask = c_types_i == ctype_i
        j_mask = c_types_i == ctype_j
        k_mask = (c_types_i!=0)*(~i_mask)*(~j_mask)
        ij_mask = i_mask + j_mask
        adj_mod = sparse.csr_matrix(~(np.expand_dims(ij_mask,1)*(ij_mask))).multiply(adj_mod)
        k_flank_outside = np.zeros(adj_mod.shape,dtype=bool)
        k_flank_outside[0] = k_mask
        k_flank_outside[:,0] = k_mask
        adj_mod = adj_mod + sparse.csr_matrix(k_flank_outside)


        n_ext_i = np.bincount(sparse.csgraph.connected_components(adj_mod)[1][i_mask])[0]
        n_ext_j = np.bincount(sparse.csgraph.connected_components(adj_mod)[1][j_mask])[0]

        return n_ext_i,n_ext_j


    def get_n_external_pairs(adj, c_types_i):
        """
        This has a special indexing.
        """
        n_external_mat = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                if i>j:
                    n_external_mat[i, j],n_external_mat[j,i] = types_xy_n_external(i + 1, j + 1, c_types_i, adj)
        return n_external_mat

    def get_external_direction(adj, c_types_i,n_ctypes_i):
        n_external_mat = get_n_external_pairs(adj, c_types_i)
        external_direction = np.zeros(3,dtype=int)
        ##ET,EX,TX i.e. E external to T, E to X, T to X
        external_direction[0] = 1*(n_external_mat[0,1] == n_ctypes_i[0]) -1*(n_external_mat[1,0] == n_ctypes_i[1])
        external_direction[1] = 1*(n_external_mat[0,2] == n_ctypes_i[0]) -1*(n_external_mat[2,0] == n_ctypes_i[2])
        external_direction[2] = 1*(n_external_mat[1,2] == n_ctypes_i[1]) -1*(n_external_mat[2,1] == n_ctypes_i[2])
        return external_direction

    """
    Scoring system:

    If cell type X is external to cell type Y, consider the reduced image of just X and Y cells.
    Then count the number of external cells in the reduced set.

    Then the final structures are: 
    
                (Ecc,Tcc,Xcc),extdir{ET,EX,TX}
    
    Go through the categories sequentially, assigning if unknown is accepted.
    0 E-T-X:      (1,1,1),(0,0,0)
    
    1 X(E(T)):    (X,1,X),(1,-1,-1)
    2 T(E(X)):    (X,X,1),(-1,1,1),
    3 X(T(E)):    (1,X,X),(-1,-1,-1)
    4 E(T(X)):    (X,X,1),(1,1,1)
    5 T(X(E)):    (1,X,X),(-1,-1,1)
    6 E(X(T)):    (X,1,X),(1,1,-1)
    
    7 X(E-T):     (1,1,X),(X,-1,-1)
    8 T(E-X):     (1,X,1),(-1,X,1)
    9 E(T-X):     (X,1,1),(1,1,X)
    
    10 T(E)-X:     (1,X,1),(-1,X,X)
    11 X(E)-T:     (1,1,X),(X,-1,X)
    12 E(X)-T:     (X,1,1),(X,1,X)
    13 T(X)-E:     (1,X,1),(X,X,1)
    14 E(T)-X:     (X,1,1),(1,X,X)
    15 X(T)-E:     (1,1,X),(X,X,-1)

    16 unsorted:   if not any of above.
    """

    score_matrix = np.array((((1,1,1),(0,0,0)),
                    ((np.nan,1,np.nan),(1,-1,-1)),
                    ((np.nan,np.nan,1),(-1,1,1)),
                    ((1,np.nan,np.nan),(-1,-1,-1)),
                    ((np.nan,np.nan,1),(1,1,1)),
                    ((1,np.nan,np.nan),(-1,-1,1)),
                    ((np.nan,1,np.nan),(1,1,-1)),
                    ((1,1,np.nan),(np.nan,-1,-1)),
                    ((1,np.nan,1),(-1,np.nan,1)),
                    ((np.nan,1,1),(1,1,np.nan)),
                    ((1,np.nan,1),(-1,np.nan,np.nan)),
                    ((1,1,np.nan),(np.nan,-1,np.nan)),
                    ((np.nan,1,1),(np.nan,1,np.nan)),
                    ((1,np.nan,1),(np.nan,np.nan,1)),
                    ((np.nan,1,1),(1,np.nan,np.nan)),
                    ((1,1,np.nan),(np.nan,np.nan,-1))))

    n_non_placeholders_score_matrix = np.sum(~np.isnan(score_matrix),axis=-1)

    def conformation_scorer(cc_i,ext_dir_i):
        index = 0
        cont = True
        while (index<16)*(cont):
            cc_true = np.nansum(cc_i == score_matrix[index][0]) == n_non_placeholders_score_matrix[index][0]
            ext_dir_true = np.nansum(ext_dir_i == score_matrix[index][1]) == n_non_placeholders_score_matrix[index][1]
            if cc_true*ext_dir_true:
                cont=False
            else:
                index+=1
        return index


    I_save_sparse = cPickle.load(bz2.BZ2File("results/soft/%d.pbz2" % iter_i, 'rb'))

    cc, n_ctypes, next, next2, next3,ext_dir = get_top_values_t(I_save_sparse)
    conf = np.array([conformation_scorer(cc_i,ext_dir_i) for (cc_i,ext_dir_i) in zip(cc,ext_dir)])

    df = pd.DataFrame({"t": np.arange(cc.shape[0]) * 1e4,
                       "N_E": n_ctypes[:, 0], "N_T": n_ctypes[:, 1], "N_X": n_ctypes[:, 2],
                       "E_cc": cc[:, 0], "T_cc": cc[:, 1], "X_cc": cc[:, 2],
                       "E_ex": next[:, 0], "T_ex": next[:, 1], "X_ex": next[:, 2],
                       "E_ex2": next2[:, 0], "T_ex2": next2[:, 1], "X_ex2": next2[:, 2],
                       "E_ex3": next3[:, 0], "T_ex3": next3[:, 1], "X_ex3": next3[:, 2],
                       "ET_ext":ext_dir[:,0],"EX_ext":ext_dir[:,1],"TX_ext":ext_dir[:,2],
                       "conformation":conf})
    df.to_csv("results/compiled/soft/%d.csv" % iter_i)

    I_save_sparse = cPickle.load(bz2.BZ2File("results/stiff/%d.pbz2" % iter_i, 'rb'))


    cc, n_ctypes, next, next2, next3,ext_dir = get_top_values_t(I_save_sparse)
    conf = np.array([conformation_scorer(cc_i,ext_dir_i) for (cc_i,ext_dir_i) in zip(cc,ext_dir)])

    df = pd.DataFrame({"t": np.arange(cc.shape[0]) * 1e4,
                       "N_E": n_ctypes[:, 0], "N_T": n_ctypes[:, 1], "N_X": n_ctypes[:, 2],
                       "E_cc": cc[:, 0], "T_cc": cc[:, 1], "X_cc": cc[:, 2],
                       "E_ex": next[:, 0], "T_ex": next[:, 1], "X_ex": next[:, 2],
                       "E_ex2": next2[:, 0], "T_ex2": next2[:, 1], "X_ex2": next2[:, 2],
                       "E_ex3": next3[:, 0], "T_ex3": next3[:, 1], "X_ex3": next3[:, 2],
                       "ET_ext":ext_dir[:,0],"EX_ext":ext_dir[:,1],"TX_ext":ext_dir[:,2],
                       "conformation":conf})
    df.to_csv("results/compiled/stiff/%d.csv" % iter_i)

    I_save_sparse = cPickle.load(bz2.BZ2File("results/scrambled/%d.pbz2" % iter_i, 'rb'))


    cc, n_ctypes, next, next2, next3,ext_dir = get_top_values_t(I_save_sparse)
    conf = np.array([conformation_scorer(cc_i,ext_dir_i) for (cc_i,ext_dir_i) in zip(cc,ext_dir)])

    df = pd.DataFrame({"t": np.arange(cc.shape[0]) * 1e4,
                       "N_E": n_ctypes[:, 0], "N_T": n_ctypes[:, 1], "N_X": n_ctypes[:, 2],
                       "E_cc": cc[:, 0], "T_cc": cc[:, 1], "X_cc": cc[:, 2],
                       "E_ex": next[:, 0], "T_ex": next[:, 1], "X_ex": next[:, 2],
                       "E_ex2": next2[:, 0], "T_ex2": next2[:, 1], "X_ex2": next2[:, 2],
                       "E_ex3": next3[:, 0], "T_ex3": next3[:, 1], "X_ex3": next3[:, 2],
                       "ET_ext":ext_dir[:,0],"EX_ext":ext_dir[:,1],"TX_ext":ext_dir[:,2],
                       "conformation":conf})
    df.to_csv("results/compiled/scrambled/%d.csv" % iter_i)
