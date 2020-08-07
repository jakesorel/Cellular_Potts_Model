import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
from CPM.CPM import CPM
import dask
from dask.distributed import Client
from matplotlib.colors import ListedColormap

def norm(x,xmin,xmax):
    return (x-xmin)/(xmax-xmin)

def do_job(x):
    l,m,N = x
    n_param_step = 8
    n_iter = 8
    p0_space, r0_space, beta_space, T_space = np.linspace(3, 10, n_param_step), np.logspace(0, 2,n_param_step), np.linspace(0, 1,n_param_step), np.logspace(0, 2, n_param_step)
    rep_space = np.arange(n_iter)
    PP, RR, BB, TT, NN = np.meshgrid(p0_space, r0_space, beta_space, T_space, rep_space, indexing="ij")
    inputs = np.array([PP.ravel(), RR.ravel(), BB.ravel(), TT.ravel(), np.arange(NN.size)]).T

    numbers = []
    with open('CPM/filenames.txt', encoding="utf-8") as f:
        for line in f:
            __, __, filE = line.split("_")  # line.split("\t") if numbers are seperated by tab
            number, __ = filE.split(".")
            numbers.append(int(number))




    beta = np.unique(inputs[:,2])[m]
    r = np.unique(inputs[:,1])[l]
    inputs = inputs[numbers]
    inputs = inputs[(inputs[:,2] == beta)&(inputs[:,1]==r)]
    # inputs[:,:-1] = inputs[:,:-1] + np.random.normal(0,1e-1,inputs[:,:-1].shape)

    sub_P_end = np.zeros((inputs.shape[0],2))


    cpm = CPM()
    cpm.make_grid(100, 100)
    cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
    n_save = int(2e2)
    for i, number in enumerate(inputs[:,-1].astype(int)):
        I_save = load_npz("results/I_save_%d.npz"%number).toarray()
        I_save = I_save.reshape((n_save,100,100))
        cpm.I_save = I_save
        sub_P_end[i] = cpm.find_subpopulations(cpm.I_save[-1])


    PP, TT = np.meshgrid(p0_space,T_space,indexing="ij")
    sub_P_end_Mat = np.zeros((n_param_step,n_param_step))
    for i, p0 in enumerate(p0_space):
        for j, T in enumerate(T_space):
            wheRe = np.nonzero((inputs[:,0] == p0)&(inputs[:,3]==T))
            print(wheRe[0].shape)
            sub_P_end_Mat[i,j] = sub_P_end[wheRe].sum(axis=-1).mean()

    sub_P_end_Mat = 2/sub_P_end_Mat

    # plt.imshow(sub_P_end_Mat,cmap=plt.cm.plasma)
    # plt.show()

    fig,ax = plt.subplots()
    p,t,val = PP.ravel(),TT.ravel(),sub_P_end_Mat.ravel()
    p,t,val = p[~np.isnan(val)],t[~np.isnan(val)],val[~np.isnan(val)]
    subPlim = val.min(),val.max()

    Levels = np.linspace(subPlim[0],subPlim[1],N)
    ax.tricontourf(p,np.log10(t),val,levels=Levels,cmap=plt.cm.plasma)
    ax.set(xlabel="$p_0$",ylabel=r"$log_{10} \ T$")
    my_cmap = ListedColormap(plt.cm.plasma(norm(Levels,subPlim[0],subPlim[1])))
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmax=subPlim[1], vmin=subPlim[0]))
    sm._A = []
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
    cl.set_label(r"$S$")
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
    fig.savefig("CPM/plots/r_%.3f_b_%.3f.pdf"%(r,beta))

do_job([-1,5,20])

def animate_param_combo(a,b,c,d):
    n_param_step = 8
    n_iter = 8
    p0_space, r0_space, beta_space, T_space = np.linspace(3, 10, n_param_step), np.logspace(0, 2,n_param_step), np.linspace(0, 1,n_param_step), np.logspace(0, 2, n_param_step)
    rep_space = np.arange(n_iter)
    PP, RR, BB, TT, NN = np.meshgrid(p0_space, r0_space, beta_space, T_space, rep_space, indexing="ij")
    inputs = np.array([PP.ravel(), RR.ravel(), BB.ravel(), TT.ravel(), np.arange(NN.size)]).T

    numbers = []
    with open('CPM/filenames.txt', encoding="utf-8") as f:
        for line in f:
            __, __, filE = line.split("_")  # line.split("\t") if numbers are seperated by tab
            number, __ = filE.split(".")
            numbers.append(int(number))

    p0 = np.unique(inputs[:,0])[a]
    r = np.unique(inputs[:,1])[b]
    beta = np.unique(inputs[:,2])[c]
    T = np.unique(inputs[:,3])[d]
    inputs = inputs[numbers]
    inputs = inputs[(inputs[:,2] == beta)&(inputs[:,1]==r)&(inputs[:,0] == p0)&(inputs[:,3]==T)]
    inputs = inputs[0]
    number = inputs[-1]



    cpm = CPM()
    cpm.make_grid(100, 100)
    cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
    n_save = int(2e2)
    I_save = load_npz("results/I_save_%d.npz"%number).toarray()
    I_save = I_save.reshape((n_save,100,100))
    cpm.I_save = I_save
    cpm.generate_image_t(res=4,col_dict={"E":"red","T":"blue","X":"green"})
    cpm.animate()



def freq_disfavourable(x):
    l,m,N = x
    n_param_step = 8
    n_iter = 8
    p0_space, r0_space, beta_space, T_space = np.linspace(3, 10, n_param_step), np.logspace(0, 2,n_param_step), np.linspace(0, 1,n_param_step), np.logspace(0, 2, n_param_step)
    rep_space = np.arange(n_iter)
    PP, RR, BB, TT, NN = np.meshgrid(p0_space, r0_space, beta_space, T_space, rep_space, indexing="ij")
    inputs = np.array([PP.ravel(), RR.ravel(), BB.ravel(), TT.ravel(), np.arange(NN.size)]).T

    numbers = []
    with open('CPM/filenames.txt', encoding="utf-8") as f:
        for line in f:
            __, __, filE = line.split("_")  # line.split("\t") if numbers are seperated by tab
            number, __ = filE.split(".")
            numbers.append(int(number))




    beta = np.unique(inputs[:,2])[m]
    r = np.unique(inputs[:,1])[l]
    inputs = inputs[numbers]
    inputs = inputs[(inputs[:,2] == beta)&(inputs[:,1]==r)]
    # inputs[:,:-1] = inputs[:,:-1] + np.random.normal(0,1e-1,inputs[:,:-1].shape)

    freq_Dis = np.zeros((inputs.shape[0],2))
    freq_Fav = np.zeros((inputs.shape[0],2))


    cpm = CPM()
    cpm.make_grid(100, 100)
    cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
    n_save = int(2e2)
    for i, number in enumerate(inputs[:,-1].astype(int)):
        I_save = load_npz("results/I_save_%d.npz"%number).toarray()
        I_save = I_save.reshape((n_save,100,100))
        cpm.I_save = I_save
        cpm.find_subpopulations_t()
        d_subP = cpm.subpopulations_t[1:] - cpm.subpopulations_t[:-1]
        beneficial = np.sum(d_subP<0)
        subopt = np.sum(d_subP>0)
        freq_Dis[i] = subopt
        freq_Fav[i] = beneficial



    PP, TT = np.meshgrid(p0_space,T_space,indexing="ij")
    freq_Dis_Mat = np.zeros((n_param_step,n_param_step))
    for i, p0 in enumerate(p0_space):
        for j, T in enumerate(T_space):
            wheRe = np.nonzero((inputs[:,0] == p0)&(inputs[:,3]==T))
            print(wheRe[0].shape)
            freq_Dis_Mat[i,j] = freq_Dis[wheRe].sum(axis=-1).mean()


    # plt.imshow(sub_P_end_Mat,cmap=plt.cm.plasma)
    # plt.show()

    fig,ax = plt.subplots()
    p,t,val = PP.ravel(),TT.ravel(),freq_Dis_Mat.ravel()
    p,t,val = p[~np.isnan(val)],t[~np.isnan(val)],val[~np.isnan(val)]
    subPlim = val.min(),val.max()

    Levels = np.linspace(subPlim[0],subPlim[1],N)
    ax.tricontourf(p,np.log10(t),val,levels=Levels,cmap=plt.cm.plasma)
    ax.set(xlabel="$p_0$",ylabel=r"$log_{10} \ T$")
    my_cmap = ListedColormap(plt.cm.plasma(norm(Levels,subPlim[0],subPlim[1])))
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmax=subPlim[1], vmin=subPlim[0]))
    sm._A = []
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
    cl.set_label(r"$F$")
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
    fig.savefig("CPM/plots/Freq_swap_r_%.3f_b_%.3f.pdf"%(r,beta))

from scipy.optimize import curve_fit

def anom_diff(t,D,alpha):
    return 4*D*t**alpha

def diff(t,D):
    return 4*D*t


def get_D_alpha(cpm,t0):
    MSD = cpm.get_MSD(t0)
    # (D,alpha),__  = curve_fit(anom_diff,np.linspace(0,1e4,MSD.size),MSD)
    D,__  = curve_fit(diff,np.linspace(0,1e4,MSD.size),MSD)

    return D

def find_MSD(x):
    l,m,N = x
    n_param_step = 8
    n_iter = 8
    p0_space, r0_space, beta_space, T_space = np.linspace(3, 10, n_param_step), np.logspace(0, 2,n_param_step), np.linspace(0, 1,n_param_step), np.logspace(0, 2, n_param_step)
    rep_space = np.arange(n_iter)
    PP, RR, BB, TT, NN = np.meshgrid(p0_space, r0_space, beta_space, T_space, rep_space, indexing="ij")
    inputs = np.array([PP.ravel(), RR.ravel(), BB.ravel(), TT.ravel(), np.arange(NN.size)]).T

    numbers = []
    with open('CPM/filenames.txt', encoding="utf-8") as f:
        for line in f:
            __, __, filE = line.split("_")  # line.split("\t") if numbers are seperated by tab
            number, __ = filE.split(".")
            numbers.append(int(number))




    beta = np.unique(inputs[:,2])[m]
    r = np.unique(inputs[:,1])[l]
    inputs = inputs[numbers]
    inputs = inputs[(inputs[:,2] == beta)&(inputs[:,1]==r)]
    # inputs[:,:-1] = inputs[:,:-1] + np.random.normal(0,1e-1,inputs[:,:-1].shape)


    cpm = CPM()
    cpm.make_grid(100, 100)
    cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
    n_save = int(2e2)

    def job(i):
        number = inputs[i,-1].astype(int)
        I_save = load_npz("results/I_save_%d.npz"%number).toarray()
        I_save = I_save.reshape((n_save,100,100))
        cpm.I_save = I_save
        cpm.get_centroids_t()
        D = get_D_alpha(cpm,0)
        return D

    n_slurm_tasks = 8
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks, memory_limit="1GB")
    lazy_results = []
    for i in np.arange(inputs.shape[0]):
    # for i in range(8):
        lazy_result = dask.delayed(job)(i)
        lazy_results.append(lazy_result)
    out = dask.compute(*lazy_results)
    DD = np.array(out).flatten()


    PP, TT = np.meshgrid(p0_space,T_space,indexing="ij")
    D_Mat = np.zeros((n_param_step,n_param_step))
    for i, p0 in enumerate(p0_space):
        for j, T in enumerate(T_space):
            wheRe = np.nonzero((inputs[:,0] == p0)&(inputs[:,3]==T))
            print(wheRe[0].shape)
            D_Mat[i,j] = DD[wheRe].sum(axis=-1).mean()


    fig,ax = plt.subplots()
    p,t,val = PP.ravel(),TT.ravel(),D_Mat.ravel()
    p,t,val = p[~np.isnan(val)],t[~np.isnan(val)],val[~np.isnan(val)]
    val[val>np.percentile(val,80)] = np.percentile(val,80)
    subPlim = val.min(),val.max()*1.1


    Levels = np.linspace(subPlim[0],subPlim[1],N)
    ax.tricontourf(p,np.log10(t),val,levels=Levels,cmap=plt.cm.plasma)
    ax.set(xlabel="$p_0$",ylabel=r"$log_{10} \ T$")
    my_cmap = ListedColormap(plt.cm.plasma(norm(Levels,subPlim[0],subPlim[1])))
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmax=subPlim[1], vmin=subPlim[0]))
    sm._A = []
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
    cl.set_label(r"$D$")
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
    fig.savefig("CPM/plots/D_r_%.3f_b_%.3f.pdf"%(r,beta))




def get_D_alpha(cpm,t0):
    MSD = cpm.get_MSD(t0)
    # (D,alpha),__  = curve_fit(anom_diff,np.linspace(0,1e4,MSD.size),MSD)
    D,__  = curve_fit(diff,np.linspace(0,1e4,MSD.size),MSD)

    return D



x = [-1,5,20]


animate_param_combo(5,-1,5,-2)

L = np.arange(8)
M = np.repeat(5,8)


n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks,memory_limit="1GB")
lazy_results = []
for inputt in np.array([L,M]).T:
    lazy_result = dask.delayed(do_job)(inputt)
    lazy_results.append(lazy_result)
dask.compute(*lazy_results)

# cpm.get_centroids_t()
# cpm.get_neighbourhood_index2()
# fig, ax = plt.subplots()
# ax.scatter(np.arange(cpm.I_save.shape[0]),cpm.neighbourhood_percentage[:,0],color="r",s=4)
# ax.scatter(np.arange(cpm.I_save.shape[0]),cpm.neighbourhood_percentage[:,1],color="b",s=4)
# ax.set(ylabel="% self-self neighbours")
# fig.show()

fig, ax = plt.subplots()
ax.plot(cpm.subpopulations_t)
fig.show()

# cpm.generate_image_t(res=4,col_dict={"E":"red","T":"blue","X":"green"})
# cpm.animate()



