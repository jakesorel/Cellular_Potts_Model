import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
from CPM.CPM import CPM
import dask
from dask.distributed import Client
from matplotlib.colors import ListedColormap


def norm(x,xmin,xmax):
    return (x-xmin)/(xmax-xmin)



n_param_step = 12
n_iter = 8
sbb, sgg = np.linspace(0, 1, n_param_step), np.linspace(0, 1, n_param_step)
rep_space = np.arange(n_iter)
BB, GG, NN = np.meshgrid(sbb, sgg, rep_space, indexing="ij")
inputs = np.array([BB.ravel(), GG.ravel(), np.arange(NN.size)]).T

numbers = []
with open('CPM/filenames.txt', encoding="utf-8") as f:
    for line in f:
        __, __, filE = line.split("_")  # line.split("\t") if numbers are seperated by tab
        number, __ = filE.split(".")
        numbers.append(int(number))


inputs = inputs[numbers]
# inputs[:,:-1] = inputs[:,:-1] + np.random.normal(0,1e-1,inputs[:,:-1].shape)

sub_P_end = np.zeros((inputs.shape[0], 2))

cpm = CPM()
cpm.make_grid(100, 100)
cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
n_save = int(2e2)
for i, number in enumerate(inputs[:, -1].astype(int)):
    I_save = load_npz("results_het/I_save_%d.npz" % number).toarray()
    I_save = I_save.reshape(int(I_save.shape[1]/I_save.shape[0]),I_save.shape[0],I_save.shape[0])
    I = I_save[-1]
    sub_P_end[i] = cpm.find_subpopulations(I)


BB, GG = np.meshgrid(sbb, sgg, indexing="ij")
sub_P_end_Mat = np.zeros((n_param_step,n_param_step))
for i, beta in enumerate(sbb):
    for j, gamma in enumerate(sgg):
        wheRe = np.nonzero((inputs[:,0] == beta)&(inputs[:,1]==gamma))
        print(wheRe[0].shape)
        sub_P_end_Mat[i, j] = sub_P_end[wheRe].sum(axis=-1).mean()

sub_P_end_Mat = 2 / sub_P_end_Mat

# plt.imshow(sub_P_end_Mat,cmap=plt.cm.plasma)
# plt.show()

fig, ax = plt.subplots()
b,g,val = BB.ravel(),GG.ravel(),sub_P_end_Mat.ravel()
b,g,val = b[~np.isnan(val)],g[~np.isnan(val)],val[~np.isnan(val)]
subPlim = val.min(), np.percentile(val,80)
val[val>=subPlim[1]] = subPlim[1]*0.99

Levels = np.linspace(subPlim[0], subPlim[1], N)
ax.tricontourf(g,b, val, levels=Levels, cmap=plt.cm.plasma)
ax.set(xlabel=r"$\sigma_{\gamma \gamma}$", ylabel=r"$\sigma_{\beta \beta}$")
my_cmap = ListedColormap(plt.cm.plasma(norm(Levels, subPlim[0], subPlim[1])))
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmax=subPlim[1], vmin=subPlim[0]))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
cl.set_label(r"$S$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
fig.savefig("sorting_het.pdf")



vmax = np.percentile(sub_P_end_Mat,80)
fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(np.flip(sub_P_end_Mat,axis=0),cmap=plt.cm.plasma,vmax = vmax,vmin=sub_P_end_Mat.min(),extent=[0,1,0,1])
ax.set(xlabel=r"$\sigma_{\gamma \gamma}$", ylabel=r"$\sigma_{\beta \beta}$")
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=vmax,vmin=sub_P_end_Mat.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax, pad=0.06, fraction=0.085, aspect=10, orientation="vertical")
cl.set_label(r"$S$")
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.7)
fig.savefig("sort_het.pdf")



def animate_param_combo(a,b,c):
    n_param_step = 12
    n_iter = 8
    sbb, sgg = np.linspace(0, 1, n_param_step), np.linspace(0, 1, n_param_step)
    rep_space = np.arange(n_iter)
    BB, GG, NN = np.meshgrid(sbb, sgg, rep_space, indexing="ij")
    inputs = np.array([BB.ravel(), GG.ravel(), np.arange(NN.size)]).T

    numbers = []
    with open('CPM/filenames.txt', encoding="utf-8") as f:
        for line in f:
            __, __, filE = line.split("_")  # line.split("\t") if numbers are seperated by tab
            number, __ = filE.split(".")
            numbers.append(int(number))


    sbb = np.unique(inputs[:,0])[a]
    sgg = np.unique(inputs[:,1])[b]
    inputs = inputs[numbers]
    inputs = inputs[(inputs[:,0] == sbb)&(inputs[:,1]==sgg)]
    inputs = inputs[c]
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




def get_normal_params(p0, r, beta, gamma,A0):
    """K = 1"""
    P0 = p0*np.sqrt(A0)
    lambda_P = A0/r
    J00 = -P0*lambda_P
    lambda_A = 1
    W = J00*np.array([[0, 0, 0],
                [0, 1, (1-beta)],
                [0, (1-beta), (1+gamma)]])
    return lambda_A,lambda_P,W,P0,A0


def do_job(inputt):
    sbb,sgg,Id = inputt
    cpm = CPM()
    cpm.make_grid(100, 100)
    lambda_A, lambda_P, W, P0, A0 = get_normal_params(p0=8, r=100, beta=0.5, gamma=0, A0=30)
    cpm.lambd_A = lambda_A
    cpm.lambd_P = lambda_P
    cpm.P0 = P0
    cpm.A0 = A0
    cpm.generate_cells(N_cell_dict={"E": 12, "T": 12})
    cpm.set_lambdP(np.array([0.0, lambda_P, lambda_P]))

    J00 = -W[1, 1]
    beta = 0.5
    gamma = 0
    see = 0  # 0.4,0.0,0.4
    sbg, sbe, sge = 0, 0, 0
    eps = 0
    cpm.make_J_ND(J00, beta, gamma, eps, sbb, sbg, sgg, sbe, sge, see)

    # cpm.make_J(W)  # ,sigma = np.ones_like(W)*0.2)
    cpm.make_init("circle", np.sqrt(cpm.A0 / np.pi) * 0.8, np.sqrt(cpm.A0 / np.pi) * 0.2)
    cpm.T = 16
    cpm.I0 = cpm.I
    cpm.run_simulation(int(1e4), int(2e2), polarise=False)
    cpm.generate_image_t(res=4,col_dict={"E":"red","T":"blue","X":"green"})
    cpm.animate()

