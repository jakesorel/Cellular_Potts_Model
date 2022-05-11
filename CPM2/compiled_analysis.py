import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42


def load_compiled_data(folder):
    files = os.listdir(folder)[:500]
    df_all = pd.DataFrame()
    for file in files:
        index = int(file.split(".csv")[0])
        df = pd.read_csv(folder+"/"+file,index_col=0)
        df = df[::10]
        df["sim"] = index
        df_all = pd.concat((df_all,df))
    df_all.index = np.arange(df_all.shape[0])
    return df_all

"""
Need to check the externalisation analysis again. May just be a zip updating problem. 

And in the analysis, need to flag up cases where cells exist in isolation. For example, could crop the simulation to the largest aggregate. 


"""

"""
There probably ought to be some kind of initialisation. 
"""



soft_df = load_compiled_data("results/compiled/soft")
soft_df["X_external"] = soft_df["X_ex2"] == soft_df["N_X"]
stiff_df = load_compiled_data("results/compiled/stiff")
stiff_df["X_external"] = stiff_df["X_ex2"] == stiff_df["N_X"]
scrambled_df = load_compiled_data("results/compiled/scrambled")
scrambled_df["X_external"] = scrambled_df["X_ex2"] == scrambled_df["N_X"]

soft_df["t_million"] = soft_df["t"]/1e6
stiff_df["t_million"] = stiff_df["t"]/1e6
scrambled_df["t_million"] = scrambled_df["t"]/1e6


fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(x="t_million",y="X_external",data=soft_df,ax=ax,label="Soft",color="#158701")
sns.lineplot(x="t_million",y="X_external",data=stiff_df,ax=ax,label="Stiff",color="grey")
ax.set(ylabel="Fraction of XEN external",xlabel="Time (x"r"$10^6$"" MCS)")
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
# ax.set_yticks(np.arange(0.7,1.05,0.1))
# ax.set_yticklabels(np.arange(70,110,10))
fig.show()
fig.savefig("plots/Fraction XEN external.pdf",dpi=300)

stiff_df["E_sorted"] = stiff_df["E_cc"]==1
soft_df["E_sorted"] = soft_df["E_cc"]==1

stiff_df["T_sorted"] = stiff_df["T_cc"]==1
stiff_df["ET_sorted"] = (stiff_df["E_sorted"] )*(stiff_df["T_sorted"] )
stiff_df["ETX_sorted"] = (stiff_df["E_sorted"] )*(stiff_df["T_sorted"] )*stiff_df["X_external"]
stiff_df["EX_sorted"] = (stiff_df["E_sorted"] )*stiff_df["X_external"]
stiff_df["TX_sorted"] = (stiff_df["T_sorted"] )*stiff_df["X_external"]

scrambled_df["E_sorted"] = scrambled_df["E_cc"]==1
scrambled_df["T_sorted"] = scrambled_df["T_cc"]==1
scrambled_df["ET_sorted"] = (scrambled_df["E_sorted"] )*(scrambled_df["T_sorted"] )
scrambled_df["ETX_sorted"] = (scrambled_df["E_sorted"] )*(scrambled_df["T_sorted"] )*stiff_df["X_external"]
scrambled_df["EX_sorted"] = (scrambled_df["E_sorted"] )*scrambled_df["X_external"]
scrambled_df["TX_sorted"] = (scrambled_df["T_sorted"] )*scrambled_df["X_external"]




fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(x="t_million",y="E_sorted",data=stiff_df,ax=ax,label="E",color="red")
sns.lineplot(x="t_million",y="T_sorted",data=stiff_df,ax=ax,label="T",color="blue")
ax.set(ylabel="Percentage sorted",xlabel="Time (x"r"$10^6$"" MCS)")
ax.set_yticks(np.arange(0,1,0.2))
ax.set_yticklabels(np.arange(0,100,20))
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
fig.savefig("plots/E vs T sorted.pdf")

fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(x="t_million",y="E_sorted",data=scrambled_df,ax=ax,label="E",color="red")
sns.lineplot(x="t_million",y="T_sorted",data=scrambled_df,ax=ax,label="T",color="blue")
ax.set(ylabel="Percentage sorted",xlabel="Time (x"r"$10^6$"" MCS)")
ax.set_yticks(np.arange(0,1,0.2))
ax.set_yticklabels(np.arange(0,100,20))
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
fig.savefig("plots/E vs T sorted scrambled.pdf")



fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(x="t_million",y="ET_sorted",data=stiff_df,ax=ax,label="ET",color=sns.color_palette("colorblind",8)[2],alpha=0.5)
sns.lineplot(x="t_million",y="EX_sorted",data=stiff_df,ax=ax,label="EX",color=sns.color_palette("colorblind",8)[0],alpha=0.5)
sns.lineplot(x="t_million",y="TX_sorted",data=stiff_df,ax=ax,label="TX",color=sns.color_palette("colorblind",8)[1],alpha=0.5)
sns.lineplot(x="t_million",y="ETX_sorted",data=stiff_df,ax=ax,label="ETX",color=sns.color_palette("colorblind",8)[4])
ax.set(ylabel="Percentage sorted",xlabel="Time (x"r"$10^6$"" MCS)")
ax.set_yticks(np.arange(0,1,0.2))

ax.set_yticklabels(np.arange(0,100,20))
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
fig.show()
fig.savefig("plots/ET EX etc sorted.pdf")



fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(x="t_million",y="ET_sorted",data=scrambled_df,ax=ax,label="ET",color=sns.color_palette("colorblind",8)[2],alpha=0.5)
sns.lineplot(x="t_million",y="EX_sorted",data=scrambled_df,ax=ax,label="EX",color=sns.color_palette("colorblind",8)[0],alpha=0.5)
sns.lineplot(x="t_million",y="TX_sorted",data=scrambled_df,ax=ax,label="TX",color=sns.color_palette("colorblind",8)[1],alpha=0.5)
sns.lineplot(x="t_million",y="ETX_sorted",data=scrambled_df,ax=ax,label="ETX",color=sns.color_palette("colorblind",8)[4])
ax.set(ylabel="Percentage sorted",xlabel="Time (x"r"$10^6$"" MCS)")
ax.set_yticks(np.arange(0,1,0.2))

ax.set_yticklabels(np.arange(0,100,20))
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
fig.show()
fig.savefig("plots/ET EX etc sorted scrambled.pdf")





def calculate_bootstrap_ci(vals):
    return sns.utils.ci(sns.algorithms.bootstrap(vals))

def calculate_bootstrap_del_ci(vals):
    mean = vals.mean()
    ci_ = sns.utils.ci(sns.algorithms.bootstrap(vals))
    ci = ci_.copy()
    ci[0] = mean-ci[0]
    ci[1] = ci[1] - mean
    return ci



fig, ax = plt.subplots(figsize=(4,4))
cols = ["red","blue","green","black","black","black","black"]
scrambled_col = "grey"
col_names = ["E_sorted","T_sorted","X_external","ET_sorted","EX_sorted","TX_sorted","ETX_sorted"]
for i, col_name in enumerate(col_names):
    val = stiff_df[100::101][col_name]
    val_scrambled = scrambled_df[100::101][col_name]
    val_ci = calculate_bootstrap_del_ci(val)
    val_scrambled_ci = calculate_bootstrap_del_ci(val_scrambled)
    ax.errorbar(x=(i, i), y=(val_scrambled.mean()*100, val.mean()*100), yerr=np.array((val_scrambled_ci,val_ci)).T*100, fmt='.', ecolor=[scrambled_col, cols[i]],
                zorder=-1)
    ax.scatter(x=(i), y=(val_scrambled.mean()*100), color=scrambled_col, s=10)
    ax.scatter(x=(i), y=(val.mean()*100), color=cols[i], s=10)
ax.set_xticks(np.arange(7))
ax.set_xticklabels(["ES","TS","XEN","ET","EX","TX","ETX"])
ax.set(xlim=[-0.5,6.5])
ax.set(ylabel="Percentage Sorted")
fig.subplots_adjust(bottom=0.4,left=0.3)
fig.savefig("plots/percentage_sorted.pdf",dpi=300)

adhesion_vals_full = np.load("adhesion_matrices/%i.npz" % 1).get("adhesion_vals")

fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(adhesion_vals_full[1:,1:],cmap=plt.cm.hot,vmin=0,vmax=4.54841)
ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=0,vmax=4.54841))
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
cl.set_label(r"$J_{ij}$")
fig.subplots_adjust(left=0.3,right=0.7)
fig.savefig("plots/adhesion_matrix_sample.pdf",dpi=300)


adhesion_vals_full = np.load("adhesion_matrices_scrambled/%i.npz" % 22).get("adhesion_vals")

fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(adhesion_vals_full[1:,1:],cmap=plt.cm.hot,vmin=0,vmax=4.54841)
ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=0,vmax=4.54841))
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
cl.set_label(r"$J_{ij}$")
fig.subplots_adjust(left=0.3,right=0.7)
fig.savefig("plots/adhesion_matrix_scrambled_sample.pdf",dpi=300)


t_span = stiff_df["t"].values


fig, ax = plt.subplots()
sns.violinplot(ax=ax,data=soft_df[1000::1001],y="X_ex2")
fig.show()


def calculate_time_until_sorted(folder):
    files = os.listdir(folder)
    sort_time = np.ones((len(files),3))*np.nan
    for i, file in enumerate(files):
        index = int(file.split(".csv")[0])
        df = pd.read_csv(folder + "/" + file, index_col=0)

        if (df["E_cc"]==1).any():
            sort_time[i,0] = df["t"][np.argmin(df["E_cc"])]
        if (df["T_cc"] == 1).any():
            sort_time[i, 1] = df["t"][np.argmin(df["T_cc"])]
        if (df["X_ex2"] == df["N_X"]).any():
            sort_time[i, 2] = df["t"][np.argmin(df["X_ex2"] != df["N_X"])]
    return sort_time

lg_sort_time = np.log(sort_time)
lg_sort_time[np.isinf(lg_sort_time)] = np.nan

fig, ax = plt.subplots()
ax.hist(sort_time[:,0])
ax.hist(sort_time[:,1])
# sns.distplot(sort_time[:,2],ax=ax)
fig.show()

def fit_expon(data_):
    data = data_[~np.isnan(data_)*~np.isinf(data_)]
    return expon.fit(data)

sort_time_soft = calculate_time_until_sorted("results/compiled/soft")
sort_time_stiff = calculate_time_until_sorted("results/compiled/stiff")

sort_time_df = pd.DataFrame({"id":np.arange(len(sort_time_soft)),"X_soft":sort_time_soft[:,2],"X_stiff":sort_time_stiff[:,2]})
mlt_sort_time_df = sort_time_df.melt(id_vars=["id"],value_vars=["X_soft","X_stiff"])
mlt_sort_time_df = mlt_sort_time_df[~np.isnan(mlt_sort_time_df["value"])]

t = np.linspace(0,1e4)
fig, ax = plt.subplots()
ax.hist(sort_time_soft[:,2],density=True,bins=12,range=(0,5e4),histtype=u'step')
ax.hist(sort_time_stiff[:,2],density=True,bins=12,range=(0,5e4),histtype=u'step')
lambd = fit_expon(sort_time_soft[:,2])[1]
ax.plot(t,1/lambd*np.exp(-t/lambd))
lambd = fit_expon(sort_time_stiff[:,2])[1]
ax.plot(t,1/lambd*np.exp(-t/lambd))
fig.show()

def sample_sort_time(vals,N=int(1e3)):
    out = np.zeros(N)
    for i in range(N):
        out[i] = fit_expon(np.random.choice(vals,len(vals),replace=True))[1]
    return out

def get_mean_ci_sort_time(vals,N=int(1e3)):
    mean = fit_expon(vals)[1]
    dist_sort_time = sample_sort_time(vals,N)
    return mean,(np.percentile(dist_sort_time,5),np.percentile(dist_sort_time,95))


soft_sort_time_fit,soft_sort_time_ci = get_mean_ci_sort_time(sort_time_soft[:,2])
stiff_sort_time_fit,stiff_sort_time_ci = get_mean_ci_sort_time(sort_time_stiff[:,2])
soft_sort_time_ci = np.array([soft_sort_time_fit - soft_sort_time_ci[0],soft_sort_time_ci[1]-soft_sort_time_fit])
stiff_sort_time_ci = np.array([stiff_sort_time_fit - stiff_sort_time_ci[0],stiff_sort_time_ci[1]-stiff_sort_time_fit])

fig, ax = plt.subplots(figsize=(3,4))
ax.errorbar(x=[0,1],y=[soft_sort_time_fit/1e3,stiff_sort_time_fit/1e3],yerr=np.array((soft_sort_time_ci,stiff_sort_time_ci)).T/1e3,fmt='.',ecolor=["#158701","grey"],zorder=-1)
ax.scatter(x=(0), y=(soft_sort_time_fit/1e3), color="#158701", s=10)
ax.scatter(x=(1), y=(stiff_sort_time_fit/1e3), color="grey", s=10)
ax.set_xticks(np.arange(2))
ax.set_xticklabels(["Soft","Stiff"])
ax.set(ylabel="XEN externalization time\n("r'$\times \ 10^3 \ $'"MCS)")
ax.set(ylim=(0,None),xlim=(-0.5,1.5))
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.5,right=0.8)
fig.savefig("plots/XEN externalisation time.pdf")

from scipy.stats import expon


def calculate_bootstrap_ci(vals):
    return sns.utils.ci(sns.algorithms.bootstrap(vals))

def calculate_mean_ci_for_t(df,column,t):
    dfi = df[df["t"]==t]
    vals = dfi[column].values
    return vals.mean(),calculate_bootstrap_ci(vals)

def calculate_mean_ci_for_ti(df,column,ti,n=1001):
    dfi = df[ti::n]
    vals = dfi[column].values
    return vals.mean(),calculate_bootstrap_ci(vals)



t_span = stiff_df["t"].values

soft_mean,soft_ci = np.zeros(1001),np.zeros((1001,2))

for ti in np.arange(1001):
    soft_mean[ti],soft_ci[ti] = calculate_mean_ci_for_ti(soft_df,"X_external",ti)
    print(ti/1001 * 100, "%")


from scipy.signal import savgol_filter

soft_mean_smooth = savgol_filter(soft_mean,101,3)


soft_mean,soft_ci = np.array([calculate_mean_ci_for_ti(soft_df,"X_external",ti) for ti in np.arange(0,1001)]).T
soft_mean,soft_ci = np.array(list(soft_mean)),np.array(list(soft_ci))
