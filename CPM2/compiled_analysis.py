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
        df = df[::100]
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


bootstrap_df = load_compiled_data("results/compiled/soft")
bootstrap_df["X_external"] = bootstrap_df["X_ex3"] == bootstrap_df["N_X"]
bootstrap_df["E_external"] = bootstrap_df["E_ex3"] == bootstrap_df["N_E"]
bootstrap_df["T_external"] = bootstrap_df["T_ex3"] == bootstrap_df["N_T"]

bootstrap_df["T_sorted"] = bootstrap_df["T_cc"] == 1
bootstrap_df["E_sorted"] = bootstrap_df["E_cc"] == 1
bootstrap_df["ET_sorted"] =( bootstrap_df["E_cc"] == 1)*( bootstrap_df["T_cc"] == 1)

fig, ax = plt.subplots()
sns.lineplot(x="t",y="E_external",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="T_external",data=bootstrap_df,ax=ax)
# sns.lineplot(x="t",y="ET_sorted",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="X_external",data=bootstrap_df,ax=ax)
# ax.set(xlim=(0,3e6))
fig.show()

bootstrap_df["X_external"] = bootstrap_df["X_ex3"] == bootstrap_df["N_X"]
fig, ax = plt.subplots()
sns.lineplot(x="t",y="X_external",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="T_sorted",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="E_sorted",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="ET_sorted",data=bootstrap_df,ax=ax)
fig.show()


soft_df = load_compiled_data("results/compiled/soft")
soft_df["X_external"] = soft_df["X_ex2"] == soft_df["N_X"]
# soft_df["X_external"] = soft_df["X_ex3"] == 6
soft_df["N"] = soft_df["N_X"] +soft_df["N_E"] + soft_df["N_T"]
soft_df["E_external"] = soft_df["E_ex3"] == soft_df["N_E"]
soft_df["T_external"] = soft_df["T_ex3"] == soft_df["N_T"]
stiff_df = load_compiled_data("results/compiled/stiff")
stiff_df["X_external"] = stiff_df["X_ex2"] == stiff_df["N_X"]
# stiff_df["X_external"] = stiff_df["X_ex3"] == 6
stiff_df["N"] = stiff_df["N_X"] +stiff_df["N_E"] + stiff_df["N_T"]
stiff_df["E_external"] = stiff_df["E_ex3"] == stiff_df["N_E"]
stiff_df["T_external"] = stiff_df["T_ex3"] == stiff_df["N_T"]

soft_df["t_million"] = soft_df["t"]/1e6
stiff_df["t_million"] = stiff_df["t"]/1e6

fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(x="t_million",y="X_external",data=soft_df,ax=ax,label="Soft",color="#158701")
sns.lineplot(x="t_million",y="X_external",data=stiff_df,ax=ax,label="Stiff",color="grey")
ax.set(ylabel="Fraction of XEN external",xlabel="Time (x"r"$10^6$"" MCS)")
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
fig.savefig("plots/Fraction XEN external.pdf",dpi=300)

stiff_df["E_sorted"] = stiff_df["E_cc"]==1
soft_df["E_sorted"] = soft_df["E_cc"]==1

stiff_df["T_sorted"] = stiff_df["T_cc"]==1
stiff_df["ET_sorted"] = (stiff_df["E_sorted"] )*(stiff_df["T_sorted"] )
stiff_df["ETX_sorted"] = (stiff_df["E_sorted"] )*(stiff_df["T_sorted"] )*stiff_df["X_external"]

stiff_df_fin = stiff_df[1000::1001]


fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(x="t_million",y="E_sorted",data=stiff_df,ax=ax,label="E",color="#158701")
sns.lineplot(x="t_million",y="T_sorted",data=stiff_df,ax=ax,label="T",color="grey")
sns.lineplot(x="t_million",y="ET_sorted",data=stiff_df,ax=ax,label="ET",color="grey")
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
fig.show()


t_span = stiff_df["t"].values


fig, ax = plt.subplots()
sns.violinplot(ax=ax,data=soft_df[1000::1001],y="X_ex2")
fig.show()




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
