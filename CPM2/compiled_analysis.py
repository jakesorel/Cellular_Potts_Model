import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def load_compiled_data(folder):
    files = os.listdir(folder)[:500]
    df_all = pd.DataFrame()
    for file in files:
        index = int(file.split(".csv")[0])
        df = pd.read_csv(folder+"/"+file,index_col=0)
        df = df[::50]
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
soft_df["X_external"] = soft_df["X_ex3"] == soft_df["N_X"]
soft_df["X_external"] = soft_df["X_ex3"] == 6
soft_df["N"] = soft_df["N_X"] +soft_df["N_E"] + soft_df["N_T"]

soft_df["E_external"] = soft_df["E_ex3"] == soft_df["N_E"]
soft_df["T_external"] = soft_df["T_ex3"] == soft_df["N_T"]

stiff_df = load_compiled_data("results/compiled/stiff")
stiff_df["X_external"] = stiff_df["X_ex3"] == stiff_df["N_X"]
stiff_df["X_external"] = stiff_df["X_ex3"] == 6
stiff_df["N"] = stiff_df["N_X"] +stiff_df["N_E"] + stiff_df["N_T"]


stiff_df["E_external"] = stiff_df["E_ex3"] == stiff_df["N_E"]
stiff_df["T_external"] = stiff_df["T_ex3"] == stiff_df["N_T"]

fig, ax = plt.subplots()
sns.lineplot(x="t",y="N",data=soft_df,ax=ax,label="Soft")
sns.lineplot(x="t",y="N",data=stiff_df,ax=ax,label="Stiff")
# sns.lineplot(x="t",y="E_cc",data=soft_df,ax=ax,label="Soft")
# sns.lineplot(x="t",y="E_cc",data=stiff_df,ax=ax,label="Stiff")
# ax.set(xlim=(0,3e6))

# sns.lineplot(x="t",y="X_external",data=bootstrap_df,ax=ax)
fig.show()


soft_df = load_compiled_data("results/compiled/soft")
stiff_df = load_compiled_data("results/compiled/stiff")
print((soft_df[soft_df["t"]==soft_df["t"].max()]["X_ex"]==6).mean())
print((stiff_df[stiff_df["t"]==stiff_df["t"].max()]["X_ex"]==6).mean())

