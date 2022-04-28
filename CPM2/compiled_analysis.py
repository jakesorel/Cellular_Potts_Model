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

bootstrap_df = load_compiled_data("results/compiled/stiff")
bootstrap_df["X_external"] = bootstrap_df["X_ex"] == 6
bootstrap_df["T_sorted"] = bootstrap_df["T_cc"] == 1
bootstrap_df["E_sorted"] = bootstrap_df["E_cc"] == 1
bootstrap_df["ET_sorted"] =( bootstrap_df["E_cc"] == 1)*( bootstrap_df["T_cc"] == 1)

fig, ax = plt.subplots()
sns.lineplot(x="t",y="E_cc",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="T_cc",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="X_ex",data=bootstrap_df,ax=ax)
fig.show()

bootstrap_df["X_external"] = bootstrap_df["X_ex"] == 6
fig, ax = plt.subplots()
sns.lineplot(x="t",y="X_external",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="T_sorted",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="E_sorted",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="ET_sorted",data=bootstrap_df,ax=ax)
fig.show()


soft_df = load_compiled_data("results/compiled/soft")
soft_df["X_external"] = soft_df["X_ex"] == 6
stiff_df = load_compiled_data("results/compiled/stiff")
stiff_df["X_external"] = stiff_df["X_ex"] == 6
fig, ax = plt.subplots()
sns.lineplot(x="t",y="X_external",data=soft_df,ax=ax,label="Soft")
sns.lineplot(x="t",y="X_external",data=stiff_df,ax=ax,label="Stiff")
# sns.lineplot(x="t",y="X_external",data=bootstrap_df,ax=ax)
fig.show()


soft_df = load_compiled_data("results/compiled/soft")
stiff_df = load_compiled_data("results/compiled/stiff")
print((soft_df[soft_df["t"]==soft_df["t"].max()]["X_ex"]==6).mean())
print((stiff_df[stiff_df["t"]==stiff_df["t"].max()]["X_ex"]==6).mean())

