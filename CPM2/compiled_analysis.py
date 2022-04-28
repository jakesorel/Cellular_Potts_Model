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

bootstrap_df = load_compiled_data("results/compiled/soft")

fig, ax = plt.subplots()
sns.lineplot(x="t",y="E_cc",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="T_cc",data=bootstrap_df,ax=ax)
sns.lineplot(x="t",y="X_ex",data=bootstrap_df,ax=ax)
fig.show()

soft_df = load_compiled_data("results/compiled/soft")
stiff_df = load_compiled_data("results/compiled/stiff")
print((soft_df[soft_df["t"]==soft_df["t"].max()]["X_ex"]==6).mean())
print((stiff_df[stiff_df["t"]==stiff["t"].max()]["X_ex"]==6).mean())

