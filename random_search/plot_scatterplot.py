import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import matplotlib.ticker as ticker


df = pd.read_json("results/df_results_mbtr.json", orient="split") 
MAE_best_idx = df[['avg_MAE']].idxmin()

alpha_opt = df.loc[MAE_best_idx, 'alpha'].iat[0]
alpha_exp_opt = df.loc[MAE_best_idx, 'alpha_exp'].iat[0]
print("alpha_exp_opt:", alpha_exp_opt)

gamma_opt = df.loc[MAE_best_idx, 'gamma'].iat[0]
gamma_exp_opt = df.loc[MAE_best_idx, 'gamma_exp'].iat[0]
print("gamma_exp_opt:", gamma_exp_opt)

sigma2_opt = df.loc[MAE_best_idx, 'sigma2'].iat[0]
sigma2_exp_opt = df.loc[MAE_best_idx, 'sigma2_exp'].iat[0]
print("sigma2_exp_opt:", sigma2_exp_opt)

sigma3_opt = df.loc[MAE_best_idx, 'sigma3'].iat[0]
sigma3_exp_opt = df.loc[MAE_best_idx, 'sigma3_exp'].iat[0]
print("sigma3_exp_opt:", sigma3_exp_opt)

s2_opt = df.loc[MAE_best_idx, 's2'].iat[0]
print("s2_opt:", s2_opt)

s3_opt = df.loc[MAE_best_idx, 's3'].iat[0]
print("s3_opt:", s3_opt)

mae_opt = df.loc[MAE_best_idx, 'avg_MAE'].iat[0]

print("length df:", len(df))
print("MAE opt:", mae_opt)

fig, ax = plt.subplots()
plt.scatter(df['alpha_exp'], df['gamma_exp'], c=df['avg_MAE'], s=70, cmap=plt.cm.viridis)#autumn)

plt.colorbar().set_label(u"MAE [eV]", size=20)
cbar_ax = fig.axes[-1]
cbar_ax.tick_params(labelsize=20)
#plt.plot(alpha_opt, gamma_opt, marker='*', markersize=15, color="red")
plt.plot(alpha_exp_opt, gamma_exp_opt, marker='*', markersize=15, color="red")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"log($\alpha$)", fontsize=20)
plt.ylabel(r"log($\gamma$)", fontsize=20)
plt.tight_layout()
plt.savefig("random_6d_1k.png", dpi=200)
plt.show()
