import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd


df = pd.read_json("results/df_results_mbtr.json", orient="split")
MAE_best_idx = df[['avg_MAE']].idxmin()
alpha_opt = df.loc[MAE_best_idx, 'alpha'].iat[0]
gamma_opt = df.loc[MAE_best_idx, 'gamma'].iat[0]
sigma2_opt = df.loc[MAE_best_idx, 'sigma2'].iat[0]
sigma3_opt = df.loc[MAE_best_idx, 'sigma3'].iat[0]
mae_opt = df.loc[MAE_best_idx, 'avg_MAE'].iat[0]

print(df)
print("alpha_opt:", alpha_opt)
print("gamma_opt:", gamma_opt)
print("sigma2_opt:", sigma2_opt)
print("sigma3_opt:", sigma3_opt)
print("MAE opt:", mae_opt)

if alpha_opt == 1E-00:
    alpha_map = 10.5
elif alpha_opt == 1E-01:
    alpha_map = 9.5
elif alpha_opt == 1E-02:
    alpha_map = 8.5
elif alpha_opt == 1E-03:
    alpha_map = 7.5
elif alpha_opt == 1E-04:
    alpha_map = 6.5
elif alpha_opt == 1E-05:
    alpha_map = 5.5
elif alpha_opt == 1E-06:
    alpha_map = 4.5
elif alpha_opt == 1E-07:
    alpha_map = 3.5
elif alpha_opt == 1E-08:
    alpha_map = 2.5
elif alpha_opt == 1E-09:
    alpha_map = 1.5
elif alpha_opt == 1E-10:
    alpha_map = 0.5

if gamma_opt == 1E-00:
    gamma_map = 10.5
elif gamma_opt == 1E-01:
    gamma_map = 9.5
elif gamma_opt == 1E-02:
    gamma_map = 8.5
elif gamma_opt == 1E-03:
    gamma_map = 7.5
elif gamma_opt == 1E-04:
    gamma_map = 6.5
elif gamma_opt == 1E-05:
    gamma_map = 5.5
elif gamma_opt == 1E-06:
    gamma_map = 4.5
elif gamma_opt == 1E-07:
    gamma_map = 3.5
elif gamma_opt == 1E-08: 
    gamma_map = 2.5 
elif gamma_opt == 1E-09:
    gamma_map = 1.5 
elif gamma_opt == 1E-10:
    gamma_map = 0.5 


pvt = pd.pivot_table(df, values='avg_MAE', index='gamma', columns='alpha')
fig, ax = plt.subplots()
heatmap = sns.heatmap(pvt, cmap=cm.viridis)
heatmap.invert_xaxis()
plt.xlabel("alpha") 
plt.ylabel("gamma")
plt.title("QM9 4D 2k")
ax.plot(alpha_map, gamma_map, marker='*', color='red', markersize=15)
plt.show()
figure = heatmap.get_figure()
fig.savefig('heatmap.png', dpi=300)

