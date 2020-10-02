import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

df = pd.read_json("results/df_results_mbtr.json", orient="split")
print("Length df:", len(df))

################# plot MAEs in alpha, gamma plane with optimal sigma2, sigma3 values #################################

df_opt = pd.DataFrame(columns=['alpha','gamma','sigma2_opt','sigma3_opt','mae_opt'])


## search best sigma2, sigma3 for evry alpha,gamma combination

for alpha in np.logspace(-10, 0, 11):
    for gamma in np.logspace(-10, 0, 11):
        df_filter = df.loc[(df['alpha']==alpha) &  (df['gamma']==gamma)]
        MAE_best_idx_filter = df_filter[['avg_MAE']].idxmin()
        mae_opt_filter = df_filter.loc[MAE_best_idx_filter, 'avg_MAE'].iat[0]
        #alpha_opt = df.loc[MAE_best_idx, 'alpha'].iat[0]
        #gamma_opt = df.loc[MAE_best_idx, 'gamma'].iat[0]
        sigma2_opt_filter = df_filter.loc[MAE_best_idx_filter, 'sigma2'].iat[0]
        sigma3_opt_filter = df_filter.loc[MAE_best_idx_filter, 'sigma3'].iat[0]
        row = [alpha, gamma, sigma2_opt_filter, sigma3_opt_filter, mae_opt_filter]
        df_opt.loc[len(df_opt)] = row

print("df_opt in alpha, gamma plane:", df_opt)

        #for sigma2 in np.logspace(-5, 0, 6):
        #    for sigma3 in np.logspace(-5, 0, 6):


MAE_best_idx = df_opt[['mae_opt']].idxmin()
mae_opt = df_opt.loc[MAE_best_idx, 'mae_opt'].iat[0]
alpha_opt = df_opt.loc[MAE_best_idx, 'alpha'].iat[0]
gamma_opt = df_opt.loc[MAE_best_idx, 'gamma'].iat[0]
sigma2_opt = df_opt.loc[MAE_best_idx, 'sigma2_opt'].iat[0]
sigma3_opt = df_opt.loc[MAE_best_idx, 'sigma3_opt'].iat[0]

print("mae_opt:", mae_opt)
print("alpha_opt in alpha, gamma plane:", alpha_opt)
print("gamma_opt in alpha, gamma plane:", gamma_opt)
print("sigma2_opt in alpha, gamma plane:", sigma2_opt)
print("sigma3_opt in alpha, gamma plane:", sigma3_opt)


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


pvt = pd.pivot_table(df_opt, index='gamma', columns='alpha', values='mae_opt')
fig, ax = plt.subplots()
heatmap = sns.heatmap(pvt, annot=True, cmap=cm.viridis)
heatmap.invert_xaxis()
plt.xlabel("alpha") 
plt.ylabel("gamma")
plt.title("QM9 4D 4k")
ax.plot(alpha_map, gamma_map, marker='*', color='red', markersize=15)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=35)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=35)
plt.tight_layout()
plt.show()
figure = heatmap.get_figure()
fig.savefig('heatmap_alpha_gamma_opt.png', dpi=300)


################# plot MAEs in sigma2, sigma3 plane with optimal alpha, gamma values #################################

df_opt = pd.DataFrame(columns=['sigma2','sigma3', 'alpha_opt', 'gamma_opt', 'mae_opt'])

for sigma2 in np.logspace(-5, 0, 6):
    for sigma3 in np.logspace(-5, 0, 6):
        df_filter = df.loc[(df['sigma2']==sigma2) &  (df['sigma3']==sigma3)]
        print(df_filter)
        MAE_best_idx_filter = df_filter[['avg_MAE']].idxmin()
        mae_opt_filter = df_filter.loc[MAE_best_idx_filter, 'avg_MAE'].iat[0]
        #alpha_opt = df.loc[MAE_best_idx, 'alpha'].iat[0]
        #gamma_opt = df.loc[MAE_best_idx, 'gamma'].iat[0]
        alpha_opt_filter = df_filter.loc[MAE_best_idx_filter, 'alpha'].iat[0]
        gamma_opt_filter = df_filter.loc[MAE_best_idx_filter, 'gamma'].iat[0]
        row = [sigma2, sigma3, alpha_opt_filter, gamma_opt_filter, mae_opt_filter]
        df_opt.loc[len(df_opt)] = row

print("df_opt in sigma2, sigma3 plane:", df_opt)

        #for sigma2 in np.logspace(-5, 0, 6):
        #    for sigma3 in np.logspace(-5, 0, 6):


MAE_best_idx = df_opt[['mae_opt']].idxmin()
mae_opt = df_opt.loc[MAE_best_idx, 'mae_opt'].iat[0]
sigma2_opt = df_opt.loc[MAE_best_idx, 'sigma2'].iat[0]
sigma3_opt = df_opt.loc[MAE_best_idx, 'sigma3'].iat[0]
alpha_opt = df_opt.loc[MAE_best_idx, 'alpha_opt'].iat[0]
gamma_opt = df_opt.loc[MAE_best_idx, 'gamma_opt'].iat[0]

print("mae_opt:", mae_opt)
print("sigma2_opt in sigma2, sigma3 plane:", sigma2_opt)
print("sigma3_opt in sigma2, sigma3 plane:", sigma3_opt)
print("alpha_opt in sigma2, sigma3 plane:", alpha_opt)
print("gamma_opt in sigma2, sigma3 plane:", gamma_opt)


################# colormap ###############################

if sigma2_opt == 1E-00:
    sigma2_map = 5.5
elif sigma2_opt == 1E-01:
    sigma2_map = 4.5
elif sigma2_opt == 1E-02:
    sigma2_map = 3.5
elif sigma2_opt == 1E-03:
    sigma2_map = 2.5
elif sigma2_opt == 1E-04:
    sigma2_map = 1.5
elif sigma2_opt == 1E-05:
    sigma2_map = 0.5

if sigma3_opt == 1E-00:
    sigma3_map = 5.5
elif sigma3_opt == 1E-01:
    sigma3_map = 4.5
elif sigma3_opt == 1E-02:
    sigma3_map = 3.5
elif sigma3_opt == 1E-03:
    sigma3_map = 2.5
elif sigma3_opt == 1E-04:
    sigma3_map = 1.5
elif sigma3_opt == 1E-05:
    sigma3_map = 0.5


pvt = pd.pivot_table(df_opt, index='sigma3', columns='sigma2', values='mae_opt')
fig, ax = plt.subplots()
heatmap = sns.heatmap(pvt, annot=True, cmap=cm.viridis)
heatmap.invert_xaxis()
plt.xlabel("sigma2") 
plt.ylabel("sigma3")
ax.plot(sigma2_map, sigma3_map, marker='*', color='red', markersize=15)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=35)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=35)
plt.tight_layout()
plt.show()
figure = heatmap.get_figure()
fig.savefig('heatmap.png', dpi=300)
