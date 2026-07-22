## Haley Canham ##
## Jan 2026 ##
## Get the importance ranking for each watershed across all 100 seeds and summarize ##


## libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
import seaborn as sns


workingPath = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full'

# get list of watersheds
# OG_file = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\Peak\\RF_AttributeTable_ARI1_Peak_StormPers_5yr50km2_nocorr.csv'.format(workingPath)
# OG_file = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\Peak\\RF_AttributeTable_ARI1_Peak_modelbounds_17_seed99_dur_burnint.csv'.format(workingPath)
#
# OG = pd.read_csv(OG_file)
#
# watersheds = OG.GAGE_ID.unique().tolist()
watersheds = [ '11458500', '11065000', '9361000']
colors = ['royalblue', 'orange', 'mediumvioletred']

all_watersheds_importances = pd.DataFrame
print('its going')

# make overall plot
r2_df = pd.DataFrame()
mse_df = pd.DataFrame()
rmse_df = pd.DataFrame()
y_pred_df = pd.DataFrame()
y_test_df = pd.DataFrame()
residuals_df = pd.DataFrame()

for x in range(0, 100):
    # read in performance files
    stats_file = '{}\\ModelRun\\Seed_{}\\Stats.csv'.format(workingPath, x)
    stat = pd.read_csv(stats_file)
    residuals_file = '{}\\ModelRun\\Seed_{}\\residuals.csv'.format(
        workingPath, x)
    residual = pd.read_csv(residuals_file)
    y_pred_file = '{}\\ModelRun\\Seed_{}\\y_pred.csv'.format(
        workingPath, x)
    y_pred = pd.read_csv(y_pred_file, index_col = 0)
    y_test_file = '{}\\ModelRun\\Seed_{}\\y_test.csv'.format(
        workingPath, x)
    y_test = pd.read_csv(y_test_file)

    # if r2_df.empty:
    r2_df['seed{}'.format(x)] = stat.R2
    rmse_df['seed{}'.format(x)] = stat.rmse
    mse_df['seed{}'.format(x)] = stat.mse
    y_pred_df['seed{}'.format(x)] = y_pred['0']
    y_test_df['seed{}'.format(x)] = y_test.PeakArea
    residuals_df['seed{}'.format(x)] = residual.PeakArea

r2_df['mean'] = r2_df.mean(axis = 1)
r2_df['std'] = r2_df.std(axis = 1)
rmse_df['mean'] = rmse_df.mean(axis = 1)
rmse_df['std'] = rmse_df.std(axis = 1)
mse_df['mean'] = mse_df.mean(axis = 1)
mse_df['std'] = mse_df.std(axis = 1)

r2_df_t = r2_df.T


#make dfs all one stack and combine
y_pred_stacked = y_pred_df.stack()
y_test_stacked = y_test_df.stack()
residuals_stacked = residuals_df.stack()

combined_df = pd.concat([y_pred_stacked, y_test_stacked], axis = 1)
combined_df.columns = ['y_pred', 'y_test']
k = gaussian_kde(np.vstack([y_pred_stacked, y_test_stacked]))
combined_df['density'] = k(combined_df.T)
combined_df['residuals'] = residuals_stacked

watersheds_r2 = []
watersheds_rmse = []
watersheds_mse = []

fig, ax = plt.subplots()
scatter = ax.scatter(np.exp(combined_df.y_test), np.exp(combined_df.y_pred), alpha = 0.05, c=combined_df['density'],
            cmap='viridis')
fig.colorbar(scatter, ax=ax, label = 'Density')
ax.set_xlabel('Actual peak (m3/s/km2)')
ax.set_ylabel('Predicted peak (m3/s/km2)')

ax.plot([0.001, 2], [0.001, 2], '--', color = 'black', lw=2)
# get individual watershed summary valeus and plot on top

for x in range(0,len(watersheds)):
# for x in range(0, 4):
    watershed = watersheds[x]
    color= colors[x]
    print(watershed)

    watershed_residuals = pd.DataFrame()
    watershed_y_pred = pd.DataFrame()
    watershed_y_test = pd.DataFrame()
    watershed_r2 = pd.DataFrame()
    watershed_rmse = pd.DataFrame()
    watershed_mse = pd.DataFrame()

    # make df of importances
    for i in range(0,100):

        # watershed_path = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\Peak\\StormPers_5yr50km2_nocorr_80_20_watersheds\\Seed_{}\\Watershed_USGS{}'.format(workingPath,i,watershed)
        watershed_path = '{}\\ModelRun_Watersheds\\Seed_{}\\Watershed_USGS{}'.format(workingPath,i,watershed)

        if os.path.exists(watershed_path):
            # print('seed_{} exists'.format(i))
            residual_file = '{}\\residuals.csv'.format(watershed_path)
            residual = pd.read_csv(residual_file)
            y_pred_file = '{}\\y_pred.csv'.format(watershed_path)
            y_pred = pd.read_csv(y_pred_file, index_col=0)
            y_test_file = '{}\\y_test.csv'.format(watershed_path)
            y_test = pd.read_csv(y_test_file)
            stats_file = '{}\\stats.csv'.format(watershed_path)
            stats = pd.read_csv(stats_file)

            watershed_residuals['seed{}'.format(i)] = residual['PeakArea']
            watershed_y_pred['seed{}'.format(i)] = y_pred['0']
            watershed_y_test['seed{}'.format(i)] = y_test['PeakArea']
            watershed_r2['seed{}'.format(i)] = stats['R2']
            watershed_rmse['seed{}'.format(i)] = stats['rmse']
            watershed_mse['seed{}'.format(i)] = stats['mse']

    watersheds_r2.append(watershed_r2.iloc[0].mean())
    watersheds_rmse.append(watershed_rmse.iloc[0].mean())
    watersheds_mse.append(watershed_mse.iloc[0].mean())


    # plot overall plot
    # Predicted vs Actual Plot


    #plot specific watershed on top

    for row in watershed_residuals.index:
        y_pred_list = watershed_y_pred.iloc[row].values.tolist()
        y_test_list = watershed_y_test.iloc[row].values.tolist()
        y_test_value = y_test_list[0]
        ax.boxplot(np.exp(y_pred_list), positions = [np.exp(y_test_value)], widths = [np.exp(y_test_value)/8],
                    showfliers=False, medianprops=dict(color='black'), patch_artist=True,
                   boxprops=dict(facecolor=color, alpha = 0.7), whis = [5,95])


# plt.xlim(0.001, 0.2)
# plt.ylim(0.001, 0.2)
# plt.title(watershed)
plt.loglog()
plt.tight_layout()
plt.savefig('{}\\watershed_performance_forpaper.png'.format(workingPath))

plt.show()
plt.close()

# watersheds_stats_summary = pd.DataFrame(list(zip(watersheds, watersheds_r2, watersheds_rmse, watersheds_mse)), columns = ['watershed', 'r2', 'rmse', 'mse'])
# watersheds_stats_summary.to_csv('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\Peak\\modelbounds_optimized_80_20_watersheds\\WatershedsR2.csv'.format(workingPath))