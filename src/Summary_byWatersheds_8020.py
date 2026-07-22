## Haley Canham ##
## Jan 2026 ##
## Get the importance ranking for each watershed across all 100 seeds and summarize ##


## libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

workingPath = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3\\RandomForest\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full'

# get list of watersheds
# OG_file = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\Peak\\RF_AttributeTable_ARI1_Peak_StormPers_5yr50km2_nocorr.csv'.format(workingPath)
OG_file = '{}\\RF_AttributeTable_ARI1_Peak_modelbounds_Optimized.csv'.format(workingPath)

OG = pd.read_csv(OG_file)

watersheds = OG.GAGE_ID.unique().tolist()

all_watersheds_importances = pd.DataFrame

for x in range(0,len(watersheds)):
# for x in range(0, 2):
    watershed = watersheds[x]
    print(watershed)

    watershed_importances = pd.DataFrame
    # make df of importances
    for i in range(0,100):

        # watershed_path = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\Peak\\StormPers_5yr50km2_nocorr_80_20_watersheds\\Seed_{}\\Watershed_USGS{}'.format(workingPath,i,watershed)
        watershed_path = '{}\\ModelRun_Watersheds\\Seed_{}\\Watershed_USGS{}'.format(workingPath,i,watershed)

        if os.path.exists(watershed_path):
            # print('seed_{} exists'.format(i))
            shap_file = '{}\\ShapValues.csv'.format(watershed_path)
            shap = pd.read_csv(shap_file, index_col=0)
            data = '{}\\ShapValues_data.csv'.format(watershed_path)
            data = pd.read_csv(data, index_col=0)

            shap_abs = shap.abs()
            shap_abs_avg = shap_abs.mean()
            shap_abs_avg = shap_abs_avg.sort_values(ascending=False)
            shap_abs_avg.name = 'seed_{}'.format(i)
            shap_abs_avg_df = shap_abs_avg.to_frame()
            shap_abs_avg_df = shap_abs_avg_df.sort_index()

            if watershed_importances.empty:
                watershed_importances = shap_abs_avg_df
            else:
                watershed_importances = watershed_importances.join(shap_abs_avg_df['seed_{}'.format(i)])

    watershed_importances_avg = watershed_importances.mean(axis = 1)
    watershed_importances_avg.name = watershed
    watershed_importances_avg = watershed_importances_avg.sort_values(ascending=False)
    watershed_importances_avg_df = watershed_importances_avg.to_frame()

    if all_watersheds_importances.empty:
        all_watersheds_importances = watershed_importances_avg_df
    else:
        all_watersheds_importances = all_watersheds_importances.join(watershed_importances_avg_df[watershed])

all_watersheds_importances_rank = all_watersheds_importances.rank(ascending=False)

# all_watersheds_importances = all_watersheds_importances.T
# all_watersheds_importances_rank = all_watersheds_importances_rank.T

# all_watersheds_importances.to_csv('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\Peak\\StormPers_5yr50km2_nocorr_80_20_watersheds\\WatershedImportances.csv'.format(workingPath))
# all_watersheds_importances_rank.to_csv('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\Peak\\StormPers_5yr50km2_nocorr_80_20_watersheds\\WatershedImportanceRanks.csv'.format(workingPath))
all_watersheds_importances.to_csv('{}\\ModelRun_Watersheds\\WatershedImportances.csv'.format(workingPath))
all_watersheds_importances_rank.to_csv('{}\\ModelRun_Watersheds\\WatershedImportanceRanks.csv'.format(workingPath))

firsts = []
seconds = []
thirds = []
fourths = []
fifths = []

for watershed in watersheds:
    firsts.append(all_watersheds_importances_rank.index[all_watersheds_importances_rank[watershed] == 1][0])
    seconds.append(all_watersheds_importances_rank.index[all_watersheds_importances_rank[watershed] == 2][0])
    thirds.append(all_watersheds_importances_rank.index[all_watersheds_importances_rank[watershed] == 3][0])
    fourths.append(all_watersheds_importances_rank.index[all_watersheds_importances_rank[watershed] == 4][0])
    fifths.append(all_watersheds_importances_rank.index[all_watersheds_importances_rank[watershed] == 5][0])

watershedfeatureRanks = pd.DataFrame(list(zip(watersheds,firsts,seconds,thirds,fourths,fifths)), columns=['GAGE_ID','first','second','third','fourth','fifth'])
# watershedfeatureRanks.to_csv('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\Peak\\StormPers_5yr50km2_nocorr_80_20_watersheds\\WatershedFeatureRanks.csv'.format(workingPath))
# watershedfeatureRanks.to_csv('{}\\ModelRun_Watersheds\\WatershedFeatureRanks.csv'.format(workingPath))

