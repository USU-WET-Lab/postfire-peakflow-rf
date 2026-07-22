## Haley Canham ##
## September 2025 ##
## Generate 100+ random watershed witholding lists - the seeds ##

## libaries
import numpy as np
import pandas as pd
import random
import os

workingPath = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3'

# watersheds_file = '{}\\RandomForest\\WatershedstoRF.csv'.format(workingPath)
# watersheds_file = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFireMultipliers\\WatsWitheld\\MultiplierWatershedsUnique_5yr_50km2.csv'.format(workingPath)
# watersheds_file = '{}\\RandomForest\\RFModels\\Optimization\\ValidationWitholdings\\Undisturbed\\Peak\\RF_AttributeTable_ARI1_Peak_cms.csv'.format(workingPath)
watersheds_file = '{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\OnlySBS\\RF_AttributeTable_ARI1_Peak_modelbounds_SBS_ForUpdatedModelRuns.csv'.format(workingPath)

data = pd.read_csv(watersheds_file)

watersheds = data.GAGE_ID.unique().tolist()
num_watersheds_remove = int(len(watersheds)*0.1) #should be 51 when using all 256 watersheds

# make folder and save lists of watersheds to test and train with
for x in range(0,100):
    watersheds_test = random.sample(watersheds, num_watersheds_remove)
    watersheds_train = list(set(watersheds) - set(watersheds_test))
    # print(watersheds_test)
    watersheds_test_df = pd.DataFrame(watersheds_test, columns=['GAGE_ID'])
    watersheds_train_df = pd.DataFrame(watersheds_train, columns=['GAGE_ID'])

    # os.mkdir('{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full\\Seeds\\90_10\\Seed_{}'.format(workingPath, x))

    watersheds_test_df.to_csv('{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\OnlySBS\\Seeds\\90_10\\Seed_{}\\wats_test.csv'.format(workingPath, x), index=False)
    watersheds_train_df.to_csv('{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\OnlySBS\\Seeds\\90_10\\Seed_{}\\wats_train.csv'.format(workingPath, x), index=False)

# for x in range(0, len(watersheds)):
#     watershed = watersheds[x]
#     print(watershed)
#     watersheds_train = [item for item in watersheds if item != watershed]
#     watersheds_test_df = pd.DataFrame([watershed], columns=['GAGE_ID'])
#     watersheds_train_df = pd.DataFrame(watersheds_train, columns=['GAGE_ID'])
#
#     # os.mkdir('{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\OnlySBS\\Seeds\\99_1\\Seed_{}'.format(workingPath, x))
#
#     watersheds_test_df.to_csv('{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\OnlySBS\\Seeds\\99_1\\Seed_{}\\wats_test.csv'.format(workingPath, x), index=False)
#     watersheds_train_df.to_csv('{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\OnlySBS\\Seeds\\99_1\\Seed_{}\\wats_train.csv'.format(workingPath, x), index=False)
