
"""
02_optimize_withholding.py - Step 2 of the post-fire peak flow pipeline

Sweeps train/test witholding percentages across all seeds generated in step 01. 
For each seed, a random forest is trained then evaluated on the test watersheds. 
The R2 and standard deviation of each seed is recorded and saved to a csv for each witholding percentage.

Reads: outputs/Seeds/<witholding>/Seed_<x>/wats_test.csv, outputs/Seeds/<witholding>/Seed_<x>/wats_train.csv

Writes: outputs/WithholdingOptimization/<witholding>.csv

Run: python src/02_optimize_withholding.py
"""
#-------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# -------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent # repo root; auto-derives, no need to edit
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'

METRIC = "PeakArea"  #modeling target (log-transformed) 
N_SEEDS = 100  # number of random train/test splits to generate
WITHOLDINGS = ['50_50', '60_40',  '70_30', '80_20', '90_10']  # list of witholding percentages to sweep across
RF_KWARGS = dict(n_estimators=100, random_state=42)  # keyword arguments for the random forest regressor

#model domain of applicability bounds (applied to the source table before splitting into train/test seeds, see README for details)
BOUNDS = dict(min_drain_sqkm = 50, min_burned_storm_depth_per = 70, 
              max_days_since_fire = 1095, min_mtbs_burnedarea_per = 20) 

#--------------------------------------------MAIN CODE BLOCK ---------------------------------------------------------------------------



# ARI = 1
metric = 'PeakArea' #Peak, Rise, DurabvThresh, VolabvThresh durationabv_1_Area
# metric_short = 'Peak'
# scenario = 'OG'
witholdings = ['50_50', '60_40',  '70_30', '80_20', '90_10']
# witholdings = ['99_1']

# postfire = 'PostFire'


workingPath = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3'
data_file = '{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full\\RF_AttributeTable_ARI1_Peak_modelbounds_Optimized.csv'.format(workingPath)
data = pd.read_csv(data_file)

# chap 3 model bounds: 50km2, 70% burned storm depth, less 3yr post-fire
data = data[(data.DRAIN_SQKM >= 50)]
data = data[(data.MTBS_burnedarea_per >= 20)]
# data = data[data.burned_storm_depth_per > thresh]
data = data[data.burned_storm_depth_per >= 70]
# data = data[data.burned_storm_int_per > thresh]
data = data[data.DaysSinceFire <= 1095]

data = data.drop(columns = ['MTBS_burnedarea_per'])

for witholding in witholdings:
    print(witholding)#, scenario)

    r2s = []
    seeds = []

    # loop through each seed
    # for x in range (0,162):
    for x in range(0, 100):
        print(x)

        watersheds_test_file = '{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full\\Seeds\\{}\\Seed_{}\\wats_test.csv'.format(workingPath, witholding, x)
        watersheds_train_file = '{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full\\Seeds\\{}\\Seed_{}\\wats_train.csv'.format(workingPath, witholding, x)
        watersheds_test_df = pd.read_csv(watersheds_test_file)
        watersheds_train_df = pd.read_csv(watersheds_train_file)
        watersheds_test = watersheds_test_df.GAGE_ID.to_list()
        watersheds_train = watersheds_train_df.GAGE_ID.to_list()

        test = data[data.GAGE_ID.isin(watersheds_test)]
        train = data[data.GAGE_ID.isin(watersheds_train)]

        X_train = train.drop(columns=[metric, 'GAGE_ID'])
        X_test = test.drop(columns=[metric, 'GAGE_ID'])
        # X_train = np.log(X_train)
        # X_test = np.log(X_test)
        # X = pd.concat([X_train, X_test], axis=0)

        # y_train = train[metric]
        # y_test = test[metric]
        y_train = np.log(train[metric])
        y_test = np.log(test[metric])
        # y = pd.concat([y_train, y_test], axis=0)

        cols = data.columns.tolist()

        features = X_train.columns.tolist()

        # Create a random forest classifier
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the classifier
        rf_regressor.fit(X_train, y_train)
        # rf_regressor.fit(X, y)

        if X_test.empty == False:
            # Make predictions on the test set
            y_pred = rf_regressor.predict(X_test)
            # y_pred = rf_regressor.predict(X)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            # mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            r2s.append(r2)
            seeds.append('seed_{}'.format(x))
            # r2 = r2_score(y, y_pred)
            print(f"Mean Squared Error: {mse}, RMSE: {rmse}")
            print(f"R-squared: {r2}")
            stats_df = pd.DataFrame(list(zip([mse], [rmse], [r2])), columns=['mse', 'rmse', 'R2'])
            # print(f"R-squared: {r2_score(y, y_pred)}")
            # stats_df.to_csv('{}\\RandomForest\\RFModels\\StormPercentileVersions\\{}\\{}\\{}\\Seed_{}\\Stats.csv'.format(workingPath,postfire, metric_short, witholding, x))
    withholding_stats_df = pd.DataFrame(list(zip(seeds, r2s)), columns = ['seeds', 'R2'])
    withholding_stats_df.to_csv('{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full\\WithholdingOptimization2\\Withholding2_{}.csv'.format(workingPath, witholding))
            
