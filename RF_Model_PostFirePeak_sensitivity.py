## Haley Canham ##
## Sep 2025 ##
## RF model runs with set seeds ##

## libaries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap
import random
import os
import math
from sklearn import metrics

ARI = 1
metric = 'PeakArea' #Peak, Rise, DurabvThresh, VolabvThresh
# metric_short = 'Peak'
# scenario = 'modelbounds_17_seed99_dur_burnint' #OG_10
# withholding = '80_20'
# print(ARI, metric, scenario, withholding)
# save_folder = '{}_{}'.format(metric,ARI)
watershed_interest = 9361000

workingPath = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full'

# threshs = [0]


#create test file
test_file = '{}\\Sensitivity\\RF_AttributeTable_Sensitivity_{}.csv'.format(
    workingPath, watershed_interest)
test_og = pd.read_csv(test_file)
test_add = pd.DataFrame(columns = test_og.columns.tolist())
for x in range(0, len(test_og.index)):
    row_to_copy = test_og.iloc[[x]]
    new_rows = pd.concat([row_to_copy]*51, ignore_index=True)

    # new_rows.at[0, 'WYT'] = 0
    # new_rows.at[1, 'WYT'] = 1

    new_rows.at[0, 'burned_storm_depth_per'] = 70
    new_rows.at[1, 'burned_storm_depth_per'] = 75
    new_rows.at[2, 'burned_storm_depth_per'] = 80
    new_rows.at[3, 'burned_storm_depth_per'] = 85
    new_rows.at[4, 'burned_storm_depth_per'] = 90
    new_rows.at[5, 'burned_storm_depth_per'] = 95
    new_rows.at[6, 'burned_storm_depth_per'] = 100

    new_rows.at[7, 'burned_storm_int_per'] = 70
    new_rows.at[8, 'burned_storm_int_per'] = 75
    new_rows.at[9, 'burned_storm_int_per'] = 80
    new_rows.at[10, 'burned_storm_int_per'] = 85
    new_rows.at[11, 'burned_storm_int_per'] = 90
    new_rows.at[12, 'burned_storm_int_per'] = 95
    new_rows.at[13, 'burned_storm_int_per'] = 100

    # new_rows.at[16, 'burnedStormDur'] = 10
    # new_rows.at[17, 'burnedStormDur'] = 20
    # new_rows.at[18, 'burnedStormDur'] = 30
    # new_rows.at[19, 'burnedStormDur'] = 40
    # new_rows.at[20, 'burnedStormDur'] = 50
    # new_rows.at[21, 'burnedStormDur'] = 60
    # new_rows.at[22, 'burnedStormDur'] = 70
    # new_rows.at[23, 'burnedStormDur'] = 80

    new_rows.at[14, 'MTBS_burnedarea_per'] = 10
    new_rows.at[15, 'MTBS_burnedarea_per'] = 20
    new_rows.at[16, 'MTBS_burnedarea_per'] = 30
    new_rows.at[17, 'MTBS_burnedarea_per'] = 40
    new_rows.at[18, 'MTBS_burnedarea_per'] = 50
    new_rows.at[19, 'MTBS_burnedarea_per'] = 60
    new_rows.at[20, 'MTBS_burnedarea_per'] = 70
    new_rows.at[21, 'MTBS_burnedarea_per'] = 80
    new_rows.at[22, 'MTBS_burnedarea_per'] = 90
    new_rows.at[23, 'MTBS_burnedarea_per'] = 100

    new_rows.at[24, 'SBS_mod_high_area_burned_per_wat'] = 10
    new_rows.at[25, 'SBS_mod_high_area_burned_per_wat'] = 20
    new_rows.at[26, 'SBS_mod_high_area_burned_per_wat'] = 30
    new_rows.at[27, 'SBS_mod_high_area_burned_per_wat'] = 40
    new_rows.at[28, 'SBS_mod_high_area_burned_per_wat'] = 50
    new_rows.at[29, 'SBS_mod_high_area_burned_per_wat'] = 60
    new_rows.at[30, 'SBS_mod_high_area_burned_per_wat'] = 70
    new_rows.at[31, 'SBS_mod_high_area_burned_per_wat'] = 80


    new_rows.at[32, 'AntecedentPrecip_mm'] = 0
    new_rows.at[33, 'AntecedentPrecip_mm'] = 12.7
    new_rows.at[34, 'AntecedentPrecip_mm'] = 25.4
    new_rows.at[35, 'AntecedentPrecip_mm'] = 38.1
    new_rows.at[36, 'AntecedentPrecip_mm'] = 50.8
    new_rows.at[37, 'AntecedentPrecip_mm'] = 63.5
    new_rows.at[38, 'AntecedentPrecip_mm'] = 76.2
    new_rows.at[39, 'AntecedentPrecip_mm'] = 88.9
    new_rows.at[40, 'AntecedentPrecip_mm'] = 101.6
    new_rows.at[41, 'AntecedentPrecip_mm'] = 114.3
    new_rows.at[42, 'AntecedentPrecip_mm'] = 127


    new_rows.at[43, 'AvgMonthly_Precip_cm'] = 1
    new_rows.at[44, 'AvgMonthly_Precip_cm'] = 5
    new_rows.at[45, 'AvgMonthly_Precip_cm'] = 10
    new_rows.at[46, 'AvgMonthly_Precip_cm'] = 15
    new_rows.at[47, 'AvgMonthly_Precip_cm'] = 20
    new_rows.at[48, 'AvgMonthly_Precip_cm'] = 25
    new_rows.at[49, 'AvgMonthly_Precip_cm'] = 30
    new_rows.at[50, 'AvgMonthly_Precip_cm'] = 35


    test_add = pd.concat([test_add, new_rows], ignore_index=True)
test = pd.concat([test_og, test_add], ignore_index=True)
test.to_csv('{}\\Sensitivity\\test_{}.csv'.format(workingPath, watershed_interest))

y_preds_df = pd.DataFrame()

# for thresh in threshs:
    # print(thresh)
    # print('ARI1 depth')
data_file = '{}\\RF_AttributeTable_ARI1_Peak_modelbounds_Optimized.csv'.format(workingPath)
data = pd.read_csv(data_file)


r2s = []

# loop through each seed
for x in range (0,100):
# for x in range(85, 86):
    print(x)

    if not os.path.exists('{}\\Sensitivity\\USGS{}\\Seed_{}'.format(workingPath, watershed_interest, x)):
        os.mkdir('{}\\Sensitivity\\USGS{}\\Seed_{}'.format(workingPath,watershed_interest, x))
        os.mkdir('{}\\Sensitivity\\USGS{}\\Seed_{}\\PD'.format(workingPath, watershed_interest, x))
        os.mkdir('{}\\Sensitivity\\USGS{}\\Seed_{}\\shapPD'.format(workingPath, watershed_interest, x))
        os.mkdir('{}\\Sensitivity\\USGS{}\\Seed_{}\\Waterfalls'.format(workingPath, watershed_interest, x))

    # watersheds_test_file = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\WatsWitheld\\PeakModel_modelbounds\\{}\\Seed_{}\\wats_test.csv'.format(workingPath, withholding, x)
    watersheds_train_file = '{}\\Seeds\\80_20\\Seed_{}\\wats_train.csv'.format(workingPath, x)
    # watersheds_test_df = pd.read_csv(watersheds_test_file)
    watersheds_train_df = pd.read_csv(watersheds_train_file)
    watersheds_train = watersheds_train_df.GAGE_ID.to_list()
    watersheds_test = watershed_interest

    if watershed_interest in watersheds_train:
        watersheds_train.remove(watershed_interest)
        print('removed from seed {}'.format(x))

    # test = data[data.GAGE_ID.isin(watersheds_test)]
    train = data[data.GAGE_ID.isin(watersheds_train)]


    X_train = train.drop(columns=[metric, 'GAGE_ID'])
    X_test = test.drop(columns=[metric, 'GAGE_ID'])

    y_train = np.log(train[metric])
    y_test = np.log(test[metric])

    # print('Train: {}, Test: {}, %Test: {}'.format(len(X_train), len(X_test), len(X_test)/len(X_train)*100))

    cols = data.columns.tolist()

    features = X_train.columns.tolist()

    # Create a random forest classifier
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the classifier
    rf_regressor.fit(X_train, y_train)
    # rf_regressor.fit(X, y)

    # Make predictions on the test set
    y_pred = rf_regressor.predict(X_test)
    # y_pred = rf_regressor.predict(X)


    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred)
    # # mse = mean_squared_error(y, y_pred)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_test, y_pred)
    # # r2 = r2_score(y, y_pred)
    # # print(f"Mean Squared Error: {mse}, RMSE: {rmse}")
    # # print(f"R-squared: {r2}")
    # stats_df = pd.DataFrame(list(zip([mse], [rmse], [r2])), columns=['mse', 'rmse', 'R2'])
    # # print(f"R-squared: {r2_score(y, y_pred)}")
    # stats_df.to_csv('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Sensitivity\\Seed_{}\\Stats.csv'.format(workingPath,metric_short, x))
    # r2s.append(r2)
    #
    # print('{} %Test: {}, R2: {}'.format(x, (len(X_test) / len(X_train) * 100), r2))

    # print('median: {}'.format(np.median(r2s)))
    # print('mean: {}'.format(np.mean(r2s)))

    # output y files
    y_test.to_csv('{}\\Sensitivity\\USGS{}\\Seed_{}\\y_test.csv'.format(workingPath,watershed_interest, x))
    y_pred_series = pd.Series(y_pred)
    y_pred_series.to_csv('{}\\Sensitivity\\USGS{}\\Seed_{}\\y_pred.csv'.format(workingPath, watershed_interest, x))

    # make df of y_preds
    y_preds_df[x] = y_pred_series



# Get feature importances
    importances = rf_regressor.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    features_sorted = []
    importances_sorted = []
    # Print ranked feature importances
    # print("Feature ranking:")
    for f in range(0,len(X_test.columns)):
    # for f in range(0, 5):
    #     print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
        features_sorted.append(features[indices[f]])
        importances_sorted.append(importances[indices[f]])

    importances_df = pd.DataFrame(list(zip(features_sorted, importances_sorted)), columns=['Feature', 'Importance'])
    importances_df.to_csv('{}\\Sensitivity\\USGS{}\\Seed_{}\\Importances.csv'.format(workingPath,watershed_interest, x))

    # Residual Plot
    # residuals = y_test - y_pred
    # residuals.to_csv('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Sensitivity\\Seed_{}\\residuals.csv'.format(workingPath, metric_short, x))

    # # residuals = y - y_pred
    # plt.scatter(((y_pred)), (residuals), alpha=0.5)
    # # plt.scatter(y, residuals, alpha=0.5)
    # plt.xlabel('Predicted Values')
    # plt.ylabel('Residuals')
    # plt.title('Residual Plot')
    # plt.axhline(y=0, color='red', linestyle='--')
    # plt.savefig('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Sensitivity\\Seed_{}\\Residuals.png'.format(workingPath, metric_short, x))
    # # plt.show()
    # plt.close()
    # #
    # # Predicted vs Actual Plot
    # plt.figure(figsize=(7, 4))
    # plt.scatter((y_test), (y_pred), alpha=0.5)
    # # plt.scatter(y, y_pred, alpha=0.5)
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    # plt.title('Predicted vs Actual Values')
    # plt.plot([0, np.exp(y_test.max())], [0, np.exp(y_test.max())], 'r--', lw=4)
    # plt.plot([(y_test.min()), (y_test.max())], [(y_test.min()), (y_test.max())], 'r--', lw=4)
    # plt.tight_layout()
    #
    # plt.savefig('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Sensitivity\\Seed_{}\\Predicted.png'.format(workingPath, metric_short,x))
    # # plt.show()
    # plt.close()

    # for col in features_sorted:
    #     # print(col)
    #     PartialDependenceDisplay.from_estimator(rf_regressor, X_test, features = [col], percentiles=(0, 1))
    #     # PartialDependenceDisplay.from_estimator(rf_regressor, X, features=[col], percentiles=(0, 1))
    #     # display.plot()
    #     plt.title(col)
    #     plt.savefig('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Plots_{}\\Seed_{}\\PD\\pd_{}.png'.format(workingPath, metric_short, scenario, x, col))
    #     plt.close()
    #
    # print('calculating shap')
    # # Create the SHAP explainer for the Random Forest model
    # explainer = shap.TreeExplainer(rf_regressor)
    #
    # # Calculate SHAP values for the test set
    # shap_values = explainer(X_test)
    # # print(shap_values)
    # print('calculated shap!')
    # shap_values_df = pd.DataFrame(shap_values.values, columns = [shap_values.feature_names])
    # shap_values_data_df = pd.DataFrame(shap_values.data, columns = [shap_values.feature_names])
    # #
    # shap_values_df.to_csv('{}\\Sensitivity\\USGS{}\\Seed_{}\\ShapValues.csv'.format(workingPath,watershed_interest, x))
    # shap_values_data_df.to_csv('{}\\Sensitivity\\USGS{}\\Seed_{}\\ShapValues_data.csv'.format(workingPath, watershed_interest, x))
    # #
    # # Generate a SHAP summary plot
    # ax = plt.gca()
    # fig4 = shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    # plt.gcf().set_size_inches(10, 12)
    # # ax.set_xlim(-0.2,0.2)
    # plt.tight_layout()
    # plt.savefig('{}\\Sensitivity\\USGS{}\\seed_{}\\Summary.png'.format(workingPath, watershed_interest, x))
    # # plt.show()
    # plt.close()
    # #


df_combined = test.join(y_preds_df)
df_combined.to_csv('{}\\Sensitivity\\ys_combined_{}.csv'.format(
        workingPath, watershed_interest))
