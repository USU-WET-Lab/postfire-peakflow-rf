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

metric = 'PeakArea' #Peak, Rise, DurabvThresh, VolabvThresh
withholding = '80_20'
print(metric, metric, withholding)
# save_folder = '{}_{}'.format(metric,ARI)

workingPath = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full'


# for thresh in threshs:
#     print(thresh)
data_file = '{}\\RF_AttributeTable_ARI1_Peak_modelbounds_Optimized.csv'.format(workingPath)
data = pd.read_csv(data_file)

data = data.drop(columns = ['ASPECT_NORTHNESS'])

# data = data[(data.DRAIN_SQKM > 50)]
# data = data[(data.burnedarea_per > 20)]
# # data = data[data.burned_storm_depth_per > thresh]
# data = data[data.burned_storm_depth_per > 70]
# # data = data[data.burned_storm_int_per > 93]
# data = data[data.DaysSinceFire < 1095]
# print(per, len(data))

r2s = []

# loop through each seed
for x in range (0,100):
# for x in range(85, 86):
    print(x)

    if not os.path.exists('{}\\ModelRun\\Seed_{}'.format(workingPath, x)):
        os.mkdir('{}\\ModelRun\\Seed_{}'.format(workingPath, x))
        os.mkdir('{}\\ModelRun\\Seed_{}\\PD'.format(workingPath, x))
        os.mkdir('{}\\ModelRun\\Seed_{}\\shapPD'.format(workingPath, x))
        os.mkdir('{}\\ModelRun\\Seed_{}\\Waterfalls'.format(workingPath, x))

    watersheds_test_file = '{}\\Seeds\\{}\\Seed_{}\\wats_test.csv'.format(workingPath, withholding, x)
    watersheds_train_file = '{}\\Seeds\\{}\\Seed_{}\\wats_train.csv'.format(workingPath, withholding, x)
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


    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    # mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # r2 = r2_score(y, y_pred)
    # print(f"Mean Squared Error: {mse}, RMSE: {rmse}")
    # print(f"R-squared: {r2}")
    stats_df = pd.DataFrame(list(zip([mse], [rmse], [r2])), columns=['mse', 'rmse', 'R2'])
    # print(f"R-squared: {r2_score(y, y_pred)}")
    stats_df.to_csv('{}\\ModelRun\\Seed_{}\\Stats.csv'.format(workingPath, x))
    r2s.append(r2)

    print('{} %Test: {}, R2: {}'.format(x, (len(X_test) / len(X_train) * 100), r2))

    # print('median: {}'.format(np.median(r2s)))
    # print('mean: {}'.format(np.mean(r2s)))

    # output y files
    y_test.to_csv('{}\\ModelRun\\Seed_{}\\y_test.csv'.format(workingPath, x))
    y_pred_series = pd.Series(y_pred)
    y_pred_series.to_csv('{}\\ModelRun\\Seed_{}\\y_pred.csv'.format(workingPath, x))

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
    importances_df.to_csv('{}\\ModelRun\\Seed_{}\\Importances.csv'.format(workingPath, x))

    # Residual Plot
    residuals = y_test - y_pred
    residuals.to_csv('{}\\ModelRun\\Seed_{}\\residuals.csv'.format(workingPath, x))

    # residuals = y - y_pred
    plt.scatter(((y_pred)), (residuals), alpha=0.5)
    # plt.scatter(y, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.savefig('{}\\ModelRun\\Seed_{}\\Residuals.png'.format(workingPath, x))
    # plt.show()
    plt.close()
    #
    # Predicted vs Actual Plot
    plt.figure(figsize=(7, 4))
    plt.scatter((y_test), (y_pred), alpha=0.5)
    # plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.plot([0, np.exp(y_test.max())], [0, np.exp(y_test.max())], 'r--', lw=4)
    plt.plot([(y_test.min()), (y_test.max())], [(y_test.min()), (y_test.max())], 'r--', lw=4)
    plt.tight_layout()

    plt.savefig('{}\\ModelRun\\Seed_{}\\Predicted.png'.format(workingPath,x))
    # plt.show()
    plt.close()

    # for col in features_sorted:
    #     # print(col)
    #     PartialDependenceDisplay.from_estimator(rf_regressor, X_test, features = [col], percentiles=(0, 1))
    #     # PartialDependenceDisplay.from_estimator(rf_regressor, X, features=[col], percentiles=(0, 1))
    #     # display.plot()
    #     plt.title(col)
    #     plt.savefig('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Plots_{}\\Seed_{}\\PD\\pd_{}.png'.format(workingPath, metric_short, scenario, x, col))
    #     plt.close()

    print('calculating shap')
    # Create the SHAP explainer for the Random Forest model
    explainer = shap.TreeExplainer(rf_regressor)

    # Calculate SHAP values for the test set
    shap_values = explainer(X_test)
    # print(shap_values)
    print('calculated shap!')
    shap_values_df = pd.DataFrame(shap_values.values, columns = [shap_values.feature_names])
    shap_values_data_df = pd.DataFrame(shap_values.data, columns = [shap_values.feature_names])
    #
    shap_values_df.to_csv('{}\\ModelRun\\Seed_{}\\ShapValues.csv'.format(workingPath, x))
    shap_values_data_df.to_csv('{}\\ModelRun\\Seed_{}\\ShapValues_data.csv'.format(workingPath, x))
    #
    # Generate a SHAP summary plot
    ax = plt.gca()
    fig4 = shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    plt.gcf().set_size_inches(10, 12)
    # ax.set_xlim(-0.2,0.2)
    plt.tight_layout()
    plt.savefig('{}\\ModelRun\\seed_{}\\Summary.png'.format(workingPath, x))
    # plt.show()
    plt.close()
    #
    # # # fig = shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    # # fig = shap.summary_plot(shap_values, X, max_display=20, show=False)
    # # plt.gcf().set_size_inches(10, 12)
    # # ax = plt.gca()
    # # ax.set_xlim(-0.2,0.2)
    # # plt.tight_layout()
    # # plt.savefig('{}\\RandomForest\\04Sep25\\PostFireMultipliers\\{}\\{}\\Allshap\\{}\\Summary_{}_{}_xlim.png'.format(workingPath, metric_short, ARI, scenario, metric, ARI))
    # # # plt.show()
    # # plt.close()
    #
    # # Generate a SHAP waterfall plot for an individual prediction
    # for i in range(0, 10):
    # # for i in range(0, len(shap_values)):
    #
    #     ax = plt.gca()
    #     shap.plots.waterfall(shap_values[i], max_display=10, show=False)
    #     plt.gcf().set_size_inches(10, 8)
    #     plt.tight_layout()
    #     plt.savefig('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Plots_{}\\Seed_{}\\Waterfalls\\Waterfalls_{}_{}.png'.format(workingPath, metric_short, scenario, x, metric, i))
    #     # plt.show()
    #     plt.close()
    #
    # for f in features:
    #
    #     ax = plt.gca()
    #     shap.plots.scatter(shap_values[:, f], show=False)
    #     plt.gcf().set_size_inches(10, 8)
    #     plt.tight_layout()
    #     plt.savefig('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\{}\\Plots_{}\\Seed_{}\\shapPD\\PD_{}.png'.format(workingPath,metric_short, scenario, x, f))
    #     plt.close()
