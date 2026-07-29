"""
05_run_final_model.py - Step 5 of the post-fire peak flow pipeline

Using the optimized feature table from step 04, we train the final model. This step trains one 
random forest per seed (with the train/test split from step 01) and for each seed: writes 
performance stats, test predictions, feature importances, two diagnostic plots, and SHAP values. 
The per-seed outputs from this step feed subsequent summary steps (08, 09, 10). The target 
is log-transformed for fitting in log space. 

Reads: data / <MODEL_TABLE> 
       outputs/seeds/<WITHOLDING>/Seed_<x>/wats_train.csv, wats_test.csv

Writes: outputs/model_run/Seed_<x>/stats.csv          (mse, rmse, R2)
        outputs/model_run/Seed_<x>/predictions.csv    (GAGE_ID, y_test, y_pred; log space)
        outputs/model_run/Seed_<x>/importances.csv    (Feature, Importance)
        outputs/model_run/Seed_<x>/shap_values.csv, shap_data.csv
        outputs/model_run/Seed_<x>/predicted_vs_actual.png, residuals.png

Run: python src/05_run_final_model.py

"""


#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
from rf_utils import (fit_rf, evaluate, ranked_importances, compute_shap,
                      shap_frames, plot_predicted_vs_actual, plot_residuals)

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent   # repo root; auto-derives, no need to edit
DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"

MODEL_TABLE = "RF_AttributeTable_PeakMag_OptimizedModel.csv"   # final optimized feature set
METRIC      = "PeakArea"    
ID_COL      = "GAGE_ID"   
N_SEEDS     = 100
WITHOLDING  = "80_20"      
RF_KWARGS   = dict(n_estimators=100, random_state=42)

# Columns to drop before fitting (kept out of the final model). See note below.
DROP_COLUMNS = ["ASPECT_NORTHNESS"]


#---------------------------------------------MAIN CODE BLOCK-------------------------------------------------------------

def main(): 
    data = pd.read_csv(DATA / MODEL_TABLE)
    if DROP_COLUMNS: 
        data = data.drop(columns = DROP_COLUMNS)
    features = [c for c in data.columns if c not in (METRIC, ID_COL)]

    for x in range(N_SEEDS): 
        seed_dir = OUTPUTS / "model_run" / f"Seed_{x}"
        seed_dir.mkdir(parents= True, exist_ok= True)

        split_dir = 

metric = 'PeakArea' 
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
  
