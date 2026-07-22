## Haley Canham ##
## Jan 2026 ##
## Summarized shap values across many seeds using abs avg ##

## libaries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

workingPath = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full\\ModelRun'

shaps = pd.DataFrame()
datas = pd.DataFrame()

# loop through each seed
for x in range (0,100):
    shap_file = '{}\\Seed_{}\\ShapValues.csv'.format(workingPath,x)
    shap = pd.read_csv(shap_file, index_col=0)
    data_file = '{}\\Seed_{}\\ShapValues_data.csv'.format(workingPath,x)
    data = pd.read_csv(data_file, index_col=0)

    if shaps.empty:
        shaps = shap
        datas = data
    else:
        shaps = pd.concat([shaps,shap])
        datas = pd.concat([datas,data])

features = shaps.columns.to_list()

# for x in range(0,len(features)):
for x in range(0, 100):

    feature = features[x]
    dic_forplot = {'data': datas[feature], 'shap': shaps[feature]}
    df_forplot = pd.DataFrame(dic_forplot)
    df_forplot = df_forplot.sort_values(by='data')
    df_forplot['shap_moving_average'] = df_forplot['shap'].rolling(window=1000, center = True).mean()
    df_forplot['shap_moving_75th'] = df_forplot['shap'].rolling(window=1000, center = True).quantile(0.75)
    df_forplot['shap_moving_25th'] = df_forplot['shap'].rolling(window=1000, center = True).quantile(0.25)
    df_forplot['shap_moving_90th'] = df_forplot['shap'].rolling(window=1000, center = True).quantile(0.9)
    df_forplot['shap_moving_10th'] = df_forplot['shap'].rolling(window=1000, center = True).quantile(0.1)

    plt.figure(figsize=(4, 3))
    plt.scatter(df_forplot['data'], df_forplot['shap'], s = 5, alpha = 0.1, c = 'grey')
    plt.plot(df_forplot['data'], df_forplot['shap_moving_average'], color = 'black')
    # plt.plot(df_forplot['data'], df_forplot['shap_moving_75th'], color = 'black')
    # plt.plot(df_forplot['data'], df_forplot['shap_moving_25th'], color = 'black')

    plt.fill_between(df_forplot['data'], df_forplot['shap_moving_75th'], df_forplot['shap_moving_25th'], color='black', alpha=0.5)
    plt.fill_between(df_forplot['data'], df_forplot['shap_moving_90th'], df_forplot['shap_moving_75th'], color='black', alpha=0.2)
    plt.fill_between(df_forplot['data'], df_forplot['shap_moving_25th'], df_forplot['shap_moving_10th'], color='black', alpha=0.2)

    plt.ylabel('Influence (shap value)')
    plt.xlabel('Feature value')
    plt.title(feature)
    plt.tight_layout()
    plt.savefig('{}\\0_SummaryPlots\\PD\\SummaryPD_{}.png'.format(workingPath, feature))
    # plt.show()
    plt.close()
