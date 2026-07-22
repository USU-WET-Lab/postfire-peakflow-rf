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
    # shap_file = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\Peak\\modelbounds_optimized_80_20\\Seed_{}\\ShapValues.csv'.format(workingPath,x)
    shap_file = '{}\\Seed_{}\\ShapValues.csv'.format(workingPath,x)
    shap = pd.read_csv(shap_file, index_col=0)
    # data_file = '{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\Peak\\modelbounds_optimized_80_20\\Seed_{}\\ShapValues_data.csv'.format(workingPath,x)
    data_file = '{}\\Seed_{}\\ShapValues_data.csv'.format(workingPath,x)
    data = pd.read_csv(data_file, index_col=0)

    if shaps.empty:
        shaps = shap
        datas = data
    else:
        shaps = pd.concat([shaps,shap])
        datas = pd.concat([datas,data])

shaps_abs = shaps.abs()
shaps_abs_avg = shaps_abs.mean()
shaps_abs_avg = shaps_abs_avg.sort_values(ascending=False)

# bar_colors_mag = ['blue', 'mediumseagreen', 'mediumseagreen', 'blue', 'blue', 'blue','mediumseagreen', 'mediumseagreen', 'mediumseagreen', 'blue',
#                   'mediumseagreen', 'blue', 'blue', 'brown', 'blue', 'brown', 'brown', 'blue']
# bar_colors_change = ['brown', 'blue', 'mediumseagreen', 'mediumseagreen','mediumseagreen', 'mediumseagreen', 'blue',
#                      'mediumseagreen', 'blue', 'blue','blue', 'brown', 'blue', 'blue', 'brown', 'blue',
#                      'mediumseagreen', 'mediumseagreen', 'mediumseagreen', 'mediumseagreen', 'mediumseagreen', 'mediumseagreen',
#                      'brown', 'mediumseagreen', 'mediumseagreen','mediumseagreen',  'mediumseagreen','mediumseagreen', 'mediumseagreen',
#                      'blue', 'mediumseagreen', 'blue','mediumseagreen','brown', 'brown' ]
bar_colors_peak = ['blue', 'mediumseagreen', 'mediumseagreen', 'blue', 'blue', 'mediumseagreen','mediumseagreen', 'mediumseagreen', 'mediumseagreen', 'brown',
                  'blue', 'blue', 'brown', 'brown', 'blue', 'brown']
shaps_abs_avg.plot(kind = 'bar', color = bar_colors_peak)

plt.tight_layout()
# plt.savefig('{}\\RandomForest\\RFModels\\StormPercentileVersions\\PostFire\\Peak\\modelbounds_optimized_80_20\\0_SummaryPlots\\SummaryBar_colored.png'.format(workingPath))
# plt.savefig('{}\\0_SummaryPlots\\SummaryBar.png'.format(workingPath))

plt.show()
