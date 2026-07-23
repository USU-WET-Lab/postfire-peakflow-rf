"""
04_summarize_variable_optimization.py - Step 4 of the post-fire peak flow pipeline
"""

#---------------------------------------------IMPORTS---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from pathlib import Path

#-------------------------------------------CONFIG: only edit this block ---------------------------------------------------------------------------


working = 'C:\\Users\\A02343538\\Box\\MyResearch\\Chap3'
workingPath = '{}\\RandomForest\\RFModels\\UpdatedModelRuns_Spring26\\Peak\\Full\\VariableOptimization'.format(working)
metric = 'Peak'
scenario = 'PostFire'
# witholding = '70_30'
# version = 'Wat25_VarOp_StormPers'

# read in seeds
seeds_list = []
directory_path = workingPath
for entry_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry_name)
        if os.path.isdir(full_path) and ('Seed_' in entry_name):
            seeds_list.append(entry_name)
# seeds_list = seeds_list[1:]
# get all stats and output as a single file
plt.figure()
all_r2_df = pd.DataFrame()
for x in seeds_list:
    vars_count_list = []
    directory_path = '{}\\{}'.format(workingPath, x)
    for entry_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry_name)
        # print(entry_name)
        if os.path.isdir(directory_path) and ('Stats_' in entry_name):
            vars_count_list.append(entry_name)
    stats_rmse = []
    stats_mse = []
    stats_r2 = []
    vars_count_num = []

    for var in vars_count_list:
        vars_num = var.split('_Vars')[1]
        vars_num = int(vars_num.split('.csv')[0])
        vars_count_num.append(vars_num)
        Stats_file = '{}\\{}\\Stats_Vars{}.csv'.format(workingPath, x, vars_num)
        Stats = pd.read_csv(Stats_file)
        stats_rmse.append(Stats.rmse[0])
        stats_mse.append(Stats.mse[0])
        stats_r2.append(Stats.R2[0])
    stats_summary = pd.DataFrame(list(zip(vars_count_num, stats_mse, stats_rmse, stats_r2)), columns = ['Vars', 'mse', 'rmse', 'R2'])
    # if all_r2_df.empty == True:

    # stats_summary.to_csv('{}\\RandomForest\\RFModels\\Optimization\\Variables\\{}\\{}\\{}\\{}\\StatsSummary.csv'.format(workingPath,scenario, metric, witholding, x))
    # make plot
    # plt.figure()
    stats_summary_sorted = stats_summary.sort_values(by=['Vars'])
    stats_summary_sorted= stats_summary_sorted.reset_index(drop=True)
    all_r2_df[x] = stats_summary_sorted['R2']

    plt.plot(stats_summary_sorted.index, stats_summary_sorted.R2)
all_r2_avg = all_r2_df.mean(axis=1)
plt.plot(all_r2_avg, label = 'Avg', color = 'black')
plt.tight_layout()
plt.savefig('{}\\VariableOptimizationR2.png'.format(workingPath))
plt.show()
all_r2_df.to_csv('{}\\VariableOptimizationR2_allseeds.csv'.format(workingPath))
all_r2_avg.to_csv('{}\\VariableOptimizationR2_avg.csv'.format(workingPath))



# get average rankings

# for var in range(15,16):
for var in range(1, 45):
    rankings_df = pd.DataFrame()
    for x in seeds_list:
        importances_file = '{}\\{}\\Importances_Vars{}.csv'.format(workingPath, x, var)
        importances = pd.read_csv(importances_file)
        rankings_df[x] = importances.Feature
    # rankings.to_csv('{}\\RandomForest\\RFModels\\Optimization\\ValidationWitholdings\\{}\\{}\\{}\\ImportanceRanks_summary.csv'.format(workingPath, scenario, metric, witholding))

    #make list of all variables (unique) in df
    vars_unique = rankings_df.Seed_1.to_list()
    for col in rankings_df.columns:
        for v in rankings_df[col]:
            if v not in vars_unique:
                vars_unique.append(v)

    # get average ranking for each variable
    rankings_vars_df = pd.DataFrame()
    for v in vars_unique:
        rankings_list = []
        print(v)
        for x in seeds_list:
            print(x)
            if v in rankings_df[x].to_list():
                rank = rankings_df[rankings_df[x] == v].index[0]
            else:
                rank = len(rankings_df.index) + 1
            rankings_list.append(rank)
        rankings_vars_df[v] = rankings_list
    rankings_vars_avg = rankings_vars_df.mean()
    rankings_vars_avg.sort_values(inplace=True)
    rankings_vars_avg.to_csv('{}\\VariableRankings\\VariableRankings_vars{}.csv'.format(workingPath, var))




