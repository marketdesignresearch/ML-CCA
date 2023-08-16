
# %%import json
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, bootstrap
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
import re
from scipy.stats import ttest_rel
from openpyxl import load_workbook

# set pd option of max columns to display and rows to display
pd.set_option('display.max_columns', 16)
pd.set_option('display.max_rows', 100)

plt.rcParams['figure.constrained_layout.use'] = True

# %% helper functions
# helper function to get mean, upper and lower CI bound from
def get_CI_normal(data,
                  significance_level_CI=0.95):
    p = 1 - (1 - significance_level_CI) / 2
    q = norm.ppf(p)
    data_summary = data.describe()

    mean = np.asarray(data_summary.loc['mean'])
    std = np.asarray(data_summary.loc['std'])
    count = np.asarray(data_summary.loc['count'])

    upper_bound = mean + q * std / np.sqrt(count)
    lower_bound = mean - q * std / np.sqrt(count)

    return mean, lower_bound, upper_bound


# helper function to get mean, upper and lower CI bound from
def get_CI_bootstrapped(data,
                        significance_level_CI=0.95,
                        print_losses=False,
                        random_state=1):
    data = pd.DataFrame(data)
    if print_losses:
        data = (1 - data) * 100
    else:
        data = data * 100
    data_summary = data.describe()
    mean = np.asarray(data_summary.loc['mean'])

    data = (data,)  # (data.iloc[:, :-1],)

    b = bootstrap(data,
                  np.mean,
                  axis=0,
                  confidence_level=significance_level_CI,
                  n_resamples=10000,
                  random_state=random_state,
                  method='percentile')

    lower_bound = b.confidence_interval.low
    upper_bound = b.confidence_interval.high
    return mean, lower_bound, upper_bound


# helper function to get mean, upper and lower CI bound from
def get_emp_quantiles(data, upper_q=0.975, lower_q=0.025):
    upper_bound = np.quantile(data, upper_q, axis=0)
    lower_bound = np.quantile(data, lower_q, axis=0)

    data_summary = data.describe()
    mean = np.asarray(data_summary.loc['mean'])

    return mean, lower_bound, upper_bound



#%% SET YOUR PARAMETERS
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
results = defaultdict(list)
domains = ['GSVM', 'LSVM', 'SRVM', 'MRVM']
mechanisms = ['CCA-MLDQ', 'CCA', 'CCA-MLDQ-C']
qmax = 100 # select between 50, 100
comparison = 'constrained' # select between 'unconstrained', 'constrained', 'constrained_vs_unconstrained_ML-CCA'
alpha_t_test = 0.05 # significance level for t-tests
remove_first_iteration = True # for efficiency path plots 
# starts at 0 thus 19 in case of remove_first_iteration=False or 18 for the first interval in case of remove_first_iteration=True
if remove_first_iteration:
    start_of_MLCA_DQ = {'GSVM': 18, 'LSVM': 18, 'SRVM': 18, 'MRVM': 48}
else:
    start_of_MLCA_DQ = {'GSVM': 19, 'LSVM': 19, 'SRVM': 19, 'MRVM': 49}

# Figure Parameters:
# To avoid type 3 fonts in figures
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True (other option to avoid type 3 fonts)
fig, ax = plt.subplots(2, 2, figsize=(7.5, 5), facecolor='w', edgecolor='k', sharex=True)
font_title = 15
font_axis_label = 13
font_tick_label = 13
font_legend = 9
print_losses = False
alpha_plot = 0.1
linewidth_means = 1
linewidth_bounds = 0.5
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# %% read results and postprocess

if comparison == 'unconstrained':
    folder_path = f'./wandb_custom_plots/raw_data/efficiency/qmax{qmax}/{comparison}'
elif comparison == 'constrained':
    folder_path = f'./wandb_custom_plots/raw_data/efficiency/qmax{qmax}/{comparison}'
elif comparison == 'constrained_vs_unconstrained_ML-CCA':
    folder_path = f'./wandb_custom_plots/raw_data/efficiency/qmax{qmax}/{comparison}'

# initialize dictionary T_TEST_DATA with None values
T_TEST_DATA = dict()

for domain in domains:
    for mechanism in mechanisms:
        for filename in os.listdir(folder_path):

            ########################################################
            # 1. CLOCK BIDS and CLOCK BIDS RAISED
            ########################################################
            if filename.endswith('.csv') and filename.startswith(f'{domain}_{mechanism}_') and 'PROFITMAX' not in filename:
                print(domain)
                print(mechanism)
                print(f'Reading file:{os.path.join(folder_path,filename)}')
                raw_data = pd.read_csv(os.path.join(folder_path,filename))
                # postprocess data
                postprocessed_data  = deepcopy(raw_data)
                postprocessed_data.drop(postprocessed_data.iloc[:,0:1], inplace=True,axis=1) # drop first iteration col

                # drop unnecessary columns

                # 1.Use Average Column (ATTENTION SOMEHOW BUGGED)
                # keep_columns = [col for col in postprocessed_data.columns if 'step' not in col and 'MIN' not in col and 'MAX' not in col]
                #

                # 2. Use __MAX Column
                keep_columns = [col for col in postprocessed_data.columns if 'step' not in col and 'MIN' not in col]
                keep_columns = [col for col in keep_columns if 'MAX' in col]
                #

                # split clock bids and raised clock bids
                clock_bids_efficiency_columns = [col for col in keep_columns if 'Raised' not in col]
                raised_efficiency_columns = [col for col in keep_columns if 'Raised' in col]
                clock_bids_efficiency = postprocessed_data[clock_bids_efficiency_columns]
                raised_efficiency = postprocessed_data[raised_efficiency_columns]

                #T-TEST-DATA
                #-----------------------------------------------------------------------------------------------------------------
                # 1. clock
                tmp_t_test_clock = deepcopy(clock_bids_efficiency)
                tmp_t_test_clock=tmp_t_test_clock.iloc[-1,:].to_frame()
                #rename values in index to only contain the seed number
                old_row_names = tmp_t_test_clock.index.values
                new_row_names = [int(re.findall(r'\d+', x)[0]) for x in old_row_names]
                tmp_t_test_clock = tmp_t_test_clock.rename(index = dict(zip(old_row_names, new_row_names)))
                if (domain,'clock') in T_TEST_DATA:
                    # concetanate last column of PD_DATA with T_TEST_DATA
                    T_TEST_DATA[(domain,'clock')] = pd.concat([T_TEST_DATA[(domain,'clock')], tmp_t_test_clock], axis=1)
                else:
                    T_TEST_DATA[(domain,'clock')] = tmp_t_test_clock
                    # rename last column of T_TEST_DATA to results_key
                T_TEST_DATA[(domain,'clock')].columns = [*T_TEST_DATA[(domain,'clock')].columns[:-1], mechanism]
                # 2. raised
                tmp_t_test_raised = deepcopy(raised_efficiency)
                tmp_t_test_raised=tmp_t_test_raised.iloc[-1,:].to_frame()
                #rename values in index to only contain the seed number
                old_row_names = tmp_t_test_raised.index.values
                new_row_names = [int(re.findall(r'\d+', x)[0]) for x in old_row_names]
                tmp_t_test_raised = tmp_t_test_raised.rename(index = dict(zip(old_row_names, new_row_names)))
                if (domain,'raised') in T_TEST_DATA:
                    # concetanate last column of PD_DATA with T_TEST_DATA
                    T_TEST_DATA[(domain,'raised')] = pd.concat([T_TEST_DATA[(domain,'raised')], tmp_t_test_raised], axis=1)
                else:
                    T_TEST_DATA[(domain,'raised')] = tmp_t_test_raised
                    # rename last column of T_TEST_DATA to results_key
                T_TEST_DATA[(domain,'raised')].columns = [*T_TEST_DATA[(domain,'raised')].columns[:-1], mechanism]
                #-----------------------------------------------------------------------------------------------------------------

                #transform into right format = list of dicts
                clock_bids_efficiency_list_of_dicts = []
                for col in clock_bids_efficiency.columns:
                    clock_bids_efficiency_list_of_dicts.append(clock_bids_efficiency[col].to_dict())
                raised_efficiency_list_of_dicts = []
                for col in raised_efficiency.columns:
                    raised_efficiency_list_of_dicts.append(raised_efficiency[col].to_dict())
                # save to results dict
                results[f'{domain}_{mechanism}_CLOCK_BIDS'] = clock_bids_efficiency_list_of_dicts
                results[f'{domain}_{mechanism}_RAISED_CLOCK_BIDS'] = raised_efficiency_list_of_dicts

            ###############
            # 2. PROFIT MAX
            ###############
            elif filename.endswith('.csv') and filename.startswith(f'{domain}_{mechanism}_PROFITMAX'):
                print(domain)
                print(mechanism)
                print(f'Reading file:{os.path.join(folder_path,filename)}')
                raw_data = pd.read_csv(os.path.join(folder_path,filename))

                # postprocess data
                postprocessed_data  = deepcopy(raw_data)
                postprocessed_data.drop(postprocessed_data.iloc[:,0:1], inplace=True,axis=1) # drop first iteration col
                # drop unnecessary columns
                
                # 1.VERSION Use Average Column (ATTENTION SOMEHOW BUGGED)
                #keep_columns = [col for col in postprocessed_data.columns if 'step' not in col and 'MIN' not in col and 'MAX' not in col]
                #

                # 2. VERSION Use __MAX Column
                keep_columns = [col for col in postprocessed_data.columns if 'step' not in col and 'MIN' not in col]
                keep_columns = [col for col in keep_columns if 'MAX' in col]
                #

                # split clock bids and raised clock bids 
                profit_max_efficiency = postprocessed_data[keep_columns]


                #T-TEST-DATA
                #-----------------------------------------------------------------------------------------------------------------
                # 3. profit max
                tmp_t_test_profitMax = deepcopy(profit_max_efficiency)
                tmp_t_test_profitMax=tmp_t_test_profitMax.iloc[-1,:].to_frame()
                #rename values in index to only contain the seed number
                old_row_names = tmp_t_test_profitMax.index.values
                new_row_names = [int(re.findall(r'\d+', x)[0]) for x in old_row_names]
                tmp_t_test_profitMax = tmp_t_test_profitMax.rename(index = dict(zip(old_row_names, new_row_names)))
                if (domain,'profitMax') in T_TEST_DATA:
                    # concetanate last column of PD_DATA with T_TEST_DATA
                    T_TEST_DATA[(domain,'profitMax')] = pd.concat([T_TEST_DATA[(domain,'profitMax')], tmp_t_test_profitMax], axis=1)
                else:
                    T_TEST_DATA[(domain,'profitMax')] = tmp_t_test_profitMax
                    # rename last column of T_TEST_DATA to results_key
                T_TEST_DATA[(domain,'profitMax')].columns = [*T_TEST_DATA[(domain,'profitMax')].columns[:-1], mechanism]
                #-----------------------------------------------------------------------------------------------------------------
                #transform into right format = list of dicts
                profit_max_efficiency_list_of_dicts = []
                for col in profit_max_efficiency.columns:
                    profit_max_efficiency_list_of_dicts.append(profit_max_efficiency[col].to_dict())
                # save to results dict
                results[f'{domain}_{mechanism}_PROFITMAX'] = profit_max_efficiency_list_of_dicts


# %% 1. Make T-Tests

for k,v in T_TEST_DATA.items():
    print(k[0],k[1])
    print(v.describe())
    if comparison == 'constrained_vs_unconstrained_ML-CCA':
        t = ttest_rel(v['CCA-MLDQ-C'], v['CCA-MLDQ'], alternative='greater',nan_policy='omit') #when comparing constrained vs unconstrained
        print(f'Paired t-test with HA: CCA_MLDQ-C > CCA_MLDQ and H0: CCA_MLDQ-C <= CCA_MLDQ (alpha={alpha_t_test}):')#when comparing constrained vs unconstrained
    else:
        t = ttest_rel(v['CCA-MLDQ'], v['CCA'], alternative='greater',nan_policy='omit')# when comparing ML-CCA vs CCA
        print(f'Paired t-test with HA: CCA_MLDQ > CCA and H0: CCA_MLDQ <= CCA (alpha={alpha_t_test}):')# when comparing ML-CCA vs CCA

    if t.pvalue <= alpha_t_test:
        print(f'Result paired t-test:  pvalue={t.pvalue} | statistic={t.statistic} | df={t.df} | REJECT H0')
    else:
        print(f'Result paired t-test:  pvalue={t.pvalue} | statistic={t.statistic} | df={t.df} | NOT REJECT H0')
    print()
print('\n')

# # save to already existing excel file
# book = load_workbook(f'./wandb_custom_plots/results/All_Results.xlsx')
# writer = pd.ExcelWriter(f'./wandb_custom_plots/results/All_Results.xlsx', engine='openpyxl') 
# writer.book = book
# writer.sheets.update(dict((ws.title, ws) for ws in book.worksheets))

# for df_name, df in T_TEST_DATA.items():
#     if unconstrained:
#         df.to_excel(writer, sheet_name=df_name[0]+df_name[1]+f'_Qmax{qmax}_unconstrained')
#     else:
#         df.to_excel(writer, sheet_name=df_name[0]+df_name[1]+f'_Qmax{qmax}')
# writer.save()


# %% 2. Efficiency Path Plot CLOCK BIDS AND CLOCK BIDS RAISED
for results_key in results.keys():

    if 'PROFITMAX' not in results_key:

        print()
        print(*['#']*50)
        print(results_key)
        if 'GSVM' in results_key:
            domain = 'GSVM'
            i,j = 0,0
        elif 'LSVM' in results_key:
            domain = 'LSVM'
            i,j = 0,1
        elif 'SRVM' in results_key:
            domain = 'SRVM'
            i,j = 1,0
        elif 'MRVM' in results_key:
            domain  = 'MRVM'
            i,j = 1,1
        else:
            raise ValueError('Unknown domain')

        print(f'Number of Seeds: {pd.DataFrame(results[results_key]).shape[0]}')
        mean, lower_bound, upper_bound = get_CI_bootstrapped(results[results_key], print_losses=print_losses)

        if remove_first_iteration:
            # avoid first iteration p=0
            mean = mean[1:]
            lower_bound = lower_bound[1:]
            upper_bound = upper_bound[1:]

        print(f'Length of each seed: {len(mean)}')
        #print()
        #print(*['*']*50)
        print(f'EFFICIENCY: LBound:{lower_bound[-1]:.2f}, Mean: {mean[-1]:.2f}, UBound: {upper_bound[-1]:.2f}           FOR PAPER: ({lower_bound[-1]:.2f}\,,\,{mean[-1]:.2f}\,,\,{upper_bound[-1]:.2f})')
        #print(*['*']*50)
        #print()
        #print(f'Mean per Iteration: {mean}')
        print(*['#']*50)
        print()
        x = np.arange(len(mean))

        if 'CCA_CLOCK_BIDS' in results_key:
            lty1 = 'g-'
            lty2 = 'g-'
            col = 'green'
            label = 'CCA clock bids'
        elif 'CCA_RAISED_CLOCK_BIDS' in results_key:
            lty1 = 'g--'
            lty2 = 'g--'
            col = 'green'
            label = 'CCA raised clock bids'
        elif 'CCA-MLDQ_CLOCK_BIDS' in results_key:
            lty1 = 'b-'
            lty2 = 'b-'
            col = 'blue'
            label = 'ML-CCA clock bids'
        elif 'CCA-MLDQ_RAISED_CLOCK_BIDS' in results_key:
            lty1 = 'b--'
            lty2 = 'b--'
            col = 'blue'
            label = 'ML-CCA raised clock bids'
        elif 'RAND-MLDQ_CLOCK_BIDS' in results_key:
            lty1 = 'y-'
            lty2 = 'y-'
            col = 'yellow'
            label = 'ML-CCA-RAND clock bids'
        elif 'RAND-MLDQ_RAISED_CLOCK_BIDS' in results_key:
            lty1 = 'y--'
            lty2 = 'y--'
            col = 'yellow'
            label = 'ML-CCA-RAND raised clock bids'
        elif 'CCA-MLDQ-C_CLOCK_BIDS' in results_key:
            lty1 = 'g-'
            lty2 = 'g-'
            col = 'green'
            label = 'ML-CCA-C clock bids'
        elif 'CCA-MLDQ-C_RAISED_CLOCK_BIDS' in results_key:
            lty1 = 'g--'
            lty2 = 'g--'
            col = 'green'
            label = 'ML-CCA-C raised clock bids'

        # start of ML queries
        ax[i,j].axvline(x=start_of_MLCA_DQ[domain], ymin=0, ymax=100, linewidth=1, color='k', linestyle='--')

        ax[i,j].plot(mean, lty1, linewidth=linewidth_means, label=label)
        ax[i,j].plot(upper_bound, lty2, linewidth=linewidth_bounds)
        ax[i,j].plot(lower_bound, lty2, linewidth=linewidth_bounds)
        ax[i,j].fill(np.append(x, x[::-1]), np.append(mean, upper_bound[::-1]), col, alpha=alpha_plot)
        ax[i,j].fill(np.append(x, x[::-1]), np.append(lower_bound, mean[::-1]), col, alpha=alpha_plot)


        ax[i,j].grid(True, which='both')
        ax[i,j].set_title(domain, fontsize=font_title)
        if print_losses:
            ax[i,j].set_yscale('log')
        else:
            ax[i,j].set_yscale('linear')
        ax[i,j].set_ylim(top=100)

        if (i,j) == (0,0):
            if print_losses:
                ax[i,j].legend(loc='upper right', fontsize=font_legend)
            else:
                ax[i,j].legend(loc='lower right', fontsize=font_legend)

        ax[i,j].tick_params(axis='both', which='major', labelsize=font_tick_label)
        ax[i,j].tick_params(axis='both', which='minor', labelsize=font_tick_label)

        if j == 0:
            if print_losses:
                ax[i,j].set_ylabel('Efficiency Losses in %', fontsize=font_axis_label)
            else:
                ax[i,j].set_ylabel('Efficiency in %', fontsize=font_axis_label)
        if i == 1:
            ax[i,j].set_xlabel('Clock Round', fontsize=font_axis_label)

        if remove_first_iteration:
            # without the first iteration
            if qmax==50:
                ax[i,j].set_xticks([0,8,18,28,38,48])
                ax[i,j].set_xticklabels([2,10,20,30,40,50])
            elif qmax==100:
                ax[i,j].set_xticks([0,18,38,58,78,98])
                ax[i,j].set_xticklabels([2,20,40,60,80,100])
        else:
            # with the first iteration
            if qmax==50:
                ax[i,j].set_xticks([0,9,19,29,39,49])
                ax[i,j].set_xticklabels([1,10,20,30,40,50])
            elif qmax==100:
                ax[i,j].set_xticks([0,19,39,59,79,99])
                ax[i,j].set_xticklabels([1,20,40,60,80,100])


plt.show()
# %% 3. PROFIT MAX Path Plot
profit_max_keys = [k for k in results.keys() if 'PROFITMAX' in k]

for results_key in profit_max_keys:

    print()
    print(*['#']*50)
    print(results_key)
    if 'GSVM' in results_key:
        domain = 'GSVM'
        i,j = 0,0
    elif 'LSVM' in results_key:
        domain = 'LSVM'
        i,j = 0,1
    elif 'SRVM' in results_key:
        domain = 'SRVM'
        i,j = 1,0
    elif 'MRVM' in results_key:
        domain  = 'MRVM'
        i,j = 1,1
    else:
        raise ValueError('Unknown domain')

    mean, lower_bound, upper_bound = get_CI_bootstrapped(results[results_key], print_losses=print_losses)

    # ALWAYS REMOVE first entry (= 0 profit max bids), since it equals last entry of raised clock bids
    mean = mean[1:]
    lower_bound = lower_bound[1:]
    upper_bound = upper_bound[1:]

    print(f'Length of results: {len(mean)}')
    #print()
    #print(*['*']*50)
    print(f'EFFICIENCY: LBound:{lower_bound[-1]:.2f}, Mean: {mean[-1]:.2f}, UBound: {upper_bound[-1]:.2f}           FOR PAPER: ({lower_bound[-1]:.2f}\,,\,{mean[-1]:.2f}\,,\,{upper_bound[-1]:.2f})')
    #print(*['*']*50)
    #print()
    #print(f'Mean per Iteration: {mean}')
    print(*['#']*50)
    print()
    x = np.arange(len(mean))

    if 'CCA_PROFITMAX' in results_key:
        lty1 = 'g-'
        lty2 = 'g-'
        col = 'green'
        label = 'CCA profit-max'
    elif 'CCA-MLDQ_PROFITMAX' in results_key:
        lty1 = 'b-'
        lty2 = 'b-'
        col = 'blue'
        label = 'ML-CCA profit-max' 
    elif 'RAND-MLDQ_PROFITMAX' in results_key:
        lty1 = 'y-'
        lty2 = 'y-'
        col = 'yellow'
        label = 'ML-CCA-RAND profit-max'
    elif 'CCA-MLDQ-C_PROFITMAX' in results_key:
        lty1 = 'g-'
        lty2 = 'g-'
        col = 'green'
        label = 'ML-CCA-C profit-max' 
    
    ax[i,j].plot(mean, lty1, linewidth=linewidth_means, label=label)
    ax[i,j].plot(upper_bound, lty2, linewidth=linewidth_bounds)
    ax[i,j].plot(lower_bound, lty2, linewidth=linewidth_bounds)
    ax[i,j].fill(np.append(x, x[::-1]), np.append(mean, upper_bound[::-1]), col, alpha=alpha_plot)
    ax[i,j].fill(np.append(x, x[::-1]), np.append(lower_bound, mean[::-1]), col, alpha=alpha_plot)


    ax[i,j].grid(True, which='both')
    ax[i,j].set_title(domain, fontsize=font_title)
    if print_losses:
        ax[i,j].set_yscale('log')
    else:
        ax[i,j].set_yscale('linear')
    #ax[i,j].set_ylim(top=100)

    if (i,j) == (0,0):
        if print_losses:
            ax[i,j].legend(loc='upper right', fontsize=font_legend)
        else:
            ax[i,j].legend(loc='lower right', fontsize=font_legend)

    ax[i,j].tick_params(axis='both', which='major', labelsize=font_tick_label)
    ax[i,j].tick_params(axis='both', which='minor', labelsize=font_tick_label)

    if j == 0:
        if print_losses:
            ax[i,j].set_ylabel('Efficiency Losses in %', fontsize=font_axis_label)
        else:
            ax[i,j].set_ylabel('Efficiency in %', fontsize=font_axis_label)
    if i == 1:
        ax[i,j].set_xlabel('Number of Profit Max Queries', fontsize=font_axis_label)

    ax[i,j].set_xticks([0,19,39,59,79,99])
    ax[i,j].set_xticklabels([1,20,40,60,80,100])


plt.show()