
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
domains = ['GSVM', 'LSVM', 'SRVM'] # no MRVM since always 0 clearing percentage
mechanisms = ['ML-CCA', 'CCA', 'ML-CCA-U']
qmax = 100 # select between 50, 100
comparison = 'constrained' # select between 'constrained', 'constrained_vs_unconstrained_ML-CCA'
alpha_t_test = 0.05 # significance level for t-test
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# %% read results and postprocess
if comparison == 'unconstrained':
    folder_path = f'./wandb_custom_plots/raw_data/clearing_prices/qmax{qmax}/unconstrained'
elif comparison == 'constrained':
    folder_path = f'./wandb_custom_plots/raw_data/clearing_prices/qmax{qmax}/constrained'
elif comparison == 'constrained_vs_unconstrained_ML-CCA':
    folder_path = f'./wandb_custom_plots/raw_data/clearing_prices/qmax{qmax}/constrained_vs_unconstrained_ML-CCA'

# initialize dictionary T_TEST_DATA with None values
T_TEST_DATA = dict()

for domain in domains:
    for mechanism in mechanisms:
        for filename in os.listdir(folder_path):

            ###############
            # 1. CLEARING PRICE DATA
            ###############
            if filename.endswith('.csv') and filename.startswith(f'{domain}_{mechanism}_'):
                print(domain)
                print(mechanism)
                print(f'Reading file:{os.path.join(folder_path,filename)}')
                raw_data = pd.read_csv(os.path.join(folder_path,filename))

                # postprocess data
                postprocessed_data  = deepcopy(raw_data)
                # drop unnecessary columns
                #keep_columns = [col for col in postprocessed_data.columns if 'step' not in col and 'MIN' not in col and 'MAX' not in col]
                keep_columns = [col for col in postprocessed_data.columns if 'step' not in col and 'MIN' not in col]
                keep_columns = [col for col in keep_columns if 'MAX' in col]
                # split clock bids and raised clock bids 
                clearing_price_data = postprocessed_data[keep_columns]

                #T-TEST-DATA
                #-----------------------------------------------------------------------------------------------------------------
                # 3. profit max
                tmp_t_test_clearingPrice = deepcopy(clearing_price_data)
                tmp_t_test_clearingPrice=tmp_t_test_clearingPrice.iloc[-1,:].to_frame()
                #rename values in index to only contain the seed number
                old_row_names = tmp_t_test_clearingPrice.index.values
                new_row_names = [int(re.findall(r'\d+', x)[0]) for x in old_row_names]
                tmp_t_test_clearingPrice = tmp_t_test_clearingPrice.rename(index = dict(zip(old_row_names, new_row_names)))
                if (domain,'clearingPrice') in T_TEST_DATA:
                    # concetanate last column of PD_DATA with T_TEST_DATA
                    T_TEST_DATA[(domain,'clearingPrice')] = pd.concat([T_TEST_DATA[(domain,'clearingPrice')], tmp_t_test_clearingPrice], axis=1)
                else:
                    T_TEST_DATA[(domain,'clearingPrice')] = tmp_t_test_clearingPrice
                    # rename last column of T_TEST_DATA to results_key
                T_TEST_DATA[(domain,'clearingPrice')].columns = [*T_TEST_DATA[(domain,'clearingPrice')].columns[:-1], mechanism]
                #-----------------------------------------------------------------------------------------------------------------


# %% Make T-Tests
for k,v in T_TEST_DATA.items():
    print(k[0],k[1])
    print(v.describe())
    if comparison == 'constrained_vs_unconstrained_ML-CCA':
        t = ttest_rel(v['ML-CCA'], v['ML-CCA-U'], alternative='greater',nan_policy='omit') #when comparing constrained vs unconstrained
        print(f'Paired t-test with HA: ML-CCA > ML-CCA-U and H0: ML-CCA <= ML-CCA-U (alpha={alpha_t_test}):')#when comparing constrained vs unconstrained
    else:
        t = ttest_rel(v['ML-CCA'], v['CCA'], alternative='greater',nan_policy='omit')# when comparing ML-CCA vs CCA
        print(f'Paired t-test with HA: ML-CCA > CCA and H0: ML-CCA <= CCA (alpha={alpha_t_test}):')# when comparing ML-CCA vs CCA
    if t.pvalue <= alpha_t_test:
        print(f'Result paired t-test:  pvalue={t.pvalue} | statistic={t.statistic} | df={t.df} | REJECT H0')
    else:
        print(f'Result paired t-test:  pvalue={t.pvalue} | statistic={t.statistic} | df={t.df} | NOT REJECT H0')
    print()
print('\n')