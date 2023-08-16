# %%
from pysats import PySats

# make sure you set the classpath before loading any other modules
PySats.getInstance()

import numpy as np
from collections import defaultdict
from pysats_ext import GenericWrapper
import pandas as pd
from matplotlib import pyplot as plt

# %%
SATS_domain = 'SRVM'
SATS_seed = 178
# %%
rec_dd = lambda: defaultdict(rec_dd)
values = rec_dd()


print(f'Seed: {SATS_seed}')
print(f'Domain: {SATS_domain}')

if SATS_domain == 'MRVM':
    SATS_auction_instance = GenericWrapper(PySats.getInstance().create_mrvm(seed=SATS_seed))  # create SATS auction instance

elif SATS_domain == 'SRVM':
    SATS_auction_instance = GenericWrapper(PySats.getInstance().create_srvm(seed=SATS_seed))  # create SATS auction instance

else:
    raise ValueError(f'SATS_domain {SATS_domain} not yet implemented')

m = len(SATS_auction_instance.get_good_ids())
capacities = SATS_auction_instance.get_capacities()
max_capacity = max(capacities.values())

for bidder in SATS_auction_instance.get_bidder_ids():
    print(f'Bidder_{bidder}')
    for item in range(m):
        print(item)
        print(capacities[item])
        item_values = []
        for c in range(capacities[item]+1):
            bundle = [0]*m
            bundle[item] = c
            item_values.append(SATS_auction_instance.calculate_value(bidder, bundle))
        # save increases for stacked barplot later
        item_increase = list(np.array(item_values[1:]) - np.array(item_values[:-1]))
        while len(item_increase) < max_capacity:
            item_increase.append(np.nan)
        values[SATS_domain][bidder][item] = item_increase


#%%

if  SATS_domain == 'MRVM':
    fig, ax = plt.subplots(4, 3, figsize=(20, 10), facecolor='w', edgecolor='k')
    fig.suptitle(f'Domain: {SATS_domain} | Seed: {SATS_seed}', fontsize=16)
    i, j = 0, 0
    row_counter = 0 
    for bidder in values[SATS_domain].keys():
        row_counter += 1
        print(f'Bidder_{bidder}')
        df = pd.DataFrame.from_dict(values[SATS_domain][bidder]).transpose()
        df.columns = [f'c_j={i}' for i in range(1,len(df.columns)+1)]
        print(df)
        ax[i, j] = df.plot.bar(stacked=True, title=f'Bidder_{bidder}', ax = ax[i, j])
        ax[i, j].set_xlabel('Item')
        ax[i, j].set_ylabel('Value')
        ax[i, j].grid()
        j += 1
        if row_counter == 3:
            row_counter = 0
            i += 1
            j = 0

    plt.tight_layout()
    plt.show()

if  SATS_domain == 'SRVM':
    fig, ax = plt.subplots(3, 3, figsize=(20, 10), facecolor='w', edgecolor='k')
    fig.suptitle(f'Domain: {SATS_domain} | Seed: {SATS_seed}', fontsize=16)
    i, j = 0, 0
    row_counter = 0 
    for bidder in values[SATS_domain].keys():
        row_counter += 1
        print(f'Bidder_{bidder}')
        df = pd.DataFrame.from_dict(values[SATS_domain][bidder]).transpose()
        df.columns = [f'c_j={i}' for i in range(1,len(df.columns)+1)]
        print(df)
        ax[i, j] = df.plot.bar(stacked=True, title=f'Bidder_{bidder}', ax = ax[i, j], legend=False)
        ax[i, j].set_xlabel('Item')
        ax[i, j].set_ylabel('Value')
        ax[i, j].grid()

        #ax[i,j].legend()
        j += 1
        if row_counter == 3:
            row_counter = 0
            i += 1
            j = 0
    
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.9, 0.1), ncol=3)

    plt.tight_layout()
    plt.show()



# %%
