# %%
from ast import RShift
from pysats import PySats
# make sure you set the classpath before loading any other modules
PySats.getInstance()

import wandb
from datetime import datetime
import os
import pandas as pd
import numpy as np 
import re
import json
from pysats_ext import GenericWrapper


from pdb import set_trace



# for SATS_domain in ['GSVM', 'LSVM']:
for SATS_domain in ['SRVM']:
    print(f'SATS_domain:{SATS_domain}')
    # Create SATS Worlds ------
    isLegacy = False
    SATS_seeds = list(range(1 + 200,1001 + 200))
    result = {} 

    # create one instance to get the number of items
    if SATS_domain == 'GSVM':
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=10001,
                                                                    isLegacyGSVM=isLegacy)
    elif SATS_domain == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=10001,
                                                                    isLegacyLSVM=isLegacy)
        
    elif SATS_domain == 'MRVM': 
        mrvm = PySats.getInstance().create_mrvm(1)
        SATS_auction_instance = GenericWrapper(mrvm)

    elif SATS_domain == 'SRVM': 
        srvm = PySats.getInstance().create_srvm(1)
        SATS_auction_instance = GenericWrapper(srvm)

    m = len(SATS_auction_instance.get_good_ids())



    result = [[] for i in range(m)]


    for SATS_seed in SATS_seeds:
        if SATS_domain == 'GSVM':
            SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_seed,
                                                                    isLegacyGSVM=isLegacy)

        elif SATS_domain == 'LSVM':
            SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_seed,
                                                                    isLegacyLSVM=isLegacy)

        elif SATS_domain == 'SRVM': 
            srvm_non_generic = PySats.getInstance().create_srvm(seed=SATS_seed)
            SATS_auction_instance = GenericWrapper(srvm_non_generic)

        elif SATS_domain == 'MRVM': # TODO: DQs not yet implemented
            mrvm_non_generic = PySats.getInstance().create_mrvm(seed=SATS_seed)
            SATS_auction_instance = GenericWrapper(mrvm_non_generic)
        else:
            pass
        # ------

        bidder_ids = SATS_auction_instance.get_bidder_ids()
        m = len(SATS_auction_instance.get_good_ids())
        n = len(bidder_ids)

        print(f'SATS_seed:{SATS_seed}')

        for item in range(m):
            # item_values = [] 
            item_bundle = [0]*m
            item_bundle[item] = 1
            for bidder_id in bidder_ids:

                item_value = SATS_auction_instance.calculate_value(bidder_id, item_bundle)

                result[item].append(item_value)

    result = np.array(result)
    result = result.mean(axis=1)

    np.save(f'{SATS_domain}_average_item_values_seeds_{SATS_seeds[0]}-{SATS_seeds[-1]}', result)
    

    # PD.loc['mean'] = PD.mean()
    # print(PD)

    # PD.to_json(f'{SATS_domain}_average_item_values_seeds{SATS_seeds[0]}-{SATS_seeds[-1]}.json')