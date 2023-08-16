# %%
from ast import RShift
from pysats import PySats
# make sure you set the classpath before loading any other modules
PySats.getInstance()
from pysats_ext import GenericWrapper

import wandb
from datetime import datetime
import os
import pandas as pd
import re
import json
from pdb import set_trace

# %%
# for SATS_domain in ['GSVM', 'LSVM']:
for SATS_domain in ['SRVM']:
    print(f'SATS_domain:{SATS_domain}')
    # Create SATS Worlds ------
    isLegacy = False
    SATS_seeds = list(range(1,101))
    result = {} 

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
        if SATS_domain in ['GSVM', 'LSVM']:
            full_bundle = [1]*m
        elif SATS_domain in ['MRVM', 'SRVM']:
            capacities = SATS_auction_instance.get_capacities()
            full_bundle = [capacities[i] for i in range(m)]

        result[f'Seed_{SATS_seed}'] = {}
        print(f'SATS_seed:{SATS_seed}')

        for bidder_id in bidder_ids:
            bidder_name = f'Bidder_{bidder_id}'
            print(bidder_name)

            null_p = [0]*m

            x_null_p = SATS_auction_instance.get_best_bundles(bidder_id, null_p, 1)
            sum_x_null_p = sum(x_null_p[0])
            v_null_p = SATS_auction_instance.calculate_value(bidder_id, x_null_p[0])

            v_full_bundle = SATS_auction_instance.calculate_value(bidder_id, full_bundle)

            if SATS_domain != 'GSVM' and v_full_bundle != v_null_p:
                raise ValueError('v_full_bundle!=v_null_p')
            
            result[f'Seed_{SATS_seed}'][bidder_name + '_max_value'] = v_null_p
            result[f'Seed_{SATS_seed}'][bidder_name + '_no_items'] = sum_x_null_p
            result[f'Seed_{SATS_seed}'][bidder_name + '_max_value_per_item'] = v_null_p/sum_x_null_p

    PD = pd.DataFrame.from_dict(result, orient='index')
    PD.loc['mean'] = PD.mean()
    print(PD)

    PD.to_json(f'{SATS_domain}_values_for_null_price_seeds{SATS_seeds[0]}-{SATS_seeds[-1]}.json')