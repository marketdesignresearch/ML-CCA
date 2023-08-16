from pysats import PySats
import numpy as np
from collections import defaultdict

# %%
SATS_domains = ['MRVM']
SATS_seeds = range(101,200)
isLegacy = False
# %%
lows = defaultdict(list)
highs= defaultdict(list)
for SATS_seed in SATS_seeds:
    print(f'Seed: {SATS_seed}')

    for SATS_domain in SATS_domains:
        print(f'Domain: {SATS_domain}')
        if SATS_domain == 'LSVM':
            SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_seed,
                                                                    isLegacyLSVM=isLegacy)  # create SATS auction instance
            print('####### ATTENTION #######')
            print('isLegacyLSVM: %s', SATS_auction_instance.isLegacy)
            print('#########################\n')

        elif SATS_domain == 'GSVM':
            SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_seed,
                                                                    isLegacyGSVM=isLegacy)  # create SATS auction instance
            print('####### ATTENTION #######')
            print('isLegacyGSVM: %s', SATS_auction_instance.isLegacy)
            print('#########################\n')

        elif SATS_domain == 'MRVM':
            SATS_auction_instance = PySats.getInstance().create_mrvm(seed=SATS_seed)  # create SATS auction instance

        elif SATS_domain == 'SRVM':
            SATS_auction_instance = PySats.getInstance().create_srvm(seed=SATS_seed)  # create SATS auction instance

        else:
            raise ValueError(f'SATS_domain {SATS_domain} not yet implemented')
        
        m = len(SATS_auction_instance.get_good_ids())
        null_bundle = np.zeros(m, dtype=np.int64)
        full_bundle = np.ones(m, dtype=np.int64)

        for bidder in SATS_auction_instance.get_bidder_ids():
            print(f'Bidder_{bidder}')
            lows[(SATS_domain,bidder)].append(SATS_auction_instance.calculate_value(bidder, null_bundle))
            highs[(SATS_domain,bidder)].append(SATS_auction_instance.calculate_value(bidder, full_bundle))

print('\n')
lows_avg = defaultdict(list)
highs_avg= defaultdict(list)
for key in lows.keys():
    highs_avg[key] = np.mean(highs[key])
    lows_avg[key] = np.mean(lows[key])
    print(f'{key}: [{lows_avg[key]}, {highs_avg[key]}]')






