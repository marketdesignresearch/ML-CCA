"""
FILE DESCRIPTION:

This file stores helper functions for mlca.

"""

# %% Libs
import logging
from collections import OrderedDict
import re
import numpy as np
# from pdb import set_trace

# %% MEASURE TIME

'''
td = timedelta object from datetime
'''
def timediff_d_h_m_s(td):
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return -(td.days), -int(td.seconds / 3600), -int(td.seconds / 60) % 60, -(td.seconds % 60)
    return td.days, int(td.seconds / 3600), int(td.seconds / 60) % 60, td.seconds % 60


# %% TRANSFORM BIDDER NAMES (str) TO BIDDER KEYS (int)

'''
Tranforms bidder_name to integer bidder_id

key = valid bidder_key (string), e.g. 'Bidder_0'
'''

def key_to_int(key):
    return (int(re.findall(r'\d+', key)[0]))

# %% 

'''
INITIAL DETERMINISTIC INCREASING DEMAND QUERY BIDS FOR LINEAR PRICES
THIS METHOD USES .get_best_bundles(bidder_id, p, 1) from SATS to return the max utility bundle given price_vector p

SATS_auction_instance = single instance of a value model
number_initial_bids = number of initial bids
start_linear_item_prices = np.array of starting linear item prices
end_linear_item_prices = np.array of ending linear item prices
bidder_id =  if not None: only generate data for the bidder with bidder_id 
'''

def init_demand_queries_mlca_increasing(SATS_auction_instance,
                                        number_initial_bids: int,
                                        start_linear_item_prices: np.array,
                                        end_linear_item_prices: np.array,
                                        bidder_id = None   #if not None: only generate data for the present bidder ids 
                                        ):

    m = len(SATS_auction_instance.get_good_ids())
    bidder_ids = SATS_auction_instance.get_bidder_ids()

    if bidder_id is not None:
        bidder_ids = [bidder_id] # only keep the bidder id of interest

    n = len(bidder_ids)
    initial_bids = OrderedDict()

    P = np.zeros((number_initial_bids,m))
    price_increments =  (end_linear_item_prices-start_linear_item_prices)/(number_initial_bids-1)
    for i in range(number_initial_bids):
        P[i,:] = start_linear_item_prices + i*price_increments

    i = 0
    for bidder_id in bidder_ids:

        X = []
        for p in P:
            X += SATS_auction_instance.get_best_bundles(bidder_id, p, 1)

        X = np.asarray(X)

        logging.info(f'Bidder_{bidder_id}: increasing INITIAL DEMAND QUERIES [X,P] with X in {X.shape}, P in {P.shape}.')
        print(f'Bidder_{bidder_id}: increasing INITIAL DEMAND QUERIES [X,P] with X in {X.shape}, P in {P.shape}.')  # TODO:delete
        initial_bids[f'Bidder_{bidder_id}'] = [X, P]
        i += 1

    print('INITIAL DEMAND QUERIES DONE')
    return initial_bids


def init_demand_queries_mlca_cca(SATS_auction_instance,
                                    capacities, 
                                    number_initial_bids: int,
                                    start_linear_item_prices: np.array,
                                    price_increment: float, 
                                    include_null_price: bool = True
                                    ):
    """
    Generates initial demand queries for MLCA by increasing the price of each item proportionally to the price increment for every overdemanded item. 
    """
    print("starting init_demand_queries_mlca_cca")

    m = len(SATS_auction_instance.get_good_ids())
    bidder_ids = SATS_auction_instance.get_bidder_ids()

    # if bidder_id is not None:
    #     bidder_ids = [bidder_id] # only keep the bidder id of interest

    n = len(bidder_ids)
    initial_bids = OrderedDict()

    for bidder_id in bidder_ids:
        initial_bids[f'Bidder_{bidder_id}'] = [[], []]

    bids_used = 0 
    if include_null_price:
        bids_used += 1
        p = np.zeros(m)
        for bidder_id in bidder_ids:
            bidder_demand = SATS_auction_instance.get_best_bundles(bidder_id, p, 1)[0]
            X, P = initial_bids[f'Bidder_{bidder_id}']
            X.append(bidder_demand)
            P.append(p)
            initial_bids[f'Bidder_{bidder_id}'] = [X, P]
    

    prices = start_linear_item_prices.copy()
    while bids_used < number_initial_bids:
        total_demand = np.zeros(m) 
        for bidder_id in bidder_ids:
            bidder_demand = SATS_auction_instance.get_best_bundles(bidder_id, prices, 1)[0]
            total_demand += bidder_demand
            
            # add the new bundle to the bidder's initial bids
            X, P = initial_bids[f'Bidder_{bidder_id}']
            X.append(bidder_demand)
            P.append(prices.copy())
            initial_bids[f'Bidder_{bidder_id}'] = [X, P]

        # raise the price of each item by the price increment if it is overdemanded
        overdemand = total_demand - capacities
        for j in range(m):
            if overdemand[j] > 0:
                prices[j] = prices[j] * (1 + price_increment)

        bids_used += 1



    for bidder_id in bidder_ids:
        X, P = initial_bids[f'Bidder_{bidder_id}']
        X = np.array(X)
        P = np.array(P)

        initial_bids[f'Bidder_{bidder_id}'] = [X, P]

        logging.info(f'Bidder_{bidder_id}: INITIAL DEMAND QUERIES [X,P] with X in {X.shape}, P in {P.shape}.')
        print(f'Bidder_{bidder_id}: INITIAL DEMAND QUERIES [X,P] with X in {X.shape}, P in {P.shape}.')

    print('INITIAL DEMAND QUERIES DONE')

    return initial_bids


# %% 

'''
INITIAL UNFORM RANDOM DEMAND QUERY BIDS FOR LINEAR PRICES
THIS METHOD USES .get_best_bundles(bidder_id, p, 1) from SATS to return the max utility bundle given price_vector p

SATS_auction_instance = single instance of a value model
number_initial_bids = number of initial bids
bidder_names = bidder_names (str)
max_linear_prices = linear prices for bidder i are sampled uniformly at random from [0,max_linear_prices[i]]
seed = seed for random initial bids
include_null_price =  if the nulll price should be included
bidder_id =  if not None: only generate data for the bidder with bidder_id
'''

def init_demand_queries_mlca_unif(SATS_auction_instance,
                                  number_initial_bids: int,
                                  max_linear_prices: dict,
                                  seed: int,
                                  include_null_price: bool = True, 
                                  bidder_id = None,
                                  price_method = 'old', 
                                  min_price_per_item = [], 
                                  max_price_per_item = [] 
                                  ):
    """
    max_linear_prices: dict with keys 'Bidder_0', 'Bidder_1', etc. and values the maximum linear price for each bidder
    seed: seed for random initial bids
    include_null_price: if the 0 price should be included 
    bidder_id: if not None: only generate data for the bidder with bidder_id
    price_method: 'old' or 'new'. 'old' uses the max_linear_prices to sample the prices. 'new' uses min_price_per_item and max_price_per_item to sample the prices.
    min_price_per_item: list of length m with the minimum price for each item
    max_price_per_item: list of length m with the maximum price for each item
    """

    m = len(SATS_auction_instance.get_good_ids())
    bidder_ids = SATS_auction_instance.get_bidder_ids()

    
    if bidder_id is not None:
        bidder_ids = [bidder_id] # only keep the bidder id of interest


    n = len(bidder_ids)
    initial_bids = OrderedDict()

    if include_null_price:
        number_initial_bids -=1

    # seed determines bidder_seeds for all bidders, e.g. seed=10 and 3 bidders generates bidder_seeds=[28,29,30]
    if seed is not None:
        bidder_seeds = list(range((seed-1) * n+1, (seed) * n+1))
    logging.debug(f'Bidder specific seeds for initial bundle-value pairs:{bidder_seeds}')

    i = 0
    for bidder_id in bidder_ids:

        np.random.seed(bidder_seeds[i])
        
        if price_method == 'old':
            P = np.random.rand(number_initial_bids,m)*max_linear_prices[f'Bidder_{bidder_id}']
        elif price_method == 'new':
            P = np.random.uniform(low = min_price_per_item, high = max_price_per_item, size = (number_initial_bids,m))

        if include_null_price:
            P = np.concatenate((np.zeros(m).reshape(1,-1), P), axis=0)

        X = []
        for p in P:
            X += SATS_auction_instance.get_best_bundles(bidder_id, p, 1)

        X = np.asarray(X)

        logging.info(f'Bidder_{bidder_id}: uniformly at random sampled INITIAL DEMAND QUERIES [X,P] with X in {X.shape}, P in {P.shape}.')
        initial_bids[f'Bidder_{bidder_id}'] = [X, P]
        i += 1

    print('INITIAL DEMAND QUERIES DONE')
    return initial_bids
# %% INITIAL VALUE QUERY BIDS

'''
FOR A SINGLE INSTANCE FOR ALL BIDDERS for MLCA
THIS METHOD USES SATS SAMPLING FROM ADMISSIBLE BUNDLE SPACE

SATS_auction_instance = single instance of a value model
number_initial_bids = number of initial bids
bidder_names = bidde _names (str)
scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set
seed = seed for random initial bids
'''

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def initial_bids_mlca_unif(SATS_auction_instance,
                           number_initial_bids,
                           bidder_names,
                           scaler=None,
                           seed=None,
                           include_full_bundle=False):

    initial_bids = OrderedDict()

    if include_full_bundle:
        number_initial_bids -=1

    # seed determines bidder_seeds for all bidders, e.g. seed=10 and 3 bidders generates bidder_seeds=[28,29,30]
    n_bidders = len(bidder_names)
    if seed is not None:
        bidder_seeds = list(range((seed-1) * n_bidders+1, (seed) * n_bidders+1))
    else:
        bidder_seeds = [None] * n_bidders
    logging.debug(f'Bidder specific seeds for initial bundle-value pairs:{bidder_seeds}')

    i = 0
    for bidder in bidder_names:

        bidder_id = key_to_int(bidder)

        # Sampling method from SATS, which incorporates bidder specific restrictions:
        # e.g. in GSVM for regional bidders only bundles of up to size 4 are sampled and for national bidders only bundles that
        # contain items from the national-circle are sampled.
        # Remark: SATS does not ensure that bundles are unique, this needs to be taken care exogenously.
        # D = (X,y) in ({0,1}^m x R_+)^number_of_bids*(m+1)
        D = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=bidder_id,
                                                                     number_of_bids=number_initial_bids,
                                                                     seed=seed),dtype=np.float32)

        # only use X from SATS generator, since then uniqueness check is easier
        X = D[:, :-1]

        M = len(SATS_auction_instance.get_good_ids())

        full_bundle = np.array([1]*M, dtype=np.float32)
        empty_bundle = np.array([0]*M, dtype=np.float32)

        # Remove full bundle and null bundle if they were drawn
        full_idx = np.where(np.all(X==full_bundle,axis=1))[0]
        empty_idx =  np.where(np.all(X==empty_bundle,axis=1))[0]
        if len(full_idx)>0:
            X = np.delete(X,full_idx,axis=0)
        if len(empty_idx)>0:
            X = np.delete(X,empty_idx,axis=0)
        #

        # UNIQUENESS
        X = np.unique(X,axis=0)
        seed_additional_bundle = None if seed is None else (10 ** 5) * seed
        while X.shape[0] != (number_initial_bids):
            logging.debug(f'Generate new bundle: only {X.shape[0]+1} are unique but you asked for:{number_initial_bids}')
            dnew = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=bidder_id,number_of_bids=1,seed=seed_additional_bundle))
            xnew = dnew[:, :-1]
            # Check until new bundle is different from FULL_BUNDLE and NULL_BUNDLE
            while np.all(xnew==full_bundle) or np.all(xnew==empty_bundle):
                if seed_additional_bundle is not None: seed_additional_bundle += 1
                dnew = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=bidder_id,number_of_bids=1,seed=seed_additional_bundle),dtype=np.float32)
                xnew = dnew[:, :-1]
                logging.debug(f'RESAMPLE additional bundle SINCE it equals null-bundle OR full-bundle:{xnew}')
            X = np.concatenate((X,xnew),axis=0)
            X = np.unique(X,axis=0)
            if seed_additional_bundle is not None: seed_additional_bundle += 1
        # --------------------------------------------

        # generate bidders' values for initial bundles
        y = np.array(SATS_auction_instance.calculate_values(bidder_id, X), dtype=np.float32)
        # --------------------------------------------

        # a) potentially include full bundle
        # --------------------------------------------
        if include_full_bundle:
            value_full_bundle = np.array([SATS_auction_instance.calculate_value(bidder_id, full_bundle)], dtype=np.float32)
            X = np.concatenate((X,full_bundle.reshape(1,-1)))
            y = np.concatenate((y,value_full_bundle))
        # --------------------------------------------

        # b) always include empty bundle since no value query needed, i.e., we know that it has value 0
        # --------------------------------------------
        value_empty_bundle = np.array([0.0], dtype=np.float32)
        X = np.concatenate((X,empty_bundle.reshape(1,-1)))
        y = np.concatenate((y,value_empty_bundle))
       # --------------------------------------------

        X = X.astype(int) # needed for MIP
        y = y.astype(np.float32)
        X, y = unison_shuffled_copies(X, y)

        assert len(np.unique(X,axis=0)) == len(X)
        logging.info('INIT BUNDLE-VALUE PAIRS ARE UNIQUE')
        logging.info(f'X in {X.shape}, y in {y.shape} (incl. empty-bundle) with include_full_bundle:{include_full_bundle}.')
        initial_bids[bidder] = [X, y]
        i += 1

    if scaler is not None:
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list(
            (key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in
            initial_bids.items()))

    return (initial_bids, scaler)

# %% FORMAT CPLEX MILP SOLUTION

'''
This function formates the solution of the winner determination problem (WDP) given elicited bids.

Mip = A solved DOcplex instance.
elicited_bids = the set of elicited bids for each bidder corresponding to the WDP.
bidder_names = bidder names (string, e.g., 'Bidder_1')
fitted_scaler = the fitted scaler used in the valuation model.
'''

def format_solution_mip_new(Mip,
                            elicited_bids,
                            bidder_names,
                            fitted_scaler,
                            generic_domain = False):
    
    if not generic_domain: 
        tmp = {'good_ids': [], 'value': 0}
    else: 
        tmp = {'allocated_bundle': np.zeros(elicited_bids[0][0].shape[0] - 1), 'value': 0}
    Z = OrderedDict()
    for bidder_name in bidder_names:
        Z[bidder_name] = tmp
    S = Mip.solution.as_dict()

    for key, cplex_value in S.items():
        if cplex_value < 1 - (10 ** -6) or cplex_value > 1 + (10 ** -6):
            logging.warning('cplex_value in solution not within integrality tolerance for 1: %s', cplex_value)
            continue 
        key = str(key)
        index = [int(x) for x in re.findall(r'\d+', key)]
        bundle = elicited_bids[index[0]][index[1], :-1]
        value = elicited_bids[index[0]][index[1], -1]

        if fitted_scaler is not None:
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug(value)
            logging.debug('WDP values for allocation scaled by: 1/%s', round(fitted_scaler.scale_[0], 8))
            value = float(fitted_scaler.inverse_transform([[value]]))
            logging.debug(value)
            logging.debug('---------------------------------------------')

        bidder = bidder_names[index[0]]
        #TODO: check if this is the desired format for the generic domains
        if generic_domain:
            # Z[bidder] = {'good_ids': {item:int(bundle[item]) for item in range(len(bundle))}, 'value': value}
            Z[bidder] = {'allocated_bundle': bundle, 'value': value}
        else:
            Z[bidder] = {'good_ids': list(np.where(bundle == 1)[0]), 'value': value}

    return Z
