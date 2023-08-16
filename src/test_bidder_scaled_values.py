from pysats import PySats
PySats.getInstance()
import wandb
from datetime import datetime
import json
import re
import numpy as np

# own libs
from mlca_demand_queries.mlca_dq_util import init_demand_queries_mlca_unif, key_to_int
from mvnns_demand_query_training.mvnn_dq_training import dq_train_mvnn
from pysats_ext import GenericWrapper
from pdb import set_trace
import argparse
from test_training import generate_random_bundles


parser = argparse.ArgumentParser()

from jnius import autoclass
CPLEXUtils = autoclass('org.marketdesignresearch.mechlib.utils.CPLEXUtils')
SolveParams = autoclass('edu.harvard.econcs.jopt.solver.SolveParam')
CPLEXUtils.SOLVER.setSolveParam(SolveParams.THREADS,8)




def bidder_type_to_bidder_id(SATS_domain,
                             bidder_type):
    bidder_id_mappings = {'GSVM': {'national': [6], 'regional': [0, 1, 2, 3, 4, 5]},
                          'LSVM': {'national': [0], 'regional': [1, 2, 3, 4, 5]},
                          'SRVM': {'national': [5, 6], 'regional': [3, 4], 'high_frequency': [2], 'local': [0, 1]},
                          'MRVM': {'national': [7, 8, 9], 'regional': [3, 4, 5, 6], 'local': [0, 1, 2]}
                          }

    bidder_id = np.random.choice(bidder_id_mappings[SATS_domain][bidder_type], size=1, replace=False)[0]
    print(f'BIDDER ID:{bidder_id}')

    return bidder_id


def get_bidder_values(SATS_domain, SATS_seed):
    TRAIN_parameters = {
                    "price_file_name": 'values_for_null_price_seeds1-100'
                    }
    
    # 1. Load the instance ------
    
    max_linear_prices = {}
    scales = {}
    if SATS_domain == 'GSVM':
        isLegacy = False
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_seed,
                                                                isLegacyGSVM=isLegacy)
        
        GSVM_national_bidder_goods_of_interest = SATS_auction_instance.get_goods_of_interest(6)
        good_capacities = [1 for _ in range(len(SATS_auction_instance.get_good_ids()))]

        V =  json.load(open(f'{SATS_domain}_{TRAIN_parameters["price_file_name"]}.json', 'r')) # AVG value per item  for the null price from GSVM_values_for_null_price.csv  
        for key in V:
            if 'max_value_per_item' in key:
                id = int(re.findall(r'\d+', key)[0])
                max_linear_prices[f'Bidder_{id}'] = V[key]['mean']
            if 'max_value' in key and 'per_item' not in key:
                id = int(re.findall(r'\d+', key)[0])
                scales[f'Bidder_{id}'] = V[key]['mean'] 

    elif SATS_domain == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_seed,
                                                            isLegacyLSVM=isLegacy)
        
        GSVM_national_bidder_goods_of_interest = None
        good_capacities = [1 for _ in range(len(SATS_auction_instance.get_good_ids()))]
        V =  json.load(open(f'{SATS_domain}_{TRAIN_parameters["price_file_name"]}.json', 'r')) # AVG value per item  or the null price from LSVM_values_for_null_price.csv
        for key in V:
            if 'max_value_per_item' in key:
                id = int(re.findall(r'\d+', key)[0])
                max_linear_prices[f'Bidder_{id}'] = V[key]['mean']
            if 'max_value' in key and 'per_item' not in key:
                id = int(re.findall(r'\d+', key)[0])
                scales[f'Bidder_{id}'] = V[key]['mean']

    elif SATS_domain in ['MRVM', 'SRVM']: 
        if SATS_domain == 'MRVM':
            non_generic_instance = PySats.getInstance().create_mrvm(seed=SATS_seed)
        else: 
            non_generic_instance = PySats.getInstance().create_srvm(seed=SATS_seed)
        SATS_auction_instance = GenericWrapper(non_generic_instance)
        GSVM_national_bidder_goods_of_interest = None
        capacities_dict = SATS_auction_instance.get_capacities()
        good_capacities = np.array([capacities_dict[i] for i in range(len(capacities_dict))])
        # set_trace()
        V =  json.load(open(f'{SATS_domain}_{TRAIN_parameters["price_file_name"]}.json', 'r')) # AVG value per item for the null price from LSVM_values_for_null_price.csv
        for key in V:
            if 'max_value_per_item' in key:
                id = int(re.findall(r'\d+', key)[0])
                max_linear_prices[f'Bidder_{id}'] = V[key]['mean']
            if 'max_value' in key and 'per_item' not in key:
                id = int(re.findall(r'\d+', key)[0])
                scales[f'Bidder_{id}'] = V[key]['mean']

    else:
        raise ValueError(f'SATS_domain:{SATS_domain} not implemented yet.')
    

    # --------
    
    # 2. For all bidders, calculate true values and scaled ones
    bidder_values_scaled = [] 
    for bidder_id in SATS_auction_instance.get_bidder_ids(): 
        bundles =  generate_random_bundles(capacities=good_capacities, number_of_bundles=50000)

        true_values_generalization = np.array([SATS_auction_instance.calculate_value(bidder_id, bundle) for bundle in bundles])

        scaled_true_values = true_values_generalization / scales[f'Bidder_{bidder_id}']
        bidder_values_scaled.append(scaled_true_values)

        print('Bidder:', bidder_id, 'scaled_values_mean:' , np.mean(scaled_true_values), 'scaled_values_std:', np.std(scaled_true_values))

    return bidder_values_scaled

def test_domain(domain, instances_to_average = 2):
    bidder_values_all_instances = []
    for i in range(instances_to_average):         
        print('Starting instance: ', 100 + i)
        bidder_values_instance = get_bidder_values(domain,100 + i)
        bidder_values_all_instances.append(bidder_values_instance)

    bidder_values_all_instances = np.array(bidder_values_all_instances)

    instance_mean = np.mean(bidder_values_all_instances, axis=2)  # shape: instances x bidders
    mean_per_bidder = np.mean(instance_mean, axis=0) # shape: bidders
    std_per_bidder = np.std(instance_mean, axis=0) # shape: bidders

    print('---- RESULTS ----')
    for i in range(len(mean_per_bidder)):
        print(f'Bidder_{i}: mean: {mean_per_bidder[i]} std: {std_per_bidder[i]}')
        


test_domain('MRVM', instances_to_average= 5)

    

        