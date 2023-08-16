# Libs
import argparse
import json
import logging
import os
import re
import numpy as np

# Own Libs
from mlca_demand_queries.mlca_dq_mechanism import mechanism
from util import StreamToLogger
# from pdb import set_trace

# in order to properly parallelize on the server side
# --------------------------------------
from pysats import PySats

# set jnius classpath before doing anything else
PySats.getInstance()
from jnius import autoclass

from pysats_ext import GenericWrapper

CPLEXUtils = autoclass('org.marketdesignresearch.mechlib.utils.CPLEXUtils')
SolveParams = autoclass('edu.harvard.econcs.jopt.solver.SolveParam')
CPLEXUtils.SOLVER.setSolveParam(SolveParams.THREADS,20)    # NOTE: From this you can change the number of threads that CPLEX uses, important for parallelization    
# --------------------------------------

def main(domain, init_method, new_query_option, qinit, seed):
    
    # 0. Set up logging
    logging.basicConfig(level=logging.INFO)

    
    # 1. SATS Parameters
    # ----------------------------------------------
    SATS_parameters = {"SATS_domain": domain,
                       "isLegacy": False,
                       "SATS_seed": seed
                        }
    # ----------------------------------------------

    # 1.5 Load the json with the HPOed parameters for the MVNNs and the one for the linear prices
    hpo_file_name = 'hpo_configs.json' # TODO: change back    
    hpo_dict =  json.load(open(hpo_file_name, 'r'))
    price_file_name = 'values_for_null_price_seeds1-100'
    average_price_file_name = 'average_item_values_seeds_201-1200'
    price_dict =  json.load(open(f'{domain}_{price_file_name}.json', 'r')) # AVG value per item   
    all_bidders_scales = {} 
    all_bidders_max_linear_prices = {}
    for key in price_dict.keys():
        if 'max_value_per_item' in key:
            id = int(re.findall(r'\d+', key)[0])
            all_bidders_max_linear_prices[id] = price_dict[key]['mean']
        if 'max_value' in key and 'per_item' not in key:
            id = int(re.findall(r'\d+', key)[0])
            all_bidders_scales[id] = price_dict[key]['mean'] 
    # ----------------------------------------------

    MVNN_parameters_all_bidders = {}
    TRAIN_parameters_all_bidders = {}
    if domain == 'GSVM':
        num_bidders = 7
        num_items = 18
    elif domain == 'LSVM':
        num_bidders = 6
        num_items = 18
    elif domain == 'MRVM':
        num_bidders = 10
        num_items = 42
    elif domain == 'SRVM':
        num_bidders = 7
        num_items = 3
    for i in range(num_bidders):
        if domain == 'GSVM':
            if i == 6:
                bidder_type = 'National'
            else:
                bidder_type = 'Regional'
        elif domain == 'LSVM':
            if i == 0: 
                bidder_type = 'National'
            else:
                bidder_type = 'Regional'
        elif domain == 'SRVM':
            if i in [0, 1]: 
                bidder_type = 'Local'
            elif i in [2]:
                bidder_type = 'High Frequency'
            elif i in [3, 4]:
                bidder_type = 'Regional'
            elif i in [5, 6]:
                bidder_type = 'National'
        elif domain == 'MRVM':
            if i in [0, 1, 2]: 
                bidder_type = 'Local'
            elif i in [3, 4, 5, 6]:
                bidder_type = 'Regional'
            elif i in [7, 8, 9]:
                bidder_type = 'National'
        
        # 2. MVNN Parameters
        # ----------------------------------------------
        MVNN_parameters = {'num_hidden_layers': hpo_dict[domain][bidder_type]['num_hidden_layers'],
                            'num_hidden_units': hpo_dict[domain][bidder_type]['num_hidden_units'],
                            'layer_type': 'MVNNLayerReLUProjected',
                            'target_max': 1, 
                            'lin_skip_connection': hpo_dict[domain][bidder_type]['lin_skip_connection'],
                            'dropout_prob': 0,
                            'init_method':'custom',
                            'random_ts': [0,1],
                            'trainable_ts': True,
                            'init_E': 1,
                            'init_Var': 0.09,
                            'init_b': 0.05,
                            'init_bias': 0.05,
                            'init_little_const': 0.1
                            }   
        # 3. Train Parameters
        # ----------------------------------------------
        if domain == 'GSVM':
            end_linear_item_prices_multiplier = 30 
        elif domain == 'LSVM':
            end_linear_item_prices_multiplier = 40 
        elif domain == 'MRVM':
            end_linear_item_prices_multiplier = 40 # TODO: FIX if we implement this method
        elif domain == 'SRVM':
            end_linear_item_prices_multiplier = 40 # TODO: FIX if we implement this method
        

        TRAIN_parameters = {"number_val_data_points": 10,
                            "max_linear_prices_multiplier": hpo_dict[domain][bidder_type]['max_linear_prices_multiplier'], # NOTE: only used for "initial_demand_query_method=random"; they actually get loaded from the file
                            "max_linear_price": all_bidders_max_linear_prices[i], 
                            "scale": all_bidders_scales[i], 
                            "start_linear_item_prices": np.zeros(num_items), 
                            "end_linear_item_prices": np.ones(num_items)* end_linear_item_prices_multiplier, # NOTE: only used for "initial_demand_query_method=increasing"
                            "price_file_name": 'values_for_null_price_seeds1-100.json',
                            "average_price_file_name": average_price_file_name,
                            'batch_size': 1,
                            'epochs': hpo_dict[domain][bidder_type]['epochs'],      
                            'l2_reg': hpo_dict[domain][bidder_type]['l2_reg'],
                            'learning_rate': hpo_dict[domain][bidder_type]['learning_rate'],
                            'clip_grad_norm': 1,
                            'use_gradient_clipping': False,
                            'scale_multiplier': hpo_dict[domain][bidder_type].get('scale_multiplier', 1),    # NOTE: only used for dynamic scaling.
                            'print_frequency': 1
                            }
        MVNN_parameters_all_bidders[f'Bidder_{i}'] = MVNN_parameters
        TRAIN_parameters_all_bidders[f'Bidder_{i}'] = TRAIN_parameters
    # ----------------------------------------------

    # 4. SET Mechanism PARAMETERS:
    mechanism_parameters = {'Qinit': qinit,  # TODO: CHANGE THIS BACK TO 20 
                            'Qmax': 100,  # TODO: change back to 100 (same as MLCA)
                            'new_query_option': new_query_option, 
                            'initial_demand_query_method': init_method, # select between 'random', 'increasing', 'cca', and 'cca_original'
                            'calculate_raised_bids': True, # if true: at every iteration the raised bids will be calculated and both efficiencies will be logged. 
                            "cca_start_linear_item_prices": np.load(f'{domain}_{average_price_file_name}.npy'), # NOTE: only used for "initial_demand_query_method=cca"
                            "cca_initial_prices_multiplier": 0.2 if domain in ['LSVM', 'MRVM'] else 1.6, # NOTE: only used for "initial_demand_query_method=cca", will multiply the initial prices. 
                            "calculate_profit_max_bids": False, # NOTE: This will calculate profit max bids for every ML-powered clock round, very expensive.
                            "calculate_profit_max_bids_unraised": False, 
                            "calculate_profit_max_bids_specific_rounds": [50, 100],  # NOTE: from this you can change in which clock rounds profit max bids will be calculated.
                            "profit_max_grid": [5, 100],   # NOTE: from this you can change the number of profit max bids that will be used
                            'parallelize_training': True,
                            'calculate_efficiency_per_iteration': True, 
                            'dynamic_scaling': False if domain in ['GSVM', 'LSVM', 'SRVM'] else True,   # only true for MRVM  
                            'hpo_file_name': hpo_file_name,
                            'W_epochs': 100 if domain in ['GSVM', 'LSVM'] else 1000,
                            'W_lr': 1 if domain in ['GSVM', 'LSVM'] else 4 if domain in ['SRVM'] else 5 * 1000000,
                            'W_lr_decay': 0.99 if domain in ['GSVM', 'LSVM', 'SRVM'] else 0.999, 
                            'W_v2_max_steps_without_improvement': 100 if domain in ['GSVM', 'LSVM', 'SRVM'] else 300,   # parameters for the new GD procedure to minimize W
                            'W_v2_lr': 1 if domain in ['GSVM', 'LSVM'] else 4 if domain in ['SRVM'] else 5 * 1000000,
                            'W_v2_lr_decay': 0.99 if domain in ['GSVM', 'LSVM', 'SRVM'] else 0.999, 
                            'W_v3_max_steps_without_improvement': 250,   # parameters for the new GD procedure to minimize W
                            'W_v3_max_steps': 300,   # parameters for the new GD procedure to minimize W  
                            'W_v3_lr': 0.01,   # NOTE: here you can change all hyperparamters for the GD procedure to minimize W in the paper
                            'W_v3_lr_decay': 0.995, 
                            'W_v3_filter_feasible': True if domain in ['GSVM', 'LSVM', 'SRVM', 'MRVM'] else False,  
                            'W_v3_feasibility_multiplier': 2 if domain in ['GSVM', 'LSVM', 'SRVM', 'MRVM'] else 0,  # punish over demand during GD by more than under-demand, because the first one leads to infeasible bids, which are harder to combine.
                            'W_v3_feasibility_multiplier_increase_factor': 1.01 if domain in ['GSVM', 'LSVM', 'SRVM', 'MRVM'] else 1,
                            'W_log_frequency': 10, # how often to log all details of W minimization, if wandb tracking is on.
                            'identical_p_threshold': 0.05 if domain in ['GSVM', 'LSVM', 'SRVM', 'MRVM'] else 0, 
                            'identical_p_threshold_decay': 0.95,
                            }
    
    if mechanism_parameters['new_query_option'] == 'cca':
        if domain in ['LSVM', 'MRVM', 'SRVM']: 
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.2
        elif domain in ['GSVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 1.6
    elif mechanism_parameters['new_query_option'] in ['gd_linear_prices_on_W', 'gd_linear_prices_on_W_v2', 'gd_linear_prices_on_W_v3']:
        if domain in ['LSVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.7
        elif domain in ['GSVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 1.6
        elif domain in ['MRVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.2
        elif domain in ['SRVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.2

    # -------------------
    if mechanism_parameters['new_query_option'] == 'cca':  
        if mechanism_parameters['Qmax'] == 100: 
            mechanism_parameters['cca_increment'] = 0.05
        elif mechanism_parameters['Qmax'] == 50:
            mechanism_parameters['cca_increment'] = 0.1025 # so that after 50 clock rounds we have the same possible price increments as in the 100 clock round case.
    elif mechanism_parameters['Qinit'] == 20:
        if domain in ['GSVM', 'LSVM', 'SRVM']:
            mechanism_parameters['cca_increment'] = 0.15  
        else: 
            mechanism_parameters['cca_increment'] = 0.22  
    elif mechanism_parameters['Qinit'] == 50:
        mechanism_parameters['cca_increment'] = 0.08
    elif mechanism_parameters['Qinit'] == 70:
        mechanism_parameters['cca_increment'] = 0.0565
    elif mechanism_parameters['Qinit'] == 60:
        mechanism_parameters['cca_increment'] = 0.0665
    else: 
        mechanism_parameters['cca_increment'] = 0.15
    

    # 5. SET MIP PARAMETERS:
    MIP_parameters = {
        'timeLimit': 3600 * 10, # Default +inf
        'MIPGap': 1e-06, # Default 1e-04
        'IntFeasTol': 1e-8, # Default 1e-5
        'FeasibilityTol': 1e-9 # Default 1e-6
    }
    # -------------------

    # Create directory for results
    # --------------------------------------
    if mechanism_parameters['initial_demand_query_method'] == 'random':
        path_addition = f'max_linear_prices_multiplier_{TRAIN_parameters_all_bidders["Bidder_0"]["max_linear_prices_multiplier"]}_'
    elif mechanism_parameters['initial_demand_query_method'] == 'increasing':
        path_addition = f'start_linear_item_prices_{TRAIN_parameters_all_bidders["Bidder_0"]["start_linear_item_prices"][0]}_end_linear_item_prices_{TRAIN_parameters_all_bidders["Bidder_0"]["end_linear_item_prices"][0]}_'
    elif mechanism_parameters['initial_demand_query_method'] == 'cca':
        path_addition = f'cca_initial_prices_multiplier_{mechanism_parameters["cca_initial_prices_multiplier"]}_increment_{mechanism_parameters["cca_increment"]}_'
    res_path = os.path.join(os.getcwd(),
                            'results',
                             f'{domain}_qinit_{mechanism_parameters["Qinit"]}_initial_demand_query_method_{mechanism_parameters["initial_demand_query_method"]}_{path_addition}new_query_option_{mechanism_parameters["new_query_option"]}',
                             f'ML_config_hpo1',  # so that we know with which config and init price method the results were generated 
                             str(seed)
                             )
    

    os.makedirs(res_path, exist_ok=True)
    # --------------------------------------


    # 6. Run mechanism
    mechanism(SATS_parameters = SATS_parameters,
              TRAIN_parameters = TRAIN_parameters_all_bidders,
              MVNN_parameters =MVNN_parameters_all_bidders,
              mechanism_parameters = mechanism_parameters,
              MIP_parameters = MIP_parameters,
              res_path = res_path, 
              wandb_tracking = True,    # NOTE: from this you can turn on/off wandb tracking
              wandb_project_name = f"ML-CCA_Domain_{domain}"   # NOTE: from this you can change the name of the wandb project 
              )
    # -------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="name of the domain to run", default= 'LSVM', type=str)
    parser.add_argument("--init_method", help="method for the Qinit queries, options: random, cca", default= 'cca', type=str)
    parser.add_argument("--qinit", help="number of initial queries", default= 20, type=int)
    parser.add_argument("--seed", help="auction instance seed to run", default= 184, type=int)
    parser.add_argument("--new_query_option", help="new query option", default= 'gd_linear_prices_on_W_v3', type=str)  # options: gd_linear_prices_on_W, gd_linear_prices_on_W_v2, gd_linear_prices_on_W_v3, cca and gd_linear_prices_on_W_v3_cheating


    args = parser.parse_args()
    domain = args.domain
    seed = args.seed
    init_method = args.init_method
    qinit = args.qinit
    new_query_option = args.new_query_option

    print(f'Starting instance on domain: {domain} with init method: {init_method}  qinit: {qinit} and seed: {seed}')
    
    
    main(domain=domain, init_method = init_method, new_query_option = new_query_option, qinit = qinit, seed=seed)


