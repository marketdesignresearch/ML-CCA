#%%
from pysats import PySats
PySats.getInstance()
import wandb
from datetime import datetime
import json
import re
import numpy as np

# own libs
from mlca_demand_queries.mlca_dq_util import init_demand_queries_mlca_unif, init_demand_queries_mlca_cca, key_to_int
from mvnns_demand_query_training.mvnn_dq_training import dq_train_mvnn
from pysats_ext import GenericWrapper
from pdb import set_trace


# %% SET PARAMETERS

# 0. W&B
# ----------------------------------------------
wandb_tracking =  False
bidder_id = 0
# ----------------------------------------------

# 1. SATS Parameters
# ----------------------------------------------
SATS_parameters = {"SATS_domain": 'MRVM',
                   "isLegacy": False,
                   "SATS_seed": 1,
                    }
# ----------------------------------------------

# 2. Training Parameters
# ----------------------------------------------
TRAIN_parameters = {"number_train_data_points": 40,
                    "number_val_data_points": 10,
                    "data_seed":1,
                    "max_linear_prices_multiplier": 5,  # sample from larger prices
                    "price_file_name": 'values_for_null_price_seeds1-100',
                    'batch_size': 1,
                    'epochs': 20,
                    'l2_reg': 1e-4,
                    'learning_rate': 0.005,
                    'clip_grad_norm': 1,
                    'use_gradient_clipping': False,
                    'print_frequency': 1
                    }
# ----------------------------------------------

# 3. MVNN Parameters
# ----------------------------------------------
MVNN_parameters = {'num_hidden_layers': 1,
                   'num_hidden_units': 20,
                   'layer_type': 'MVNNLayerReLUProjected',
                   'target_max': 1, # TODO: check
                   'lin_skip_connection': False, # TODO: discuss if we ever want True
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
# ----------------------------------------------


# 4. MIP Parameters
MIP_parameters = {
        'timeLimit': 3600 * 10, # Default +inf
        'MIPGap': 1e-06, # Default 1e-04
        'IntFeasTol': 1e-8, # Default 1e-5
        'FeasibilityTol': 1e-9 # Default 1e-6
    }
# ----------------------------------------------


def generate_random_bundles(capacities, number_of_bundles): 
    """
    Generates a list of bundles with random items, respecting the capacities of the items.
    """
    bundles = [] 
    for _ in range(number_of_bundles):
        bundle = np.random.randint(capacities + 1)  # +1 because randint is exclusive
        bundles.append(bundle)
    return np.array(bundles)

def train_mvnn(SATS_parameters,
               TRAIN_parameters,
               MVNN_parameters,
               MIP_parameters,
               wandb_tracking,
               init_method = 'random', # options: 'random', 'cca' 
               MECHANISM_parameters = None, # only needed if init_method = 'cca'
               bidder_id = None  # if not None, only train for this bidder
               ):
    
    # Create SATS World ------
    SATS_domain = SATS_parameters['SATS_domain']
    isLegacy = SATS_parameters['isLegacy']
    SATS_seed = SATS_parameters['SATS_seed']

    print('-----Inside train_mvnn(), wandb_tracking = ', wandb_tracking)

    max_linear_prices = {}
    scales = {}
    if SATS_domain == 'GSVM':
        isLegacy = False
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_seed,
                                                                isLegacyGSVM=isLegacy)
        
        GSVM_national_bidder_goods_of_interest = SATS_auction_instance.get_goods_of_interest(6)
        good_capacities = np.array([1 for _ in range(len(SATS_auction_instance.get_good_ids()))])

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
        good_capacities = np.array([1 for _ in range(len(SATS_auction_instance.get_good_ids()))])
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

    # add those to SATS_parameters
    TRAIN_parameters['max_linear_prices'] = max_linear_prices
    TRAIN_parameters['scale'] = scales[f'Bidder_{bidder_id}']
    SATS_parameters['GSVM_national_bidder_goods_of_interest'] = GSVM_national_bidder_goods_of_interest
    # ------


    # Create Initial DQ Training and Validation Data ------
    if init_method == 'random':
        train_DQdata = init_demand_queries_mlca_unif(SATS_auction_instance,
                                                    number_initial_bids = TRAIN_parameters['number_train_data_points'] + TRAIN_parameters['number_val_data_points'],
                                                    max_linear_prices = {k:TRAIN_parameters['max_linear_prices_multiplier']*v for k,v in max_linear_prices.items()},
                                                    seed = TRAIN_parameters['data_seed'], 
                                                    bidder_id = bidder_id 
                                                    )
    
    elif init_method == 'cca':
        DQdata_train_only = init_demand_queries_mlca_cca(SATS_auction_instance,
                                        capacities = good_capacities, 
                                        number_initial_bids = TRAIN_parameters['number_train_data_points'],
                                        start_linear_item_prices = MECHANISM_parameters['cca_start_linear_item_prices'] * MECHANISM_parameters['cca_initial_prices_multiplier'],
                                        price_increment = MECHANISM_parameters['cca_increment'],
                                        include_null_price = True
                                        )
        DQdata_train_only = DQdata_train_only[f'Bidder_{bidder_id}']  # only keep the data that corresponds to the bidder_id in question 

        final_prices = DQdata_train_only[1][-1]
        
        # the validation set is drawn using random prices, either with the old or new method 
        DQdata_val_only = init_demand_queries_mlca_unif(SATS_auction_instance,
                                                    number_initial_bids = TRAIN_parameters['number_val_data_points'],
                                                    max_linear_prices = {k:TRAIN_parameters['max_linear_prices_multiplier']*v for k,v in max_linear_prices.items()},  #TODO: make this proper
                                                    seed = TRAIN_parameters['data_seed'], 
                                                    bidder_id = bidder_id , 
                                                    price_method= TRAIN_parameters.get('val_price_method', 'old'),
                                                    min_price_per_item= final_prices * TRAIN_parameters['val_points_multipliers'][0],
                                                    max_price_per_item= final_prices * TRAIN_parameters['val_points_multipliers'][1]
                                                    )
        
        #  combine the data so that they have the same format as in the case where the method is random. 
        train_DQdata = {f'Bidder_{bidder_id}': [np.concatenate((DQdata_train_only[0], DQdata_val_only[f'Bidder_{bidder_id}'][0])), np.concatenate((DQdata_train_only[1], DQdata_val_only[f'Bidder_{bidder_id}'][1]))]} 

    
    # ------
    print('Inside train mvnn(), train_DQdata finished')
    # set_trace()

    # 4. W&B tracking
    # ----------------------------------------------
    if wandb_tracking:
        wandb.init(project=f"MVNN_DQTRAIN_{SATS_domain}",
                name = f'Run_{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}',
                config={**SATS_parameters,**TRAIN_parameters,**MVNN_parameters},
                reinit=True
    )
        wandb.define_metric("epochs")
    #else:
        #wandb = None
    # ----------------------------------------------

    # Training ------
    all_models = {}
    all_metrics = {}
    print(f'Starting training on {SATS_domain}')
    for bidder_name in train_DQdata.keys():
        print(f'Training MVNN for {bidder_name}')

        X_train = train_DQdata[bidder_name][0][:TRAIN_parameters['number_train_data_points']]
        P_train = train_DQdata[bidder_name][1][:TRAIN_parameters['number_train_data_points']]
        

        X_val = train_DQdata[bidder_name][0][TRAIN_parameters['number_train_data_points']:]
        P_val = train_DQdata[bidder_name][1][TRAIN_parameters['number_train_data_points']:]

        if MECHANISM_parameters['dynamic_scaling']:
           implied_values = np.array([ np.dot(P_train[i], X_train[i]) for i in range(len(P_train))])
           new_scale = np.max(implied_values) * TRAIN_parameters['scale_multiplier']
           print(f'New scaling factor for {bidder_name} is {new_scale / scales[bidder_name]} times the old one!')
           
           scales[bidder_name] = new_scale
           TRAIN_parameters['scale'] = new_scale

        P_train_scaled = P_train/scales[bidder_name]
        P_val_scaled = P_val/scales[bidder_name]


        # create bundles for the validation set that will only be used for generalization performance measures
        if TRAIN_parameters['number_gen_val_points'] > 0:
            bundles_gen = generate_random_bundles(capacities=good_capacities, number_of_bundles=TRAIN_parameters['number_gen_val_points'])

            X_val_gen_only = bundles_gen
            P_val_gen_only = np.zeros((TRAIN_parameters['number_gen_val_points'], P_val.shape[1]))
        else: 
            X_val_gen_only = None
            P_val_gen_only = None


        mvnn, metrics = dq_train_mvnn(SATS_auction_instance = SATS_auction_instance, 
                                    capacity_generic_goods= good_capacities,
                                    P_train = P_train_scaled,
                                    X_train = X_train,
                                    P_val = P_val_scaled,
                                    X_val = X_val,
                                    P_val_gen_only = P_val_gen_only,
                                    X_val_gen_only = X_val_gen_only,
                                    SATS_parameters = SATS_parameters,
                                    TRAIN_parameters = TRAIN_parameters,
                                    MVNN_parameters = MVNN_parameters,
                                    MIP_parameters = MIP_parameters,
                                    bidder_id = key_to_int(bidder_name),
                                    bidder_scale = scales[bidder_name],
                                    GSVM_national_bidder_goods_of_interest = GSVM_national_bidder_goods_of_interest,
                                    wandb_tracking = wandb_tracking
                                    )

    if wandb_tracking:
        wandb.finish()
    # ------

    all_models[bidder_name] = mvnn
    all_metrics[bidder_name] = metrics

    return all_models, all_metrics

# %%


if __name__ == '__main__':
    all_models, all_metrics = train_mvnn(SATS_parameters=SATS_parameters,
                                         TRAIN_parameters=TRAIN_parameters,
                                         MVNN_parameters=MVNN_parameters,
                                         MIP_parameters=MIP_parameters,
                                         wandb_tracking=wandb_tracking, 
                                         bidder_id= bidder_id)