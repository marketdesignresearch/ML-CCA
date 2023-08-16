
# %% Libs
from pysats import PySats
import wandb
from datetime import datetime
import os
from pdb import set_trace
import json 
import re
import numpy as np

# %% own Libs
from mlca_demand_queries.mlca_dq_util import init_demand_queries_mlca_unif, key_to_int, init_demand_queries_mlca_increasing
from mvnns_demand_query_training.mvnn_dq_training import dq_train_mvnn
from demand_query_generation import minimize_W


# %% Create SATS World ------
SATS_domain = 'GSVM'
isLegacy = False
SATS_seed = 1
max_linear_prices = {}
scales = {}

if SATS_domain == 'GSVM':
    isLegacy = False
    SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_seed,
                                                             isLegacyGSVM=isLegacy)
    
    GSVM_national_bidder_goods_of_interest = SATS_auction_instance.get_goods_of_interest(6)

    V =  json.load(open('GSVM_values_for_null_price_seeds1-100.json', 'r')) # AVG value per item  for the null price from GSVM_values_for_null_price.csv
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
    V =  json.load(open('LSVM_values_for_null_price_seeds1-100.json', 'r')) # AVG value per item  for the null price from LSVM_values_for_null_price.csv
    for key in V:
        if 'max_value_per_item' in key:
            id = int(re.findall(r'\d+', key)[0])
            max_linear_prices[f'Bidder_{id}'] = V[key]['mean']
        if 'max_value' in key and 'per_item' not in key:
            id = int(re.findall(r'\d+', key)[0])
            scales[f'Bidder_{id}'] = V[key]['mean']

elif SATS_domain == 'SRVM': # TODO: DQs not yet implemented
    SATS_auction_instance = PySats.getInstance().create_srvm(seed=SATS_seed)
    GSVM_national_bidder_goods_of_interest = None

elif SATS_domain == 'MRVM': # TODO: DQs not yet implemented
    SATS_auction_instance = PySats.getInstance().create_mrvm(seed=SATS_seed)
    GSVM_national_bidder_goods_of_interest = None
else:
    pass
# ------


initial_demand_query_method = 'increasing'
number_initial_bids = 20
    
if initial_demand_query_method == 'random':
    # Create Initial RANDOM DQ Training Data ------
    init_data_seed = 1
    max_linear_prices_multiplier = 1
    DQdata = init_demand_queries_mlca_unif(SATS_auction_instance,
                                        number_initial_bids = number_initial_bids,
                                        max_linear_prices = {k:max_linear_prices_multiplier*v for k,v in max_linear_prices.items()}, # TODO: check max_linear_prices setting
                                        seed = init_data_seed,
                                        include_null_price=True,
                                        bidder_id = None )
elif initial_demand_query_method == 'increasing':
    # Create Initial INCREASING DQ Training Data ------
    m = len(SATS_auction_instance.get_good_ids())
    start_linear_item_prices = np.zeros(m),
    end_linear_item_prices= np.ones(m)*50
    DQdata2 = init_demand_queries_mlca_increasing(SATS_auction_instance = SATS_auction_instance,
                                                number_initial_bids = number_initial_bids,
                                                start_linear_item_prices = start_linear_item_prices,
                                                end_linear_item_prices= end_linear_item_prices,
                                                bidder_id = None 
    )
else:
    raise NotImplementedError(f'initial_demand_query_method:{initial_demand_query_method} not implemented')


#%% Training ------
models = [] # list of trained mvnn models
scale_list = [] # list of scales for each bidder
for bidder_name in DQdata.keys():
    print(f'Training MVNN for {bidder_name}')

    X = DQdata[bidder_name][0]
    P = DQdata[bidder_name][1]
    P_scaled = P/scales[bidder_name]
    scale_list.append(scales[bidder_name])

    wandb.init(
    # set the wandb project where this run will be logged
    project=f"MVNN_TRAINING_ON_DQ_{SATS_domain}",
    name = f'Run_{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}_bidder_{bidder_name}',
    # track hyperparameters and run metadata
    config={
    "SATS_domain": SATS_domain,
    "isLegacy": isLegacy,
    "SATS_seed": SATS_seed,
    "number_initial_bids": number_initial_bids,
    "max_linear_prices_multiplier":max_linear_prices_multiplier,
    "max_linear_prices": max_linear_prices,
    "init_data_seed":init_data_seed,
    },
    reinit=True
)

    mvnn, scale, loss = dq_train_mvnn(price_vectors = P_scaled,
                                      demand_responses = X,
                                      SATS_domain = SATS_domain,
                                      bidder_id = key_to_int(bidder_name),
                                      GSVM_national_bidder_goods_of_interest = GSVM_national_bidder_goods_of_interest,
                                      batch_size = 1,
                                      epochs = 20,
                                      l2_reg = 1e-4,
                                      learning_rate = 0.005,
                                      clip_grad_norm = 1,
                                      use_gradient_clipping = False,
                                      print_frequency = 1,
                                      wandb = wandb
                                    )
    mvnn.transform_weights()  # transform weights to ensure they are legal MVNNs
    models.append((key_to_int(bidder_name), mvnn))
# ------


initial_price_vector = P[0]


price_vector_final = minimize_W(models, initial_price_vector, scale_list, SATS_domain, GSVM_national_bidder_goods_of_interest)