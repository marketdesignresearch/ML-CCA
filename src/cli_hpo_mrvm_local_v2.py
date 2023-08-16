import wandb
import numpy as np 
import random
# from pdb import set_trace
from test_training import train_mvnn
import torch 
import argparse 
import matplotlib.pyplot as plt 


# from pdb import set_trace



bidder_type = 'local'
instances_to_average = 5   # TODO: change back to 5

parser = argparse.ArgumentParser()

from jnius import autoclass
CPLEXUtils = autoclass('org.marketdesignresearch.mechlib.utils.CPLEXUtils')
SolveParams = autoclass('edu.harvard.econcs.jopt.solver.SolveParam')
CPLEXUtils.SOLVER.setSolveParam(SolveParams.THREADS,6)


#train parameter arguments
parser.add_argument("--l2_reg", help="l2 regularization", type=float, default= 1e-4)
parser.add_argument("--learning_rate", help="learning rate", default= 0.005, type=float)
parser.add_argument("--epochs", help="number of epochs", default= 3, type=int)   
parser.add_argument("--batch_size", help="batch size", type=int, default = 1)

parser.add_argument("--max_linear_prices_multiplier", help="max linear prices multiplier", default= 2 * 3, type=float)  # NOTE: This used to be 2! 
parser.add_argument("--use_gradient_clipping", help="use gradient clipping", default= 'false', type=str)
parser.add_argument("--price_file_name", help="name of the price file, without domain", default= 'values_for_null_price_seeds1-100', type=str)

# MVNN parameter arguments
parser.add_argument("--num_hidden_layers", help="number of hidden layers", default= 1, type=int)
parser.add_argument("--num_hidden_units", help="number of hidden units", default= 20, type=int)
parser.add_argument("--dropout_prob", help="dropout probability", default= 0, type=float)
parser.add_argument("--lin_skip_connection", help="initialization method", default= 'false', type=str)

parser.add_argument("--scale_multiplier", help="scale multiplier, used only with dynamic scaling", default= 1, type=float)



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


# Define training function that takes in hyperparameter values from `wandb.config` and uses them to train a model and return metric
def main(): 

    # 0. Set random seeds
    torch.manual_seed(0)
    
    config = parser.parse_args()

    # 1. SATS Parameters
    # ----------------------------------------------
    SATS_parameters = {"SATS_domain": 'MRVM',
                    "isLegacy": False,
                    "SATS_seed": 1,
                        }
    # ----------------------------------------------

    # 2. Training Parameters
    # ----------------------------------------------
    TRAIN_parameters = {"number_train_data_points": 50,
                        "data_seed": 1,
                        "number_val_data_points": 200,   # NOTE: THIS WAS 500 FOR THE OTHER DOMAINS 
                        "val_points_multipliers": (0.25, 1.25), # NOTE: this only affects the PV dataset, not the generalization dataset
                        "number_gen_val_points": 50000,
                        "instances_averaged": instances_to_average,
                        "max_linear_prices_multiplier": config.max_linear_prices_multiplier,  # sample from larger prices
                        "price_file_name": config.price_file_name,
                        'batch_size': config.batch_size,
                        'epochs': config.epochs,
                        'l2_reg': config.l2_reg,
                        'learning_rate': config.learning_rate,
                        'clip_grad_norm': 1,
                        'use_gradient_clipping': config.use_gradient_clipping,
                        'scale_multiplier': config.scale_multiplier,  # only used with dynamic scaling
                        'print_frequency': 1, 
                        }
    # ----------------------------------------------

    # 3. MVNN Parameters
    # ----------------------------------------------
    MVNN_parameters = {'num_hidden_layers': config.num_hidden_layers,
                    'num_hidden_units': config.num_hidden_units,
                    'layer_type': 'MVNNLayerReLUProjected',
                    'target_max': 1, # TODO: check
                    'lin_skip_connection': config.lin_skip_connection, 
                    'dropout_prob': config.dropout_prob,
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
    # ----------------------------------------------
    MIP_parameters = {
        'timeLimit': 3600 * 10, # Default +inf
        'MIPGap': 1e-06, # Default 1e-04
        'IntFeasTol': 1e-8, # Default 1e-5
        'FeasibilityTol': 1e-9 # Default 1e-6
    }  
    # ----------------------------------------------

    # 4.5 Mechanism Parameters (only needed for CCA)
    # ----------------------------------------------
    mechanism_parameters = { 
                            "cca_start_linear_item_prices": np.load(f'{SATS_parameters["SATS_domain"]}_average_item_values_seeds_201-1200.npy'), # NOTE: only used for "initial_demand_query_method=cca"
                            "cca_initial_prices_multiplier": 0.2, # NOTE: only used for "initial_demand_query_method=cca", will multiply the initial prices. 
                            "cca_increment" : 0.075, 
                            "dynamic_scaling": True, 
                            }
    # ---------------------------------------------- 
    
    # 5. Run
    # ----------------------------------------------
    run = wandb.init(project=f'MRVM-HPO-Biddertype_{bidder_type}_v2.8', 
                    config={**SATS_parameters,**TRAIN_parameters,**MVNN_parameters},
                    reinit=True)

    

    # note: setting WANDB tracking to False in order to use an external tracker 
    # for hyperparameter optimization
    metrics_all_runs = [] 
    for i in range(instances_to_average):
        print(f'Starting run {i}')

        # set random seed
        seed_offset = i
        SATS_parameters['SATS_seed'] = SATS_parameters['SATS_seed'] + seed_offset
        TRAIN_parameters['data_seed'] = TRAIN_parameters['data_seed'] + seed_offset

        bidder_id = bidder_type_to_bidder_id(SATS_domain= SATS_parameters['SATS_domain'], bidder_type=bidder_type)
        all_models, all_metrics = train_mvnn(SATS_parameters=SATS_parameters,
                                            TRAIN_parameters=TRAIN_parameters,
                                            MVNN_parameters=MVNN_parameters,
                                            MIP_parameters=MIP_parameters,
                                            wandb_tracking=False, 
                                            init_method= 'cca', 
                                            MECHANISM_parameters= mechanism_parameters,
                                            bidder_id=bidder_id)  
    
    # 5. Log metrics in the new W&B logger

    # set_trace()

        metrics = all_metrics[f'Bidder_{bidder_id}']
        metrics_all_runs.append(metrics)


    for epoch in range(config.epochs):
        avg_metrics = {k: np.mean([metrics_all_runs[j][epoch][k] for j in range(instances_to_average)]) for k in metrics_all_runs[0][epoch].keys()}
        # set_trace()
        wandb.log({f"train_loss_dq_scaled": avg_metrics["train_scaled_dq_loss"], 
                    f"val_loss_dq_scaled": avg_metrics["scaled_dq_loss"], 
                    f"mean_regret": avg_metrics["mean_regret"], 
                    f"mean_regret_scaled": avg_metrics["mean_regret_scaled"],
                    f"val_r2_PV": avg_metrics["r2"], 
                    f"val_r2c_PV": avg_metrics["r2_centered"],
                    f"val_KT_PV": avg_metrics["kendall_tau"], 
                    f"val_MAE_PV": avg_metrics["mae"],
                    f"val_r2_G": avg_metrics["r2_generalization"], 
                    f"val_r2c_G": avg_metrics["r2_centered_generalization"],
                    f"val_KT_G": avg_metrics["kendall_tau_generalization"], 
                    f"val_MAE_G": avg_metrics["mae_generalization"],
                    f"train_KT": avg_metrics["kendall_tau_train"],
                    f"train_MAE": avg_metrics["mae_train"],
                    f"train_r2c": avg_metrics["r2_centered_train"],
                    f"mae_scaled": avg_metrics["mae_scaled"],
                    f"mae_scaled_generalization": avg_metrics["mae_scaled_generalization"],
                    f"mae_scaled_train": avg_metrics["mae_scaled_train"],
                    "epochs": epoch})
        
    true_values = metrics_all_runs[-1][config.epochs -1]['scaled_true_values_generalization']
    predicted_values = metrics_all_runs[-1][config.epochs - 1]['scaled_predicted_values_generalization']

    true_values_train = metrics_all_runs[-1][config.epochs -1]['scaled_true_values_train']
    predicted_values_train = metrics_all_runs[-1][config.epochs - 1]['scaled_predicted_values_train']
    inferred_values_train = metrics_all_runs[-1][config.epochs - 1]['scaled_inferred_values_train']

    true_values_PV = metrics_all_runs[-1][config.epochs -1]['scaled_true_values']  
    predicted_values_PV = metrics_all_runs[-1][config.epochs - 1]['scaled_predicted_values']
    inferred_values_PV = metrics_all_runs[-1][config.epochs - 1]['scaled_inferred_values']

    
    plt.figure(figsize=(9,9))
    plt.scatter(true_values, predicted_values, c='crimson', s = 0.4, alpha = 0.4,facecolors='none', edgecolors='r', label = 'Generalization Dataset')
    plt.scatter(true_values_train, predicted_values_train, c='blue', s = 7, alpha = 0.8,facecolors='none', edgecolors='b', label = 'Training Dataset')
    plt.scatter(true_values_PV, predicted_values_PV, c='green', s = 7, alpha = 0.8,facecolors='none', edgecolors='g', label = 'PV Dataset')
    plt.scatter(true_values_train, inferred_values_train, c='pink', s = 7, alpha = 0.8,facecolors='none', edgecolors='k', label = 'Inferred Values Train Set')
    plt.scatter(true_values_PV, inferred_values_PV, c='orange', s = 7, alpha = 0.8,facecolors='none', edgecolors='k', label = 'Inferred Values PV Set')

    # the point of the last CCA iteration, closet to where the prices ended up
    plt.scatter([true_values_train[-1]], [predicted_values_train[-1]], c='pink', s = 30, alpha = 0.8,facecolors='none', edgecolors='k', label = 'Last CCA Iteration')


    plt.legend(loc='upper left', fontsize=15)

    # plt.yscale('log')
    # plt.xscale('log')

    p1 = max(max(predicted_values), max(true_values))
    p2 = min(min(predicted_values), min(true_values))
    plt.plot([p1, p2], [p1, p2], ls="--", c=".3")
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title('True vs. Predicted Values', fontsize=15)
    plt.axis('equal')

    wandb.log({"true_vs_predicted_plot" : wandb.Image(plt)})


        
    run.finish()


main()

# # Start sweep agent with the function defined above
# wandb.agent(sweep_id, function=main, count=2)