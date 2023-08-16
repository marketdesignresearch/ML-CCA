import wandb
import numpy as np 
import random
# from pdb import set_trace
from test_training import train_mvnn
import torch 
import argparse 
import matplotlib.pyplot as plt 
import matplotlib
import json


from pdb import set_trace



instances_to_average = 1   

parser = argparse.ArgumentParser()

from jnius import autoclass
CPLEXUtils = autoclass('org.marketdesignresearch.mechlib.utils.CPLEXUtils')
SolveParams = autoclass('edu.harvard.econcs.jopt.solver.SolveParam')
CPLEXUtils.SOLVER.setSolveParam(SolveParams.THREADS,6)



parser.add_argument("--domain", help="domain to use", default= 'MRVM', type=str)
parser.add_argument("--bidder_type", help="bidder_type", default= 'National', type=str)



def bidder_type_to_bidder_id(SATS_domain,
                             bidder_type):
    bidder_id_mappings = {'GSVM': {'National': [6], 'Regional': [0, 1, 2, 3, 4, 5]},
                          'LSVM': {'National': [0], 'Regional': [1, 2, 3, 4, 5]},
                          'SRVM': {'National': [5, 6], 'Regional': [3, 4], 'High Frequency': [2], 'Local': [0, 1]},
                          'MRVM': {'National': [7, 8, 9], 'Regional': [3, 4, 5, 6], 'Local': [0, 1, 2]}
                          }

    bidder_id = np.random.choice(bidder_id_mappings[SATS_domain][bidder_type], size=1, replace=False)[0]
    print(f'BIDDER ID:{bidder_id}')

    return bidder_id


# Define training function that takes in hyperparameter values from `wandb.config` and uses them to train a model and return metric
def main(): 

    # 0. Set random seeds
    torch.manual_seed(0)
    
    arg_config = parser.parse_args()
    domain = arg_config.domain
    bidder_type = arg_config.bidder_type
    hpo_file_name = 'hpo_configs.json'    
    hpo_dict =  json.load(open(hpo_file_name, 'r'))
    epochs = hpo_dict[domain][bidder_type]['epochs']  # TODO: change back to the dictionary
    # epochs = 1

    # 1. SATS Parameters
    # ----------------------------------------------
    SATS_parameters = {"SATS_domain": domain,
                    "isLegacy": False,
                    "SATS_seed": 1,
                        }
    # ----------------------------------------------

    # 2. Training Parameters
    # ----------------------------------------------
    TRAIN_parameters = {"number_train_data_points": 50,
                        "data_seed": 1,
                        "number_val_data_points": 500,   
                        "val_points_multipliers": (0.75, 1.5), # NOTE: this only affects the PV dataset, not the generalization dataset. Only affects the new method. 
                        "val_price_method": 'old', # options: 'old', new 
                        "number_gen_val_points": 50000,
                        "instances_averaged": instances_to_average,
                        "max_linear_prices_multiplier": hpo_dict[domain][bidder_type]['max_linear_prices_multiplier'],  # sample from larger prices, oly affects the 'old' method. 
                        "price_file_name": 'values_for_null_price_seeds1-100',
                        'batch_size': 1,
                        'epochs':  hpo_dict[domain][bidder_type]['epochs'],  # TODO: change back to the dictionary
                        'l2_reg':  hpo_dict[domain][bidder_type]['l2_reg'],
                        'learning_rate':  hpo_dict[domain][bidder_type]['learning_rate'],
                        'clip_grad_norm': 1,
                        'use_gradient_clipping': False,
                        'scale_multiplier':  hpo_dict[domain][bidder_type].get('scale_multiplier', 1),  # only used with dynamic scaling
                        'print_frequency': 1, 
                        }
    # ----------------------------------------------

    # 3. MVNN Parameters
    # ----------------------------------------------
    MVNN_parameters = {'num_hidden_layers': hpo_dict[domain][bidder_type]['num_hidden_layers'],
                    'num_hidden_units': hpo_dict[domain][bidder_type]['num_hidden_units'],
                    'layer_type': 'MVNNLayerReLUProjected',
                    'target_max': 1, # TODO: check
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
    run = wandb.init(project=f'PLOT-HPO-{SATS_parameters["SATS_domain"]}-Biddertype_{bidder_type}_v2.9', 
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


    for epoch in range(epochs):
        avg_metrics = {k: np.mean([metrics_all_runs[j][epoch][k] for j in range(instances_to_average)]) for k in metrics_all_runs[0][epoch].keys()}
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
        
    true_values = metrics_all_runs[-1][epochs -1]['scaled_true_values_generalization']
    predicted_values = metrics_all_runs[-1][epochs - 1]['scaled_predicted_values_generalization']

    true_values_train = metrics_all_runs[-1][epochs -1]['scaled_true_values_train']
    predicted_values_train = metrics_all_runs[-1][epochs - 1]['scaled_predicted_values_train']
    inferred_values_train = metrics_all_runs[-1][epochs - 1]['scaled_inferred_values_train']

    true_values_PV = metrics_all_runs[-1][epochs -1]['scaled_true_values']  
    predicted_values_PV = metrics_all_runs[-1][epochs - 1]['scaled_predicted_values']
    inferred_values_PV = metrics_all_runs[-1][epochs - 1]['scaled_inferred_values']

    
    plt.figure(figsize=(9,9))
    plt.scatter(true_values, predicted_values, s = 20, alpha = 0.3, facecolors='none', edgecolors='r', marker = 'o', label = 'Validation Set 1')
    plt.scatter(true_values_train, predicted_values_train,  s = 30, alpha = 0.8,facecolors='none', edgecolors='b', label = 'Training Set')
    plt.scatter(true_values_train, inferred_values_train, s = 30, alpha = 0.8, marker = 'v',facecolors='none', edgecolors='b', label = 'Inferred Values Training Set')
    plt.scatter(true_values_PV, predicted_values_PV, s = 30, alpha = 0.6,facecolors='none', edgecolors='g', label = 'Validation Set 2')
    plt.scatter(true_values_PV, inferred_values_PV,  s = 30, alpha = 0.6,facecolors='none', edgecolors='g', marker='v', label = 'Inferred Values Validation Set 2')

    # the point of the last CCA iteration, closet to where the prices ended up
    plt.scatter([true_values_train[-1]], [predicted_values_train[-1]], c='k', s = 90, marker = '*', alpha = 1,facecolors='none', edgecolors='k', label = 'Last CCA Iteration')


    legend = plt.legend(loc='upper left', fontsize=15)

    # plt.yscale('log')
    # plt.xscale('log')

    p1 = max(max(predicted_values), max(true_values))
    p2 = min(min(predicted_values), min(true_values))
    p1 = 3
    p2 = 0
    plt.plot([p1, p2], [p1, p2], ls="--", c=".3")
    plt.xlabel('True Values', fontsize=15 + 5)
    plt.ylabel('Predictions', fontsize=15 + 5)
    # plt.title('True vs. Predicted Values', fontsize=15)
    plt.axis('equal')
    plt.grid()

    # set both axes to start at zero 
    # plt.xlim(0, max(max(predicted_values), max(true_values)))
    # plt.ylim(0, max(max(predicted_values), max(true_values)))
    plt.xlim(0, 3)
    plt.ylim(0, 3)

    # Get the handles (line2D instances) and labels from the legend
    handles, labels = legend.legendHandles, legend.get_texts()


    # set the right ticks, ensure that it is the right fonttype 
    plt.xticks([0,1,2,3])
    plt.yticks([0,1,2,3])
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    # Set the size of the symbols in the legend
    for handle in handles:
        handle.set_sizes([100])

    # Change tick sizes
    # Get the current axes
    ax = plt.gca()

    # Increase the tick sizes
    ax.tick_params(axis='both', which='both', labelsize=16, width=2, length=6)


    plt.show()
    plt.savefig(f'./learning_figures/{SATS_parameters["SATS_domain"]}_bidder_{bidder_type}_true_vs_predicted_plot_v2_test.pdf')

    wandb.log({"true_vs_predicted_plot" : wandb.Image(plt)})


        
    run.finish()


main()

# # Start sweep agent with the function defined above
# wandb.agent(sweep_id, function=main, count=2)