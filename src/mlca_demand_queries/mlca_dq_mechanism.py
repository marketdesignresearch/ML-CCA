# Libs
import json
import logging
from datetime import datetime
import random
import torch
import numpy as np
from numpyencoder import NumpyEncoder
from pysats import PySats
# make sure you set the classpath before loading any other modules
PySats.getInstance()
import os
# Own Libs
from mlca_demand_queries.mlca_dq_economies import MLCA_DQ_ECONOMIES
from pysats_ext import GenericWrapper
# from pdb import set_trace



# %% MECHANISM
def mechanism(SATS_parameters: dict,
              TRAIN_parameters: dict,
              MVNN_parameters: dict,
              mechanism_parameters: dict,
              MIP_parameters: dict,
              res_path: str, 
              wandb_tracking: bool,
              wandb_project_name: str
              ):

    SATS_seed = SATS_parameters['SATS_seed']
    SATS_domain = SATS_parameters['SATS_domain']
    Qinit =mechanism_parameters['Qinit']
    Qmax = mechanism_parameters['Qmax']
    new_query_option = mechanism_parameters['new_query_option']
    isLegacy = SATS_parameters['isLegacy']
    calc_efficiency_per_iteration = mechanism_parameters['calculate_efficiency_per_iteration']
    initial_demand_query_method = mechanism_parameters['initial_demand_query_method']
    

    # Save config dict
    config_dict = {'SATS_parameters':SATS_parameters,
                    'TRAIN_parameters':TRAIN_parameters,
                    'MVNN_parameters':MVNN_parameters,
                    'mechanism_parameters':mechanism_parameters,
                    'MIP_parameters':MIP_parameters
                    }
    
    json.dump(config_dict,
            open(os.path.join(res_path,'config.json'), 'w'),
            indent=4,
            sort_keys=False,
            separators=(', ', ': '),
            ensure_ascii=False,
            cls=NumpyEncoder)

    start = datetime.now()

    # SEEDING ------------------
    np.random.seed(SATS_seed)
    torch.manual_seed(SATS_seed)
    random.seed(SATS_seed)
    # ---------------------------

    logging.warning('START MLCA:')
    logging.warning('-----------------------------------------------')
    logging.warning(f'Model: {SATS_domain}')
    logging.warning(f'Seed SATS Instance: {SATS_seed}')
    logging.warning(f'Qinit:{Qinit}')
    logging.warning(f'Qmax: {Qmax}')
    logging.warning(f'new_query_option: {new_query_option}')
    logging.warning(f'initial_demand_query_method: {initial_demand_query_method}')
    logging.warning('')

    # Instantiate Economies
    logging.warning('Instantiate SATS Instance')
    if SATS_domain == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_seed,
                                                                 isLegacyLSVM=isLegacy)  # create SATS auction instance
        logging.warning('####### ATTENTION #######')
        logging.warning('isLegacyLSVM: %s', SATS_auction_instance.isLegacy)
        logging.warning('#########################\n')
        GSVM_national_bidder_goods_of_interest = None

    elif SATS_domain == 'GSVM':
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_seed,
                                                                 isLegacyGSVM=isLegacy)  # create SATS auction instance
        logging.warning('####### ATTENTION #######')
        logging.warning('isLegacyGSVM: %s', SATS_auction_instance.isLegacy)
        logging.warning('#########################\n')
        GSVM_national_bidder_goods_of_interest = SATS_auction_instance.get_goods_of_interest(6) # national bidder is bidder 6

    elif SATS_domain == 'MRVM':
        mrvm_non_generic = PySats.getInstance().create_mrvm(seed=SATS_seed)  # create SATS auction instance
        SATS_auction_instance = GenericWrapper(mrvm_non_generic) # wrap non-generic auction instance
        GSVM_national_bidder_goods_of_interest = None


    elif SATS_domain == 'SRVM':
        srvm_non_generic = PySats.getInstance().create_srvm(seed=SATS_seed)  # create SATS auction instance
        SATS_auction_instance = GenericWrapper(srvm_non_generic) # wrap non-generic auction instance
        GSVM_national_bidder_goods_of_interest = None

    else:
        raise ValueError(f'SATS_domain {SATS_domain} not yet implemented')
    
    SATS_parameters['GSVM_national_bidder_goods_of_interest'] = GSVM_national_bidder_goods_of_interest

    # create economy instance
    E = MLCA_DQ_ECONOMIES(SATS_auction_instance = SATS_auction_instance,
                          SATS_parameters = SATS_parameters,
                          TRAIN_parameters = TRAIN_parameters,
                          MVNN_parameters= MVNN_parameters,
                          mechanism_parameters = mechanism_parameters,
                          start_time=start, 
                          wandb_tracking = wandb_tracking,
                          wandb_project_name = wandb_project_name
                          )
    # set NN parameters
    E.set_ML_parameters(parameters=MVNN_parameters) 
    # set MIP parameters
    E.set_MIP_parameters(parameters=MIP_parameters)  

    # SAMPLE INITIAL DEMAND QUERIES
    E.set_initial_dqs(method = initial_demand_query_method)

    # Calculate efficient allocation given current elicited bids
    if calc_efficiency_per_iteration: 
        E.calculate_efficiency_per_iteration()


    # Global while loop: check if for all bidders one addtitional auction round is feasible
    for iteration in range(1, (E.Qmax-E.Qinit) + 1):

        # Increment iteration
        E.mlca_iteration += 1

        # logging info
        E.get_info()

        # Reset attributes
        logging.info('RESET: ML Models')
        E.reset_ML_models()

        # Train ML Models
        E.estimation_step()

        # DQ generation (only in Main Economy), also updates the elicited DQ object
        E.generate_dq()
        
        # check if the DQ found clears the market.
        if E.found_clearing_prices: 
            logging.warning('EARLY STOPPING - CLEARED THE MARKET')
            logging.info('')
            logging.info('CALCULATE FINAL CLEARING ALLOCATION')
            logging.info('---------------------------------------------')

            # we have clearing prices -> we do not need to calculate the final allocation based on inferred bids. 
            E.calculate_clearing_allocation(E.demand_vector_per_iteration[iteration], E.price_vector_per_iteration[iteration], is_final_allocation = True)
            #E.final_allocation_efficiency = E.calculate_efficiency_of_allocation(E.final_allocation, E.final_allocation_scw, verbose=1)

            # Save results per iteration 
            E.calc_time_spent()
            E.save_results(res_path)
            E.extend_per_iteration_results()
            break

        # Calculate efficieny per iteration
        if calc_efficiency_per_iteration:
            E.calculate_efficiency_per_iteration()

        # Save results per iteration
        E.calc_time_spent() # Calculate timings
        E.save_results(res_path)

    # allocation & payments
    if not E.found_clearing_prices:
        E.calculate_final_allocation()
        #E.final_allocation_efficiency = E.calculate_efficiency_of_allocation(E.final_allocation, E.final_allocation_scw, verbose=1)
        E.calculate_vcg_payments()

    # Calculate timings
    E.calc_time_spent()

    # FINAL SAVING OF RESULTS
     # save results here but NO wandb tracking (since this was done already in the iteration forloop)
    E.save_results(res_path, no_wandb_logging=True)

    
    #Save final wandb table
    E.wandb_final_table()

    # Final Info
    E.get_info(final_summary=True)

    return
