# Libs
import copy
import json
import logging
import multiprocessing
from functools import partial
from joblib import Parallel, delayed
import os
import time
from collections import OrderedDict, defaultdict
from datetime import timedelta, datetime
import numpy as np
from numpyencoder import NumpyEncoder
import wandb

# Own Libs
from mlca_demand_queries.mlca_dq_util import key_to_int, format_solution_mip_new, init_demand_queries_mlca_unif, timediff_d_h_m_s, init_demand_queries_mlca_increasing, init_demand_queries_mlca_cca
from mlca_demand_queries.mlca_dq_wdp import MLCA_DQ_WDP
from mlca_demand_queries.mlca_dq_wdp_generic import MLCA_DQ_WDP_GENERIC
from mvnns_demand_query_training.mvnn_dq_training import dq_train_mvnn, dq_train_mvnn_parallel
from demand_query_generation import minimize_W, minimize_W_v2, minimize_W_v3, minimize_W_cheating


# from pdb import set_trace

class MLCA_DQ_ECONOMIES:

    def __init__(self,
                SATS_auction_instance,
                SATS_parameters,
                TRAIN_parameters,
                MVNN_parameters,
                mechanism_parameters,
                start_time, 
                wandb_tracking,
                wandb_project_name):

        # STATIC ATTRIBUTES
        self.SATS_auction_instance = SATS_auction_instance  # auction instance from SATS: LSVM, GSVM or MRVM generated via PySats.py.
        self.SATS_auction_instance_allocation = None  # true efficient allocation of auction instance
        self.SATS_auction_instance_scw = None  # SATS_auction_instance.get_efficient_allocation()[1]  # social welfare of true efficient allocation of auction instance
        self.SATS_auction_instance_seed = SATS_parameters['SATS_seed']  # auction instance seed from SATS
        self.bidder_ids = list(SATS_auction_instance.get_bidder_ids())  # bidder ids in this auction instance.
        self.bidder_names = list('Bidder_{}'.format(bidder_id) for bidder_id in self.bidder_ids)
        self.N = len(self.bidder_ids)  # number of bidders
        self.good_ids = set(SATS_auction_instance.get_good_ids())  # good ids in this auction instance
        self.M = len(self.good_ids)  # number of items
        self.Qinit = mechanism_parameters['Qinit']  # number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure, different per bidder)
        self.Qmax = mechanism_parameters['Qmax']  # maximal number of possible value queries in the preference elicitation algorithm (PEA) per bidder
        self.final_allocation = None  # mlca allocation
        self.final_allocation_scw = None  # true social welfare of mlca allocation
        self.final_allocation_efficiency = None  # efficiency of mlca allocation
        self.MIP_parameters = None  # MIP parameters
        self.mlca_iteration = 0  # mlca iteration tracker
        self.revenue = 0  # sum of payments
        self.relative_revenue = None  # relative revenue cp to SATS_auction_instance_scw
        self.found_clearing_prices = False # boolean: True if clearing prices have been found
        self.wandb_tracking = wandb_tracking  # wandb tracking

        # -- Attributes mostly used for environments supporting generic goods
        if SATS_parameters['SATS_domain'] in ['LSVM', 'GSVM']:
            self.generic_domain = False
            self.good_capacities = np.array([1 for _ in range(self.M)])
        elif SATS_parameters['SATS_domain'] in ['MRVM', 'SRVM']:
            self.generic_domain = True
            capacities_dictionary = SATS_auction_instance.get_capacities()
            self.good_capacities = np.array([capacities_dictionary[good_id] for good_id in self.good_ids])  #self.good_capacities[j]: Capacity of good j 
        
        # Time statistics
        self.number_of_mips_solved = {'Price Generation': 0 ,'WDP': 0}
        self.total_time_elapsed_distr = {'WDP MIP': 0, 'ML TRAIN': 0, 'Price Vector Generation': 0,'OTHER': 0}
        self.total_time_price_vector_generation = 0
        self.total_time_wdp_mip = 0
        self.total_time_ml_train = 0
        self.total_time_price_generation = 0
        
        # Statistics about clearing error per iteration and the corresponding price vectors
        self.clearing_error_per_iteration = OrderedDict()
        self.predicted_clearing_error_per_iteration = OrderedDict()
        self.price_vector_per_iteration = OrderedDict()
        self.identical_price_vector_per_iteration = OrderedDict()  # stores if the price vector decided on was already encountered in a previous iteration
        self.found_clearing_prices_per_iteration = OrderedDict()  # stores if the price vector decided on was already encountered in a previous iteration
        self.demand_vector_per_iteration = OrderedDict()
        self.perturbed_prices = False 
        self.feasible_allocation = False

        self.new_query_option = mechanism_parameters['new_query_option']
        self.parallelize_training = mechanism_parameters['parallelize_training']

        self.scales = {bidder_name: TRAIN_parameters[bidder_name]['scale'] for bidder_name in TRAIN_parameters.keys()}
        self.max_linear_prices = {bidder_name: TRAIN_parameters[bidder_name]['max_linear_price'] for bidder_name in TRAIN_parameters.keys()}
        self.dynamic_scaling = mechanism_parameters['dynamic_scaling']

        self.start_time = start_time
        self.end_time = None

        self.main_economy = self.bidder_ids

        self.SATS_parameters = SATS_parameters
        self.TRAIN_parameters = TRAIN_parameters
        self.MVNN_parameters = MVNN_parameters
        self.mechanism_parameters = mechanism_parameters
        
        self.marginal_economies_allocations = {}
        self.marginal_economies_scw = {}
        self.marginal_economies_inferred_scw = {}
        self.scw_per_iteration = OrderedDict()  # storage for scw per auction round
        self.efficiency_per_iteration = OrderedDict()  # storage for efficiency stat per auction round
        self.allocation_per_iteration = OrderedDict()  # storage for allocation per auction round
        self.inferred_scw_per_iteration = OrderedDict()  # storage for inferred scw per auction round
        self.MIP_relative_gap_per_iteration = OrderedDict()  # storage for relative gap per auction round
        self.MIP_time_per_iteration = OrderedDict()  # storage for MIP time per auction round
        self.MIP_unsatisfied_constraints_per_iteration = OrderedDict()  # storage for number of unsatisfied constraints per auction round

        self.unique_bundles_bid_on_per_iteration = OrderedDict()  # storage for unique bundles per auction round [auction_round][bidder_name] -> number of bundles
        
        # storage for the extra metrics for the raised bids
        self.efficiency_per_iteration_raised_bids = OrderedDict()  # storage for efficiency raised per auction round
        self.scw_per_iteration_raised_bids = OrderedDict()  # storage for scw raised per auction round
        self.allocation_per_iteration_raised_bids = OrderedDict()  # storage for allocation raised per auction round
        self.MIP_relative_gap_per_iteration_raised_bids = OrderedDict()  # storage for relative gap for raised clock bids per auction round
        self.MIP_time_per_iteration_raised_bids = OrderedDict()  # storage for MIP time for raised clock bids per auction round
        self.MIP_unsatisfied_constraints_per_iteration_raised_bids = OrderedDict()  # storage for number of unsatisfied constraints for raised clock bids per auction round
        self.final_allocation_raised_bids = None  
        self.final_allocation_scw_raised_bids = None
        self.final_allocation_efficiency_raised_bids = None
        self.marginal_economies_allocations_raised_bids = {}
        self.marginal_economies_scw_raised_bids = {}
        self.marginal_economies_inferred_scw_raised_bids = {}

        # storage for the extra metrics for profit max  # NOTE: Those are all dictionaries with key being number of bids and value the underlying metric, in its original format. 
        self.efficiency_per_iteration_profit_max = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}  # storage for efficiency raised per auction round
        self.scw_per_iteration_profit_max = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}  # storage for scw raised per auction round
        self.allocation_per_iteration_profit_max = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}  # storage for allocation raised per auction round
        self.final_allocation_profit_max = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}   
        self.final_allocation_scw_profit_max = {}
        self.final_allocation_efficiency_profit_max = {}
        self.marginal_economies_allocations_profit_max = {}
        self.marginal_economies_scw_profit_max = {}
        self.marginal_economies_inferred_scw_profit_max = {}
        self.marginal_economies_allocations_profit_max = {}
        self.marginal_economies_scw_profit_max = {}
        self.marginal_economies_inferred_scw_profit_max = {}
        self.profit_max_bids_combined = {}        
        self.revenue_profit_max = {}
        self.relative_revenue_profit_max = {}

        # same as for profit max, but without first raising the clock bids
        self.efficiency_per_iteration_profit_max_unraised = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}  # storage for efficiency raised per auction round
        self.scw_per_iteration_profit_max_unraised = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}  # storage for scw raised per auction round
        self.allocation_per_iteration_profit_max_unraised = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}  # storage for allocation raised per auction round
        self.final_allocation_profit_max_unraised = {j: OrderedDict() for j in self.mechanism_parameters['profit_max_grid']}   
        self.final_allocation_scw_profit_max_unraised = {}
        self.final_allocation_efficiency_profit_max_unraised = {}
        self.marginal_economies_allocations_profit_max_unraised = {}
        self.marginal_economies_scw_profit_max_unraised = {}
        self.marginal_economies_inferred_scw_profit_max_unraised = {}
        self.marginal_economies_allocations_profit_max_unraised = {}
        self.marginal_economies_scw_profit_max_unraised = {}
        self.marginal_economies_inferred_scw_profit_max_unraised = {}
        self.profit_max_bids_combined_unraised = {}        
        self.revenue_profit_max_unraised = {}
        self.relative_revenue_profit_max_unraised = {}

        # storage for the extra metrics for both profit maxes for specific rounds
        self.efficiency_per_iteration_profit_max_specific_round = OrderedDict()
        self.efficiency_per_iteration_profit_max_unraised_specific_round = OrderedDict()
        self.MIP_relative_gap_per_iteration_profit_max_specific_round = OrderedDict()
        self.MIP_time_per_iteration_profit_max_specific_round = OrderedDict()
        self.MIP_unsatisfied_constraints_per_iteration_profit_max_specific_round = OrderedDict()
        self.scw_per_iteration_profit_max_specific_round = OrderedDict()
        self.scw_per_iteration_profit_max_unraised_specific_round = OrderedDict()
        self.profit_max_bids_specific_round = {}
        self.profit_max_bids_unraised_specific_round = {}  # does not need to be a dictionary, since only the last entries will ever be used. 

        self.marginal_economies_allocations_profit_max_specific_round = {}  #NOTE: Do not correspond to a specific round, the name just comes from the fact 
        self.marginal_economies_scw_profit_max_specific_round = {}          # that those were calculated from the specific round profit max method. 
        self.marginal_economies_inferred_scw_profit_max_specific_round = {}
        self.marginal_economies_allocations_profit_max_unraised_specific_round = {}
        self.marginal_economies_scw_profit_max_unraised_specific_round = {}
        self.marginal_economies_inferred_scw_profit_max_unraised_specific_round = {}

        self.nn_seed = self.SATS_auction_instance_seed * 10 ** 4  # seed for training the MVNNs, shifted by 10**6 (i.e. one has 10**6 fits capacity until its overlapping with self.nn_seed+1)
        self.mip_logs = defaultdict(list)    # contains the logs of the WDP MIPs. 
        self.train_logs = {}
        self.total_time_elapsed = None
        self.fitted_scaler = None
        

        # DYNAMIC PER BIDDER
        self.vcg_payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # VCG-style payments, calculated at the end
        self.vcg_payments_raised_bids = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # VCG-style payments, calculated at the end
        self.vcg_payments_profit_max = {j: OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                                self.bidder_ids)) for j in self.mechanism_parameters['profit_max_grid']}
        self.vcg_payments_profit_max_unraised = {j: OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                                self.bidder_ids)) for j in self.mechanism_parameters['profit_max_grid']}
        self.clearing_payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # clearing-style payments, calculated at the end
        self.clearing_payments_raised_bids = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))
        self.clearing_payments_profit_max = {j: OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids)) for j in self.mechanism_parameters['profit_max_grid']}
        self.clearing_payments_profit_max_unraised = {j: OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids)) for j in self.mechanism_parameters['profit_max_grid']}
        self.clearing_payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))
        self.elicited_dqs = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # R=(R_1,...,R_n) elicited bids per bidder
        self.ML_parameters = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # DNNs parameters as in the Class NN described.
        


        self.ML_models = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))

        if self.wandb_tracking: 
            merged_train_mvnn_dict = {bidder_name : {**TRAIN_parameters[bidder_name], **MVNN_parameters[bidder_name]} for bidder_name in TRAIN_parameters.keys()}
            wandb.init(project=wandb_project_name,
                        name = f'Seed_{self.SATS_auction_instance_seed}_Run_{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}',
                        config={**SATS_parameters,**merged_train_mvnn_dict, **mechanism_parameters},
                        reinit=True)
            
            wandb.define_metric("iterations")
            wandb.define_metric("Number of Elicited Bids") 
            wandb.define_metric("Clock Round")
            wandb.define_metric('Efficiency Clock Bids', step_metric="iterations")
            wandb.define_metric('Efficiency Clock Bids Per Clock Round', step_metric="Clock Round")
            wandb.define_metric('Clearing Error', step_metric="iterations")
            wandb.define_metric('Predicted Clearing Error', step_metric="iterations")
            wandb.define_metric('Identical Price Vector', step_metric="iterations")
            wandb.define_metric('Price Vector Sum', step_metric="iterations")
            # wandb.define_metric(f'Price Vector', step_metric="iterations")
            wandb.define_metric("Inferred SCW", step_metric="iterations")
            wandb.define_metric("SCW", step_metric="iterations")
            wandb.define_metric("Found Clearing Prices", step_metric="iterations")
            wandb.define_metric("WDP Relative Gap Clock Bids", step_metric="iterations")
            wandb.define_metric("WDP Time Clock Bids", step_metric="iterations")
            wandb.define_metric("WDP Unsatisfied Constraints Clock Bids", step_metric="iterations")
            wandb.define_metric('Perturbed Prices', step_metric="iterations")
            wandb.define_metric('Feasible Allocation', step_metric="iterations")
            
            for item in self.good_ids: 
                wandb.define_metric(f'Price Good {item}', step_metric="iterations")
                wandb.define_metric(f'Extended Prices Good {item}', step_metric="Number of Elicited Bids") 

            for bidder_id in self.bidder_ids:
                wandb.define_metric(f'Bidder {bidder_id} unique bundles bid on', step_metric="iterations")
                wandb.define_metric(f'Winning round bid for bidder {bidder_id}', step_metric="Clock Round")

            
            if self.mechanism_parameters['calculate_raised_bids']:
                wandb.define_metric('Efficiency Raised Clock Bids', step_metric="iterations")
                wandb.define_metric('Efficiency Raised Clock Bids Per Clock Round', step_metric="Clock Round")
                wandb.define_metric("SCW Raised Bids", step_metric="iterations")
                wandb.define_metric("WDP Relative Gap Raised Clock Bids", step_metric="iterations")
                wandb.define_metric("WDP Time Raised Clock Bids", step_metric="iterations")
                wandb.define_metric("WDP Unsatisfied Constraints Raised Clock Bids", step_metric="iterations")

            if self.mechanism_parameters['calculate_profit_max_bids']:
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    wandb.define_metric(f'Efficiency Profit Max {number_of_profit_max_bids} Bids', step_metric="iterations")
                    wandb.define_metric(f'SCW Profit Max {number_of_profit_max_bids} Bids', step_metric="iterations")
                
            if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    wandb.define_metric(f'Efficiency Profit Max {number_of_profit_max_bids} Bids (unraised clock bids)', step_metric="iterations")
                    wandb.define_metric(f'SCW Profit Max {number_of_profit_max_bids} Bids (unraised clock bids)', step_metric="iterations")

            for profit_max_rounds in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']:
                    wandb.define_metric(f'Efficiency Profit Max ({profit_max_rounds} clock rounds)', step_metric="profit max bids")
                    wandb.define_metric(f'SCW Profit Max ({profit_max_rounds} clock rounds)', step_metric="profit max bids")
                    wandb.define_metric(f'Efficiency Profit Max ({profit_max_rounds} clock rounds) (unraised clock bids)', step_metric="profit max bids")
                    wandb.define_metric(f'SCW Profit Max ({profit_max_rounds} clock rounds) (unraised clock bids)', step_metric="profit max bids")
                    wandb.define_metric(f'WDP Relative Gap Profit Max ({profit_max_rounds} clock rounds)', step_metric="profit max bids")
                    wandb.define_metric(f'WDP Time Profit Max ({profit_max_rounds} clock rounds)', step_metric="profit max bids")
                    wandb.define_metric(f'WDP Unsatisfied Constraints Profit Max ({profit_max_rounds} clock rounds)', step_metric="profit max bids")
                
    def log_Qinit_efficiency(self):
        logging.info('log Qinit efficiency')
        for clock_round in range(1, self.Qinit + 1): 
            print(f'Current clock round: {clock_round}')
            # set_trace()
            inferred_bids = self.calculate_inferred_bids_per_round(clock_round)

            allocation, true_value, inferred_value, details = self.solve_WDP(inferred_bids, self.MIP_parameters, verbose=1)  # find the efficient allocation with respect to the inferred bids
            self.number_of_mips_solved['WDP'] += 1
            efficiency = self.calculate_efficiency_of_allocation(allocation=allocation, allocation_scw=true_value) 

            # create a dictionary with the round in which each agent bid on the bundle they won 
            winning_round_bids = {}
            for bidder_id in self.bidder_ids:
                bundle_won = allocation[f'Bidder_{bidder_id}']['allocated_bundle']
                bundle_won = np.array(bundle_won)
                bundles_bid_on = self.elicited_dqs[f'Bidder_{bidder_id}'][0][:clock_round] 
                # get the last round where the bundle won was bid on
                round_matches = (bundle_won == bundles_bid_on).sum(axis = 1) == bundle_won.shape[0]
                if np.sum(round_matches) > 0:
                    last_round = np.max(np.where(round_matches)[0]) + 1 # because indexes start from 0, but clock rounds/iterations start from 1
                else:
                    last_round = 0 # if the bundle was never bid on (i.e., the agent got the empty bundle) set the last round to 0
                winning_round_bids[f'Winning round bid for bidder {bidder_id}'] = last_round

            wandb_dict = {
                "Efficiency Clock Bids Per Clock Round": efficiency,
                "Clock Round": clock_round
            }
            wandb_dict.update(winning_round_bids)

            if self.mechanism_parameters['calculate_raised_bids']:
                raised_bids = self.raise_inferred_bids_per_round(clock_round)
                
                allocation_raised, true_value_raised, inferred_value_raised, details_raised = self.solve_WDP(raised_bids, self.MIP_parameters, verbose=1)  # find the efficient allocation with respect to the inferred bids
                self.number_of_mips_solved['WDP'] += 1

                efficiency_raised = self.calculate_efficiency_of_allocation(allocation=allocation_raised, allocation_scw=true_value_raised)

                wandb_dict['Efficiency Raised Clock Bids Per Clock Round'] = efficiency_raised
                print(f'Clock round: {clock_round}, Efficiency: {efficiency} Efficiency Raised: {efficiency_raised}')

            wandb.log(wandb_dict)
            
        return 

    def extend_per_iteration_results(self):

        # for clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']:
        #     if clock_round >= self.mlca_iteration:
        #         self.efficiency_per_iteration_profit_max_specific_round[clock_round] = [self.efficiency_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]  
        #         self.efficiency_per_iteration_profit_max_unraised_specific_round[clock_round] = [self.efficiency_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]

        #         self.scw_per_iteration_profit_max_specific_round[clock_round] = [self.scw_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
        #         self.scw_per_iteration_profit_max_unraised_specific_round[clock_round] = [self.scw_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
        
        for future_iter in range(self.mlca_iteration+1,(self.Qmax-self.Qinit)+1):
            logging.warning(f'Extending json results for iteration {future_iter}')
            self.clearing_error_per_iteration[future_iter] = self.clearing_error_per_iteration[self.mlca_iteration]
            self.predicted_clearing_error_per_iteration[future_iter] = self.predicted_clearing_error_per_iteration[self.mlca_iteration]
            self.price_vector_per_iteration[future_iter] = self.price_vector_per_iteration[self.mlca_iteration]
            self.identical_price_vector_per_iteration[future_iter] = self.identical_price_vector_per_iteration[self.mlca_iteration]
            self.found_clearing_prices_per_iteration[future_iter] = self.found_clearing_prices_per_iteration[self.mlca_iteration]
            self.demand_vector_per_iteration[future_iter] = self.demand_vector_per_iteration[self.mlca_iteration]
            self.scw_per_iteration[future_iter] = self.scw_per_iteration[self.mlca_iteration]
            self.efficiency_per_iteration[future_iter] = self.efficiency_per_iteration[self.mlca_iteration]
            self.MIP_relative_gap_per_iteration[future_iter] = 0  # if we have found clearing prices -> we do not solve WDPs, set those metrics to 0
            self.MIP_time_per_iteration[future_iter] = 0
            self.MIP_unsatisfied_constraints_per_iteration[future_iter] = 0
            self.allocation_per_iteration[future_iter] = self.allocation_per_iteration[self.mlca_iteration]
            self.inferred_scw_per_iteration[future_iter] = self.inferred_scw_per_iteration[self.mlca_iteration]
            self.unique_bundles_bid_on_per_iteration[future_iter] = self.unique_bundles_bid_on_per_iteration[self.mlca_iteration]

            
            if self.mechanism_parameters['calculate_raised_bids']:
                # if the market cleared (using the non-raised bids)
                # the raised results should be identical, i.e., we never even reach that phase. 
                self.efficiency_per_iteration_raised_bids[future_iter] = self.efficiency_per_iteration[self.mlca_iteration]
                self.MIP_relative_gap_per_iteration_raised_bids[future_iter] = 0
                self.MIP_time_per_iteration_raised_bids[future_iter] = 0
                self.MIP_unsatisfied_constraints_per_iteration_raised_bids[future_iter] = 0
                self.scw_per_iteration_raised_bids[future_iter] = self.scw_per_iteration[self.mlca_iteration]

            if self.mechanism_parameters['calculate_profit_max_bids']:
                # if the market cleared (using the non-raised bids)
                # the profit max results should be identical, i.e., we never even reach that phase.
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    self.efficiency_per_iteration_profit_max[number_of_profit_max_bids][future_iter] = self.efficiency_per_iteration[self.mlca_iteration]
                    self.scw_per_iteration_profit_max[number_of_profit_max_bids][future_iter] = self.scw_per_iteration[self.mlca_iteration]

            if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
                # if the market cleared (using the non-raised bids)
                # the profit max results should be identical, i.e., we never even reach that phase.
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    self.efficiency_per_iteration_profit_max_unraised[number_of_profit_max_bids][future_iter] = self.efficiency_per_iteration[self.mlca_iteration]
                    self.scw_per_iteration_profit_max_unraised[number_of_profit_max_bids][future_iter] = self.scw_per_iteration[self.mlca_iteration]

            future_clock_round = future_iter + self.Qinit
            
            if future_clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']:
                self.efficiency_per_iteration_profit_max_specific_round[future_clock_round] = [self.efficiency_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]  
                self.efficiency_per_iteration_profit_max_unraised_specific_round[future_clock_round] = [self.efficiency_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
                self.MIP_relative_gap_per_iteration_profit_max_specific_round[future_clock_round] = [0 for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
                self.MIP_time_per_iteration_profit_max_specific_round[future_clock_round] = [0 for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
                self.MIP_unsatisfied_constraints_per_iteration_profit_max_specific_round[future_clock_round] = [0 for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]

                
                self.scw_per_iteration_profit_max_specific_round[future_clock_round] = [self.scw_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
                self.scw_per_iteration_profit_max_unraised_specific_round[future_clock_round] = [self.scw_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]



            logging.warning(f'Extending wandb logged results for iteration {future_iter}')
            if self.wandb_tracking:
                price_dict_to_save =  {f'Price Good {i}': self.price_vector_per_iteration[future_iter][i] for i in self.good_ids}
                wandb_dict = {"Efficiency Clock Bids": self.efficiency_per_iteration[future_iter],
                            "WDP Relative Gap Clock Bids": self.MIP_relative_gap_per_iteration[future_iter],
                            "WDP Time Clock Bids": self.MIP_time_per_iteration[future_iter],
                            "WDP Unsatisfied Constraints Clock Bids": self.MIP_unsatisfied_constraints_per_iteration[future_iter],
                            "Clearing Error": self.clearing_error_per_iteration[future_iter], 
                            "Predicted Clearing Error": self.predicted_clearing_error_per_iteration[future_iter],
                            "Identical Price Vector": 1 if self.identical_price_vector_per_iteration[future_iter] else 0, 
                            "Price Vector Sum": np.sum(self.price_vector_per_iteration[future_iter]), 
                            "Inferred SCW": self.inferred_scw_per_iteration[future_iter],
                            "SCW": self.scw_per_iteration[future_iter],
                            "Found Clearing Prices": 1 if self.found_clearing_prices_per_iteration[future_iter] else 0,
                            "Perturbed Prices": 1 if self.perturbed_prices else 0,
                            "Feasible Allocation": 1 if self.feasible_allocation else 0, 
                            "iterations": future_iter,
                            "Clock Round": future_iter + self.Qinit,
                            "Efficiency Clock Bids Per Clock Round": self.efficiency_per_iteration[future_iter],
                            }
                
                wandb_dict.update(price_dict_to_save)

                # Add the winning round bids to the wandb dict
                winning_round_bids = {}
                for bidder_id in self.bidder_ids:
                    winning_round_bids[f'Winning round bid for bidder {bidder_id}'] = self.mlca_iteration + self.Qinit 
                wandb_dict.update(winning_round_bids)

                # add the unique bundles that each agent bid on. 
                unique_bundles_bid_on_dict_to_save = {f'Bidder {bidder_id} Unique Bundles Bid On': self.unique_bundles_bid_on_per_iteration[future_iter][f'Bidder_{bidder_id}'] for bidder_id in self.bidder_ids}
                wandb_dict.update(unique_bundles_bid_on_dict_to_save)


                if self.mechanism_parameters['calculate_raised_bids']:
                    wandb_dict.update({"Efficiency Raised Clock Bids": self.efficiency_per_iteration_raised_bids[future_iter],
                                    "Efficiency Raised Clock Bids Per Clock Round": self.efficiency_per_iteration_raised_bids[future_iter],
                                    "WDP Relative Gap Raised Clock Bids": self.MIP_relative_gap_per_iteration_raised_bids[future_iter],
                                    "WDP Time Raised Clock Bids": self.MIP_time_per_iteration_raised_bids[future_iter],
                                    "WDP Unsatisfied Constraints Raised Clock Bids": self.MIP_unsatisfied_constraints_per_iteration_raised_bids[future_iter],
                                    "SCW Raised Bids": self.scw_per_iteration_raised_bids[future_iter]})
                    
                if self.mechanism_parameters['calculate_profit_max_bids']:
                    for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                        wandb_dict.update({f'Efficiency Profit Max {number_of_profit_max_bids} Bids': self.efficiency_per_iteration_profit_max[number_of_profit_max_bids][future_iter],
                                        f'SCW Profit Max {number_of_profit_max_bids} Bids': self.scw_per_iteration_profit_max[number_of_profit_max_bids][future_iter]})
                        
                if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
                    for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                        wandb_dict.update({f'Efficiency Profit Max {number_of_profit_max_bids} Bids (unraised clock bids)': self.efficiency_per_iteration_profit_max_unraised[number_of_profit_max_bids][future_iter],
                                        f'SCW Profit Max {number_of_profit_max_bids} Bids (unraised clock bids)': self.scw_per_iteration_profit_max_unraised[number_of_profit_max_bids][future_iter]})

                wandb.log(wandb_dict)

                if future_clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']:
                    for number_of_profit_max_bids in range(self.mechanism_parameters['profit_max_grid'][-1]+1):
                        profit_max_dict_round = {
                            f'Efficiency Profit Max ({future_clock_round} clock rounds)': self.efficiency_per_iteration_profit_max_specific_round[future_clock_round][number_of_profit_max_bids],
                            f'WDP Relative Gap Profit Max ({future_clock_round} clock rounds)': self.MIP_relative_gap_per_iteration_profit_max_specific_round[future_clock_round][number_of_profit_max_bids],
                            f'WDP Time Profit Max ({future_clock_round} clock rounds)': self.MIP_time_per_iteration_profit_max_specific_round[future_clock_round][number_of_profit_max_bids],
                            f'WDP Unsatisfied Constraints Profit Max ({future_clock_round} clock rounds)': self.MIP_unsatisfied_constraints_per_iteration_profit_max_specific_round[future_clock_round][number_of_profit_max_bids],
                            f'SCW Profit Max ({future_clock_round} clock rounds)': self.scw_per_iteration_profit_max_specific_round[future_clock_round][number_of_profit_max_bids],
                            f'Efficiency Profit Max ({future_clock_round} clock rounds) (unraised clock bids)': self.efficiency_per_iteration_profit_max_unraised_specific_round[future_clock_round][number_of_profit_max_bids],
                            f'SCW Profit Max ({future_clock_round} clock rounds) (unraised clock bids)': self.scw_per_iteration_profit_max_unraised_specific_round[future_clock_round][number_of_profit_max_bids],
                            'profit max bids': number_of_profit_max_bids,
                        }
                        wandb.log(profit_max_dict_round)

            
                for j in range(self.get_number_of_elicited_dqs()[0]+ 1,self.Qmax+1):
                    extended_price_dict_to_save = {f'Extended Prices Good {i}': self.elicited_dqs['Bidder_0'][1][self.get_number_of_elicited_dqs()[0]-1,i] for i in self.good_ids}  
                    extended_price_dict_to_save["Number of Elicited Bids"] = j
                    wandb.log(extended_price_dict_to_save)


    def wandb_final_table(self):
        columns=["SATS Seed", "Efficiency in %", "Relative (VCG) Revenue in %", "Clearing Error", "Total Time Elapsed (sec)"]
        data = [self.SATS_auction_instance_seed, self.final_allocation_efficiency, self.relative_revenue, self.clearing_error_per_iteration[len(self.clearing_error_per_iteration)], (self.end_time - self.start_time).total_seconds()]
        
        if self.mechanism_parameters['calculate_raised_bids']:
            columns = columns + ["Efficiency Raised Clock Bids in %", "Relative (VCG) Revenue Raised Clock Bids in %"]
            data = data + [self.final_allocation_efficiency_raised_bids, self.relative_revenue_raised_bids]

        if self.mechanism_parameters['calculate_profit_max_bids']:
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                columns = columns + [f"Efficiency Profit Max {number_of_profit_max_bids} Bids in %", f"Relative (VCG) Revenue Profit Max {number_of_profit_max_bids} Bids in %"]
                data = data + [self.final_allocation_efficiency_profit_max[number_of_profit_max_bids], self.relative_revenue_profit_max[number_of_profit_max_bids]]

        if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                columns = columns + [f"Efficiency Profit Max {number_of_profit_max_bids} Bids (unraised clock bids) in %", f"Relative (VCG) Revenue Profit Max {number_of_profit_max_bids} Bids (unraised clock bids) in %"]
                data = data + [self.final_allocation_efficiency_profit_max_unraised[number_of_profit_max_bids], self.relative_revenue_profit_max_unraised[number_of_profit_max_bids]]
       
            
        wandb_table = wandb.Table(data=[data], columns=columns)
        wandb.log({"final_table": wandb_table})


    def save_results(self,
                     res_path,
                     no_wandb_logging=False
                     ):
        """
        Saves results in the res_path directory as a dictionary, as well as logs the results to wandb.
        """
        print(f'Saving results to {res_path}, current iteration: {self.mlca_iteration}')
        save_dict = OrderedDict()

        save_dict['Efficiency'] = self.final_allocation_efficiency
        save_dict['Efficiency per Iteration'] = self.efficiency_per_iteration
        save_dict['SCW per Iteration'] = self.scw_per_iteration
        save_dict['Inferred SCW per Iteration'] = self.inferred_scw_per_iteration
        save_dict['Clearing Error per Iteration'] = self.clearing_error_per_iteration
        save_dict['Predicted Clearing Error per Iteration'] = self.predicted_clearing_error_per_iteration
        save_dict['Price Vector per Iteration'] = self.price_vector_per_iteration
        save_dict['Identical Price Vector per Iteration'] = self.identical_price_vector_per_iteration
        save_dict['Found Clearing Prices'] = self.found_clearing_prices
        save_dict['Final SCW'] = self.final_allocation_scw,  # Potentially None for intermediate iteration
        save_dict['Final Allocation'] = self.final_allocation,  # Potentially None for intermediate iteration
        save_dict['Marginal Economies Allocations'] = self.marginal_economies_allocations
        save_dict['Marginal Economies SCW'] = self.marginal_economies_scw
        save_dict['Marginal Economies Inferred SCW'] = self.marginal_economies_inferred_scw
        save_dict['SATS Optimal SCW'] = self.SATS_auction_instance_scw
        save_dict['SATS Efficient Allocation'] = self.SATS_auction_instance_allocation

        save_dict['VCG Payments'] = self.vcg_payments
        save_dict['Clearing Payments'] = self.clearing_payments

        save_dict['Revenue'] = self.revenue
        save_dict['Relative Revenue'] = self.relative_revenue
        
        save_dict['Total Time Elapsed'] = self.total_time_elapsed
        save_dict['Total Time Elapsed Distribution'] = self.total_time_elapsed_distr
        save_dict['Optimization MIPs'] = self.number_of_mips_solved
        save_dict['ML Statistics'] = self.train_logs
        save_dict['Elicited DQs'] = self.elicited_dqs
        save_dict['Allocation per Iteration'] = self.allocation_per_iteration
        

        if self.mechanism_parameters['calculate_raised_bids']:
            save_dict['Efficiency per Iteration Raised Bids'] = self.efficiency_per_iteration_raised_bids
            save_dict['SCW per Iteration Raised Bids'] = self.scw_per_iteration_raised_bids

        if self.mechanism_parameters['calculate_profit_max_bids']:
            save_dict['Efficiency per Iteration Profit Max'] = self.efficiency_per_iteration_profit_max
            save_dict['SCW per Iteration Profit Max'] = self.scw_per_iteration_profit_max

        if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
            save_dict['Efficiency per Iteration Profit Max Unraised'] = self.efficiency_per_iteration_profit_max_unraised
            save_dict['SCW per Iteration Profit Max Unraised'] = self.scw_per_iteration_profit_max_unraised

        if self.wandb_tracking and not no_wandb_logging:

            if self.mlca_iteration == 1:

                # Call the function that will log results for the first Qinit clock rounds. 
                self.log_Qinit_efficiency()


                # Only log those results for iteration 0! 
                wandb_init_dict = {"Efficiency Clock Bids": self.efficiency_per_iteration[0],
                           "WDP Relative Gap Clock Bids": self.MIP_relative_gap_per_iteration[0],
                           "WDP Time Clock Bids": self.MIP_time_per_iteration[0],
                           "WDP Unsatisfied Constraints Clock Bids": self.MIP_unsatisfied_constraints_per_iteration[0],
                           "Inferred SCW": self.inferred_scw_per_iteration[0],
                           "SCW": self.scw_per_iteration[0],
                           "Perturbed Prices": 0, 
                           "iterations": 0, 
                           "Clock Round": self.Qinit,
                           "Efficiency Clock Bids Per Clock Round": self.efficiency_per_iteration[0],}
                

                if self.mechanism_parameters['calculate_raised_bids']:
                    wandb_init_dict["Efficiency Raised Clock Bids"] = self.efficiency_per_iteration_raised_bids[0]
                    wandb_init_dict["Efficiency Raised Clock Bids Per Clock Round"] = self.efficiency_per_iteration_raised_bids[0]
                    wandb_init_dict["WDP Relative Gap Raised Clock Bids"] = self.MIP_relative_gap_per_iteration_raised_bids[0]
                    wandb_init_dict["WDP Time Raised Clock Bids"] = self.MIP_time_per_iteration_raised_bids[0]
                    wandb_init_dict["WDP Unsatisfied Constraints Raised Clock Bids"] = self.MIP_unsatisfied_constraints_per_iteration_raised_bids[0]
                    wandb_init_dict["SCW Raised Bids"] = self.scw_per_iteration_raised_bids[0]

                if self.mechanism_parameters['calculate_profit_max_bids']:
                    for j in self.mechanism_parameters['profit_max_grid']:
                        wandb_init_dict[f"Efficiency Profit Max {j} Bids"] = self.efficiency_per_iteration_profit_max[j][0]
                        wandb_init_dict[f"SCW Profit Max {j} Bids"] = self.scw_per_iteration_profit_max[j][0]

                if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
                    # set_trace()
                    for j in self.mechanism_parameters['profit_max_grid']:
                        wandb_init_dict[f"Efficiency Profit Max {j} Bids (unraised clock bids)"] = self.efficiency_per_iteration_profit_max_unraised[j][0]
                        wandb_init_dict[f"SCW Profit Max  {j} Bids (unraised clock bids)"] = self.scw_per_iteration_profit_max_unraised[j][0]
                
                
                unique_bundles_bid_on_dict_to_save = {f'Bidder {bidder_id} Unique Bundles Bid On': self.unique_bundles_bid_on_per_iteration[0][f'Bidder_{bidder_id}'] for bidder_id in self.bidder_ids}
                wandb_init_dict.update(unique_bundles_bid_on_dict_to_save)

                
                wandb.log(wandb_init_dict)
                

                for j in range(1, self.get_number_of_elicited_dqs()[0]):
                    extended_price_dict_to_save = {f'Extended Prices Good {i}': self.elicited_dqs['Bidder_0'][1][j-1,i] for i in self.good_ids}  #NOTE: only works for increasing prices/CCA
                    extended_price_dict_to_save["Number of Elicited Bids"] = j
                    wandb.log(extended_price_dict_to_save)


            # also log the profit max results for iteration 0, if you have to. 
            clock_round = self.Qinit
            if clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']: 
                for number_of_profit_max_bids in range(self.mechanism_parameters['profit_max_grid'][-1]+1):
                    profit_max_dict_round = {
                        f'Efficiency Profit Max ({clock_round} clock rounds)': self.efficiency_per_iteration_profit_max_specific_round[clock_round][number_of_profit_max_bids],
                        f'SCW Profit Max ({clock_round} clock rounds)': self.scw_per_iteration_profit_max_specific_round[clock_round][number_of_profit_max_bids],
                        f'Efficiency Profit Max ({clock_round} clock rounds) (unraised clock bids)': self.efficiency_per_iteration_profit_max_unraised_specific_round[clock_round][number_of_profit_max_bids],
                        f'SCW Profit Max ({clock_round} clock rounds) (unraised clock bids)': self.scw_per_iteration_profit_max_unraised_specific_round[clock_round][number_of_profit_max_bids],
                        'profit max bids': number_of_profit_max_bids,
                    }
                    wandb.log(profit_max_dict_round)

            wandb_dict = {"Efficiency Clock Bids": self.efficiency_per_iteration[self.mlca_iteration],
                        "WDP Relative Gap Clock Bids": self.MIP_relative_gap_per_iteration[self.mlca_iteration],
                        "WDP Time Clock Bids": self.MIP_time_per_iteration[self.mlca_iteration],
                        "WDP Unsatisfied Constraints Clock Bids": self.MIP_unsatisfied_constraints_per_iteration[self.mlca_iteration],
                        "Clearing Error": self.clearing_error_per_iteration[self.mlca_iteration], 
                        "Predicted Clearing Error": self.predicted_clearing_error_per_iteration[self.mlca_iteration],
                        "Identical Price Vector": 1 if self.identical_price_vector_per_iteration[self.mlca_iteration] else 0, 
                        "Price Vector Sum": np.sum(self.price_vector_per_iteration[self.mlca_iteration]), 
                        "Inferred SCW": self.inferred_scw_per_iteration[self.mlca_iteration],
                        "SCW": self.scw_per_iteration[self.mlca_iteration],
                        "Found Clearing Prices": 1 if self.found_clearing_prices_per_iteration[self.mlca_iteration] else 0, 
                        "Perturbed Prices": 1 if self.perturbed_prices else 0, 
                        "Feasible Allocation": 1 if self.feasible_allocation else 0,
                        "iterations": self.mlca_iteration, 
                        "Clock Round": self.mlca_iteration + self.Qinit,
                        "Efficiency Clock Bids Per Clock Round": self.efficiency_per_iteration[self.mlca_iteration]
                        }
            
            price_dict_to_save =  {f'Price Good {i}': self.price_vector_per_iteration[self.mlca_iteration][i] for i in self.good_ids}

            # create a dictionary with the round in which each agent bid on the bundle they won 
            winning_round_bids = {}
            for bidder_id in self.bidder_ids:
                bundle_won = self.allocation_per_iteration[self.mlca_iteration][f'Bidder_{bidder_id}']['allocated_bundle']
                bundle_won = np.array(bundle_won)
                bundles_bid_on = self.elicited_dqs[f'Bidder_{bidder_id}'][0]
                # get the last round where the bundle won was bid on
                round_matches = (bundle_won == bundles_bid_on).sum(axis = 1) == bundle_won.shape[0]
                if np.sum(round_matches) > 0:
                    last_round = np.max(np.where(round_matches)[0]) + 1 # because indexes start from 0, but clock rounds/iterations start from 1
                else:
                    last_round = 0 # if the bundle was never bid on (i.e., the agent got the empty bundle) set the last round to 0
                winning_round_bids[f'Winning round bid for bidder {bidder_id}'] = last_round

            wandb_dict.update(winning_round_bids)

            
            if self.mechanism_parameters['calculate_raised_bids']:
                wandb_dict["Efficiency Raised Clock Bids"] = self.efficiency_per_iteration_raised_bids[self.mlca_iteration]
                wandb_dict["Efficiency Raised Clock Bids Per Clock Round"] = self.efficiency_per_iteration_raised_bids[self.mlca_iteration]
                wandb_dict["WDP Relative Gap Raised Clock Bids"] = self.MIP_relative_gap_per_iteration_raised_bids[self.mlca_iteration]
                wandb_dict["WDP Time Raised Clock Bids"] = self.MIP_time_per_iteration_raised_bids[self.mlca_iteration]
                wandb_dict["WDP Unsatisfied Constraints Raised Clock Bids"] = self.MIP_unsatisfied_constraints_per_iteration_raised_bids[self.mlca_iteration]
                wandb_dict["SCW Raised Bids"] = self.scw_per_iteration_raised_bids[self.mlca_iteration]

            if self.mechanism_parameters['calculate_profit_max_bids']:
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    wandb_dict[f'Efficiency Profit Max {number_of_profit_max_bids} Bids'] = self.efficiency_per_iteration_profit_max[number_of_profit_max_bids][self.mlca_iteration]
                    wandb_dict[f'SCW Profit Max {number_of_profit_max_bids} Bids'] = self.scw_per_iteration_profit_max[number_of_profit_max_bids][self.mlca_iteration]

            if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    wandb_dict[f'Efficiency Profit Max {number_of_profit_max_bids} Bids (unraised clock bids)'] = self.efficiency_per_iteration_profit_max_unraised[number_of_profit_max_bids][self.mlca_iteration]
                    wandb_dict[f'SCW Profit Max {number_of_profit_max_bids} Bids (unraised clock bids)'] = self.scw_per_iteration_profit_max_unraised[number_of_profit_max_bids][self.mlca_iteration]
            
            wandb_dict.update(price_dict_to_save)

            print('Current iteration: ', self.mlca_iteration)
            # set_trace()
            unique_bundles_bid_on_dict_to_save = {f'Bidder {bidder_id} Unique Bundles Bid On': self.unique_bundles_bid_on_per_iteration[self.mlca_iteration][f'Bidder_{bidder_id}'] for bidder_id in self.bidder_ids}
            wandb_dict.update(unique_bundles_bid_on_dict_to_save)


            wandb.log(wandb_dict)

            extended_price_dict_to_save = {f'Extended Prices Good {i}': self.elicited_dqs['Bidder_0'][1][self.get_number_of_elicited_dqs()[0]-1,i] for i in self.good_ids}  #NOTE: only works for increasing prices/cca
            extended_price_dict_to_save["Number of Elicited Bids"] = self.get_number_of_elicited_dqs()[0]
            wandb.log(extended_price_dict_to_save)

            clock_round = self.mlca_iteration + self.Qinit
            if clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']: 
                for number_of_profit_max_bids in range(self.mechanism_parameters['profit_max_grid'][-1]+1):
                    profit_max_dict_round = {
                        f'Efficiency Profit Max ({clock_round} clock rounds)': self.efficiency_per_iteration_profit_max_specific_round[clock_round][number_of_profit_max_bids],
                        f'WDP Relative Gap Profit Max ({clock_round} clock rounds)': self.MIP_relative_gap_per_iteration_profit_max_specific_round[clock_round][number_of_profit_max_bids],
                        f'WDP Time Profit Max ({clock_round} clock rounds)': self.MIP_time_per_iteration_profit_max_specific_round[clock_round][number_of_profit_max_bids],
                        f'WDP Unsatisfied Constraints Profit Max ({clock_round} clock rounds)': self.MIP_unsatisfied_constraints_per_iteration_profit_max_specific_round[clock_round][number_of_profit_max_bids],
                        f'SCW Profit Max ({clock_round} clock rounds)': self.scw_per_iteration_profit_max_specific_round[clock_round][number_of_profit_max_bids],
                        f'Efficiency Profit Max ({clock_round} clock rounds) (unraised clock bids)': self.efficiency_per_iteration_profit_max_unraised_specific_round[clock_round][number_of_profit_max_bids],
                        f'SCW Profit Max ({clock_round} clock rounds) (unraised clock bids)': self.scw_per_iteration_profit_max_unraised_specific_round[clock_round][number_of_profit_max_bids],
                        'profit max bids': number_of_profit_max_bids,
                    }
                    wandb.log(profit_max_dict_round)
                    


        json.dump(save_dict, open(os.path.join(res_path, 'results.json'), 'w'), indent=4,
                sort_keys=False, separators=(', ', ': '),
                ensure_ascii=False, cls=NumpyEncoder)

    def get_info(self,
                 final_summary=False
                 ):

        if final_summary:
            logging.warning('SUMMARY')
            if self.wandb_tracking: 
                wandb.finish()
        else:
            logging.warning('INFO')
        logging.warning('-----------------------------------------------')
        logging.warning('Seed Auction Instance: %s', self.SATS_auction_instance_seed)
        logging.warning('Iteration: %s', self.mlca_iteration)
        logging.warning('Qinit: %s | Qmax: %s', self.Qinit, self.Qmax)
        if final_summary:
            logging.warning('EFFICIENCY: {} %'.format(round(self.final_allocation_efficiency, 4) * 100))
            logging.warning(f'TOTAL TIME ELAPSED: {self.total_time_elapsed[0]}')
            logging.warning(f'WDP MIP TIME: {self.total_time_elapsed_distr["WDP MIP"][0]} ({100 * self.total_time_elapsed_distr["WDP MIP"][1]}%) | Price Generation TIME: {self.total_time_elapsed_distr["Price Vector Generation"][0]} ({100 * self.total_time_elapsed_distr["Price Vector Generation"][1]}%) | ML TRAIN TIME: {self.total_time_elapsed_distr["ML TRAIN"][0]} ({100 * self.total_time_elapsed_distr["ML TRAIN"][1]}%) | OTHER TIME: {self.total_time_elapsed_distr["OTHER"][0]} ({100 * self.total_time_elapsed_distr["OTHER"][1]}%)')
            logging.warning(f'NORMAL OPTIMIZATIONS: {self.number_of_mips_solved}')
            # logging.warning(f'MIP avg REL.GAP: {np.mean(self.mip_logs["rel. gap"])} | MIP HIT TIME LIMIT: {int(sum(self.mip_logs["hit_limit"]))}/{len(self.mip_logs["hit_limit"])}')
            logging.warning('FINISHED')
        else:
            logging.warning('Efficiency given elicited bids from iteration 0-%s: %s\n',self.mlca_iteration - 1, self.efficiency_per_iteration.get(self.mlca_iteration - 1))
            

    def calc_time_spent(self):

        self.end_time = datetime.now()
        if self.start_time:
            tot_seconds = (self.end_time - self.start_time).total_seconds()
            wdp_mip_seconds = self.total_time_wdp_mip
            ml_train_seconds = self.total_time_ml_train
            price_vector_generation_seconds = self.total_time_price_vector_generation
            other_seconds = tot_seconds - wdp_mip_seconds - ml_train_seconds - price_vector_generation_seconds
            self.total_time_elapsed_distr['WDP MIP'] = (
                '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(timedelta(seconds=wdp_mip_seconds))),
                round(wdp_mip_seconds / tot_seconds, ndigits=2),
                wdp_mip_seconds)
            self.total_time_elapsed_distr['ML TRAIN'] = (
                '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(timedelta(seconds=ml_train_seconds))),
                round(ml_train_seconds / tot_seconds, ndigits=2),
                ml_train_seconds)
            self.total_time_elapsed_distr['Price Vector Generation'] = (
                '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(timedelta(seconds=price_vector_generation_seconds))),
                round(price_vector_generation_seconds / tot_seconds, ndigits=2),
                ml_train_seconds)
            self.total_time_elapsed_distr['OTHER'] = (
                '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(timedelta(seconds=other_seconds))),
                round(other_seconds / tot_seconds, ndigits=2),
                other_seconds)
            self.total_time_elapsed = ('{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(self.end_time - self.start_time)),
                                       tot_seconds)
        else:
            logging.warning('start_time not specififed')

    def get_number_of_elicited_dqs(self):
        return [len(self.elicited_dqs[bidder][0]) for bidder in self.bidder_names]
    

    def merge_dicts(self, *dict_args):
        """
        Given any number of dictionaries, shallow copy and merge into a new dict,
        precedence goes to key-value pairs in latter dictionaries.
        """
        dict_args = dict_args[0]
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result
    
    
    def calculate_inferred_bids(self): 
        inferred_bids = {} 
        self.unique_bundles_bid_on_per_iteration[self.mlca_iteration] = {}
        for bidder in self.bidder_names:
            X = self.elicited_dqs[bidder][0]
            P = self.elicited_dqs[bidder][1]
            V = np.array([np.dot(X[i], P[i]) for i in range(X.shape[0])]) # possibly reshape(-1, 1) in the end. 

            # make the array of inferred bids have unique rows, because the same bundle could have been requested at different prices
            X_unique = np.unique(X, axis=0)  # get all unique Xs
            
            V_unique = np.array([np.max(V[np.where((X == X_unique[i]).all(axis = 1))[0]]) for i in range(X_unique.shape[0])]) # tell Haskell I said hi. 


            inferred_bids[bidder] = [X_unique, V_unique]
            print(f'For bidder {bidder} inferred bid shape after removing duplicates is : {inferred_bids[bidder][0].shape}')
            self.unique_bundles_bid_on_per_iteration[self.mlca_iteration][bidder] = X_unique.shape[0]

        self.inferred_bids = inferred_bids

    def raise_inferred_bids(self):
        """
        Raises the inferred bids to their true value, same as the "clock bids raised" heuristic in the MLCA paper.  
        """
        inferred_bids_raised = {} 
        for bidder_id in range(len(self.bidder_names)):
            bidder_name = f'Bidder_{bidder_id}'
            X = self.inferred_bids[bidder_name][0] 
            X_unique = np.unique(X, axis=0)  # get all unique Xs  
            V = [] 
            # for every bidder, for all of her unique inferred bids, calculate the value of that bundle
            for i in range(X_unique.shape[0]):
                raised_bid = self.SATS_auction_instance.calculate_value(bidder_id, X_unique[i])
                V.append(raised_bid)
            inferred_bids_raised[bidder_name] = [X_unique, np.array(V)]
        
        self.inferred_bids_raised = inferred_bids_raised

    def calculate_inferred_bids_per_round(self, round_number): 
        inferred_bids = {}
        for bidder in self.bidder_names:
            X = self.elicited_dqs[bidder][0][:round_number]
            P = self.elicited_dqs[bidder][1][:round_number]
            V = np.array([np.dot(X[i], P[i]) for i in range(X.shape[0])]) # possibly reshape(-1, 1) in the end. 

            # make the array of inferred bids have unique rows, because the same bundle could have been requested at different prices
            X_unique = np.unique(X, axis=0)  # get all unique Xs
            
            V_unique = np.array([np.max(V[np.where((X == X_unique[i]).all(axis = 1))[0]]) for i in range(X_unique.shape[0])]) # tell Haskell I said hi. 


            inferred_bids[bidder] = [X_unique, V_unique]
            # set_trace()
            print(f'For bidder {bidder} inferred bid shape up to round {round_number} after removing duplicates is : {inferred_bids[bidder][0].shape}')

        return inferred_bids

    def raise_inferred_bids_per_round(self, round_number):
        """
        Raises the inferred bids to their true value, same as the "clock bids raised" heuristic in the MLCA paper.  
        Does not log any results, as this is to be called recursively for all Qinit first clock rounds. 
        """
        inferred_bids_raised = {} 
        for bidder_id in range(len(self.bidder_names)):
            bidder_name = f'Bidder_{bidder_id}'
            X = self.elicited_dqs[bidder_name][0][:round_number]   
            V = [] 
            # for every bidder, for all of her unique inferred bids, calculate the value of that bundle
            for i in range(X.shape[0]):
                raised_bid = self.SATS_auction_instance.calculate_value(bidder_id, X[i])
                V.append(raised_bid)
            inferred_bids_raised[bidder_name] = [X, np.array(V)]
            # set_trace()
        
        return inferred_bids_raised

    def calculate_profit_max_bids(self): 
        """
        Calculates the profit-maximizing bids for all bidders, for the current price vector. 
        """
        # profit_max_bids = {j: {} for j in self.mechanism_parameters['profit_max_grid']}
        profit_max_bids = {}  
        if self.mlca_iteration != 0:
            price_vector = self.price_vector_per_iteration[self.mlca_iteration]
        else: 
            logging.warning('No price vector yet, using the last from the extended bids')
            price_vector = self.elicited_dqs['Bidder_0'][1][-1]

        start = datetime.now()
        for bidder_id in range(len(self.bidder_names)):
            logging.info(f'Calculating profit-max bids for bidder {bidder_id}...')
            bidder_name = f'Bidder_{bidder_id}'
            #NOTE: this assumes that the profit max grid is increasing, and so the last entry is the total number of profit-max bids we need to sample
            demand_response_total = self.SATS_auction_instance.get_best_bundles(bidder_id, price_vector, self.mechanism_parameters['profit_max_grid'][-1], allow_negative = True) 

            V = [] 
            # for all of those bids: get the bidder's value for that bundle
            for i in range(len(demand_response_total)):
                V.append(self.SATS_auction_instance.calculate_value(bidder_id, demand_response_total[i]))
            
            
            # # store those in the profit_max_bids dict
            # for j in self.mechanism_parameters['profit_max_grid']:
            #     profit_max_bids[j][bidder_name] = [demand_response_total[:j], V[:j]]
            profit_max_bids[bidder_name] = [demand_response_total, V]

        end = datetime.now()
        logging.info(f'Calculating profit-max bids took {(end - start)} seconds')

        self.profit_max_bids = profit_max_bids

    
    def calculate_efficiency_per_iteration(self):  
        logging.info('')
        logging.info('Calculate current efficiency:')
        self.calculate_inferred_bids()    # transform the demand responses of the agents into inferred bids
        allocation, true_value, inferred_value, details = self.solve_WDP(self.inferred_bids, self.MIP_parameters, verbose=1)  # find the efficient allocation with respect to the inferred bids
        self.number_of_mips_solved['WDP'] += 1

        
        # calculate the value of the resulting allocation with respect to the true values of the agents 
        self.allocation_per_iteration[self.mlca_iteration] = allocation
        efficiency = self.calculate_efficiency_of_allocation(allocation=allocation, allocation_scw=true_value)  
        self.efficiency_per_iteration[self.mlca_iteration] = efficiency
        self.MIP_relative_gap_per_iteration[self.mlca_iteration] = details['Relative Gap']
        self.MIP_time_per_iteration[self.mlca_iteration] = details['Time']
        self.MIP_unsatisfied_constraints_per_iteration[self.mlca_iteration] = details['Unsatisfied Constraints']
        logging.info(f'Current efficiency: {efficiency}')

        self.inferred_scw_per_iteration[self.mlca_iteration] = inferred_value
        self.scw_per_iteration[self.mlca_iteration] = true_value

        if self.mechanism_parameters['calculate_raised_bids']:
            self.raise_inferred_bids()
            allocation_raised, true_value_raised, inferred_value_raised, details = self.solve_WDP(self.inferred_bids_raised, self.MIP_parameters, verbose=1) # find the efficient allocation with respect to the raised bids
            self.number_of_mips_solved['WDP'] += 1

            self.allocation_per_iteration_raised_bids[self.mlca_iteration] = allocation_raised
            efficiency_raised = self.calculate_efficiency_of_allocation(allocation=allocation_raised, allocation_scw=true_value_raised)
            self.efficiency_per_iteration_raised_bids[self.mlca_iteration] = efficiency_raised
            self.MIP_relative_gap_per_iteration_raised_bids[self.mlca_iteration] = details['Relative Gap']
            self.MIP_time_per_iteration_raised_bids[self.mlca_iteration] = details['Time']
            self.MIP_unsatisfied_constraints_per_iteration_raised_bids[self.mlca_iteration] = details['Unsatisfied Constraints']
            logging.info(f'Current efficiency (raised bids): {efficiency_raised}')
            # set_trace()

            self.scw_per_iteration_raised_bids[self.mlca_iteration] = true_value_raised

        current_clock_round = self.Qinit + self.mlca_iteration 

        if self.mechanism_parameters['calculate_profit_max_bids'] or self.mechanism_parameters['calculate_profit_max_bids_unraised'] or current_clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']:
            self.calculate_profit_max_bids()
            
            # the profit max bids get added to the bids/raised bids of the clock phase! 

            initial_bids = copy.deepcopy(self.inferred_bids_raised)
            initial_bids_unraised = copy.deepcopy(self.inferred_bids)

            if self.mechanism_parameters['calculate_profit_max_bids']:
                for profit_max_bids_number in self.mechanism_parameters['profit_max_grid']:
                    # combine the right number of profit max bids with the the (raised) clock bids

                    total_bids = {bidder_name: [np.concatenate((initial_bids[bidder_name][0], self.profit_max_bids[bidder_name][0][:profit_max_bids_number])), np.concatenate((initial_bids[bidder_name][1], self.profit_max_bids[bidder_name][1][:profit_max_bids_number]))] for bidder_name in self.bidder_names}

                    self.profit_max_bids_combined[profit_max_bids_number] = copy.deepcopy(total_bids)  # store a copy of the combined bids for final payment calculation
                    
                    # calculate the efficiency for the profit-max bids
                    allocation_profit_max, true_value_profit_max, inferred_value_profit_max, details = self.solve_WDP(total_bids, self.MIP_parameters, verbose=1) 
                    self.number_of_mips_solved['WDP'] += 1

                    self.allocation_per_iteration_profit_max[profit_max_bids_number][self.mlca_iteration] = allocation_profit_max
                    efficiency_profit_max = self.calculate_efficiency_of_allocation(allocation=allocation_profit_max, allocation_scw=true_value_profit_max)
                    self.efficiency_per_iteration_profit_max[profit_max_bids_number][self.mlca_iteration] = efficiency_profit_max
                    logging.info(f'Current efficiency (profit max bids, {profit_max_bids_number}): {efficiency_profit_max}')
                    # set_trace()

                    self.scw_per_iteration_profit_max[profit_max_bids_number][self.mlca_iteration] = true_value_profit_max
                
            if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
                    for profit_max_bids_number in self.mechanism_parameters['profit_max_grid']:
                # combine the right number of profit max bids with the the (raised) clock bids

                        total_bids_unraised = {bidder_name: [np.concatenate((initial_bids_unraised[bidder_name][0], self.profit_max_bids[bidder_name][0][:profit_max_bids_number])), np.concatenate((initial_bids_unraised[bidder_name][1], self.profit_max_bids[bidder_name][1][:profit_max_bids_number]))] for bidder_name in self.bidder_names}

                        self.profit_max_bids_combined_unraised[profit_max_bids_number] = copy.deepcopy(total_bids)  # store a copy of the combined bids for final payment calculation
                        
                        # calculate the efficiency for the profit-max bids
                        allocation_profit_max_unraised, true_value_profit_max_unraised, inferred_value_profit_max_unraised, details = self.solve_WDP(total_bids_unraised, self.MIP_parameters, verbose=1) 
                        self.number_of_mips_solved['WDP'] += 1

                        self.allocation_per_iteration_profit_max_unraised[profit_max_bids_number][self.mlca_iteration] = allocation_profit_max_unraised
                        efficiency_profit_max_unraised = self.calculate_efficiency_of_allocation(allocation=allocation_profit_max_unraised, allocation_scw=true_value_profit_max_unraised)
                        self.efficiency_per_iteration_profit_max_unraised[profit_max_bids_number][self.mlca_iteration] = efficiency_profit_max_unraised
                        logging.info(f'Current efficiency (profit max bids (unraised clock bids), {profit_max_bids_number}): {efficiency_profit_max_unraised}')
                        # set_trace()

                        self.scw_per_iteration_profit_max_unraised[profit_max_bids_number][self.mlca_iteration] = true_value_profit_max_unraised

        if current_clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']:
            logging.info(f'Reached clock round {current_clock_round}, calculating profit-max bids!')
            # for every possible number of profit-max bids
            self.efficiency_per_iteration_profit_max_specific_round[current_clock_round] = [self.efficiency_per_iteration_raised_bids[self.mlca_iteration]]  
            self.efficiency_per_iteration_profit_max_unraised_specific_round[current_clock_round] = [self.efficiency_per_iteration[self.mlca_iteration]]

            self.scw_per_iteration_profit_max_specific_round[current_clock_round] = [self.scw_per_iteration_raised_bids[self.mlca_iteration]]
            self.scw_per_iteration_profit_max_unraised_specific_round[current_clock_round] = [self.scw_per_iteration[self.mlca_iteration]]

            self.MIP_relative_gap_per_iteration_profit_max_specific_round[current_clock_round] = [self.MIP_relative_gap_per_iteration_raised_bids[self.mlca_iteration]]
            self.MIP_time_per_iteration_profit_max_specific_round[current_clock_round] = [self.MIP_time_per_iteration_raised_bids[self.mlca_iteration]]
            self.MIP_unsatisfied_constraints_per_iteration_profit_max_specific_round[current_clock_round] = [self.MIP_unsatisfied_constraints_per_iteration_raised_bids[self.mlca_iteration]]

            for profit_max_bids_number in range(1, self.mechanism_parameters['profit_max_grid'][-1] + 1):
                start = datetime.now()

                # calculate the efficiency for the profit-max bids
                initial_bids = copy.deepcopy(self.inferred_bids_raised)
                total_bids = {bidder_name: [np.concatenate((initial_bids[bidder_name][0], self.profit_max_bids[bidder_name][0][:profit_max_bids_number])), np.concatenate((initial_bids[bidder_name][1], self.profit_max_bids[bidder_name][1][:profit_max_bids_number]))] for bidder_name in self.bidder_names}
                
                # set_trace()
                allocation_profit_max, true_value_profit_max, inferred_value_profit_max, details = self.solve_WDP(total_bids,  self.MIP_parameters, verbose=1) 
                self.number_of_mips_solved['WDP'] += 1
   
                efficiency_profit_max = self.calculate_efficiency_of_allocation(allocation=allocation_profit_max, allocation_scw=true_value_profit_max)
                self.efficiency_per_iteration_profit_max_specific_round[current_clock_round].append(efficiency_profit_max)
                self.scw_per_iteration_profit_max_specific_round[current_clock_round].append(true_value_profit_max)
                self.MIP_relative_gap_per_iteration_profit_max_specific_round[current_clock_round].append(details['Relative Gap'])
                self.MIP_time_per_iteration_profit_max_specific_round[current_clock_round].append(details['Time'])
                self.MIP_unsatisfied_constraints_per_iteration_profit_max_specific_round[current_clock_round].append(details['Unsatisfied Constraints'])


                logging.info(f'Current efficiency clock round {current_clock_round}, {profit_max_bids_number} profit max bids: {efficiency_profit_max}')
                
                
                # calculate the efficiency for the profit-max bids (unraised clock bids)
                initial_bids_unraised = copy.deepcopy(self.inferred_bids)
                total_bids_unraised = {bidder_name: [np.concatenate((initial_bids_unraised[bidder_name][0], self.profit_max_bids[bidder_name][0][:profit_max_bids_number])), np.concatenate((initial_bids_unraised[bidder_name][1], self.profit_max_bids[bidder_name][1][:profit_max_bids_number]))] for bidder_name in self.bidder_names}
                

                if profit_max_bids_number in self.mechanism_parameters['profit_max_grid']:
                    self.profit_max_bids_specific_round[profit_max_bids_number] = copy.deepcopy(total_bids)  # store a copy of the combined bids for final payment calculation
                    self.profit_max_bids_unraised_specific_round[profit_max_bids_number] = copy.deepcopy(total_bids_unraised)  # store a copy of the combined bids for final payment calculation
                        
                allocation_profit_max_unraised, true_value_profit_max_unraised, inferred_value_profit_max_unraised, details = self.solve_WDP(total_bids_unraised, self.MIP_parameters, verbose=1) 
                self.number_of_mips_solved['WDP'] += 1

                efficiency_profit_max_unraised = self.calculate_efficiency_of_allocation(allocation=allocation_profit_max_unraised, allocation_scw=true_value_profit_max_unraised)
                self.efficiency_per_iteration_profit_max_unraised_specific_round[current_clock_round].append(efficiency_profit_max_unraised)
                self.scw_per_iteration_profit_max_unraised_specific_round[current_clock_round].append(true_value_profit_max_unraised)
                
                end = datetime.now()
                logging.info(f'Current efficiency (profit max bids (unraised clock bids), {profit_max_bids_number}: {efficiency_profit_max_unraised}')
                logging.info(f'Calculating both WDPs for {profit_max_bids_number} profit-max bids took {end-start} seconds')


        return efficiency
    

    def set_ML_parameters(self,
                          parameters
                          ):

        logging.debug('Set ML parameters')
        self.ML_parameters = OrderedDict(parameters)

    def set_MIP_parameters(self,
                           parameters
                           ):

        logging.debug('Set MIP parameters')
        self.MIP_parameters = parameters

    def set_initial_dqs(self,
                         method):  

        logging.info('INITIALIZE BIDS')
        logging.info('-----------------------------------------------')
        
        if method == 'random':
            logging.warning(f'Initial bids method: RANDOM UNIFORM')
            self.elicited_dqs = init_demand_queries_mlca_unif(SATS_auction_instance = self.SATS_auction_instance,
                                                            number_initial_bids= self.Qinit,
                                                            max_linear_prices= {k:self.TRAIN_parameters[k]['max_linear_prices_multiplier']*v for k,v in self.max_linear_prices.items()},
                                                            seed= self.SATS_auction_instance_seed,
                                                            include_null_price = True, 
                                                            bidder_id = None   #if not None: only generate data for the present bidder ids 
                                                            )
        elif method == 'increasing':
            logging.warning(f'Initial bids method: INCREASING')
            self.elicited_dqs = init_demand_queries_mlca_increasing(SATS_auction_instance = self.SATS_auction_instance,
                                                                    number_initial_bids= self.Qinit,
                                                                    start_linear_item_prices= self.TRAIN_parameters[self.bidder_names[0]]['start_linear_item_prices'], # NOTE: same for all bidders
                                                                    end_linear_item_prices=self.TRAIN_parameters[self.bidder_names[0]]['end_linear_item_prices'], # NOTE: same for all bidders
                                                                    bidder_id = None   #if not None: only generate data for the present bidder ids
                                                                    )
        elif method == 'cca':
            logging.warning(f'Initial bids method: CCA')
            self.elicited_dqs = init_demand_queries_mlca_cca(SATS_auction_instance = self.SATS_auction_instance,
                                                            capacities= self.good_capacities,
                                                            number_initial_bids= self.Qinit,
                                                            start_linear_item_prices = self.mechanism_parameters['cca_start_linear_item_prices'] * self.mechanism_parameters['cca_initial_prices_multiplier'],
                                                            price_increment = self.mechanism_parameters['cca_increment'],
                                                            include_null_price = True
                                                            )
        else: 
            raise ValueError(f'Initial bids method {method} not recognized')
    
        

    def reset_ML_models(self):
        delattr(self, 'ML_models')
        self.ML_models = OrderedDict(list((bidder_id, None) for bidder_id in self.bidder_ids))


    def solve_SATS_auction_instance(self):
        self.SATS_auction_instance_allocation, self.SATS_auction_instance_scw = self.SATS_auction_instance.get_efficient_allocation()


    def estimation_step(self):
        if self.mechanism_parameters['new_query_option'] == 'cca':
            # if the new query method is the CCA -> next price vector does not depend on ML models, 
            # thus we do not need to train them and can completely skip this step. 
            logging.info('ESTIMATION STEP: CCA')
            logging.info('-----------------------------------------------')
            logging.info('skipping estimation step as ML models are not needed')
            return
        
        elif self.mechanism_parameters['new_query_option'] == 'gd_linear_prices_on_W_v3_cheating':
            # if the new query method is the CCA -> next price vector does not depend on ML models, 
            # thus we do not need to train them and can completely skip this step. 
            logging.info('ESTIMATION STEP: GD on W - CHEATING')
            logging.info('-----------------------------------------------')
            logging.info('skipping estimation step as ML models are not needed')
            return

        start_estimation = datetime.now()
        logging.info('ESTIMATION STEP: main economy')
        logging.info('-----------------------------------------------')

        training_seeds = self.update_nn_training_seeds(number_of_seeds=len(self.bidder_names))

        # if we have dynamic scaling: Set the scales in each round equal to the maximum inferred value of that bidder times the scale multiplier
        if self.dynamic_scaling: 
            new_scales = {}
            for bidder in self.bidder_names:
                inferred_values = self.inferred_bids[bidder][1]
                max_inferred_value = np.max(inferred_values)
                if max_inferred_value > 0:
                    new_scales[bidder] = max_inferred_value * self.TRAIN_parameters[bidder]['scale_multiplier']
                else :
                    new_scales[bidder] = self.scales[bidder]  # if the max inferred value is 0, keep the old scale
                self.TRAIN_parameters[bidder]['scale'] = new_scales[bidder]

            for bidder in self.bidder_names:
                logging.info(f'Ratio of new scale to old scale for bidder {bidder}: {new_scales[bidder] / self.scales[bidder]}')
            self.scales = new_scales


        # OPTION 1: PARALLEL TRAINING
        # NEW VERSION FOR MLCA DQ 
        # ----------------------------------------------------------------------------
        if self.parallelize_training:
            pool_args = self.bidder_ids
            m = Parallel(n_jobs=-1)(
                    delayed(
                        partial(dq_train_mvnn_parallel,
                            capacity_generic_goods=self.good_capacities,
                            elicited_dqs=self.elicited_dqs,   # NOTE: Parallel version works with unscaled data. 
                            dqs_val_data=None, # Must be None for now
                            scales=self.scales,
                            SATS_parameters=self.SATS_parameters,
                            TRAIN_parameters=self.TRAIN_parameters,
                            MVNN_parameters=self.MVNN_parameters,
                            MIP_parameters=self.MIP_parameters,
                            GSVM_national_bidder_goods_of_interest=self.SATS_parameters['GSVM_national_bidder_goods_of_interest'],
                            wandb_tracking=False,
                            num_cpu_per_job=1
                        )
                    )
                    (bidder_id) for bidder_id in pool_args
                )
                                    
            models = self.merge_dicts(m)
        else:
            # OPTION 2: SEQUENTIAL TRAINING
            # ----------------------------------------------------------------------------
            # NEW VERSION FOR MLCA DQ 
            models = {}
            for bidder, seed in zip(self.bidder_names, training_seeds):
                logging.info(f"Estimation Step for {bidder}")
                train_start_time = time.time()
                model, train_metrics = dq_train_mvnn(SATS_auction_instance = self.SATS_auction_instance,
                                                    capacity_generic_goods = self.good_capacities,
                                                    P_train = self.elicited_dqs[bidder][1] / self.scales[bidder],  # NOTE: method requires scaled P_train! scale the data to the range [0,1]
                                                    X_train = self.elicited_dqs[bidder][0],
                                                    P_val = None,  
                                                    X_val = None,
                                                    P_val_gen_only= None,
                                                    X_val_gen_only= None,
                                                    SATS_parameters = self.SATS_parameters,
                                                    TRAIN_parameters = self.TRAIN_parameters[bidder],
                                                    MVNN_parameters = self.MVNN_parameters[bidder],
                                                    MIP_parameters = self.MIP_parameters,
                                                    bidder_id = key_to_int(bidder),  
                                                    bidder_scale = self.scales[bidder],
                                                    GSVM_national_bidder_goods_of_interest = self.SATS_parameters['GSVM_national_bidder_goods_of_interest'],  
                                                    wandb_tracking = False)
                train_end_time = time.time()
                train_metrics["train_time_elapsed"] = train_end_time - train_start_time
                logging.info(f'Training time for bidder {bidder}: {train_metrics["train_time_elapsed"]}')
                # {bidder: [mvnn, train_logs]}   # original format from MLCA -> we keep the same format 
                models[bidder] = [model, train_metrics] 



        # Add train metrics
        trained_models = {}
        if not self.train_logs.get(self.mlca_iteration):
            self.train_logs[self.mlca_iteration] = OrderedDict(
                list(('Bidder_{}'.format(bidder_id), []) for bidder_id in self.bidder_ids))

        for bidder_id, (model, train_logs) in models.items():
            trained_models[bidder_id] = model
            self.train_logs[self.mlca_iteration][bidder_id] = train_logs
            # total time for ml training
            self.total_time_ml_train += train_logs["train_time_elapsed"]

        self.ML_models = trained_models

        end_estimation = datetime.now()
        logging.info('Elapsed Time: {}d {}h:{}m:{}s\n'.format(*timediff_d_h_m_s(end_estimation - start_estimation)))
        return

    def update_nn_training_seeds(self,
                                 number_of_seeds
                                 ):

        training_seeds = list(range(self.nn_seed, self.nn_seed + number_of_seeds))
        self.nn_seed += number_of_seeds  # update

        return training_seeds


    def generate_dq(self):
        logging.info('GENERATE DQs FOR ALL BIDDERS')
        logging.info(f'DQ method: {self.mechanism_parameters["new_query_option"]}')
        logging.info('-----------------------------------------------\n')

        start = datetime.now()
        if self.mechanism_parameters["new_query_option"] in ["gd_linear_prices_on_W", "gd_linear_prices_on_W_v2", "gd_linear_prices_on_W_v3" ]:
            scale_list = [] # list of scales for each bidder
            bidder_models = [] # list of tuples (bidder_id, model)
            
            # create list of bidder models and scales for W minimization
            for bidder_name in self.elicited_dqs.keys():
                P = self.elicited_dqs[bidder_name][1]
                scale_list.append(self.scales[bidder_name])
                bidder_models.append((key_to_int(bidder_name), self.ML_models[bidder_name]))


            # get the price vector that minimizes the W -> this is the price vector with the lowest CE
            if self.mechanism_parameters["new_query_option"] == "gd_linear_prices_on_W":
                price_vector, predicted_ce, mips_solved = minimize_W(bidder_models, 
                                                                    initial_price_vector = P[0], 
                                                                    capacities = self.good_capacities,
                                                                    scale = scale_list, 
                                                                    SATS_domain = self.SATS_parameters['SATS_domain'],
                                                                    GSVM_national_bidder_goods_of_interest = self.SATS_parameters['GSVM_national_bidder_goods_of_interest'],
                                                                    W_epochs = self.mechanism_parameters['W_epochs'],
                                                                    lr = self.mechanism_parameters['W_lr'],
                                                                    lr_decay = self.mechanism_parameters['W_lr_decay'],
                                                                    MIP_parameters = self.MIP_parameters,
                                                                    )
            elif self.mechanism_parameters["new_query_option"] == "gd_linear_prices_on_W_v2":
                # Start from a price vector where the price of each item is drawn uniformly from [0, 2]* previous round price
                starting_price_vector = np.random.uniform(low = 0, high = 2 * P[-1])
                
                price_vector, predicted_ce, mips_solved = minimize_W_v2(bidder_models, 
                                                                    starting_price_vector = starting_price_vector, # does not start from 0 prices, but instead from the prices of the last iteration. 
                                                                    capacities = self.good_capacities,
                                                                    scale = scale_list, 
                                                                    SATS_domain = self.SATS_parameters['SATS_domain'],
                                                                    GSVM_national_bidder_goods_of_interest = self.SATS_parameters['GSVM_national_bidder_goods_of_interest'],
                                                                    max_steps_without_improvement = self.mechanism_parameters['W_v2_max_steps_without_improvement'],
                                                                    lr = self.mechanism_parameters['W_v2_lr'],
                                                                    lr_decay = self.mechanism_parameters['W_v2_lr_decay'],
                                                                    MIP_parameters = self.MIP_parameters,
                                                                    )
                
            elif self.mechanism_parameters["new_query_option"] == "gd_linear_prices_on_W_v3":  
                # Now start from a price vector where the price of each item is drawn uniformly from [0.75, 1.25] * previous round price
                
                starting_price_vector = np.random.uniform(low = 0.9 * P[self.Qinit - 1], high = 1.1 * P[self.Qinit - 1])
                
                price_vector, predicted_ce, mips_solved, all_Ws, all_CEs, all_CEs_norm_1, is_feasible = minimize_W_v3(bidder_models, 
                                                                    starting_price_vector = starting_price_vector, 
                                                                    capacities = self.good_capacities,
                                                                    scale = scale_list, 
                                                                    SATS_domain = self.SATS_parameters['SATS_domain'],
                                                                    GSVM_national_bidder_goods_of_interest = self.SATS_parameters['GSVM_national_bidder_goods_of_interest'],
                                                                    max_steps_without_improvement = self.mechanism_parameters['W_v3_max_steps_without_improvement'],
                                                                    max_steps = self.mechanism_parameters['W_v3_max_steps'],
                                                                    lr = self.mechanism_parameters['W_v3_lr'],
                                                                    lr_decay = self.mechanism_parameters['W_v3_lr_decay'],
                                                                    MIP_parameters = self.MIP_parameters,
                                                                    filter_feasible= self.mechanism_parameters['W_v3_filter_feasible'], 
                                                                    feasibility_multiplier= self.mechanism_parameters['W_v3_feasibility_multiplier'],
                                                                    feasibility_multiplier_increase_factor= self.mechanism_parameters['W_v3_feasibility_multiplier_increase_factor']
                                                                    )
                
                # calculate the relative change for each item compared to the price vector of all previous rounds 
                self.perturbed_prices  = False 
                previous_price_vectors = self.elicited_dqs['Bidder_0'][1][1:]  # so that we exclude the prive vector at zero prices. 
                absolute_differences = np.abs(previous_price_vectors - price_vector)
                relative_differences = absolute_differences / previous_price_vectors
                relative_differences_l_infty = np.max(relative_differences, axis = 1)
                price_vector_copy = price_vector.copy()
                self.feasible_allocation = is_feasible

                
                iteration_threshold = self.mechanism_parameters['identical_p_threshold'] *  (self.mechanism_parameters['identical_p_threshold_decay'] ** self.mlca_iteration) 
                # as the rounds progress, the prices can be closer together without perturbing

                randomness_scale = iteration_threshold / 2
                while min(relative_differences_l_infty) < iteration_threshold:
                    self.perturbed_prices = True
                    logging.info(f'Not enough change in prices, perturbing...')
                    price_vector = np.random.uniform( low = max(0,(1 - randomness_scale)) * price_vector_copy, high = (1 + randomness_scale) * price_vector_copy)
                    absolute_differences = np.abs(previous_price_vectors - price_vector)
                    relative_differences = absolute_differences / previous_price_vectors
                    relative_differences_l_infty = np.max(relative_differences, axis = 1)

                    randomness_scale = randomness_scale * 1.1

                
                
                if (self.mlca_iteration % 10 == 1) and self.wandb_tracking: # we start counting iterations from 1, not form 0. 
                    wandb.define_metric(f'W per GD step iteration {self.mlca_iteration}',  step_metric="GD Step")
                    wandb.define_metric(f'CE per GD step iteration {self.mlca_iteration}', step_metric="GD Step")
                    wandb.define_metric(f'CE norm 1 per GD step iteration {self.mlca_iteration}', step_metric="GD Step")

                    for i in range(len(all_Ws)):
                        wandb.log({f'W per GD step iteration {self.mlca_iteration}': all_Ws[i], f'CE per GD step iteration {self.mlca_iteration}': all_CEs[i], f'CE norm 1 per GD step iteration {self.mlca_iteration}': all_CEs_norm_1[i], "GD Step": i})
                
            self.number_of_mips_solved['Price Generation'] += mips_solved

            # Check that the price vector has not been used before
            if np.any(np.all(self.elicited_dqs['Bidder_0'][1] == price_vector, axis=1)):
                logging.info('Price vector already used, perturbing...')
                price_vector = price_vector + 0.01 * np.random.rand(price_vector.shape[0])
                self.identical_price_vector_per_iteration[self.mlca_iteration] = True
            else: 
                self.identical_price_vector_per_iteration[self.mlca_iteration] = False

        elif self.mechanism_parameters['new_query_option'] == 'cca': 

            m = len(self.SATS_auction_instance.get_good_ids())
            bidder_ids = self.SATS_auction_instance.get_bidder_ids()
            predicted_ce = 0 

            # get the last price vector
            price_vector = self.elicited_dqs['Bidder_0'][1][-1].copy()
            # set_trace()
            
            # get the total demand at the last price vector 
            total_demand = np.zeros(m)

            for bidder_id in bidder_ids:
                total_demand += self.elicited_dqs[f'Bidder_{bidder_id}'][0][-1]

            # get the new price vector
            over_demand = total_demand - self.good_capacities
            price_increment = self.mechanism_parameters['cca_increment']

            for j in range(m):
                if over_demand[j] > 0:
                    price_vector[j] = price_vector[j] * (1 + price_increment)

            # Check that the price vector has not been used before
            if np.any(np.all(self.elicited_dqs['Bidder_0'][1] == price_vector, axis=1)):
                logging.info('Price vector already used...')
                self.identical_price_vector_per_iteration[self.mlca_iteration] = True
            else: 
                self.identical_price_vector_per_iteration[self.mlca_iteration] = False

        
        elif self.mechanism_parameters["new_query_option"] == "gd_linear_prices_on_W_v3_cheating":  
            # Now start from a price vector where the price of each item is drawn uniformly from [0.9, 1.1] * the final CCA round price. 
            bidder_name = list(self.elicited_dqs.keys())[0]
            P = self.elicited_dqs[bidder_name][1]
            starting_price_vector = np.random.uniform(low = 0.9 * P[self.Qinit - 1], high = 1.1 * P[self.Qinit - 1])
            
            price_vector, predicted_ce, mips_solved, all_Ws, all_CEs, all_CEs_norm_1 = minimize_W_cheating(self.SATS_auction_instance, 
                                                                self.bidder_ids, 
                                                                starting_price_vector = starting_price_vector, 
                                                                capacities = self.good_capacities,
                                                                max_steps_without_improvement = self.mechanism_parameters['W_v3_max_steps_without_improvement'],
                                                                max_steps = self.mechanism_parameters['W_v3_max_steps'],
                                                                lr = self.mechanism_parameters['W_v3_lr'],
                                                                lr_decay = self.mechanism_parameters['W_v3_lr_decay'],
                                                                )
            # price_vector, predicted_ce, mips_solved, all_Ws, all_CEs = minimize_W_v3(bidder_models, starting_price_vector = starting_price_vector, capacities = self.good_capacities,scale = scale_list, SATS_domain = self.SATS_parameters['SATS_domain'],GSVM_national_bidder_goods_of_interest = self.SATS_parameters['GSVM_national_bidder_goods_of_interest'],max_steps_without_improvement = self.mechanism_parameters['W_v3_max_steps_without_improvement'],lr = self.mechanism_parameters['W_v3_lr'],lr_decay = self.mechanism_parameters['W_v3_lr_decay'],MIP_parameters = self.MIP_parameters,)
            
            if (self.mlca_iteration % 10 == 1) and self.wandb_tracking: # we start counting iterations from 1, not form 0. 
                wandb.define_metric(f'W per GD step iteration {self.mlca_iteration}',  step_metric="GD Step")
                wandb.define_metric(f'CE per GD step iteration {self.mlca_iteration}', step_metric="GD Step")
                wandb.define_metric(f'CE norm 1 per GD step iteration {self.mlca_iteration}', step_metric="GD Step")

                for i in range(len(all_Ws)):
                    wandb.log({f'W per GD step iteration {self.mlca_iteration}': all_Ws[i], f'CE per GD step iteration {self.mlca_iteration}': all_CEs[i], f'CE norm 1 per GD step iteration {self.mlca_iteration}': all_CEs_norm_1[i], "GD Step": i})
               
            # Check that the price vector has not been used before
            if np.any(np.all(self.elicited_dqs['Bidder_0'][1] == price_vector, axis=1)):
                logging.info('Price vector already used...')
                self.identical_price_vector_per_iteration[self.mlca_iteration] = True
            else: 
                self.identical_price_vector_per_iteration[self.mlca_iteration] = False


        else: 
            raise ValueError(f'Unknown new query option: {self.mechanism_parameters["new_query_option"]}')

            
        # ask each bidder the demand query for that price vector 
        real_demand = np.zeros(price_vector.shape[0])
        real_demand_vector = [] 
        for bidder_id in self.bidder_ids:
            try:
                demand_response = np.array(self.SATS_auction_instance.get_best_bundles(bidder_id, price_vector, 1, allow_negative = True)[0]) # convert to np array
            except:
                demand_response = self.elicited_dqs[f'Bidder_{bidder_id}'][0][-1].copy() # if the bidder does not respond, use the last demand response

            real_demand += demand_response
            real_demand_vector.append(demand_response)
        
            # update the elicited_dqs with the new price vector and demand response
            self.elicited_dqs[f'Bidder_{bidder_id}'][1] = np.concatenate((self.elicited_dqs[f'Bidder_{bidder_id}'][1], price_vector.reshape(1, -1)), axis=0)
            self.elicited_dqs[f'Bidder_{bidder_id}'][0] = np.concatenate((self.elicited_dqs[f'Bidder_{bidder_id}'][0], demand_response.reshape(1, -1)), axis=0)

        
        over_demand = real_demand - self.good_capacities
        real_ce = np.sum(over_demand**2)
        print(f'Predicted CE: {predicted_ce}  Real CE: {real_ce}')
        self.clearing_error_per_iteration[self.mlca_iteration] = real_ce
        self.predicted_clearing_error_per_iteration[self.mlca_iteration] = predicted_ce
        self.price_vector_per_iteration[self.mlca_iteration] = price_vector
        self.demand_vector_per_iteration[self.mlca_iteration] = real_demand_vector

        end = datetime.now()
        self.total_time_price_vector_generation += (end - start).total_seconds()


        if real_ce == 0:
            logging.info('Found clearing prices!')
            self.found_clearing_prices = True
            self.found_clearing_prices_per_iteration[self.mlca_iteration] = True 
        else:
            self.found_clearing_prices_per_iteration[self.mlca_iteration] = False 

        return 



    def calculate_final_allocation(self):

        '''
        Calculate the final allocation of the mechanism
        '''

        logging.info('')
        logging.info('CALCULATE FINAL WDP ALLOCATION')
        logging.info('---------------------------------------------')
        logging.info('')
        
        self.calculate_inferred_bids()    # transform the demand responses of the agents into inferred bids

        start = datetime.now()
        allocation, true_value, inferred_value, details = self.solve_WDP(self.inferred_bids, self.MIP_parameters, verbose=1)  # find the efficient allocation with respect to the inferred bids
        self.number_of_mips_solved['WDP'] += 1

        end = datetime.now()
        self.total_time_wdp_mip += (end - start).total_seconds()
        
        # calculate the value of the resulting allocation with respect to the true values of the agents 
        #self.allocation_per_iteration[self.mlca_iteration] = allocation
        efficiency = self.calculate_efficiency_of_allocation(allocation=allocation, allocation_scw=true_value)  
        
        # Save the allocation and the social welfare of the final allocation
        self.final_allocation = allocation
        self.final_allocation_scw = true_value
        self.final_allocation_efficiency = efficiency


        if self.mechanism_parameters['calculate_raised_bids']:
            self.raise_inferred_bids()  # raise the inferred bids to the true values of the agents
            start = datetime.now()
            allocation_raised_bids, true_value_raised_bids, inferred_value_raised_bids, details = self.solve_WDP(self.inferred_bids_raised, self.MIP_parameters, verbose=1)  # find the efficient allocation with respect to the raised bids
            self.number_of_mips_solved['WDP'] += 1
            end = datetime.now()
            self.total_time_wdp_mip += (end - start).total_seconds()
            efficiency_raised_bids = self.calculate_efficiency_of_allocation(allocation=allocation_raised_bids, allocation_scw=true_value_raised_bids)

            self.final_allocation_raised_bids = allocation_raised_bids
            self.final_allocation_scw_raised_bids= true_value_raised_bids
            self.final_allocation_efficiency_raised_bids = efficiency_raised_bids

        if self.mechanism_parameters['calculate_profit_max_bids']: 
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']: 
                profit_max_bids = self.profit_max_bids_combined[number_of_profit_max_bids]
                start = datetime.now()
                allocation_profit_max_bids, true_value_profit_max_bids, inferred_value_profit_max_bids, details = self.solve_WDP(profit_max_bids, self.MIP_parameters, verbose=1)  # find the efficient allocation with respect to the (raised) bids of the clock round + profit-max bids
                self.number_of_mips_solved['WDP'] += 1
                end = datetime.now()
                self.total_time_wdp_mip += (end - start).total_seconds()
                efficiency_profit_max_bids = self.calculate_efficiency_of_allocation(allocation=allocation_profit_max_bids, allocation_scw=true_value_profit_max_bids)

                self.final_allocation_profit_max[number_of_profit_max_bids] = allocation_profit_max_bids
                self.final_allocation_scw_profit_max[number_of_profit_max_bids] = true_value_profit_max_bids
                self.final_allocation_efficiency_profit_max[number_of_profit_max_bids] = efficiency_profit_max_bids

        if self.mechanism_parameters['calculate_profit_max_bids_unraised']: 
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']: 
                profit_max_bids_unraised = self.profit_max_bids_combined_unraised[number_of_profit_max_bids]
                start = datetime.now()
                allocation_profit_max_bids_unraised, true_value_profit_max_bids_unraised, inferred_value_profit_max_bids_unraised, details = self.solve_WDP(profit_max_bids_unraised, self.MIP_parameters, verbose=1)  # find the efficient allocation with respect to the (raised) bids of the clock round + profit-max bids
                self.number_of_mips_solved['WDP'] += 1
                end = datetime.now()
                self.total_time_wdp_mip += (end - start).total_seconds()
                efficiency_profit_max_bids_unraised = self.calculate_efficiency_of_allocation(allocation=allocation_profit_max_bids_unraised, allocation_scw=true_value_profit_max_bids_unraised)

                self.final_allocation_profit_max_unraised[number_of_profit_max_bids] = allocation_profit_max_bids_unraised
                self.final_allocation_scw_profit_max_unraised[number_of_profit_max_bids] = true_value_profit_max_bids_unraised
                self.final_allocation_efficiency_profit_max_unraised[number_of_profit_max_bids] = efficiency_profit_max_bids_unraised

        return 



    def calculate_allocation_value(self, allocation):
        """
        Given an allocation in the format returned by solve_WDP, calculates its value.
        Additionally, it sets the true value of each bidder's allocated bundle in the allocation dictionary.
        """

        allocation_transformed = {} # dictionary that will store the allocation in a more convenient format
        allocation_value = 0 
        for bidder_name in allocation.keys():
            allocation_transformed[bidder_name] = {}
            inferred_value = allocation[bidder_name]['value']
             # NOTE: value can lead to bugs. The allocation dictionary will store the true value of the bundle in the key 'true_value', and inferred in 'inferred_value'

            if not self.generic_domain:
                allocated_bundle_indices  = allocation[bidder_name]['good_ids']
                allocated_bundle = [0 for i in range(len(self.SATS_auction_instance.get_good_ids()))]
                for j in allocated_bundle_indices:
                    allocated_bundle[j] = 1
            else: 
               allocated_bundle = allocation[bidder_name]['allocated_bundle']
               allocated_bundle_indices = np.where(allocated_bundle >= 1)[0] # NOTE: commpletely irrelevant field for generic domains


            true_bundle_value = self.SATS_auction_instance.calculate_value(key_to_int(bidder_name), allocated_bundle)
            logging.info(f'True value of {bidder_name}\'s bundle is {true_bundle_value} while inferred value is {inferred_value}')

            # update the allocation dictionary with the true value
            allocation_transformed[bidder_name]['inferred_value'] = inferred_value
            allocation_transformed[bidder_name]['true_value'] = true_bundle_value
            allocation_transformed[bidder_name]['good_ids'] = allocated_bundle_indices
            allocation_transformed[bidder_name]['allocated_bundle'] = allocated_bundle

            allocation_value += true_bundle_value

        return allocation_value, allocation_transformed
    
    
    def make_bids_unique(self, elicited_bids):
        '''
        Makes bids unique by adding a small random value to each bid
        '''
        elicited_bids_unique = {}
        for bidder_name, bids in elicited_bids.items():
            X, V = elicited_bids[bidder_name]
    
    
            X_unique = np.unique(X, axis=0)  # get all unique Xs
            
            V_unique = np.array([np.max(V[np.where((X == X_unique[i]).all(axis = 1))[0]]) for i in range(X_unique.shape[0])]) # tell Haskell I said hi. 


            elicited_bids_unique[bidder_name] = [X_unique, V_unique]

        return elicited_bids_unique

    def solve_WDP(self,
                  elicited_bids,
                  MIP_parameters, 
                  verbose=0 
                  ):

        '''
        REMARK: objective always rescaled to true original values
        '''

        elicited_bids = self.make_bids_unique(elicited_bids)
        bidder_names = list(elicited_bids.keys())
        if verbose == 1: logging.debug('Solving WDP based on elicited bids for bidder: %s', bidder_names)
        
        if not self.generic_domain:
            elicited_bundle_value_pairs = [np.concatenate((bids[0], np.asarray(bids[1]).reshape(-1, 1)), axis=1) for
                                        bidder, bids in
                                        elicited_bids.items()]  # transform self.elicited_bids into format for WDP class
            wdp = MLCA_DQ_WDP(elicited_bundle_value_pairs, MIP_parameters)
            wdp.initialize_mip(verbose=0)
            wdp.solve_mip(verbose)
            inferred_value = wdp.Mip.objective_value  # the value of the allocation with respect to the inferred bids, needed for VCG payments
            
            details = wdp.get_solve_details()   # get the details of the solved MIP


            allocation = format_solution_mip_new(Mip=wdp.Mip,
                                            elicited_bids=elicited_bundle_value_pairs,
                                            bidder_names=bidder_names,
                                            fitted_scaler=self.fitted_scaler,
                                            generic_domain=False)
        else:
            elicited_bundle_value_pairs = [np.concatenate((bids[0], np.asarray(bids[1]).reshape(-1, 1)), axis=1) for
                                        bidder, bids in
                                        elicited_bids.items()]  # transform self.elicited_bids into format for WDP class
            wdp = MLCA_DQ_WDP_GENERIC(elicited_bundle_value_pairs, MIP_parameters, capacity_generic_items= self.good_capacities)
            wdp.initialize_mip(verbose=0)
            wdp.solve_mip(verbose)
            inferred_value = wdp.Mip.objective_value  # the value of the allocation with respect to the inferred bids, needed for VCG payments
            
            details = wdp.get_solve_details()   # get the details of the solved MIP

            allocation = format_solution_mip_new(Mip=wdp.Mip,
                                            elicited_bids=elicited_bundle_value_pairs,
                                            bidder_names=bidder_names,
                                            fitted_scaler=self.fitted_scaler,
                                            generic_domain=self.generic_domain)          


        allocation_value, allocation_transformed = self.calculate_allocation_value(allocation) # calculate the true value of the allocation (not the inferred value)
        
        if allocation_value < inferred_value - 0.1: 
            print('WARNING: allocation value is smaller than inferred value. This should not happen. Check the code!!!')
            # set_trace()
            raise ValueError
        return (allocation_transformed, allocation_value, inferred_value, details)


    def calculate_efficiency_of_allocation(self,
                                           allocation,
                                           allocation_scw,
                                           verbose=0
                                           ):

        self.solve_SATS_auction_instance()
        efficiency = allocation_scw / self.SATS_auction_instance_scw
        if verbose == 1:
            logging.debug('Calculating efficiency of input allocation. Inferred bids are:')
            for key, value in allocation.items():
                logging.debug('%s %s', key, value['good_ids'])  # note: the efficiency is wrt. the true value of this allocation, not the inferred values. 
            logging.debug('Social Welfare: %s', allocation_scw)
            logging.debug('Efficiency of allocation: %s', efficiency)
        return (efficiency)
    

    def calculate_clearing_allocation(self, demand_vector, price_vector, is_final_allocation):
        """
        This function will be called in case a clearing price vector is found. 
        It will convert the demand vector into an appropriate allocation dictionary.
        """
        allocation = {}
        total_value = 0
        total_inferred_value = 0
        # build the allocation object
        for bidder_id in range(len(self.bidder_names)):
            allocation[f'Bidder_{bidder_id}'] = {'good_ids': [i for i in range(len(demand_vector[bidder_id])) if demand_vector[bidder_id][i] == 1]}
            allocation[f'Bidder_{bidder_id}']['value'] = np.dot(demand_vector[bidder_id], price_vector)  # just for bookkeeping, not used
            allocation[f'Bidder_{bidder_id}']['inferred_value'] = np.dot(demand_vector[bidder_id], price_vector)
            allocation[f'Bidder_{bidder_id}']['allocated_bundle'] = demand_vector[bidder_id]
            total_inferred_value += allocation[f'Bidder_{bidder_id}']['inferred_value'] 
            true_value = self.SATS_auction_instance.calculate_value(bidder_id, demand_vector[bidder_id])
            allocation[f'Bidder_{bidder_id}']['true_value'] = true_value
            total_value += true_value

        # Update the per iteration attributes 
        self.scw_per_iteration[self.mlca_iteration] = total_value
        self.inferred_scw_per_iteration[self.mlca_iteration] = total_inferred_value
        self.allocation_per_iteration[self.mlca_iteration] = allocation
        efficiency = self.calculate_efficiency_of_allocation(allocation=allocation, allocation_scw=total_value)  
        logging.warning(f'Found clearing allocation. Final efficiency: {efficiency}')
        self.efficiency_per_iteration[self.mlca_iteration] = efficiency
        self.MIP_relative_gap_per_iteration[self.mlca_iteration] = 0
        self.MIP_time_per_iteration[self.mlca_iteration] = 0
        self.MIP_unsatisfied_constraints_per_iteration[self.mlca_iteration] = 0

        # in case the market clears: The auction stops, so the clearing allocation is also 
        # the allocation for the supplementary round heuristics. 
        if self.mechanism_parameters['calculate_raised_bids']:
            self.efficiency_per_iteration_raised_bids[self.mlca_iteration] = efficiency
            self.MIP_relative_gap_per_iteration_raised_bids[self.mlca_iteration] = 0
            self.MIP_time_per_iteration_raised_bids[self.mlca_iteration] = 0
            self.MIP_unsatisfied_constraints_per_iteration_raised_bids[self.mlca_iteration] = 0
            self.scw_per_iteration_raised_bids[self.mlca_iteration] = total_value

        if self.mechanism_parameters['calculate_profit_max_bids']:
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                self.efficiency_per_iteration_profit_max[number_of_profit_max_bids][self.mlca_iteration] = efficiency
                self.scw_per_iteration_profit_max[number_of_profit_max_bids][self.mlca_iteration] = total_value

        if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                self.efficiency_per_iteration_profit_max_unraised[number_of_profit_max_bids][self.mlca_iteration] = efficiency
                self.scw_per_iteration_profit_max_unraised[number_of_profit_max_bids][self.mlca_iteration] = total_value

        current_clock_round = self.mlca_iteration + self.Qinit
        if current_clock_round in self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']:
            self.efficiency_per_iteration_profit_max_specific_round[current_clock_round] = [self.efficiency_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]  
            self.efficiency_per_iteration_profit_max_unraised_specific_round[current_clock_round] = [self.efficiency_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]

            self.MIP_relative_gap_per_iteration_profit_max_specific_round[current_clock_round] = [0 for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
            self.MIP_time_per_iteration_profit_max_specific_round[current_clock_round] = [0 for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
            self.MIP_unsatisfied_constraints_per_iteration_profit_max_specific_round[current_clock_round] = [0 for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]

            self.scw_per_iteration_profit_max_specific_round[current_clock_round] = [self.scw_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]
            self.scw_per_iteration_profit_max_unraised_specific_round[current_clock_round] = [self.scw_per_iteration[self.mlca_iteration] for _ in range(self.mechanism_parameters['profit_max_grid'][-1] + 1)]

        
        # if this is also the final allocation, store it in the class
        if is_final_allocation:
            self.final_allocation_efficiency = efficiency
            self.final_allocation = allocation
            self.final_allocation_scw = total_value
            self.revenue = total_inferred_value # this is under the assumption that we charge the clearing prices in case the market cleared
            self.relative_revenue = self.revenue / self.SATS_auction_instance_scw 
            # also set clearing payment (note they equal 'inferred values calculated above')
            for bidder_name in self.bidder_names: 
                self.clearing_payments[bidder_name] = self.final_allocation[bidder_name]['inferred_value']

            if self.mechanism_parameters['calculate_raised_bids']:
                self.final_allocation_efficiency_raised_bids = efficiency
                self.final_allocation_raised_bids = allocation
                self.final_allocation_scw_raised_bids = total_value
                self.revenue_raised_bids = np.sum(price_vector)
                self.relative_revenue_raised_bids = self.revenue_raised_bids / self.SATS_auction_instance_scw
                # also set clearing payment (note they equal 'inferred values calculated above')
                for bidder_name in self.bidder_names: 
                    self.clearing_payments_raised_bids[bidder_name] = self.final_allocation_raised_bids[bidder_name]['inferred_value']

            if self.mechanism_parameters['calculate_profit_max_bids']:
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    self.final_allocation_efficiency_profit_max[number_of_profit_max_bids] = efficiency
                    self.final_allocation_profit_max[number_of_profit_max_bids] = allocation
                    self.final_allocation_scw_profit_max[number_of_profit_max_bids] = total_value
                    self.revenue_profit_max[number_of_profit_max_bids] = np.sum(price_vector)
                    self.relative_revenue_profit_max[number_of_profit_max_bids] = self.revenue_profit_max[number_of_profit_max_bids] / self.SATS_auction_instance_scw
                    # also set clearing payment (note they equal 'inferred values calculated above')
                    for bidder_name in self.bidder_names: 
                        self.clearing_payments_profit_max[number_of_profit_max_bids][bidder_name] = self.final_allocation_profit_max[number_of_profit_max_bids][bidder_name]['inferred_value']
            
            if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    self.final_allocation_efficiency_profit_max_unraised[number_of_profit_max_bids] = efficiency
                    self.final_allocation_profit_max_unraised[number_of_profit_max_bids] = allocation
                    self.final_allocation_scw_profit_max_unraised[number_of_profit_max_bids] = total_value
                    self.revenue_profit_max_unraised[number_of_profit_max_bids] = np.sum(price_vector)
                    self.relative_revenue_profit_max_unraised[number_of_profit_max_bids] = self.revenue_profit_max_unraised[number_of_profit_max_bids] / self.SATS_auction_instance_scw
                    # also set clearing payment (note they equal 'inferred values calculated above')
                    for bidder_name in self.bidder_names: 
                        self.clearing_payments_profit_max_unraised[number_of_profit_max_bids][bidder_name] = self.final_allocation_profit_max_unraised[number_of_profit_max_bids][bidder_name]['inferred_value']

            self.calculate_inferred_bids() # calculate the inferred bids of the final allocation for proper logging. 

        return allocation

    def calculate_vcg_payments(self):

        logging.info('')
        logging.info('CALCULATE VCG PAYMENTS')
        logging.info('---------------------------------------------')

        if self.mechanism_parameters['calculate_profit_max_bids']: 
            self.marginal_economies_allocations_profit_max = {j: {} for j in self.mechanism_parameters['profit_max_grid']}
            self.marginal_economies_scw_profit_max = {j: {} for j in self.mechanism_parameters['profit_max_grid']}
            self.marginal_economies_inferred_scw_profit_max = {j: {} for j in self.mechanism_parameters['profit_max_grid']}

        if self.mechanism_parameters['calculate_profit_max_bids_unraised']: 
            self.marginal_economies_allocations_profit_max_unraised = {j: {} for j in self.mechanism_parameters['profit_max_grid']}
            self.marginal_economies_scw_profit_max_unraised = {j: {} for j in self.mechanism_parameters['profit_max_grid']}
            self.marginal_economies_inferred_scw_profit_max_unraised = {j: {} for j in self.mechanism_parameters['profit_max_grid']}

        # (i) calculate the allocation (and apparent social welfare) of all marginal economies 
        for bidder_id in self.SATS_auction_instance.get_bidder_ids():
            print('Calculating marginal allocation without bidder: ', bidder_id)
            # Create the marginal inferred bids by removing all bids of the bidder in question 
            marginal_inferred_bids = copy.deepcopy(self.inferred_bids)
            marginal_inferred_bids[f'Bidder_{bidder_id}'][1] = np.array([0])
            marginal_inferred_bids[f'Bidder_{bidder_id}'][0] = np.array([[0 for i in range(len(self.SATS_auction_instance.get_good_ids()))]])
                    
            # calculate the marginal allocation and its apparent social welfare
            marginal_allocation, marginal_true_value, marginal_inferred_value, details = self.solve_WDP(marginal_inferred_bids, self.MIP_parameters, verbose=1)
            self.number_of_mips_solved['WDP'] += 1

            # set the corresponding values in the dictionaries for payment rules downstream 
            # the key is the bidder_id of the missing bidder from the marginal economy
            self.marginal_economies_allocations[f'Bidder_{bidder_id}'] = marginal_allocation
            self.marginal_economies_scw[f'Bidder_{bidder_id}'] = marginal_true_value
            self.marginal_economies_inferred_scw[f'Bidder_{bidder_id}'] = marginal_inferred_value

            # if we are calculating raised bids, we also need to calculate the marginal allocation for the raised bids
            if self.mechanism_parameters['calculate_raised_bids']:
                marginal_inferred_bids_raised = copy.deepcopy(self.inferred_bids_raised)
                marginal_inferred_bids_raised[f'Bidder_{bidder_id}'][1] = np.array([0])
                marginal_inferred_bids_raised[f'Bidder_{bidder_id}'][0] = np.array([[0 for i in range(len(self.SATS_auction_instance.get_good_ids()))]])

                # calculate the marginal allocation and its social welfare for the raised bids
                marginal_allocation_raised_bids, marginal_true_value_raised_bids, marginal_inferred_value_raised_bids, details = self.solve_WDP(marginal_inferred_bids_raised, self.MIP_parameters, verbose=1)
                self.number_of_mips_solved['WDP'] += 1

                self.marginal_economies_allocations_raised_bids[f'Bidder_{bidder_id}'] = marginal_allocation_raised_bids
                self.marginal_economies_scw_raised_bids[f'Bidder_{bidder_id}'] = marginal_true_value_raised_bids
                self.marginal_economies_inferred_scw_raised_bids[f'Bidder_{bidder_id}'] = marginal_inferred_value_raised_bids

            # if we are using profit max, we also need to calculate the marginal allocation for the profit max bids
            if self.mechanism_parameters['calculate_profit_max_bids']: 

                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    marginal_inferred_bids_profit_max = copy.deepcopy(self.profit_max_bids_combined[number_of_profit_max_bids])
                    marginal_inferred_bids_profit_max[f'Bidder_{bidder_id}'][1] = np.array([0])
                    marginal_inferred_bids_profit_max[f'Bidder_{bidder_id}'][0] = np.array([[0 for i in range(len(self.SATS_auction_instance.get_good_ids()))]])

                    # calculate the marginal allocation and its social welfare for the raised bids
                    marginal_allocation_profit_max_bids, marginal_true_value_profit_max_bids, marginal_inferred_value_profit_max_bids, details = self.solve_WDP(marginal_inferred_bids_profit_max, self.MIP_parameters, verbose=1)
                    self.number_of_mips_solved['WDP'] += 1

                    self.marginal_economies_allocations_profit_max[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_allocation_profit_max_bids
                    self.marginal_economies_scw_profit_max[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_true_value_profit_max_bids
                    self.marginal_economies_inferred_scw_profit_max[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_inferred_value_profit_max_bids

            # same for profit max without raising the clock bids. 
            if self.mechanism_parameters['calculate_profit_max_bids_unraised']: 

                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    marginal_inferred_bids_profit_max_unraised = copy.deepcopy(self.profit_max_bids_combined_unraised[number_of_profit_max_bids])
                    marginal_inferred_bids_profit_max_unraised[f'Bidder_{bidder_id}'][1] = np.array([0])
                    marginal_inferred_bids_profit_max_unraised[f'Bidder_{bidder_id}'][0] = np.array([[0 for i in range(len(self.SATS_auction_instance.get_good_ids()))]])

                    # calculate the marginal allocation and its social welfare for the raised bids
                    marginal_allocation_profit_max_bids_unraised, marginal_true_value_profit_max_bids_unraised, marginal_inferred_value_profit_max_bids_unraised, details = self.solve_WDP(marginal_inferred_bids_profit_max, self.MIP_parameters, verbose=1)
                    self.number_of_mips_solved['WDP'] += 1

                    self.marginal_economies_allocations_profit_max_unraised[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_allocation_profit_max_bids_unraised
                    self.marginal_economies_scw_profit_max_unraised[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_true_value_profit_max_bids_unraised
                    self.marginal_economies_inferred_scw_profit_max_unraised[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_inferred_value_profit_max_bids_unraised

            # NOTE: All this needs a conceptual change. We may have no profit max bids if the market claread and this can crush us. 
            # Should also switch to vcg-nearest core payments.             
            # # if len(self.mechanism_parameters['calculate_profit_max_bids_specific_rounds']) and not self.found_clearing_prices:  # if you've found clearing prices -> these may not be set, and do not affect revenue.
            #     round_number = self.mechanism_parameters['calculate_profit_max_bids_specific_rounds'][-1]
            #     for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
            #         marginal_inferred_bids_profit_max = copy.deepcopy(self.profit_max_bids_specific_round[number_of_profit_max_bids])
            #         marginal_inferred_bids_profit_max[f'Bidder_{bidder_id}'][1] = np.array([0])
            #         marginal_inferred_bids_profit_max[f'Bidder_{bidder_id}'][0] = np.array([[0 for i in range(len(self.SATS_auction_instance.get_good_ids()))]])

            #         # calculate the marginal allocation and its social welfare for the raised bids
            #         marginal_allocation_profit_max_bids, marginal_true_value_profit_max_bids, marginal_inferred_value_profit_max_bids = self.solve_WDP(marginal_inferred_bids_profit_max, verbose=1)
            #         self.number_of_mips_solved['WDP'] += 1

            #         self.marginal_economies_allocations_profit_max_specific_round[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_allocation_profit_max_bids
            #         self.marginal_economies_scw_profit_max_specific_round[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_true_value_profit_max_bids
            #         self.marginal_economies_inferred_scw_profit_max_specific_round[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_inferred_value_profit_max_bids

                    
            #         marginal_inferred_bids_profit_max_unraised = copy.deepcopy(self.profit_max_bids_unraised_specific_round[number_of_profit_max_bids])
            #         marginal_inferred_bids_profit_max_unraised[f'Bidder_{bidder_id}'][1] = np.array([0])
            #         marginal_inferred_bids_profit_max_unraised[f'Bidder_{bidder_id}'][0] = np.array([[0 for i in range(len(self.SATS_auction_instance.get_good_ids()))]])

            #         # calculate the marginal allocation and its social welfare for the raised bids
            #         marginal_allocation_profit_max_bids_unraised, marginal_true_value_profit_max_bids_unraised, marginal_inferred_value_profit_max_bids_unraised = self.solve_WDP(marginal_inferred_bids_profit_max, verbose=1)
            #         self.number_of_mips_solved['WDP'] += 1

            #         self.marginal_economies_allocations_profit_max_unraised_specific_round[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_allocation_profit_max_bids_unraised
            #         self.marginal_economies_scw_profit_max_unraised_specific_round[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_true_value_profit_max_bids_unraised
            #         self.marginal_economies_inferred_scw_profit_max_unraised_specific_round[number_of_profit_max_bids][f'Bidder_{bidder_id}'] = marginal_inferred_value_profit_max_bids_unraised




        # (ii) calculate VCG terms for this economy
        final_allocation = self.final_allocation
        if self.mechanism_parameters['calculate_raised_bids']:
            final_allocation_raised_bids = self.final_allocation_raised_bids
        if self.mechanism_parameters['calculate_profit_max_bids']:
            final_allocation_profit_max = self.final_allocation_profit_max
        if self.mechanism_parameters['calculate_profit_max_bids_unraised']:
            final_allocation_profit_max_unraised = self.final_allocation_profit_max_unraised

        for bidder in self.bidder_names:
            
            apparent_scw_marginal_economy  = self.marginal_economies_inferred_scw[bidder]        
            main_economy_apparent_scw_without_bidder = sum([final_allocation[i]['inferred_value'] for i in final_allocation.keys() if i != bidder])
            self.vcg_payments[bidder] = round(apparent_scw_marginal_economy - main_economy_apparent_scw_without_bidder, 2)
            logging.info(f'Payment {bidder}: {apparent_scw_marginal_economy:.2f} - {main_economy_apparent_scw_without_bidder:.2f}  =  {self.vcg_payments[bidder]:.2f}')
            print(f'Payment {bidder}: {apparent_scw_marginal_economy:.2f} - {main_economy_apparent_scw_without_bidder:.2f}  =  {self.vcg_payments[bidder]:.2f}')
            

            # if we are calculating raised bids, we also need to calculate the VCG terms for the raised bids
            if self.mechanism_parameters['calculate_raised_bids']:
                apparent_scw_marginal_economy_raised_bids  = self.marginal_economies_inferred_scw_raised_bids[bidder]        
                main_economy_apparent_scw_without_bidder_raised_bids = sum([final_allocation_raised_bids[i]['inferred_value'] for i in final_allocation_raised_bids.keys() if i != bidder])
                self.vcg_payments_raised_bids[bidder] = round(apparent_scw_marginal_economy_raised_bids - main_economy_apparent_scw_without_bidder_raised_bids, 2)
                logging.info(f'Payment raised bids {bidder}: {apparent_scw_marginal_economy_raised_bids:.2f} - {main_economy_apparent_scw_without_bidder_raised_bids:.2f}  =  {self.vcg_payments_raised_bids[bidder]:.2f}')

            # if we are using profit max, we also need to calculate the VCG terms for the profit max bids
            if self.mechanism_parameters['calculate_profit_max_bids']:
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    apparent_scw_marginal_economy_profit_max  = self.marginal_economies_inferred_scw_profit_max[number_of_profit_max_bids][bidder]        
                    main_economy_apparent_scw_without_bidder_profit_max = sum([final_allocation_profit_max[number_of_profit_max_bids][i]['inferred_value'] for i in self.marginal_economies_allocations_profit_max[number_of_profit_max_bids].keys() if i != bidder])
                    self.vcg_payments_profit_max[number_of_profit_max_bids][bidder] = round(apparent_scw_marginal_economy_profit_max - main_economy_apparent_scw_without_bidder_profit_max, 2)
                    logging.info(f'Payment profit max ({number_of_profit_max_bids} bids) bidder {bidder}: {apparent_scw_marginal_economy_profit_max:.2f} - {main_economy_apparent_scw_without_bidder_profit_max:.2f}  =  {self.vcg_payments_profit_max[number_of_profit_max_bids][bidder]:.2f}')              

            
            # same logic for profit max with unraised clock bids
            if self.mechanism_parameters['calculate_profit_max_bids_unraised']: 
                for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                    apparent_scw_marginal_economy_profit_max_unraised  = self.marginal_economies_inferred_scw_profit_max_unraised[number_of_profit_max_bids][bidder]        
                    main_economy_apparent_scw_without_bidder_profit_max_unraised = sum([final_allocation_profit_max_unraised[number_of_profit_max_bids][i]['inferred_value'] for i in self.marginal_economies_allocations_profit_max_unraised[number_of_profit_max_bids].keys() if i != bidder])
                    self.vcg_payments_profit_max_unraised[number_of_profit_max_bids][bidder] = round(apparent_scw_marginal_economy_profit_max_unraised - main_economy_apparent_scw_without_bidder_profit_max_unraised, 2)
                    logging.info(f'Payment profit max UNRAISED ({number_of_profit_max_bids} bids) bidder {bidder}: {apparent_scw_marginal_economy_profit_max_unraised:.2f} - {main_economy_apparent_scw_without_bidder_profit_max_unraised:.2f}  =  {self.vcg_payments_profit_max[number_of_profit_max_bids][bidder]:.2f}')


        self.revenue = sum([self.vcg_payments[i] for i in self.bidder_names])
        self.relative_revenue = self.revenue / self.SATS_auction_instance_scw
        logging.info('Revenue: {} | {}% of SCW in efficient allocation\n'.format(self.revenue, self.relative_revenue * 100))

        if self.mechanism_parameters['calculate_raised_bids']:
            self.revenue_raised_bids = sum([self.vcg_payments_raised_bids[i] for i in self.bidder_names])
            self.relative_revenue_raised_bids = self.revenue_raised_bids / self.SATS_auction_instance_scw
            logging.info('Revenue raised bids: {} | {}% of SCW in efficient allocation\n'.format(self.revenue_raised_bids, self.relative_revenue_raised_bids * 100))

        if self.mechanism_parameters['calculate_profit_max_bids']: 
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                self.revenue_profit_max[number_of_profit_max_bids] = sum([self.vcg_payments_profit_max[number_of_profit_max_bids][i] for i in self.bidder_names])
                self.relative_revenue_profit_max[number_of_profit_max_bids] = self.revenue_profit_max[number_of_profit_max_bids] / self.SATS_auction_instance_scw
                logging.info('Revenue profit max ({} bids): {} | {}% of SCW in efficient allocation\n'.format(number_of_profit_max_bids, self.revenue_profit_max[number_of_profit_max_bids], self.relative_revenue_profit_max[number_of_profit_max_bids] * 100))

        if self.mechanism_parameters['calculate_profit_max_bids_unraised']: 
            for number_of_profit_max_bids in self.mechanism_parameters['profit_max_grid']:
                self.revenue_profit_max_unraised[number_of_profit_max_bids] = sum([self.vcg_payments_profit_max_unraised[number_of_profit_max_bids][i] for i in self.bidder_names])
                self.relative_revenue_profit_max_unraised[number_of_profit_max_bids] = self.revenue_profit_max_unraised[number_of_profit_max_bids] / self.SATS_auction_instance_scw
                logging.info('Revenue profit max UNRAISED ({} bids): {} | {}% of SCW in efficient allocation\n'.format(number_of_profit_max_bids, self.revenue_profit_max[number_of_profit_max_bids], self.relative_revenue_profit_max[number_of_profit_max_bids] * 100))
        
        
