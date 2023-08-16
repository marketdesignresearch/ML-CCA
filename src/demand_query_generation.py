import sys
from pysats import PySats
import wandb
from datetime import datetime
import numpy as np
import torch
from mvnns.mvnn import MVNN
from milps.gurobi_mip_mvnn_single_bidder_util_max import GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX
from milps.gurobi_mip_mvnn_generic_single_bidder_util_max import GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX
# from pdb import set_trace 



def minimize_W(bidder_models,
            initial_price_vector,
            capacities,
            scale,
            SATS_domain,
            GSVM_national_bidder_goods_of_interest,
            W_epochs,
            lr,
            lr_decay,
            MIP_parameters):
    """
    bidder_models: list of tuples (bidder_id, bidder_model)
    initial_price_vector: np.array of initial prices
    capacities: list of capacities for each good
    scale: list of scale factors for each bidder model
    SATS_domain: string
    GSVM_national_bidder_goods_of_interest: list of goods of interest for the national bidder in GSVM
    W_epochs: number of epochs to run the W minimization
    lr: learning rate
    lr_decay: learning rate decay
    """

    # Create the initial solver models so that then you only update prices and solve
    solver_models = []
    number_of_bidders = len(bidder_models)
    for (bidder_id, bidder_model) in bidder_models:
        bidder_model.transform_weights()
        bidder_model.eval()
        if SATS_domain in ['GSVM', 'LSVM']:
            solver = GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX(bidder_model, SATS_domain=SATS_domain, bidder_id=bidder_id, GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest)
        elif SATS_domain in ['MRVM', 'SRVM']:
            solver = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(model=bidder_model)
        else:
            raise ValueError(f'SATS_domain {SATS_domain} not yet implemented')
        print(initial_price_vector)
        solver.generate_mip(initial_price_vector)
        solver_models.append(solver)

    # Start minimizing W 
    price_vector = initial_price_vector
    best_price_vector = initial_price_vector 
    best_ce = np.inf
    for epoch in range(W_epochs):
        total_predicted_demand = np.zeros(price_vector.shape[0])
        for (i, solver) in enumerate(solver_models):
            # Solve the MIP for the current price vector
            # Multiply the price vector by the scale factor because all mvnns are scaled to have outputs in [0,1]
            price_vector_scaled = price_vector / scale[i]
            solver.update_prices_in_objective(price_vector_scaled)
            
            try:
                predicted_demand = np.array(solver.solve_mip(outputFlag=False,
                                                        verbose = False,
                                                        timeLimit = MIP_parameters["timeLimit"],
                                                        MIPGap = MIP_parameters["MIPGap"],
                                                        IntFeasTol = MIP_parameters["IntFeasTol"],
                                                        FeasibilityTol = MIP_parameters["FeasibilityTol"],
                                                        )
                                        )
            except:
                print('Error in MIP solution, skipping this agent')  # NOTE: THis has never occured in more than 200 runs thus far. 
                
            
            total_predicted_demand += predicted_demand  # add the predicted demand to the total predicted demand



        # Update the price vector
        over_demand = total_predicted_demand - capacities
        print(f'Epoch: {epoch} Total predicted demand: {total_predicted_demand} CE: {np.sum(over_demand**2)}') 
        if np.sum(over_demand**2) < best_ce:
            best_ce = np.sum(over_demand**2)
            best_price_vector = price_vector
        price_vector = price_vector + lr * over_demand
        lr = lr * lr_decay

        if np.sum(over_demand**2) == 0:
            print('Found PREDICTED clearing prices, stopping W optimization')
            break

    print(f'Returning best price vector: {best_price_vector} with CE: {best_ce}')
    mips_solved = (epoch + 1) * number_of_bidders
    return best_price_vector, best_ce, mips_solved



def minimize_W_v2(bidder_models,
            starting_price_vector,
            capacities,
            scale,
            SATS_domain,
            GSVM_national_bidder_goods_of_interest,
            max_steps_without_improvement,
            lr,
            lr_decay,
            MIP_parameters,
            price_scale = 1,
            projected_GD = False,
            previous_prices = None,
            price_elasticity = 0.15
            ):
    """
    bidder_models: list of tuples (bidder_id, bidder_model)
    initial_price_vector: np.array of initial prices
    capacities: list of capacities for each good
    scale: list of scale factors for each bidder model
    SATS_domain: string
    GSVM_national_bidder_goods_of_interest: list of goods of interest for the national bidder in GSVM
    W_epochs: number of epochs to run the W minimization
    lr: learning rate
    lr_decay: learning rate decay
    MIPS_parameters: dict with MIP parameters
    price_scale: float, scale the price vector by this factor. This will enable us to use the "same" learning_rate etc for different domains. 
    projected_GD: bool, if True, use projected gradient descent
    previous_prices: np.array of prices of the previous round, used for projected gradient descent
    price_elasticity: float, price elasticity of d
    """

    # Create the initial solver models so that then you only update prices and solve
    solver_models = []
    number_of_bidders = len(bidder_models)
    for (bidder_id, bidder_model) in bidder_models:
        bidder_model.transform_weights()
        bidder_model.eval()
        if SATS_domain in ['GSVM', 'LSVM']:
            solver = GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX(bidder_model, SATS_domain=SATS_domain, bidder_id=bidder_id, GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest)
        elif SATS_domain in ['MRVM', 'SRVM']:
            solver = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(model=bidder_model)
        else:
            raise ValueError(f'SATS_domain {SATS_domain} not yet implemented')
        print(starting_price_vector)
        solver.generate_mip(starting_price_vector)
        solver_models.append(solver)

    # Start minimizing W 
    price_vector = starting_price_vector
    best_price_vector = starting_price_vector 
    best_ce = np.inf
    
    steps_without_improvement = 0
    all_steps = 0 
    mips_solved = 0

    while steps_without_improvement < max_steps_without_improvement:
        total_predicted_demand = np.zeros(price_vector.shape[0])
        for (i, solver) in enumerate(solver_models):
            # Solve the MIP for the current price vector
            # Multiply the price vector by the scale factor because all mvnns are scaled to have outputs in [0,1]
            price_vector_scaled = price_vector / scale[i]
            solver.update_prices_in_objective(price_vector_scaled)
            
            try:
                predicted_demand = np.array(solver.solve_mip(outputFlag=False,
                                                        verbose = False,
                                                        timeLimit = MIP_parameters["timeLimit"],
                                                        MIPGap = MIP_parameters["MIPGap"],
                                                        IntFeasTol = MIP_parameters["IntFeasTol"],
                                                        FeasibilityTol = MIP_parameters["FeasibilityTol"],
                                                        )
                                        )
            except:
                print('Error in MIP solution, skipping this agent')  # NOTE: THis has never occured in more than 200 runs thus far. 
                
            
            total_predicted_demand += predicted_demand  # add the predicted demand to the total predicted demand


        over_demand = total_predicted_demand - capacities
        
        steps_without_improvement += 1
        all_steps += 1

        print(f'After {all_steps} iterations -> Total predicted demand: {total_predicted_demand} CE: {np.sum(over_demand**2)}') 
        if np.sum(over_demand**2) < best_ce:
            # if the price vector is better -> update the best price vector and reset the steps without improvement
            best_ce = np.sum(over_demand**2)
            best_price_vector = price_vector
            steps_without_improvement = 0

        # Update the price vector
        price_vector = price_vector + lr * over_demand

        if projected_GD:
            # Project the price vector onto the feasible set
            price_vector = np.maximum(price_vector, previous_prices * (1 - price_elasticity))
            price_vector = np.minimum(price_vector, previous_prices * (1 + price_elasticity))


        lr = lr * lr_decay

        if np.sum(over_demand**2) == 0:
            print('Found PREDICTED clearing prices, stopping W optimization')
            break
    
    mips_solved = all_steps * number_of_bidders

    print(f'Returning best price vector: {best_price_vector} with CE: {best_ce}')
    return best_price_vector, best_ce, mips_solved


def minimize_W_v3(bidder_models,
            starting_price_vector,
            capacities,
            scale,
            SATS_domain,
            GSVM_national_bidder_goods_of_interest,
            max_steps_without_improvement,
            max_steps, 
            lr,
            lr_decay,
            MIP_parameters, 
            filter_feasible,
            feasibility_multiplier, 
            feasibility_multiplier_increase_factor
            ):
    """
    bidder_models: list of tuples (bidder_id, bidder_model)
    starting_price_vector: np.array of initial prices
    capacities: list of capacities for each good
    scale: list of scale factors for each bidder model
    SATS_domain: string
    GSVM_national_bidder_goods_of_interest: list of goods of interest for the national bidder in GSVM
    W_epochs: number of epochs to run the W minimization
    lr: learning rate
    lr_decay: learning rate decay
    MIPS_parameters: dict with MIP parameters
    price_scale: float, scale the price vector by this factor. This will enable us to use the "same" learning_rate etc for different domains. 
    filter_feasible: bool, if True, only return price vectors that yield (predicted) feasible allocations
    feasibility_multiplier: the Lagrange multiplier for the feasibility constraint, will affect the gradient 
    """

    # Create the initial solver models so that then you only update prices and solve
    solver_models = []
    number_of_bidders = len(bidder_models)
    for (bidder_id, bidder_model) in bidder_models:
        bidder_model.transform_weights()
        bidder_model.eval()
        if SATS_domain in ['GSVM', 'LSVM']:
            solver = GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX(bidder_model, SATS_domain=SATS_domain, bidder_id=bidder_id, GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest)
        elif SATS_domain in ['MRVM', 'SRVM']:
            solver = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(model=bidder_model)
        else:
            raise ValueError(f'SATS_domain {SATS_domain} not yet implemented')
        print(starting_price_vector)
        solver.generate_mip(starting_price_vector)
        solver_models.append(solver)

    # Start minimizing W 
    price_vector = starting_price_vector
    best_price_vector = starting_price_vector 
    best_W = np.inf
    best_price_vector_is_feasible = False

    # also hold the best price vector etc for no over-demand
    best_price_vector_no_over_demand = starting_price_vector
    best_W_no_over_demand = np.inf
    best_price_vector_no_over_demand_is_feasible = False
    
    steps_without_improvement = 0
    all_steps = 0 
    mips_solved = 0
    all_Ws = [] 
    all_CEs = []
    all_CEs_norm_1 = [] 
    feasible_list = [] # will store for each W if it was feasible or not 

    while (steps_without_improvement < max_steps_without_improvement) and (all_steps < max_steps):
        total_predicted_demand = np.zeros(price_vector.shape[0])
        indirect_revenue = np.dot(price_vector, capacities)
        current_W = indirect_revenue

        for (i, solver) in enumerate(solver_models):
            # Solve the MIP for the current price vector
            # Multiply the price vector by the scale factor because all mvnns are scaled to have outputs in [0,1]
            price_vector_scaled = price_vector / scale[i]
            solver.update_prices_in_objective(price_vector_scaled)
            
            try:
                predicted_demand = np.array(solver.solve_mip(outputFlag=False,
                                                        verbose = False,
                                                        timeLimit = MIP_parameters["timeLimit"],
                                                        MIPGap = MIP_parameters["MIPGap"],
                                                        IntFeasTol = MIP_parameters["IntFeasTol"],
                                                        FeasibilityTol = MIP_parameters["FeasibilityTol"],
                                                        )
                                        )
                
                # get the objective value 
                objective_value = solver.mip.getAttr('objVal')
                # convert that to indirect utility: 
                indirect_utility = objective_value * scale[i]
                # add the agent's indirect utility to W
                current_W += indirect_utility


            except:
                print('Error in MIP solution, skipping this agent')  # NOTE: THis has never occured in more than 200 runs thus far. 
                current_W = 2 * current_W # penalize the crushed agent by doubling its W
                
            
            total_predicted_demand += predicted_demand  # add the predicted demand to the total predicted demand


        over_demand = total_predicted_demand - capacities

        all_Ws.append(current_W)
        all_CEs_norm_1.append(np.sum(np.abs(over_demand)))
        all_CEs.append(np.sum(over_demand**2))

        # check if the allocation is feasible, i.e., no item has positive over-demand 
        if np.all(over_demand <= 0):
            feasible = True
        else:
            feasible = False
        feasible_list.append(feasible)
        
        steps_without_improvement += 1

        print(f'After {all_steps} iterations -> Total predicted demand: {total_predicted_demand} CE: {np.sum(over_demand**2)} and current W: {current_W}') 
        if current_W < best_W:
            # if the price vector is better -> update the best price vector and reset the steps without improvement
            print(f'Improving W from: {best_W} to {current_W}!')
            best_W = current_W
            ce_at_best_W = np.sum(over_demand**2)
            best_price_vector = price_vector
            best_price_vector_is_feasible = feasible
            steps_without_improvement = 0

        if feasible and current_W < best_W_no_over_demand:
            # if the price vector is better -> update the best price vector and reset the steps without improvement
            print(f'---> Found better FEASIBLE W from: {best_W_no_over_demand} to {current_W}!')
            best_W_no_over_demand = current_W
            ce_at_best_W_no_over_demand = np.sum(over_demand**2)
            best_price_vector_no_over_demand = price_vector
            steps_without_improvement = 0
            best_price_vector_no_over_demand_is_feasible = feasible

        # --- Update the price vector ---
        price_vector = price_vector * ( 1 + (lr_decay ** all_steps) * lr * over_demand ) 
        all_steps += 1

        #further update the price vector only for the items with positive over-demand
        over_demand_positive = np.maximum(over_demand, 0)  
        price_vector = price_vector * ( 1 + feasibility_multiplier * (lr_decay ** all_steps) * lr * over_demand_positive )

        if not best_price_vector_no_over_demand_is_feasible: 
            feasibility_multiplier = feasibility_multiplier * feasibility_multiplier_increase_factor # increase the feasibility multiplier if we have not found a feasible price vector yet
            print(f'After {all_steps} iterations -> Increasing feasibility multiplier to {feasibility_multiplier}')


        if np.sum(over_demand**2) == 0:
            best_W = current_W
            ce_at_best_W = np.sum(over_demand**2)
            best_price_vector = price_vector
            steps_without_improvement = 0
            print('Found PREDICTED clearing prices, stopping W optimization')
            return best_price_vector, ce_at_best_W, mips_solved, all_Ws, all_CEs, all_CEs_norm_1, True # if CE = 0 -> the price vector is feasible
    
    mips_solved = all_steps * number_of_bidders


    if filter_feasible and best_price_vector_no_over_demand_is_feasible:
        print(f'Returning best price vector with no overdemand: {best_price_vector_no_over_demand} with CE: {ce_at_best_W_no_over_demand} and W: {best_W_no_over_demand}')
        return best_price_vector_no_over_demand, ce_at_best_W_no_over_demand, mips_solved, all_Ws, all_CEs, all_CEs_norm_1, best_price_vector_no_over_demand_is_feasible
    
    # If we did not find a feasible price vector -> return the one with the lowest W 
    print(f'Returning best price vector: {best_price_vector} with CE: {ce_at_best_W} and W: {best_W}')
    return best_price_vector, ce_at_best_W, mips_solved, all_Ws, all_CEs, all_CEs_norm_1, best_price_vector_is_feasible


def minimize_W_cheating(SATS_auction_instance, 
                        bidder_ids, 
                        starting_price_vector, 
                        capacities ,
                        max_steps_without_improvement,
                        max_steps,
                        lr,
                        lr_decay,
                        ):
    """
    bidder_models: list of tuples (bidder_id, bidder_model)
    initial_price_vector: np.array of initial prices
    capacities: list of capacities for each good
    scale: list of scale factors for each bidder model
    SATS_domain: string
    GSVM_national_bidder_goods_of_interest: list of goods of interest for the national bidder in GSVM
    W_epochs: number of epochs to run the W minimization
    lr: learning rate
    lr_decay: learning rate decay
    MIPS_parameters: dict with MIP parameters
    price_scale: float, scale the price vector by this factor. This will enable us to use the "same" learning_rate etc for different domains. 
    projected_GD: bool, if True, use projected gradient descent
    previous_prices: np.array of prices of the previous round, used for projected gradient descent
    price_elasticity: float, price elasticity of d
    """

    # Start minimizing W 
    price_vector = starting_price_vector
    best_price_vector = starting_price_vector 
    best_W = np.inf
    
    steps_without_improvement = 0
    all_steps = 0 
    mips_solved = 0
    all_Ws = [] 
    all_CEs = []
    all_CEs_norm_1 = [] 

    while (steps_without_improvement < max_steps_without_improvement) and (all_steps < max_steps):
        total_predicted_demand = np.zeros(price_vector.shape[0])
        indirect_revenue = np.dot(price_vector, capacities)
        current_W = indirect_revenue

        for bidder_id in bidder_ids:
            # Solve the MIP for the current price vector
           
            demand_response = np.array(SATS_auction_instance.get_best_bundles(bidder_id, price_vector, 1, allow_negative = True)[0]) # convert to np array
            
            
            bundle_value = SATS_auction_instance.calculate_value(bidder_id, demand_response)
            price = np.dot(price_vector, demand_response)

            indirect_utility = bundle_value - price
            # add the agent's indirect utility to W
            current_W += indirect_utility

            total_predicted_demand += demand_response  # add the predicted demand to the total predicted demand



        over_demand = total_predicted_demand - capacities

        all_Ws.append(current_W)
        all_CEs_norm_1.append(np.sum(np.abs(over_demand)))
        all_CEs.append(np.sum(over_demand**2))
        
        steps_without_improvement += 1

        print(f'After {all_steps} iterations -> Total predicted demand: {total_predicted_demand} CE: {np.sum(over_demand**2)} and current W: {current_W}') 
        if current_W < best_W:
            # if the price vector is better -> update the best price vector and reset the steps without improvement
            print(f'Improving W from: {current_W} to {best_W}!')
            best_W = current_W
            ce_at_best_W = np.sum(over_demand**2)
            best_price_vector = price_vector
            steps_without_improvement = 0

        # Update the price vector
        # price_vector = price_vector + lr * over_demand

        price_vector = price_vector * ( 1 + (lr_decay ** all_steps) * lr * over_demand ) 
        all_steps += 1


        if np.sum(over_demand**2) == 0:
            best_W = current_W
            ce_at_best_W = np.sum(over_demand**2)
            best_price_vector = price_vector
            steps_without_improvement = 0
            print('Found PREDICTED clearing prices, stopping W optimization')
            break
    
    mips_solved = all_steps * len(bidder_ids)

    print(f'Returning best price vector: {best_price_vector} with CE: {ce_at_best_W} and W: {best_W}')
    return best_price_vector, ce_at_best_W, mips_solved, all_Ws, all_CEs, all_CEs_norm_1