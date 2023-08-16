# %% PACKAGES
import numpy as np
import pandas as pd
import torch
import logging

# own modules
from mlca_demand_queries.mlca_dq_wdp_generic import MLCA_DQ_WDP_GENERIC
from mlca_demand_queries.mlca_dq_util import format_solution_mip_new

# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', None)
# clear existing logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# log debug to console
logging.basicConfig(level=logging.DEBUG, format='', datefmt='%H:%M:%S')


# %% TEST GENERIC WDP TOY EXAMPLE
#----------------------------------------------------------------------------------------------
print(*['*']*50)
print('TEST 1: GENERIC WDP TOY EXAMPLE')
# %% SET PARAMETERS:
m = 3 # number of generic items
capacity_generic_items = np.array([10,10,10])
print('CAPACITIES of GENERIC ITEMS:')
print(capacity_generic_items)

# %% Create Elicited Generic Bids For each Bidder
print('GENERATING TOY EXAMPLE feasible GENERIC BUNDLES:')
n = 3 # number of bidders
bidder_names = ['Bidder_'+str(i) for i in range(n)]
elicited_bids = []

print('Bidder_0')
bids_0 = np.array([[3,9,5,10],[3,10,5,11],[0,10,5,5]])
print(bids_0)
print('Bidder_1')
bids_1 = np.array([[4,1,5,5],[1,1,1,1]])
print(bids_1)
print('Bidder_2')
bids_2 = np.array([[3,0,0,3],[1,2,3,4]])
print(bids_2)

elicited_bids = [bids_0, bids_1, bids_2]
# %% Solve GENERIC WDP
MIP_parameters = {
        'timeLimit': 3600 * 10, # Default +inf
        'MIPGap': 1e-06, # Default 1e-04
        'IntFeasTol': 1e-8, # Default 1e-5
        'FeasibilityTol': 1e-9 # Default 1e-6
    }

wdp = MLCA_DQ_WDP_GENERIC(bids=elicited_bids,
                          MIP_parameters=MIP_parameters,
                          capacity_generic_items=capacity_generic_items)
wdp.initialize_mip(verbose=1)
wdp.solve_mip(verbose=1)
details = wdp.get_solve_details()   # get the details of the solved MIP
print(f'Details:{details}')

print('\n')
print('SOLUTION:')
print(wdp.x_star)

print('\n')
print('OPTIMAL ALLOCATION')
wdp.print_optimal_allocation()

assert(wdp.Mip.objective_value == 18.0)
assert((wdp.x_star[0] == np.array([3,9,5])).all())
assert((wdp.x_star[1] == np.array([4,1,5])).all())
assert((wdp.x_star[2] == np.array([3,0,0])).all())
# %% TEST format_solution_mip_new for generic WDP
allocation = format_solution_mip_new(Mip=wdp.Mip,
                                    elicited_bids=elicited_bids,
                                    bidder_names=bidder_names,
                                    fitted_scaler=None,
                                    generic_domain=True)
print('\n')
print('OPTIMAL ALLOCATION after format_solution_mip_new():')
print(allocation)

assert(allocation['Bidder_0']['good_ids'] == {0: 3, 1: 9, 2: 5})
assert(allocation['Bidder_0']['value'] == 10)

assert(allocation['Bidder_1']['good_ids'] == {0: 4, 1: 1, 2: 5})
assert(allocation['Bidder_1']['value'] == 5)

assert(allocation['Bidder_2']['good_ids'] == {0: 3, 1: 0, 2: 0})
assert(allocation['Bidder_2']['value'] == 3)
print(' ')
print(*['*']*50)
print('\n\n')
#----------------------------------------------------------------------------------------------


# %% TEST GENERIC WDP SYNTHETIC DATA
#----------------------------------------------------------------------------------------------
print(*['*']*50)
print('TEST 2: GENERIC WDP SYNTHETIC DATA')

#%% SET PARAMETERS:
m = 5 # number of generic items
capacity_generic_items = np.random.randint(low=1, high=20, size=m,dtype=np.int64)
print('CAPACITIES of GENERIC ITEMS:')
print(capacity_generic_items)

# %% Create Elicited Generic Bids For each Bidder
print('GENERATING RANDOM feasible GENERIC BUNDLES:')
n = 2 # number of bidders
bidder_names = ['Bidder_'+str(i) for i in range(n)]
elicited_bids = []
for i in range(n):
    n_bids = 100
    elicited_bids_bidder = []
    for _  in range(n_bids):
        rand_bundle = np.random.randint(low=[0]*m, high=capacity_generic_items+1)
        #print(rand_bundle)
        rand_value = np.random.normal(loc=np.sum(rand_bundle), scale=1, size=1)
        #print(rand_value[0])
        elicited_bids_bidder.append(np.concatenate((rand_bundle,rand_value)))
        assert((rand_bundle <= capacity_generic_items).all())

    elicited_bids_bidder = np.asarray(elicited_bids_bidder)
    print(bidder_names[i])
    print(f'elicited_bids.shape:{elicited_bids_bidder.shape}')
    print(f'elicited_bids:{elicited_bids_bidder}')
    elicited_bids.append(elicited_bids_bidder)

# %% Solve GENERIC WDP
MIP_parameters = {
        'timeLimit': 3600 * 10, # Default +inf
        'MIPGap': 1e-06, # Default 1e-04
        'IntFeasTol': 1e-8, # Default 1e-5
        'FeasibilityTol': 1e-9 # Default 1e-6
    }

wdp = MLCA_DQ_WDP_GENERIC(bids=elicited_bids,
                          MIP_parameters=MIP_parameters,
                          capacity_generic_items=capacity_generic_items)
wdp.initialize_mip(verbose=1)
wdp.solve_mip(verbose=1)
details = wdp.get_solve_details()   # get the details of the solved MIP
print(f'Details:{details}')

print('\n')
print('SOLUTION:')
print(wdp.x_star)

print('\n')
print('OPTIMAL ALLOCATION')
wdp.print_optimal_allocation()

# %% TEST format_solution_mip_new for generic WDP
allocation = format_solution_mip_new(Mip=wdp.Mip,
                                    elicited_bids=elicited_bids,
                                    bidder_names=bidder_names,
                                    fitted_scaler=None,
                                    generic_domain=True)
print('\n')
print('OPTIMAL ALLOCATION after format_solution_mip_new():')
print(allocation)
print(' ')
print(*['*']*50)
#----------------------------------------------------------------------------------------------