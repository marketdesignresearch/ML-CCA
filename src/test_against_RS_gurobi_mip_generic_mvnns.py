# %% PACKAGES
import numpy as np
import pandas as pd
import torch
import logging

# own modules
from mvnns.mvnn import MVNN
from mvnns.mvnn_generic import MVNN_GENERIC
from milps.gurobi_mip_mvnn_generic_single_bidder_util_max import GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX as MIP
from milps.gurobi_mip_mvnn_single_bidder_util_max import GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX as MIP2

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

#%% SET PARAMETERS:
m = 30 # number of generic items
prices = np.zeros(m) #np.random.rand(m)/100
print('PRICES of GENERIC ITEMS:')
print(prices)
capacity_generic_items = np.random.randint(low=1, high=20, size=m,dtype=np.int64)
print('CAPACITIES of GENERIC ITEMS:')
print(capacity_generic_items)

# %% Create Random MVNN
MVNN = MVNN_GENERIC(input_dim=len(capacity_generic_items),
                    num_hidden_layers=3,
                    num_hidden_units=100,
                    layer_type='MVNNLayerReLUProjected',
                    target_max=1,
                    dropout_prob=0,
                    init_method='custom',
                    random_ts=[0, 1],
                    trainable_ts=False,
                    init_E=1,
                    init_Var=0.09,
                    init_b=0.05,
                    init_bias=0.05,
                    init_little_const=0.1,
                    lin_skip_connection=False,
                    capacity_generic_items=capacity_generic_items # NEW!!!!!!!!!!!!!!!!
        )

MVNN.transform_weights()

# %% TODO: TRAIN & TEST MVNN on SATS DATA

# %% Solve GENERIC MVNN MIP
X = MIP(MVNN)
X.generate_mip(prices = prices,
            MIPGap = None,
            verbose = False)

sol_MIP = X.solve_mip(outputFlag = False,
                verbose = True,
                timeLimit = np.inf,
                MIPGap = 1e-04,
                IntFeasTol = 1e-5,
                FeasibilityTol = 1e-6
                )
obj_MIP = X.mip.ObjVal

# %% Create TEST Dataset
print('GENERATING RANDOM feasible GENERIC BUNDLES:')
n = 10000
X_test = []
for _  in range(n):
    rand_bundle = np.random.randint(low=[0]*m, high=capacity_generic_items+1)
    X_test.append(rand_bundle)
    #print(rand_bundle)
    #print((rand_bundle <= capacity_generic_items).all())
    assert((rand_bundle <= capacity_generic_items).all())

X_test = np.asarray(X_test, dtype=np.float32)
X_test = torch.from_numpy(X_test)
print(f'X_test.shape:{X_test.shape}')
print(f'X_test:{X_test}')

# %% Make MVNN predictions TEST Dataset
MVNN_preds = []
print('Making MVNN predictions for all random feasible bundles:')
MVNN_pred_test = MVNN(X_test).detach().data.numpy().flatten()
print(f'MVNN_pred_test: {MVNN_pred_test}')
print('\n')

# %% CHECK 1: MVNN_pred_test-prices*X_test <= obj_MIP
X_test_np = X_test.detach().data.numpy()
MVNN_utility_test = MVNN_pred_test - X_test_np@prices
check1 = (MVNN_utility_test <= obj_MIP).all()
print(f'CHECK1: "MVNN_pred_test-prices*X_test <= obj_MIP? -> {check1}')
print()
assert(check1)

# %% CHECK 2: obj_MIP - max(MVNN_pred_test-prices*X_test)  
check2 = obj_MIP- max(MVNN_utility_test)
print(f'CHECK2: obj_MIP - max(MVNN_pred_test-prices*X_test) = {check2}')
print()

# %% CHECK 3: MVNN(sol_MIP)-prices*sol_MIP == obj_MIP
sol_MIP_torch = torch.from_numpy(np.array(sol_MIP, dtype=np.float32))
MVNN_pred_sol_MIP = MVNN(sol_MIP_torch).detach().data.numpy()
check3 = MVNN_pred_sol_MIP - np.dot(prices,sol_MIP) == obj_MIP
print(f'CHECK3: "MVNN(sol_MIP)-prices*sol_MIP == obj_MIP? -> {check3}')
print('Exact Diff')
print(f'obj_MIP - MVNN(sol_MIP)-prices*sol_MIP = {obj_MIP - (MVNN_pred_sol_MIP - np.dot(prices,sol_MIP))}')
print('')

# %% CHECK 4: If prices == 0 then sol_MIP == capacity_generic_items?
if max(prices)<=0:
    check4 = (sol_MIP == capacity_generic_items).all()
    print(f'CHECK4: "If prices == 0 then sol_MIP == capacity_generic_items? -> {check4}')
    print('')