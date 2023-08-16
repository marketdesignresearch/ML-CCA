# %% PACKAGES
import numpy as np
import pandas as pd
import torch
from torchinfo import summary
import logging
import time

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

# %% Toy Examples
capacity_generic_items = np.array([2, 1], dtype=np.int64)
                                  
# GENERIC MVNN1
#------------------------------------------------------------------------------------------------------------------------
MVNN1 = MVNN_GENERIC(input_dim=len(capacity_generic_items),
                num_hidden_layers=2,
                num_hidden_units=2,
                layer_type='MVNNLayerReLUProjected',
                target_max=1,
                dropout_prob=0.0,
                init_method='glorot_sqrt',
                random_ts=[1, 1],
                trainable_ts=False,
                init_E=0.0,
                init_Var=0.0,
                init_b=0.0,
                init_bias=0.0,
                init_little_const=1e-8,
                lin_skip_connection=False,
                capacity_generic_items=capacity_generic_items)

# HL-0 = Generic Transformation (specified above)

# HL-1
MVNN1.layers[1].weight.data = torch.from_numpy(np.array([[1, 1], [0, 1]], dtype=np.float32))
MVNN1.layers[1].bias.data = torch.from_numpy(np.array([-0.5, 0], dtype=np.float32))
MVNN1.layers[1].ts = torch.from_numpy(np.array([1, 0.5], dtype=np.float32))
# HL-2
MVNN1.layers[2].weight.data = torch.from_numpy(np.array([[0.5, 0], [1, 2]], dtype=np.float32))
MVNN1.layers[2].bias.data = torch.from_numpy(np.array([0, -0.75], dtype=np.float32))
MVNN1.layers[2].ts = torch.from_numpy(np.array([0.5, 1], dtype=np.float32))

# Lin. Skip. Layer
#MVNN1.lin_skip_layer.weight.data = torch.from_numpy(np.array([[2, 1, 0.5]], dtype=np.float32))

# OUTPUT-Layer
MVNN1.output_layer.weight.data = torch.from_numpy(np.array([[3, 1]], dtype=np.float32))
#------------------------------------------------------------------------------------------------------------------------

MVNN1.print_parameters()
# %%
# MVNN2
#------------------------------------------------------------------------------------------------------------------------
MVNN2 = MVNN_GENERIC(input_dim=len(capacity_generic_items),
                num_hidden_layers=1,
                num_hidden_units=3,
                layer_type='MVNNLayerReLUProjected',
                target_max=1,
                dropout_prob=0.0,
                init_method='glorot_sqrt',
                random_ts=[1, 1],
                trainable_ts=False,
                init_E=0.0,
                init_Var=0.0,
                init_b=0.0,
                init_bias=0.0,
                init_little_const=1e-8,
                lin_skip_connection=False,
                capacity_generic_items=capacity_generic_items)

# HL-0 = Generic Transformation (specified above)

# HL-1
MVNN2.layers[1].weight.data = torch.from_numpy(np.array([[0.5, 1], [2, 1], [0, 2] ], dtype=np.float32))
MVNN2.layers[1].bias.data = torch.from_numpy(np.array([-1, 0, -0.5], dtype=np.float32))
MVNN2.layers[1].ts = torch.from_numpy(np.array([1, 1, 1], dtype=np.float32))

# Lin. Skip. Layer
# MVNN2.lin_skip_layer.weight.data = torch.from_numpy(np.array([[0.5, 1, 2]], dtype=np.float32))

# OUTPUT-Layer
MVNN2.output_layer.weight.data = torch.from_numpy(np.array([[1, 1, 0]], dtype=np.float32))
#------------------------------------------------------------------------------------------------------------------------
# %% PREDICTIONS
print('\nMVNN1:')
print(MVNN1)
print(summary(MVNN1))
print()
print('Predictions:')
print(f'[0, 0]: {MVNN1(torch.from_numpy(np.array([[0, 0]], dtype=np.float32)))}')
assert (MVNN1(torch.from_numpy(np.array([[0, 0]], dtype=np.float32))) == torch.tensor([[0]]))
print(f'[0, 1]: {MVNN1(torch.from_numpy(np.array([[0, 1]], dtype=np.float32)))}')
assert (MVNN1(torch.from_numpy(np.array([[0, 1]], dtype=np.float32))) == torch.tensor([[1.5]]))
print(f'[1, 1]: {MVNN1(torch.from_numpy(np.array([[1, 1]], dtype=np.float32)))}')
assert (MVNN1(torch.from_numpy(np.array([[1, 1]], dtype=np.float32))) == torch.tensor([[2.5]]))
print(f'[2, 1]: {MVNN1(torch.from_numpy(np.array([[2, 1]], dtype=np.float32)))}')
assert (MVNN1(torch.from_numpy(np.array([[2, 1]], dtype=np.float32))) == torch.tensor([[2.5]]))
print(f'[1, 0]: {MVNN1(torch.from_numpy(np.array([[1, 0]], dtype=np.float32)))}')
assert (MVNN1(torch.from_numpy(np.array([[1, 0]], dtype=np.float32))) == torch.tensor([[0]]))
print(f'[2, 0]: {MVNN1(torch.from_numpy(np.array([[2, 0]], dtype=np.float32)))}')
assert (MVNN1(torch.from_numpy(np.array([[2, 0]], dtype=np.float32))) == torch.tensor([[0.75]]))
print('\n')

print('\nMVNN2:')
print(MVNN2)
print(summary(MVNN2))
print()
print('Predictions:')
print(f'[0, 0]: {MVNN2(torch.from_numpy(np.array([[0, 0]], dtype=np.float32)))}')
assert (MVNN2(torch.from_numpy(np.array([[0, 0]], dtype=np.float32))) == torch.tensor([[0]]))
print(f'[0, 1]: {MVNN2(torch.from_numpy(np.array([[0, 1]], dtype=np.float32)))}')
assert (MVNN2(torch.from_numpy(np.array([[0, 1]], dtype=np.float32))) == torch.tensor([[1]]))
print(f'[1, 1]: {MVNN2(torch.from_numpy(np.array([[1, 1]], dtype=np.float32)))}')
assert (MVNN2(torch.from_numpy(np.array([[1, 1]], dtype=np.float32))) == torch.tensor([[1.25]]))
print(f'[2, 1]: {MVNN2(torch.from_numpy(np.array([[2, 1]], dtype=np.float32)))}')
assert (MVNN2(torch.from_numpy(np.array([[2, 1]], dtype=np.float32))) == torch.tensor([[1.5]]))
print(f'[1, 0]: {MVNN2(torch.from_numpy(np.array([[1, 0]], dtype=np.float32)))}')
assert (MVNN2(torch.from_numpy(np.array([[1, 0]], dtype=np.float32))) == torch.tensor([[1]]))
print(f'[2, 0]: {MVNN2(torch.from_numpy(np.array([[2, 0]], dtype=np.float32)))}')
assert (MVNN2(torch.from_numpy(np.array([[2, 0]], dtype=np.float32))) == torch.tensor([[1]]))

# %% 1 NEW GENERIC MIP MVNN1
prices = np.array([1/2, 1], dtype=np.float32)
X = MIP(MVNN1)
X.generate_mip(prices = prices,
            MIPGap = None,
            verbose = True)

s = X.solve_mip(outputFlag = False,
                verbose = True,
                timeLimit = np.inf,
                MIPGap = 1e-04,
                IntFeasTol = 1e-5,
                FeasibilityTol = 1e-6
                )

assert(s == [1,1])
assert(X.mip.ObjVal == 1.0)
time.sleep(1.1) # since MIPs are saved with name depending on seconds

# %% 2 NEW GENERIC MIP MVNN1 WITH forbidden bundle
prices = np.array([1/2, 1], dtype=np.float32)
X = MIP(MVNN1)
X.generate_mip(prices = prices,
            MIPGap = None,
            verbose = True)

# NEW: add forbidden bundle
# --------------------------------------------------
forbidden_bundle = np.array([1, 1], dtype=np.int64)
X.add_forbidden_bundle(forbidden_bundle) # -> s = [0,1] or [2,1] with obj=0.75
forbidden_bundle = np.array([2, 1], dtype=np.int64)
X.add_forbidden_bundle(forbidden_bundle) # -> s = [0,1] with obj=0.75
forbidden_bundle = np.array([0, 1], dtype=np.int64)
X.add_forbidden_bundle(forbidden_bundle) # -> s = [0,0] with obj=0
#---------------------------------------------------


s = X.solve_mip(outputFlag = False,
                verbose = True,
                timeLimit = np.inf,
                MIPGap = 1e-04,
                IntFeasTol = 1e-5,
                FeasibilityTol = 1e-6
                )

assert(s == [0,0])
assert(X.mip.ObjVal == 0.0)
time.sleep(1.1) # since MIPs are saved with name depending on seconds

# %% 3 NEW GENERIC MIP MVNN2
prices = np.array([1/4, 1.25], dtype=np.float32)
X = MIP(MVNN2)
X.generate_mip(prices = prices,
            MIPGap = None,
            verbose = True)

s = X.solve_mip(outputFlag = False,
                verbose = True,
                timeLimit = np.inf,
                MIPGap = 1e-04,
                IntFeasTol = 1e-5,
                FeasibilityTol = 1e-6
                )

assert(s == [1,0])
assert(X.mip.ObjVal == 0.75)
time.sleep(1.1) # since MIPs are saved with name depending on seconds

# %% 3 NEW GENERIC MIP MVNN2 WITH forbidden bundle
prices = np.array([1/4, 1.25], dtype=np.float32)
X = MIP(MVNN2)
X.generate_mip(prices = prices,
            MIPGap = None,
            verbose = True)

# NEW: add forbidden bundle
# --------------------------------------------------
forbidden_bundle = np.array([1, 0], dtype=np.int64) 
X.add_forbidden_bundle(forbidden_bundle) #-> s=[2,0] with obj=0.5
#---------------------------------------------------

s = X.solve_mip(outputFlag = False,
                verbose = True,
                timeLimit = np.inf,
                MIPGap = 1e-04,
                IntFeasTol = 1e-5,
                FeasibilityTol = 1e-6
                )

assert(s == [2,0])
assert(X.mip.ObjVal == 0.50)
print('\n')
print('CHECK ASSERTION: OK')