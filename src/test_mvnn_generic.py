# %%
import numpy as np
import pandas as pd
import torch
from torchinfo import summary
import logging
from matplotlib import pyplot as plt
# own modules
from mvnns.mvnn_generic import MVNN_GENERIC
from mvnns.mvnn import MVNN

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', None)

# clear existing logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# log debug to console

logging.basicConfig(level=logging.DEBUG, format='', datefmt='%H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True

# %% Create Training Dataset
capacity_generic_items = np.array([1,5,1,1,10,1,1,1,3,1,1,1,4,1,1,1])
n_train = 100
monotone_y_values = True

x_train = []
for capacity in capacity_generic_items:
    col = np.random.randint(low=0, high=capacity+1, size=(n_train,1))
    col = np.array(col, dtype=np.float32)
    x_train.append(col)

x_train = np.concatenate(x_train, axis=1)
x_train_tilde = x_train/capacity_generic_items
x_train_tilde = np.array(x_train_tilde, dtype=np.float32)

if monotone_y_values:
    y_train = x_train.sum(axis=1)/capacity_generic_items.sum() # monotone relationship for targets
else:
    y_train = np.random.rand(n_train,1) # random targets

y_train = y_train.reshape(-1,1)
y_train = np.array(y_train, dtype=np.float32)

print(f'Shape of x_train:{x_train.shape}')
print(f'x_train:{x_train}')
print(f'Shape of x_train_tilde:{x_train_tilde.shape}')
print(f'x_train_tilde:{x_train_tilde}')
print(f'Shape of y_train:{y_train.shape}')
print(f'y_train:{y_train}')

# %% Create Generic MVNN

generic_MVNN = MVNN_GENERIC(input_dim=len(capacity_generic_items),
                    num_hidden_layers=2,
                    num_hidden_units=50,
                    layer_type='MVNNLayerReLUProjected',
                    target_max=1,
                    dropout_prob=0,
                    init_method='custom',
                    random_ts=[0, 1],
                    trainable_ts=True,
                    init_E=1,
                    init_Var=0.09,
                    init_b=0.05,
                    init_bias=0.05,
                    init_little_const=0.1,
                    lin_skip_connection=True,
                    capacity_generic_items=capacity_generic_items # NEW!!!!!!!!!!!!!!!!
                    )

MVNN = MVNN(input_dim=len(capacity_generic_items),
                    num_hidden_layers=2,
                    num_hidden_units=50,
                    layer_type='MVNNLayerReLUProjected',
                    target_max=1,
                    dropout_prob=0,
                    init_method='custom',
                    random_ts=[0, 1],
                    trainable_ts=True,
                    init_E=1,
                    init_Var=0.09,
                    init_b=0.05,
                    init_bias=0.05,
                    init_little_const=0.1,
                    lin_skip_connection=True,
                    )

generic_MVNN.print_parameters()
print(summary(generic_MVNN))

# %% define loss and optimizer
loss_function = torch.nn.L1Loss()
optimizer = torch.optim.Adam(generic_MVNN.parameters(),lr=0.002,weight_decay=1e-3)
optimizer_tilde = torch.optim.Adam(MVNN.parameters(),lr=0.002,weight_decay=1e-3)

# set model to training mode
generic_MVNN.train()
MVNN.train()
# %% training loop
for epoch in range(1,5000):

    # 1. zero the parameter gradients
    optimizer.zero_grad()
    optimizer_tilde.zero_grad() 

    # 2. predict full GD
    # Generic MVNN
    x = torch.from_numpy(x_train)
    y_pred = generic_MVNN(x)
    # MVNN with scaled data
    x_tilde = torch.from_numpy(x_train_tilde)
    y_pred_tilde = MVNN(x_tilde)

    # 3. calculate loss for epoch
    y = torch.from_numpy(y_train)
    # Generic MVNN
    loss_epoch = loss_function(y,y_pred)
    # MVNN with scaled data
    loss_epoch_tilde = loss_function(y,y_pred_tilde)
    if epoch%100 ==0:
        print(f'epoch: {epoch}, MAE NEW Generic MVNN:{loss_epoch}, MAE MVNN WITH SCALED DATA:{loss_epoch_tilde}')

    # 4. calculate gradients
    loss_epoch.backward()
    loss_epoch_tilde.backward()

    # 5. gradient descent step
    optimizer.step()
    optimizer_tilde.step()

 
generic_MVNN.transform_weights()
MVNN.transform_weights()
# %% Inspect Parameters Again
generic_MVNN.print_parameters()
print(summary(generic_MVNN))

# %% testing (on training set) and plotting
generic_MVNN.eval()
MVNN.eval()

x = torch.from_numpy(x_train)  
y = torch.from_numpy(y_train)
y_pred = generic_MVNN(x).detach().data.numpy()
plt.scatter(y_train, y_pred, label='MVNN with generic transformation')

x_tilde = torch.from_numpy(x_train_tilde)  
y_pred_tilde = MVNN(x_tilde).detach().data.numpy()
plt.scatter(y_train, y_pred_tilde, color='red', label='MVNN with scaled data')

plt.legend(loc='best')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot([0,1],[0,1])
plt.show()

# %% Inspect Parameters Again
generic_MVNN.print_parameters()
print(summary(generic_MVNN))
