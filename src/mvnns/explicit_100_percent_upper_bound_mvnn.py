import logging

import torch
import torch.nn as nn
from mvnns.layers import *


class Explicit100UpperBoundMVNN(nn.Module):

    def __init__(self, input_dim: int, layer_type: str, target_max: float, X_train, y_train, *args, **kwargs):

        super(Explicit100UpperBoundMVNN, self).__init__()

        fc_layer = eval(layer_type)
        self.output_activation_function = torch.nn.Identity()
        self._layer_type = layer_type
        self._num_hidden_layers = 2
        self.input_dim = input_dim
        self._target_max = target_max

        self._set_training_data(X_train=X_train,
                                y_train=y_train)

        self.layers = torch.nn.ModuleList([])

        fc1 = fc_layer(input_dim,
                       self.ntrain - 1,
                       init_method='glorot_default',
                       random_ts=[1, 1],
                       trainable_ts=False,
                       use_brelu=True,
                       bias=True,
                       init_E=None,
                       init_Var=None,
                       init_b=None,
                       init_bias=None,
                       init_little_const=None)  # bias ist set to (0,...,0) later, cannot use bias=False since MIP needs access to a bias
        self.layers.append(fc1)
        fc2 = fc_layer(self.ntrain - 1,
                       self.ntrain - 1,
                       init_method='glorot_default',
                       random_ts=[1, 1],
                       trainable_ts=False,
                       use_brelu=True,
                       bias=True,
                       init_E=None,
                       init_Var=None,
                       init_b=None,
                       init_bias=None,
                       init_little_const=None)
        self.layers.append(fc2)

        self.output_layer = fc_layer(self.ntrain - 1,
                                     1,
                                     init_method='glorot_default',
                                     random_ts=[1, 1],
                                     trainable_ts=False,
                                     use_brelu=False,
                                     bias=False,
                                     init_E=None,
                                     init_Var=None,
                                     init_b=None,
                                     init_bias=None,
                                     init_little_const=None
                                     )

        self.dataset_info = None
        self._set_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def transform_weights(self):
        pass

    def _set_training_data(self,
                           X_train,
                           y_train):

        # changed from numpy logic to torch logic
        # X_train = X_train.numpy()
        # y_train = y_train.numpy()

        # Check if full bundle and null bundle is contained
        # full_bundle = np.array([1]*self.input_dim, dtype=np.float32)
        full_bundle = torch.ones(self.input_dim)
        # empty_bundle = np.array([0]*self.input_dim, dtype=np.float32)
        empty_bundle = torch.zeros(self.input_dim)

        # full_idx = np.where(np.all(X_train==full_bundle,axis=1))[0]
        # empty_idx =  np.where(np.all(X_train==empty_bundle,axis=1))[0]

        full_idx = torch.where((X_train == full_bundle).all(dim=1))[0]
        empty_idx = torch.where((X_train == empty_bundle).all(dim=1))[0]

        if len(full_idx) == 0:
            raise ValueError(
                'Full Bundle (1,...,1) not contained in X_train -> Explicit-100%-upper-UB cannot be initialized.')
        if len(full_idx) > 1:
            logging.warning('Multiple Full Bundles (1,...,1) contained in X_train -> please Check.')
        if len(empty_idx) == 0:
            logging.info('Adding null bundle (0,...,0) for 100%-Explicit-upper-UB')
            # X_train = np.concatenate((X_train,empty_bundle.reshape(1,-1)))
            # value_empty_bundle = np.array([0.0], dtype=np.float32)
            # y_train = np.concatenate((y_train,value_empty_bundle))
            X_train = torch.cat((X_train, torch.reshape(empty_bundle, (1, -1))), dim=0)
            y_train = torch.cat((y_train, torch.tensor([0.0])))
        if len(empty_idx) > 1:
            logging.warning('Multiple Null Bundles (0,...,0) contained in X_train -> please Check.')

        # self.X_train = torch.from_numpy(X_train)
        # self.y_train = torch.from_numpy(y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.ntrain = self.X_train.shape[0]  # inlcuding null AND full bundle

    def _set_weights(self):

        y, idx = torch.sort(self.y_train)
        X = self.X_train[idx, :][:-1, :]  # all except full bundle

        fc1, fc2 = self.layers[0], self.layers[1]

        # First HL
        fc1.weight.data = 1.0 - X
        fc1.bias.data = torch.zeros(len(X), dtype=torch.float32)

        # Second HL
        fc2.weight.data = torch.tril(torch.ones(self.ntrain - 1, self.ntrain - 1, dtype=torch.float32))
        fc2.bias.data = - torch.arange(start=0, end=(self.ntrain - 1), dtype=torch.float32)

        # Output Layer
        self.output_layer.weight.data = (y[1:] - y[:-1]).reshape(1, -1).float()
