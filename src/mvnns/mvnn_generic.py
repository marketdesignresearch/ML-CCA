import numpy as np
import torch
from torchinfo import summary
import torch.nn as nn
from mvnns.layers import *


class MVNN_GENERIC(nn.Module):

    def __init__(self,
                 input_dim: int,
                 num_hidden_layers: int,
                 num_hidden_units: int,
                 dropout_prob: float,
                 layer_type: str,
                 target_max: float,
                 init_method: str,
                 random_ts: tuple,
                 trainable_ts: bool,
                 init_E: float,
                 init_Var: float,
                 init_b: float,
                 init_bias: float,
                 init_little_const: float,
                 lin_skip_connection: bool,
                 capacity_generic_goods: np.array,
                 *args, **kwargs):

        super(MVNN_GENERIC, self).__init__()

        fc_layer = eval(layer_type)

        self.output_activation_function = torch.nn.Identity()
        self._layer_type = layer_type
        self._num_hidden_layers = num_hidden_layers
        self._target_max = target_max
        self.capacity_generic_goods = capacity_generic_goods

        self.layers = []

        # NEW: Generic Transformation with requires_grad=False
        #------------------------
        generic_trafo_layer = nn.Linear(in_features = input_dim,
                                        out_features = input_dim,
                                        bias = False
                                        )
        
        generic_trafo_layer_weight = np.diag(1/self.capacity_generic_goods)
        generic_trafo_layer_weight = generic_trafo_layer_weight.astype(np.float32)
        generic_trafo_layer.weight.data = torch.from_numpy(generic_trafo_layer_weight)

        for param in generic_trafo_layer.parameters():
            param.requires_grad = False


        self.layers.append(generic_trafo_layer)
        #------------------------


        fc1 = fc_layer(input_dim,
                       num_hidden_units,
                       init_method=init_method,
                       random_ts=random_ts,
                       trainable_ts=trainable_ts,
                       use_brelu=True,
                       bias=True,
                       init_E=init_E,
                       init_Var=init_Var,
                       init_b=init_b,
                       init_bias=init_bias,
                       init_little_const=init_little_const
                       )

        self.layers.append(fc1)
        for _ in range(num_hidden_layers - 1):
            self.layers.append(
                fc_layer(num_hidden_units,
                         num_hidden_units,
                         init_method=init_method,
                         random_ts=random_ts,
                         trainable_ts=trainable_ts,
                         use_brelu=True,
                         bias=True,
                         init_E=init_E,
                         init_Var=init_Var,
                         init_b=init_b,
                         init_bias=init_bias,
                         init_little_const=init_little_const
                         )
            )

        self.layers = torch.nn.ModuleList(self.layers)
        self.dropouts = torch.nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in range(len(self.layers))])

        self.output_layer = fc_layer(num_hidden_units,
                                     1,
                                     init_method=init_method,
                                     random_ts=random_ts,
                                     trainable_ts=trainable_ts,
                                     bias=False,
                                     use_brelu=False,
                                     init_E=init_E,
                                     init_Var=init_Var,
                                     init_b=init_b,
                                     init_bias=init_bias,
                                     init_little_const=init_little_const
                                     )
        if lin_skip_connection:
            self.lin_skip_layer = fc_layer(input_dim,
                                           1,
                                           init_method='zero',
                                           random_ts=None,
                                           trainable_ts=False,
                                           bias=False,
                                           use_brelu=False,
                                           init_E=None,
                                           init_Var=None,
                                           init_b=None,
                                           init_bias=None,
                                           init_little_const=None
                                           )
        self.dataset_info = None

    def forward(self, x):
        if hasattr(self, 'lin_skip_layer'):
            x_in = x
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = dropout(x)

        # Output layer
        if hasattr(self, 'lin_skip_layer'):
            x = self.output_activation_function(self.output_layer(x)) + self.lin_skip_layer(x_in)
        else:
            x = self.output_activation_function(self.output_layer(x))
        return x

    def set_dropout_prob(self, dropout_prob):
        for dropout in self.dropouts:
            dropout.p = dropout_prob

    def transform_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'transform_weights'):
                layer.transform_weights()
        if hasattr(self.output_layer, 'transform_weights'):
            self.output_layer.transform_weights()

    
    def print_parameters(self):
        i = 0
        for layer in self.layers:
            print(f'Layer {i}: {layer}')
            print('layer.weight')
            print(f'Shape: {layer.weight.data.shape}')
            print(f'Values: {layer.weight.data}')
            if layer.bias is not None:
                print('layer.bias')
                print(f'Shape: {layer.bias.data.shape}')
                print(f'Values: {layer.bias.data}')
            for name, param in layer.named_parameters():
                print(f'{name} requires_grad={param.requires_grad}')
            i += 1
            print()
        print(f'Output Layer')
        print('output_layer.weight')
        print(f'Shape: {self.output_layer.weight.data.shape}')
        print(f'Values: {self.output_layer.weight.data}')
        if self.output_layer.bias is not None:
                print('output_layer.bias.bias')
                print(f'Shape: {self.output_layer.bias.data.shape}')
                print(f'Values: {self.output_layer.bias.data}')
        for name, param in self.output_layer.named_parameters():
                print(f'{name} requires_grad={param.requires_grad}')
