import torch
import torch.nn as nn
from mvnns.layers import *


class MVNN(nn.Module):

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
                 *args, **kwargs):

        super(MVNN, self).__init__()

        fc_layer = eval(layer_type)

        self.output_activation_function = torch.nn.Identity()
        self._layer_type = layer_type
        self._num_hidden_layers = num_hidden_layers
        self._target_max = target_max

        self.layers = []

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
