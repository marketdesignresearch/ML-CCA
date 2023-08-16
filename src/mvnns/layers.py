from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats


def bReLU(x: torch.Tensor, t: torch.Tensor = 1):
    return -torch.relu(t - torch.relu(x)) + t


class MVNNLayerReLUProjected(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 init_method,
                 random_ts,
                 trainable_ts,
                 use_brelu,
                 bias,
                 init_E,
                 init_Var,
                 init_b,
                 init_bias,
                 init_little_const,
                 ):

        super(MVNNLayerReLUProjected, self).__init__(in_features, out_features, bias)

        self.weight_init(method=init_method,
                         init_E=init_E,
                         init_Var=init_Var,
                         init_b=init_b,
                         init_bias=init_bias,
                         init_little_const=init_little_const
                         )

        self.use_brelu = use_brelu

        if self.use_brelu:
            ts = (random_ts[1] - random_ts[0]) * torch.rand(out_features) + torch.ones(out_features) * random_ts[0]

            if trainable_ts:
                self.ts = torch.nn.Parameter(ts, requires_grad=True)
            else:
                self.ts = ts

    def forward(self, input):
        self.transform_weights()
        if self.bias is not None:
            out = nn.functional.linear(input, weight=self.weight, bias=self.bias)
        else:
            out = nn.functional.linear(input, weight=self.weight, bias=None)
        if self.use_brelu:
            out = bReLU(out, self.ts)
        return out

    def transform_weights(self):
        self.weight.data.clamp_(min=0)
        if self.use_brelu:
            self.ts.data.clamp_(min=0)
        if self.bias is not None:
            self.bias.data.clamp_(max=0)

    def weight_init(self,
                    method,
                    init_E,  # only for method=='custom'
                    init_Var,  # only for method=='custom'
                    init_b,  # only for method=='custom'
                    init_bias,  # only for method=='custom'
                    init_little_const  # only for method=='custom'
                    ):

        if method == 'custom':

            '''
            # PARAMETERS:
            # --------------
            # (1) init_E: gives (approximately) the expected value of MVNN(1,1,...,1) (i.e. the predicted value of the full bundle) at initialization of the network, if all cutoffs t=1.
            # if you normalize the data such that the full bunde has always value 1, init_E=1 is a good choice. If you choose random_ts~Unif(0,1), then init_E in [1,2] is recommended.
            # if the values of your MVNN are in a different order of magnitude, you should adjust init_E accordingly
            # (2) init_Var: gives (approximately) the variance of MVNN(1,1,...,1) at initialization if all cutoffs t=1. Typically we select values between 1/50 and 1. If init_E is in a different order of magnitude, init_Var should probably scale approximately with sqrt(init_E)
            # (3) init_b: init_b=0.05 is recommended. The "big" weights that are sampled from Unif(0,b). Typically b is calculated such that we get the chosen expectation and variance at the initilaization, but init_b is an upper lower bound for b.
            # (4) init_bias: init_bias=0.05 is recommended. All the biases are sampled uniformly from [0,init_bias].
            # (5) init_little_const: init_little_const=0.1 is recommended. Very technical parameter, probably not that important. If you set it to 0 it can happen that some parameters are initialized as exactly 0.
            #
            # Note that alle parameters have to be chosen non-negativ!
            '''

            c = init_E + init_bias / 2
            n = self.weight.shape[-1]
            v = init_Var

            b = max((3 * c ** 2 + 3 * n * v) / (2 * c * n) + init_little_const / n, init_b) if n > c ** 2 / (
                    3 * v) else 2 * c / n
            a = (3 * c ** 2 - 4 * b * c * n + b ** 2 * n ** 2 + 3 * n * v) * (2 * c - b * n * (
                    1 - (4 * c ** 2 - 4 * b * c * n + b ** 2 * n ** 2) / (
                    3 * c ** 2 - 4 * b * c * n + b ** 2 * n ** 2 + 3 * n * v))) / (
                        n * (4 * c ** 2 - 4 * b * c * n + b ** 2 * n ** 2)) if n > c ** 2 / (3 * v) else 0
            p = 1 - (4 * c ** 2 - 4 * b * c * n + b ** 2 * n ** 2) / (
                    3 * c ** 2 - 4 * b * c * n + b ** 2 * n ** 2 + 3 * n * v) if n > c ** 2 / (3 * v) else 1

            weight = []
            for rvs in scipy_stats.bernoulli.rvs(p, size=np.prod(self.weight.shape)):
                weight.append(float(np.random.uniform(0, a, 1) if rvs == 0 else float(np.random.uniform(0, b, 1))))
            weight = np.array(weight).reshape(self.weight.shape)
            weight = weight.astype(np.float32)
            self.weight.data = torch.from_numpy(weight)

            if self.bias is not None:
                bias = []
                for _ in range(np.prod(self.bias.shape)):
                    bias.append(float(np.random.uniform(-init_bias, 0)))
                bias = np.array(bias).reshape(self.bias.shape)
                bias = bias.astype(np.float32)
                self.bias.data = torch.from_numpy(bias)

        elif method == 'glorot_sqrt':
            self.weight.data = self.weight.data.abs() / torch.sqrt(6 * torch.tensor(sum(self.weight.shape)))
            if self.bias is not None:
                self.bias.data = -self.bias.data.abs() / torch.sqrt(6 * torch.tensor(sum(self.bias.shape)))

        elif method == 'glorot_default':
            self.weight.data = self.weight.data.abs()
            if self.bias is not None:
                self.bias.data = -self.bias.data.abs()

        elif method == 'zero':
            self.weight.data = torch.zeros_like(self.weight.data)
            if self.bias is not None:
                self.bias.data = torch.zeros_like(self.bias.data)
        else:
            raise NotImplementedError(f'Initialization method:{method} not implemented.')
