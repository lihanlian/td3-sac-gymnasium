import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils

class TanhTransform(pyd.transforms.Transform):
    '''
    what are these used for?
    '''
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5*(x.log1p() - (-x).log1p())
    
    def __eq__(self, other):
        return isinstance(other, TanhTransform)
    
    def _call(self, x):
        return x.tanh()
    
    def _inverse(self, y):
        return self.atanh(y)
    
    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))
    
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2*action_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        # .chunk(2, dim=1) splits the tensor into 2 pieces along dimension 1.
        # equivalent to mu = x[:, :action_dim], log_std = x[:, action_dim:]
        mu, log_std = self.trunk(obs).chunk(2, dim=1)

        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

class TD3Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, action_range):
        super().__init__()
        # hidden_dim = 256, hidden_depth = 2
        self.net =  utils.mlp(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            hidden_depth=hidden_depth,
            output_mod=nn.Tanh()      # this replaces torch.tanh at the end
        )
        self.max_action = action_range[1]
        # self.max_action = max_action

    def forward(self, state):
        # net(state) is in [-1, 1] thanks to Tanh
        return self.max_action * self.net(state)

