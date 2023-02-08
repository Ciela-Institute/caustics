import torch
from torch.distributions import transform_to, biject_to


class Parameter:
    def __init__(self, constraints, prior=None, ):
        ...

    def keys(self):
        ...

    def values(self):
        ...

    def items(self):
        ...

    def prior(self):
        ...

    def constraints(self):
        ...

    def unconstrained_value(self):
        ...

    def constrained_value(self):
        ...

    def init(self):
        # initialize the value from the prior, if no prior is given, assume a Uniform prior over the constraints
        ...

    def __repr__(self):
        ...




