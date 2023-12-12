from flax import linen as nn

from typing import Any

class ActivationModule(nn.Module):
    fn: Any


    def __call__(self, x):
        return self.fn(x)


class ActivationFactory:
    @staticmethod
    def create(name):
        if name == 'relu':
            fn = nn.relu
        elif name == 'tanh':
            fn = nn.tanh
        elif name == 'celu':
            fn = nn.celu
        elif name == 'gelu':
            fn = nn.gelu
        elif name == 'elu':
            fn = nn.elu
        elif name == 'silu':
            fn = nn.silu
        elif name == 'softplus':
            fn = nn.softplus
        elif name == 'prelu':
            fn = nn.activation.PReLU()
        else:
            raise Exception(f'Unknown activation name: {name}')
        return ActivationModule(fn)
