import flax.linen as nn

class Noop(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x
