'''
Generalized Entropy F(\mu) = (1/m-1) \int \mu^m dx
'''
from cwgf.problems.problem_base import ProblemBase
import jax
import jax.numpy as jnp

class GeneralizedEntropy(ProblemBase):
    def __init__(self, *,
                 dim,
                 m,
                 prior,
                 total_time,
                 uniform_scale,
                 volume_scale):
        '''
        Args:
          dim:
            Dimension.
          m:
            m > 0 parameter of the geenralized entropy
          prior:
            Prior distribution, instance of Distribution
          sample:
            Optional sample function: [rng, D] -> (B, D).
          total_time:
            Total flow time.
          uniform_scale: 
            scaling parameter for the bound of the uniform sampler
        '''
        self.dim = dim
        self.prior = prior
        self.total_time = total_time
        self.volume_scale = volume_scale
        self.uniform_scale = uniform_scale
        self.m = m


    def get_prior(self):
        return self.prior


    def get_total_time(self):
        return self.total_time


    def get_dim(self):
        return self.dim


    def require_log_p(self):
        return True


    def eval_F(self, X, log_p):
        p = jnp.exp(log_p)
        return (1/(self.m-1))*(p**(self.m-1)).squeeze()


    def eval_dF(self, X, log_p):
        p = jnp.exp(log_p)
        return (self.m/(self.m-1))*(p**(self.m-1)).squeeze()


    def compute_dual_stats(self, x_batch):
        max_bound = jnp.max(jnp.abs(x_batch))
        return {'max_bound': max_bound}


    def dual_sample(self, rng, batch_size, stats):
        minval = -self.uniform_scale*stats['max_bound']
        maxval = self.uniform_scale*stats['max_bound']
        return jax.random.uniform(rng, shape=(2*batch_size,self.dim),
                minval=minval, maxval=maxval)

    def dual_potential(self, p_x, x):
        '''
        The output of the potential network might not be the actual potential.
        Here return the actual potential corresponding to p_x.
        '''
        raise Exception('Not implemented!')

    def dual_A(self, p_x, x, stats):
        '''
        Args:
          p_x: ()
          x: (D,)
        Returns:
          A: ()
        '''
        h = p_x

        vol = (2*self.volume_scale*self.uniform_scale*stats['max_bound'])**self.dim
        return (self.m/(self.m-1))*(h/vol)**(self.m-1)


    def dual_B(self, p_z, z, stats):
        h = p_z
        vol = (2*self.volume_scale*self.uniform_scale*stats['max_bound'])**self.dim
        B = (h**(self.m))*((1/vol)**(self.m-1))
        return B

