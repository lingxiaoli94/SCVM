'''
KL divergence for \mu with density q:
  KL(\mu || p) = \EE_{X\sim\mu}[\log q(X) - \log p(X)]
               = -Ent(\mu) - \EE_{X\sim\mu}[\log p(X)]
'''
from scvm.problems.problem_base import ProblemBase

import jax
import jax.numpy as jnp

from scvm.problems.distribution import \
    gaussian_unnormalized_log_p, gaussian_log_Z, gaussian_sample
from scvm.solvers.utils import jax_div, jax_div_hutchinson

class KLDivergence(ProblemBase):
    def __init__(self, *,
                 dim,
                 prior,
                 target_log_p=None,
                 sample_fn=None,
                 hutchinson=False,
                 dual_mode='importance',
                 total_time):
        '''
        Either target_log_p or sample_fn need to be provided.

        There are two dual modes:
          * If dual_mode == 'importance', then we use importance sampling,
            and only target_log_p will be used.
          * If dual_mode == 'empirical', then target_log_p is not used and
            only sample_fn is used.

        Args:
          dim:
            Dimension.
          prior:
            Prior distribution, instance of Distribution
          target_log_p:
            Optional target log density function: (D,) -> ().
            If unnormalized, eval_F will be off by a constant but the
            gradient flow is identity to that of the normalized one.
          sample_fn:
            Optional sample_fn function: (rng, D) -> (B, D).
          total_time:
            Total flow time.
        '''
        assert((target_log_p is not None) or (sample_fn is not None))
        self.dim = dim
        self.prior = prior
        self.total_time = total_time
        self.target_log_p = target_log_p
        if target_log_p is not None:
            self.target_grad_log_p = jax.grad(target_log_p)
        self.sample_fn = sample_fn
        self.hutchinson = hutchinson
        self.dual_mode = dual_mode


    def get_prior(self):
        return self.prior


    def get_total_time(self):
        return self.total_time


    def get_dim(self):
        return self.dim


    def require_log_p(self):
        return True


    def require_noise(self):
        return self.hutchinson


    def eval_F(self, x, log_p):
        assert(self.target_log_p is not None)
        return log_p - self.target_log_p(x)


    def eval_dF(self, x, log_p):
        assert(self.target_log_p is not None)
        return log_p - self.target_log_p(x) + 1


    def compute_v_goal(self, x, t, info):
        log_p_fn = info['log_p_fn']
        grad_log_p_fn = jax.grad(log_p_fn, argnums=2)
        return -(grad_log_p_fn(info['params'], t, x) -
                 self.target_grad_log_p(x))


    def compute_v_goal_with_score(self, x, t, score, info):
        # score = info['scores']
        return -(score -
                 self.target_grad_log_p(x))


    def compute_v_dot_ibp(self, x, t, info):
        '''
        info should have v_fn: (params, t, x) -> (D,)
        '''
        v_fn = info['v_fn']
        params = info['params']

        v = v_fn(params, t, x)
        if self.hutchinson:
            noise = info['noise']
            div = jax_div_hutchinson(v_fn, argnums=2)(
                params, t, x, eps=noise)
        else:
            div = jax_div(v_fn, argnums=2)(params, t, x)
        result = (v * self.target_grad_log_p(x)).sum(-1)
        result += div
        return result


    def compute_dual_stats(self, x_batch):
        '''
        Fit a gaussian to the batch of samples.
        Args:
          x_batch: (B, D)
        '''
        assert(self.target_grad_log_p is not None)
        assert(x_batch.ndim == 2)

        mean = jnp.mean(x_batch, 0) # (D,)
        if self.dual_mode == 'empirical':
            return {'mean': mean} # just throw in something to avoid empty dict
        cov = jnp.cov(x_batch, rowvar=False) # (D, D)

        cov_sqrt = jnp.linalg.cholesky(cov)
        cov_inv = jnp.linalg.inv(cov_sqrt @ cov_sqrt.T)
        log_Z = gaussian_log_Z(cov_sqrt)

        return {
            'mean': mean,
            'cov_sqrt': cov_sqrt,
            'cov_inv': cov_inv,
            'log_Z': log_Z
        }


    def dual_sample(self, rng, batch_size, stats):
        if self.dual_mode == 'empirical':
            assert(self.sample_fn is not None)
            return self.sample_fn(rng, batch_size)
        return gaussian_sample(rng, batch_size,
                               stats['mean'], stats['cov_sqrt'])


    def pos_act(self, p_x):
        return jax.nn.softplus(p_x) + 1e-8


    def dual_potential(self, p_x, x, stats):
        h = self.pos_act(p_x)
        log_h = jnp.log(h)
        if self.dual_mode == 'importance':
            log_q = self.target_log_p(x)
            log_gamma = gaussian_unnormalized_log_p(
                x, stats['mean'], stats['cov_inv']) + stats['log_Z']
            log_h += log_gamma - log_q
        return log_h + 1


    def dual_A(self, p_x, x, stats):
        '''
        Args:
          p_x: ()
          x: (D,)
        Returns:
          A: ()
        '''
        h = self.pos_act(p_x)
        log_h = jnp.log(h)
        if self.dual_mode == 'empirical':
            return log_h + 1
        log_q = self.target_log_p(x)
        log_gamma = gaussian_unnormalized_log_p(
            x, stats['mean'], stats['cov_inv']) + stats['log_Z']
        return log_h + log_gamma - log_q + 1


    def dual_B(self, p_z, z, stats):
        h = self.pos_act(p_z)
        return h

