'''
General time dependant Fokker-Planck Equation with interaction term
v_t(x,\mu) = b_t(x) - D_t(x)\nabla log p_t(x) - \nabla W \times \mu_t(x)
'''
from cwgf.problems.problem_base import ProblemBase

import jax
import jax.numpy as jnp
from functools import partial
from cwgf.problems.distribution import \
    gaussian_unnormalized_log_p, gaussian_log_Z, gaussian_sample
from cwgf.solvers.utils import jax_div

class TimeFPE(ProblemBase):
    def __init__(self, *,
                 dim,
                 prior,
                 b,
                 D,
                 W=None,
                 grad_W=None,
                 sample_fn=None,
                 total_time):
        '''
        Args:
          dim:
            Dimension.
          prior:
            Prior distribution, instance of Distribution
          b:
            b(t,x) function (Gradient of the potential in the classical FPE).
          D:
            D(t,x) Time dependant Diffution function.
          W:
            W(t,x,y) Time dependant Interaction function. Defaullt : None
          sample_fn:
            Optional sample_fn function: (rng, D) -> (B, D).
          total_time:
            Total flow time.
        '''

        self.dim = dim
        self.prior = prior
        self.total_time = total_time
        self.W = W
        self.D = D
        self.b = b
        self.sample_fn = sample_fn
        if W is not None :
            self.grad_W = jax.grad(self.W,argnums=1)
        elif grad_W is not None :
            self.grad_W = grad_W
        else :
            self.grad_W  = None


    def get_prior(self):
        return self.prior


    def get_total_time(self):
        return self.total_time


    def get_dim(self):
        return self.dim


    def require_log_p(self):
        return True


    def compute_v_goal(self, x, t, info):
        log_p_fn = info['log_p_fn']
        xt = info['samples']
        grad_log_p_fn = jax.grad(log_p_fn, argnums=2)
        if self.grad_W is not None:
            return self.b(t,x) - self.D(t,x) @ grad_log_p_fn(info['params'], t, x) - self.grad_W(t,x,xt)
        else:
            return self.b(t,x) - self.D(t,x) @ grad_log_p_fn(info['params'], t, x)


    def compute_v_goal_with_score(self, x, t, score, info):
        if self.grad_W is not None:
            xt = info['samples']
            return self.b(t,x) - self.D(t,x) @ score - self.grad_W(t,x,xt)
        else:
            return self.b(t,x) - self.D(t,x) @ score


    def compute_v_dot_ibp(self, x, t, info):
        '''
        info should have v_fn: (params, t, x) -> (D,)
        '''
        v_fn = info['v_fn']
        params = info['params']
        xt = info['samples']
        Dv_fn = lambda params,t,x : self.D(t,x) @ v_fn(params, t, x)
        v = v_fn(params, t, x)
        div = jax_div(Dv_fn, argnums=2)(params, t, x)
        if self.grad_W is not None:
            result = (v * (self.b(t,x) - self.grad_W(t,x,xt))).sum(-1)
        else :
            result = (v * (self.b(t,x))).sum(-1)
        result += div
        return result


    def SDE_sampler(self, rng, batch_size, num_steps, include_init=False):
        init_x = self.prior.sample(rng, batch_size)
        time_steps = jnp.linspace(0,self.total_time,num_steps)
        step_size = time_steps[1] - time_steps[0]

        @partial(jax.jit)
        def forward_step(x, t, x_batch, z):
            if self.grad_W is not None:
                x = x + step_size*(self.b(t,x) - self.grad_W(t,x,x_batch)) + jnp.sqrt(2*step_size*self.D(t,x)) @ z
            else:
                x = x +step_size*self.b(t,x) + jnp.sqrt(2*step_size*self.D(t,x)) @ z
            return x

        forward_step_vmap = jax.vmap(forward_step, in_axes=(0, None, None, 0))
        x_list = []
        x = init_x
        if include_init:
            x_list.append(init_x)
        for t in time_steps:
            rng, t_rng = jax.random.split(rng, 2)
            z_batch = jax.random.normal(t_rng, x.shape)
            x = forward_step_vmap(x, t, x, z_batch)
            x_list.append(x)
        return x_list, step_size
