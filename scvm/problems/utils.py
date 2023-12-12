import jax
import jax.numpy as jnp
import numpy as np
import functools
from jax import random
from functools import partial

from collections import namedtuple
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC

# faster implementation
@partial(jax.jit, static_argnums=(1,2))
def metropolis_kernel(rng_key, log_density, stepsize, position, log_prob):
    key, subkey = jax.random.split(rng_key)
    move_proposals = jax.random.normal(key, shape=position.shape) * stepsize
    proposal = position + move_proposals
    proposal_log_prob = log_density(proposal)

    log_uniform = jnp.log(jax.random.uniform(subkey))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
    return key, (position, log_prob)

@partial(jax.jit, static_argnums=(1,2,3,4))
def my_metropolis_hastings(rng_key, log_density, num_warmup, num_samples, stepsize, x_0, verbose=False):

    def mh_step(carry, x):
        rng_key, (position, log_prob) = carry
        rng_key, (position, log_prob) = metropolis_kernel(rng_key, log_density, stepsize, position, log_prob)
        return (rng_key, (position, log_prob)), (position, log_prob)

    carry = (rng_key, (x_0, log_density(x_0)))
    carry, samples = jax.lax.scan(mh_step, carry, None, num_samples+num_warmup)
    return samples[0][num_warmup:]

# useful implementation for verbose
MHState = namedtuple("MHState", ["u", "rng_key"])
class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "u"
    def __init__(self, potential_fn, step_size=0.1):
        self.potential_fn = potential_fn
        self.step_size = step_size

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        return MHState(init_params, rng_key)

    def sample(self, state, model_args, model_kwargs):
        u, rng_key = state
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
        accept_prob = jnp.exp(self.potential_fn(u_proposal)-self.potential_fn(u))
        u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
        return MHState(u_new, rng_key)

def metropolis_hastings(rng_key, log_density, num_warmup, num_samples, stepsize, x_0, verbose = False):
    kernel = MetropolisHastings(log_density, step_size=stepsize)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=verbose)
    mcmc.run(rng_key, init_params=x_0)
    samples = mcmc.get_samples()
    return samples


if __name__ == '__main__' :

    # if dim < 4 :
    #     num_warmup_sampler = 1000
    #     stepsize_sampler = 0.1
    # elif dim < 7 :
    #     num_warmup_sampler = 2000
    #     stepsize_sampler = 0.05
    # elif dim < 10 :
    #     num_warmup_sampler = 5000
    #     stepsize_sampler = 0.05
    # elif dim < 13 :
    #     num_warmup_sampler = 10000
    #     stepsize_sampler = 0.05
    # else :
    #     num_warmup_sampler = 10000
    #     stepsize_sampler = 0.01

    from scvm.problems.distribution import Barenblatt
    t0 = 1e-3
    m = 2
    dim = 9
    p0_bound = 0.25
    x0 = jnp.zeros((1,dim))
    init_samples = False
    t = 0.02
    init_dist = Barenblatt(t, t0, x0, m, dim, p0_bound, init_samples=init_samples, num_warmup_sampler=2000, stepsize_sampler=0.1)
    samples = init_dist.sample(jax.random.PRNGKey(0), 1000)
