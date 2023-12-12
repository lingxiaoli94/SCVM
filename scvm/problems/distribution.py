from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
import functools
import math
from jax import random
from functools import partial
from scvm.problems.utils import my_metropolis_hastings, metropolis_hastings

class Distribution(ABC):
    @abstractmethod
    def sample(self, rng, batch_size):
        '''
        Args:
          rng:
            Jax's rng.

        Returns:
          (B, D), samples
        '''
        pass


    def log_p(self, X):
        '''
        Args:
          X: (D,), a single point.
        '''
        pass

    def p(self, X):
        '''
        Args:
          X: (D,), a single point.
        '''
        pass


    def log_p_batch(self, X):
        '''
        Args:
          X: (B, D), a batch of points.
        '''
        return jax.vmap(lambda X: self.log_p(X))(X)

    def p_batch(self, X):
        '''
        Args:
          X: (B, D), a batch of points.
        '''
        return jax.vmap(lambda X: self.p(X))(X)



class FuncDistribution(Distribution):
    def __init__(self, sample_fn,
                 log_p_fn=None,
                 log_p_batch_fn=None):
        self.sample_fn = sample_fn
        self.log_p_fn = log_p_fn
        self.log_p_batch_fn = log_p_batch_fn

    def sample(self, rng, batch_size):
        return self.sample_fn(rng, batch_size)

    def p(self, X):
        if self.log_p_fn is None :
            return None
        return jnp.exp(self.log_p_fn(X))

    def log_p(self, X):
        if self.log_p_fn is None:
            return None
        return self.log_p_fn(X)

    def p_batch(self, X):
        # p_batch_fn can contain jitted code.
        if self.log_p_batch_fn is not None:
            return jnp.exp(self.log_p_batch(X))
        if self.log_p_fn is None:
            return None
        return super().p_batch(X)

    def log_p_batch(self, X):
        # log_p_batch_fn can contain jitted code.
        if self.log_p_batch_fn is not None:
            return self.log_p_batch_fn(X)
        if self.log_p_fn is None:
            return None
        return super().log_p_batch(X)


class StdGaussian(Distribution):
    def __init__(self, dim):
        self.dim = dim
        self.mean = jnp.zeros((dim,))
        self.cov = jnp.eye(dim)

    def sample(self, rng, batch_size):
        return jax.random.normal(rng, (batch_size, self.dim))


    def log_p(self, X):
        return jax.scipy.stats.multivariate_normal.logpdf(
            X, self.mean, self.cov)


def gaussian_unnormalized_log_p(X, mean, cov_inv):
    X_centered = X - mean
    tmp = -0.5 * jnp.matmul(jnp.expand_dims(X_centered, -2),
                            jnp.matmul(cov_inv,
                                       jnp.expand_dims(X_centered, -1)))
    tmp = jnp.squeeze(tmp, (-2, -1))
    return tmp


def gaussian_log_Z(cov_sqrt):
    dim = cov_sqrt.shape[-1]
    log_Z = (-dim/2 * np.log(2 * np.pi) -
             np.linalg.slogdet(cov_sqrt)[1])
    return log_Z


def gaussian_sample(rng, batch_size, mean, cov_sqrt):
    dim = cov_sqrt.shape[-1]
    Z = jax.random.normal(rng, (batch_size, dim))
    X = jnp.squeeze(jnp.expand_dims(cov_sqrt, 0) @
                    jnp.expand_dims(Z, -1), -1)
    return X + mean


class Gaussian(Distribution):
    def __init__(self, mean, cov_sqrt):
        '''
        Args:
          mean: (D,)
          cov_sqrt: (D, D), actual covariance is (cov_sqrt @ cov_sqrt^T).
        '''
        self.dim = mean.shape[0]
        self.mean = mean
        self.cov_sqrt = cov_sqrt

        self.log_Z = gaussian_log_Z(self.cov_sqrt)
        self.cov_inv = jnp.linalg.inv(self.cov_sqrt @ self.cov_sqrt.T)


    def sample(self, rng, batch_size):
        return gaussian_sample(rng, batch_size,
                               self.mean, self.cov_sqrt)


    def log_p(self, X):
        return gaussian_unnormalized_log_p(
            X, self.mean, self.cov_inv) + self.log_Z


    def get_cov(self):
        return self.cov_sqrt @ self.cov_sqrt.T


class OUSolution:
    def __init__(self, A, b):
        self.dim = A.shape[0]
        self.w, self.v = np.linalg.eig(A)
        self.A_inv = np.linalg.inv(A)
        self.b = b


    def get_solution(self, t):
        I = np.eye(self.dim)
        mean = (I - self.get_neg_exp(t)) @ self.b
        neg_2_exp = self.get_neg_exp(2 * t)
        cov = self.A_inv @ (I - neg_2_exp) + neg_2_exp
        cov_sqrt = np.linalg.cholesky(cov)
        return Gaussian(mean, cov_sqrt)


    def get_neg_exp(self, t):
        '''
        Returns: e^{-At}
        '''
        return self.v @ np.diag(np.exp(-self.w * t)) @ self.v.T


class Gaussian_cov(Distribution):
    def __init__(self, mean, cov):
        '''
        Args:
          mean: (D,)
          cov: (D, D)
        '''
        self.dim = mean.shape[0]
        self.mean = mean
        self.cov = cov

        self.log_Z = gaussian_log_Z(self.cov_sqrt)
        self.cov_inv = jnp.linalg.inv(self.cov)


    def sample(self, rng, batch_size):
        return gaussian_sample(rng, batch_size,
                               self.mean, self.cov_sqrt)


    def log_p(self, X):
        return gaussian_unnormalized_log_p(
            X, self.mean, self.cov_inv) + self.log_Z



class Mixture(Distribution):
    def __init__(self, mixtures, weights):
        '''
        Args:
          mixtures: a list of Distribution.
          weights: weights of mixture, sum up to 1.
        '''
        self.mixtures = mixtures
        self.num_mixture = len(mixtures)
        self.weights = weights
        self.logit_weights= jnp.log(weights)

        assert(self.weights.shape[0] == self.num_mixture)

        self.select_fn = jax.vmap(lambda s_all, c: s_all[:, c])


    def sample(self, rng, batch_size):
        choices = jax.random.categorical(rng, self.logit_weights,
                                         axis=-1,
                                         shape=(batch_size,))

        rngs = jax.random.split(rng, self.num_mixture + 1)
        rng = rngs[0]
        rngs = rngs[1:]

        samples_each = []
        for i, mixture in enumerate(self.mixtures):
            samples_each.append(mixture.sample(rngs[i], batch_size))
        samples_all = jnp.stack(samples_each, -1) # (B, D, M)

        # samples[i, :] = samples_all[i, :, choices[i]]
        samples = self.select_fn(samples_all, choices)
        return samples


    def log_p(self, X):
        '''
        log(\sum_i w_i exp(log_p_i(X)))
        '''
        log_p_each = []
        for mixture in self.mixtures:
            log_p_each.append(mixture.log_p(X))
        log_p_all = jnp.stack(log_p_each, -1) # (M,)
        log_p = jax.scipy.special.logsumexp(log_p_all,
                                            axis=-1, b=self.weights) # ()
        log_p -= math.log(self.num_mixture)
        return log_p


class FuzzyPointCloud(Distribution):
    def __init__(self, points, bandwidth,
                 replace=False,
                 exact_sample=False):
        '''
        Args:
          points: (B, D)
          exact_sample: if True, then do not add noise
        '''
        self.points = jnp.array(points)
        self.bandwidth = bandwidth
        self.replace = replace
        self.exact_sample = exact_sample


    def sample(self, rng, batch_size):
        rng1, rng2 = jax.random.split(rng)
        if self.replace:
            replace = True
        else:
            replace = batch_size > self.points.shape[0]
        choices = jax.random.choice(rng1, len(self.points),
                                    replace=replace,
                                    shape=(batch_size,))
        points = self.points[choices]
        if not self.exact_sample:
            points += self.bandwidth * jax.random.normal(rng2, points.shape)
        return points


    def log_p(self, x):
        diff = x - self.points # (B, D)
        lpdfs = jax.scipy.stats.norm.logpdf(jnp.reshape(diff, -1),
                                            loc=0, scale=self.bandwidth)
        lpdfs = jnp.reshape(lpdfs, self.points.shape).sum(-1) # (B,)
        log_p = jax.scipy.special.logsumexp(lpdfs)
        log_p -= math.log(self.points.shape[0])
        return log_p


class Barenblatt(Distribution):
    def __init__(self, t, t0, x0, m, dim, bound, C = 0., init_samples = True, size_init = int(1e6), num_warmup_sampler=1000, stepsize_sampler=0.1, MH_code = 'pyro'):
        self.alpha = dim/(dim*(m-1)+2)
        self.beta = (m-1)*self.alpha/(2*dim*m)
        self.t = t
        self.t0 = t0
        self.x0 = x0
        self.dim = dim
        if C > 0 :
            self.C = C
        else :
            self.C = self.beta * (bound**2) * t0**(-2 * self.alpha / self.dim)
        self.m = m
        self.init_samples = init_samples
        self.num_warmup_sampler = num_warmup_sampler
        self.stepsize_sampler = stepsize_sampler
        self.MH_code = MH_code
        if init_samples :
            self.samples_init = self.my_sample(jax.random.PRNGKey(0), num_warmup_sampler, size_init, self.stepsize_sampler)

        self.bound = np.sqrt(self.C / (self.beta*(self.t + self.t0)**(-2 * self.alpha / self.dim)))

    def my_sample(self, rng, batch_size, num_warmup, stepsize):
        x0 = jnp.zeros(self.dim)
        if self.MH_code == 'pyro':
            samples = metropolis_hastings(rng, self.log_p, num_warmup, batch_size, stepsize, x0).reshape(batch_size,self.dim)
        else :
            samples = my_metropolis_hastings(rng, self.log_p, num_warmup, batch_size, stepsize, x0).reshape(batch_size,self.dim)
        return samples

    def sample(self,rng,batch_size):
        if self.init_samples :
            samples = jax.random.choice(rng,self.samples_init,(batch_size,1)).reshape(batch_size,self.dim)
        else :
            samples = self.my_sample(rng, batch_size, self.num_warmup_sampler, self.stepsize_sampler)
        return samples

    def p(self,X):
        A = self.C - self.beta * (jnp.linalg.norm(X-self.x0,axis=-1)**2) * (self.t+self.t0)**(-2 * self.alpha / self.dim)
        p = ((self.t+self.t0)**(-self.alpha))*(A * (A > 0))**(1/(self.m-1))
        return p

    def log_p(self, X):
        return jnp.log(self.p(X)+1e-15)


class NL_FP_solution(Distribution):
    def __init__(self, m, V, C, dim, num_warmup_sampler=1000, stepsize_sampler=0.1):
        self.m = m
        self.dim = dim
        self.V = V
        self.C = C
        self.num_warmup_sampler = num_warmup_sampler
        self.stepsize_sampler = stepsize_sampler

    def sample(self, rng, batch_size):
        x0 = jnp.zeros(self.dim)
        samples = metropolis_hastings(rng, self.log_p, self.num_warmup_sampler, batch_size, self.stepsize_sampler, x0, init=None).reshape(batch_size,self.dim)
        return samples

    def p(self,X):
        A = self.C - ((self.m-1)/self.m)*self.V(X)
        return (A * (A > 0))**(1/(self.m-1))

    def log_p(self, X):
        return jnp.log(self.p(X)+1e-15)

