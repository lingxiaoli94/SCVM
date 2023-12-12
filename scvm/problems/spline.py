from cwgf.problems.problem_base import ProblemBase

import jax
import jax.numpy as jnp

from cwgf.solvers.utils import jax_div

class Spline(ProblemBase):
    def __init__(self, *,
                 use_ot_loss,
                 ot_threshold,
                 debiased,
                 ot_eps,
                 dim,
                 prior,
                 key_timestamps,
                 key_dists,
                 time_bandwidth,
                 multiplier,
                 total_time):
        self.use_ot_loss = use_ot_loss
        self.ot_threshold = ot_threshold
        self.debiased = debiased
        self.ot_eps = ot_eps
        self.dim = dim
        self.prior = prior
        self.key_timestamps = key_timestamps
        self.key_dists = key_dists
        self.num_key = len(self.key_dists)
        self.key_grad_log_ps = [
            jax.grad(dist.log_p)
            for dist in self.key_dists
        ]
        self.time_bandwidth = time_bandwidth
        self.multiplier = multiplier
        self.total_time = total_time


    def get_dim(self):
        return self.dim


    def get_prior(self):
        return self.prior


    def require_log_p(self):
        return not self.use_ot_loss


    def require_rng(self):
        return self.use_ot_loss


    def get_total_time(self):
        return self.total_time


    def compute_v_goal(self, x, t, info):
        assert(False)


    def impl_v_goal_batch(self):
        return self.use_ot_loss


    def compute_w2_grad(self, x, y):
        from ott.geometry import pointcloud
        from ott.solvers.linear import sinkhorn
        from ott.tools.sinkhorn_divergence import sinkhorn_divergence
        def ot_cost(x, y):
            geom = pointcloud.PointCloud(x, y,
                                         epsilon=self.ot_eps,
                                         relative_epsilon=True)
            if not self.debiased:
                ot = sinkhorn.solve(geom, threshold=self.ot_threshold)
                return ot.reg_ot_cost
            else:
                ot = sinkhorn_divergence(geom, x=geom.x, y=geom.y,
                                         static_b=True,
                                         epsilon=self.ot_eps,
                                         sinkhorn_kwargs={'threshold': self.ot_threshold})
                return ot.divergence

        return jax.grad(ot_cost, argnums=0)(x, y)


    def compute_v_goal_batch(self, x, t, info):
        '''
        Args:
          x: (B, D)
          t: scalar
        Returns:
          (B, D)
        '''
        assert(self.use_ot_loss)
        rng = info['rng']
        batch_size = x.shape[0]
        v_goal = 0
        for i in range(self.num_key):
            w_i = self.compute_w_i(i, t)
            samples_i = self.key_dists[i].sample(rng, batch_size)
            v_i = -self.compute_w2_grad(x, samples_i) # (B, D)
            v_goal += w_i * v_i
        # jax.debug.print('{v_goal}', v_goal=v_goal)
        return v_goal * self.multiplier


    def compute_w_i(self, i, t):
        diff = jnp.abs(self.key_timestamps[i] - t)
        t1 = self.time_bandwidth * 0.4
        t2 = self.time_bandwidth * 0.5
        return jnp.where(
            diff < t1,
            1.0,
            jnp.where(diff < t2,
                      (t2 - diff) / (t2 - t1),
                      0.0))
        # return jax.scipy.stats.norm.pdf(
        #     t, loc=self.key_timestamps[i],
        #     scale=self.time_bandwidth)


    def compute_v_dot_ibp(self, x, t, info):
        assert(not self.use_ot_loss)
        v_fn = info['v_fn']
        params = info['params']

        v = v_fn(params, t, x)
        div = jax_div(v_fn, argnums=2)(params, t, x)

        result = 0
        for i in range(self.num_key):
            w_i = self.compute_w_i(i, t)
            result += w_i * ((v * self.key_grad_log_ps[i](x)).sum(-1) +
                             div)
        return result * self.multiplier
