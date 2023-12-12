'''
Discrete forward Euler by Boffi 2022.
'''

from cwgf.solvers.solver_base import SolverBase
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from cwgf.solvers.models.snn import SNN
from cwgf.solvers.utils import jax_div, jax_div_hutchinson
from cwgf.problems.kl import KLDivergence
from cwgf.problems.tFPE import TimeFPE
from cwgf.problems.distribution import FuzzyPointCloud

class DFE(SolverBase):
    def __init__(self, *,
                 num_particle,
                 dt,
                 num_inner_step,
                 learning_rate,
                 save_dt,
                 pretrain_tol,
                 **kwargs):
        super().__init__(**kwargs)

        assert(isinstance(self.problem, TimeFPE) or
               isinstance(self.problem, KLDivergence))
        if isinstance(self.problem, TimeFPE):
            assert(self.problem.grad_W is None)

        self.num_particle = num_particle
        self.dt = dt
        self.num_inner_step = num_inner_step
        self.learning_rate = learning_rate
        self.save_dt = save_dt
        self.pretrain_tol = pretrain_tol
        self.save_freq = int(round(save_dt / dt))
        self.score = SNN(dim=self.problem.get_dim(),
                         num_layer=4,
                         layer_size=128,
                         activation_layer='celu',
                         embed_time_dim=0)


    def create_functions(self):
        def s_fn(params, x):
            # (D,) -> (D,)
            return self.score.apply({'params': params}, 0, x)
        self.s_vmap = jax.vmap(s_fn,
                               in_axes=(None, 0)) # params, (B, D) -> (B, D)
        s_div = jax_div(s_fn, argnums=1) # params, (D,) -> ()
        self.s_div_vmap = jax.vmap(s_div,
                                   in_axes=(None, 0)) # params, (B, D) -> (B,)


    def run(self):
        self.create_functions()

        rng = jax.random.PRNGKey(self.seed)
        total_time = self.problem.get_total_time()
        num_timestep = int(round(total_time / self.dt))
        dim = self.problem.get_dim()
        optimizer = optax.adam(
            learning_rate=self.learning_rate)
        prior = self.problem.get_prior()

        rng, score_rng = jax.random.split(rng)
        s_params = self.score.init(
            score_rng,
            0,
            jnp.zeros([dim])
        )['params']

        def score_loss_fn(s_params, x):
            s = self.s_vmap(s_params, x) # (B, D)
            s_div = self.s_div_vmap(s_params, x) # (B,)
            loss = (s ** 2).sum(-1) + 2 * s_div
            loss = loss.mean()
            return loss

        @jax.jit
        def train_score_step(params, opt_state, x):
            loss, grads = jax.value_and_grad(score_loss_fn)(params, x)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        prior_score = jax.grad(prior.log_p) # (D,) -> (D,)
        prior_score_vmap = jax.vmap(prior_score) # (B, D) -> (B, D)
        def pretrain_loss_fn(s_params, x):
            s_pred = self.s_vmap(s_params, x) # (B, D)
            s_goal = prior_score_vmap(x) # (B, D)
            norm = (s_goal ** 2).sum(-1).mean()
            return ((s_pred - s_goal) ** 2).sum(-1).mean() / norm

        @jax.jit
        def pretrain_step(params, opt_state, x):
            loss, grads = jax.value_and_grad(pretrain_loss_fn)(params, x)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        @jax.jit
        def compute_velocity(t, x, s):
            if isinstance(self.problem, KLDivergence):
                b_t = jax.vmap(self.problem.target_grad_log_p)(x) # (B, D)
                Ds_t = s # (B, D)
            else:
                b_t = jax.vmap(self.problem.b, in_axes=(None, 0))(t, x) # (B, D)
                D_t = jax.vmap(self.problem.D, in_axes=(None, 0))(t, x) # (B, D, D)
                Ds_t = jnp.squeeze(jnp.matmul(D_t, jnp.expand_dims(s, -1)), -1) # (B, D) TODO: check

            return b_t - Ds_t

        s_opt_state = optimizer.init(s_params)
        pretrain_count = 0
        while True:
            rng, sample_rng = jax.random.split(rng)
            samples = prior.sample(sample_rng, self.num_particle)

            s_params, s_opt_state, loss = pretrain_step(
                s_params, s_opt_state, samples
            )
            pretrain_count += 1
            if loss < self.pretrain_tol:
                break
        print('Pretraining takes {} iterations.'.format(pretrain_count))

        # Actual algorithm begins here.
        s_opt_state = optimizer.init(s_params)
        rng, sample_rng = jax.random.split(rng)
        samples = prior.sample(sample_rng, self.num_particle)
        self.sample_list = [samples]

        train_range = trange(num_timestep)
        for i in train_range:
            cur_t = i * self.dt
            for j in range(self.num_inner_step):
                s_params, s_opt_state, loss = train_score_step(
                    s_params, s_opt_state, samples
                )

            cur_score = self.s_vmap(s_params, samples) # (B, D)
            cur_velocity = compute_velocity(cur_t, samples, cur_score) # (B, D)
            samples = samples + self.dt * cur_velocity # (B, D)

            if (i + 1) % self.save_freq == 0:
                self.sample_list.append(samples)


        self.global_step = num_timestep
        self._validate()


    def extract_solution(self, t1):
        i = round(t1 / self.save_dt)
        t1_err = abs(i * self.save_dt - t1)
        if t1_err > 1e-4:
            print(('Extracting solution at t={} but nearest'
                  ' existing solution is for t={}!').format(t1, i*self.save_dt))
        if i >= len(self.sample_list):
            return None

        samples = self.sample_list[i]

        # Bandwidth by Scott's rule.
        h = np.power(samples.shape[0], -1. / (samples.shape[1] + 4))
        return FuzzyPointCloud(samples, h,
                               replace=True,
                               exact_sample=True)


    def eval_multi_t(self, rng, timesteps, val_num_sample):
        samples0 = self.sample_list[0]
        inds = jax.random.choice(rng, samples0.shape[0],
                                 [val_num_sample])
        x0 = samples0[inds]

        xt = []
        for t in timesteps:
            i = round(t / self.save_dt)
            t_err = abs(i * self.save_dt - t)
            assert(t_err < 1e-4)
            assert(i < len(self.sample_list))

            xt.append(self.sample_list[i][inds])

        xt = jnp.stack(xt, 1) # (B, T, D)

        return {
            'x0': x0,
            'xt': xt
        }
