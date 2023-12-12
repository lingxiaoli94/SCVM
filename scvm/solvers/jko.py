'''
JKO of Morkov et al. 2021 and Fan et al. 2022.
'''
from scvm.solvers.solver_base import SolverBase

import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import trange
import wandb
from jax.experimental.host_callback import id_print
from scvm.solvers.flow_view import create_flow_view
from scvm.problems.distribution import FuncDistribution
from scvm.solvers.models.dummy import Dummy

class JKO(SolverBase):
    def __init__(self, *,
                 jko_len,
                 pretrain,
                 val_final_only,
                 jko_lr_decay,
                 thp,
                 flow,
                 potential,
                 optimizer,
                 **kwargs):
        '''
        Since time is discrete, we use flow and its view as pushforward
        only with t=1. The training state will have one set of flow parameters
        for each timestamp.

        If potential is Dummy, then this corresponds to Morkov et al. 2021.
        Otherwise, this is the primal-dual JKO from Fan et al. 2022.

        Args:
          jko_len:
            Number of JKO steps. If jko_len == 1, then equivalent to directly
            minimizing the functional.
          pretrain: Whether to pretrain to regress the identity map.
          flow: Pushforward network.
          potential: Potential network.
          thp.train_num_step: Number of training step for each timestamp.
          thp.val_freq: Validation frequency in terms of JKO steps.
          thp.save_freq: Saving frequency in terms of JKO steps.
        '''
        super().__init__(**kwargs)

        self.jko_len = jko_len
        self.jko_h = self.problem.get_total_time() / jko_len # JKO step size
        self.jko_lr_decay = jko_lr_decay
        self.pretrain = pretrain
        self.val_final_only = val_final_only

        self.thp = thp
        self.flow = flow
        if hasattr(flow, 'ignore_time') and not flow.ignore_time:
            raise Exception('Flow has the option to ignore time yet'
                  ' ignore_time=False!')
        if not isinstance(potential, Dummy):
            self.primal_dual = True
            self.potential = potential
        else:
            self.primal_dual = False
        self.flow_view = create_flow_view(flow)
        self.optimizer = optimizer


    def _pretrain(self, rng, params):
        def mse_fn(params, x):
            y = self.forward_vmap(params, x, False, False)
            return ((x - y) ** 2).sum(-1).mean()

        opt = optax.adam(learning_rate=1e-3)
        def pretrain_step(params, opt_state, x):
            loss, grads = jax.value_and_grad(mse_fn, argnums=0)(
                params, x
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        pretrain_step = jax.jit(pretrain_step)

        opt_state = opt.init(params)
        train_range = trange(2000)
        for j in train_range:
            rng, i_rng = jax.random.split(rng)
            x = self.problem.get_prior().sample(i_rng, self.thp.train_batch_size)
            params, opt_state, loss = pretrain_step(params, opt_state, x)
            train_range.set_description('Pretrain | Loss: {:.6f}'.format(
                loss
            ))
        return params


    def _train(self):
        rng = jax.random.PRNGKey(self.seed)

        rng, flow_rng = jax.random.split(rng)
        f_params = self.flow.init(flow_rng,
                                  *self.flow_view.get_init_args(
                                      self.problem.get_dim()
                                  ))['params']
        if self.primal_dual:
            rng, potential_rng = jax.random.split(rng)
            p_params = self.potential.init(
                potential_rng,
                0.,
                jnp.zeros([self.problem.get_dim()])
            )['params']

        if self.pretrain:
            rng, pretrain_rng = jax.random.split(rng)
            f_params = self._pretrain(pretrain_rng, f_params)

        self.param_list = []
        for i in range(self.jko_len):
            if self.jko_lr_decay:
                if self.primal_dual:
                    lr = 1e-3 if i < (self.jko_len // 2) else 4e-4
                else:
                    lr = 5e-3 if i < (self.jko_len // 2) else 2e-3
                self.optimizer = optax.adam(learning_rate=lr)

            f_opt_state = self.optimizer.init(f_params)
            if self.primal_dual:
                p_opt_state = self.optimizer.init(p_params)

            train_range = trange(self.thp.train_num_step)
            if self.primal_dual:
                # Compute mean and covariance.
                rng, g_rng = jax.random.split(rng)
                x = self.sample_till(g_rng, self.thp.train_batch_size,
                                     i, need_log_p=False) # (B, D)
                stats = self.problem.compute_dual_stats(x)

            for j in train_range:
                rng, i_rng = jax.random.split(rng)
                train_dict = {
                    'f_params': f_params,
                    'f_opt_state': f_opt_state,
                    'jko_step': i,
                    'rng': i_rng
                }
                if self.primal_dual:
                    train_dict.update({
                        'p_params': p_params,
                        'p_opt_state': p_opt_state,
                        'stats': stats,
                    })
                result_dict = self.train_step(train_dict)
                f_params = result_dict['f_params']
                f_opt_state = result_dict['f_opt_state']
                if not self.primal_dual:
                    train_range.set_description('JKO #{} | Loss: {:.6f}'.format(
                        i, result_dict['loss']
                    ))
                else:
                    p_params = result_dict['p_params']
                    p_opt_state = result_dict['p_opt_state']
                    train_range.set_description(
                        'JKO #{} | P Loss: {:.6f} | F Loss: {:.6f}'.format(
                            i, result_dict['p_loss'], result_dict['f_loss']
                        ))

            self.param_list.append(f_params)

            if self.primal_dual:
                self.p_params = p_params

            if (i == self.jko_len - 1) or (not self.val_final_only
                                           and  self.thp.should_validate(i+1)):
                self._validate()



    @property
    def global_step(self):
        return len(self.param_list)


    def get_current_end_time(self):
        return len(self.param_list) * self.jko_h


    def extract_solution(self, t1):
        i = round(t1 / self.jko_h)
        t1_err = abs(i * self.jko_h - t1)
        if t1_err > 1e-4:
            print(('Extracting solution at t={} but nearest'
                  ' existing solution is for t={}!').format(t1, i*self.jko_h))
        if i == 0:
            return self.problem.get_prior()

        # Otherwise use first i pushforwards.

        if i > len(self.param_list):
            # print(i,len(self.param_list))
            return None

        def sample_fn(rng, batch_size):
            return self.sample_till(rng, batch_size, i, need_log_p=False)

        def log_p_fn(x):
            log_p = 0
            for k in reversed(range(i)):
                x, ldj = self.forward_fn(
                    self.param_list[k], x, True, True)
                log_p += ldj
            log_p += self.problem.prior.log_p(x)
            return log_p

        return FuncDistribution(sample_fn=sample_fn, log_p_fn=log_p_fn)


    def _create_functions(self):
        need_log_p = self.problem.require_log_p()

        def forward_fn(f_params, x0, reverse, ldj):
            '''
            Use dummy time t=1.0. Some model will use it (e.g. VNN) while
            the others will ignore it (with ignore_time=True).
            '''
            return self.flow_view.forward(f_params, 1., x0, reverse, ldj)

        self.forward_fn = jax.jit(forward_fn,
                                  static_argnames=['reverse', 'ldj'])
        forward_vmap = jax.vmap(forward_fn,
                              in_axes=(None, 0, None, None))
        self.forward_vmap = jax.jit(forward_vmap,
                                    static_argnames=['reverse', 'ldj'])
        '''
        forward_vmap: (f_params, x0, reverse, ldj) -> x1 or (x1, ldj_t):
          x0: (B, D)
          x1: (B, D)
          ldj_t: (B,)
        '''

        if self.primal_dual:
            def p_fn(p_params, x):
                return self.potential.apply({'params': p_params}, 1.0, x)

            self.p_vmap = jax.vmap(p_fn,
                                   in_axes=(None, 0))
            '''
            p_vmap: (p_params, x) -> p:
            x: (B, D)
            p: (B,)
            '''

            self.dual_A_vmap = jax.vmap(self.problem.dual_A,
                                        in_axes=(0, 0, None))
            self.dual_B_vmap = jax.vmap(self.problem.dual_B,
                                        in_axes=(0, 0, None))

        self._create_train_step()




    def sample_till(self, rng, batch_size, jko_step, need_log_p):
        prior = self.problem.get_prior()
        x = prior.sample(rng, batch_size)
        if need_log_p:
            log_p = prior.log_p_batch(x)
        # Push x forward, the slowest step.
        for k in range(jko_step):
            x = self.forward_vmap(self.param_list[k], x, False, need_log_p)
            if need_log_p:
                x, ldj = x
                log_p -= ldj

        if need_log_p:
            return x, log_p
        return x


    def _create_train_step(self):
        if self.primal_dual:
            def potential_loss_fn(f_params, p_params, x, z, stats):
                x_push = self.forward_vmap(f_params, x, False, False) # (B, D)
                A = self.dual_A_vmap(
                    self.p_vmap(p_params, x_push), x_push, stats).mean()
                B = self.dual_B_vmap(
                    self.p_vmap(p_params, z), z, stats).mean()
                loss = A - B # (B,)
                # Negation since we need to maximize.
                return -loss.mean()

            def flow_loss_fn(f_params, p_params, x, stats):
                x_push = self.forward_vmap(f_params, x, False, False) # (B, D)
                A = self.dual_A_vmap(
                    self.p_vmap(p_params, x_push), x_push, stats)
                loss = A
                if self.jko_len > 1: # if 1, then ignore the proximal term
                    loss += (1. / (2*self.jko_h) *
                             ((x_push - x) ** 2).sum(-1)) # (B,)
                return loss.mean()

            def train_step_impl(f_params, f_opt_state,
                                p_params, p_opt_state,
                                x, z, stats):
                p_loss = 0
                for j in range(self.thp.train_potential_rep):
                    loss, grads = jax.value_and_grad(
                        potential_loss_fn, argnums=1)(
                            f_params, p_params,
                            x, z, stats
                        )
                    updates, p_opt_state = self.optimizer.update(
                        grads, p_opt_state, p_params
                    )
                    p_params = optax.apply_updates(p_params, updates)
                    p_loss += loss
                p_loss /= self.thp.train_potential_rep

                f_loss = 0
                for j in range(self.thp.train_flow_rep):
                    loss, grads = jax.value_and_grad(
                        flow_loss_fn, argnums=0)(
                            f_params, p_params,
                            x, stats
                        )
                    updates, f_opt_state = self.optimizer.update(
                        grads, f_opt_state, f_params
                    )
                    f_params = optax.apply_updates(f_params, updates)
                    f_loss += loss
                f_loss /= self.thp.train_flow_rep

                return (f_params, f_opt_state, f_loss,
                        p_params, p_opt_state, p_loss)
            train_step_impl = jax.jit(train_step_impl)

            def train_step(train_dict):
                f_params = train_dict['f_params']
                f_opt_state = train_dict['f_opt_state']
                p_params = train_dict['p_params']
                p_opt_state = train_dict['p_opt_state']

                stats = train_dict['stats']
                x_rng, z_rng = jax.random.split(train_dict['rng'])
                x = self.sample_till(x_rng,
                                     self.thp.train_batch_size,
                                     train_dict['jko_step'],
                                     need_log_p=False)
                z = self.problem.dual_sample(z_rng,
                                             self.thp.train_batch_size,
                                             stats)
                f_params, f_opt_state, f_loss, \
                    p_params, p_opt_state, p_loss = train_step_impl(
                        f_params, f_opt_state,
                        p_params, p_opt_state,
                        x, z, stats)
                return {
                    'f_params': f_params,
                    'f_opt_state': f_opt_state,
                    'f_loss': f_loss,
                    'p_params': p_params,
                    'p_opt_state': p_opt_state,
                    'p_loss': p_loss,
                }


        else:
            need_log_p = self.problem.require_log_p()

            def loss_fn(f_params, x, log_p):
                '''
                \EE[ F(T(x)) + 1/(2h) ||T(x) - x||^2 ].

                Args:
                  x: (B, D), samples from last step.
                  log_p: (B,), log p of those samples.
                '''
                x_push = self.forward_vmap(f_params, x, False, need_log_p) # (B, D)
                if need_log_p:
                    x_push, ldj = x_push
                    log_p = log_p - ldj
                else:
                    log_p = None

                loss = jax.vmap(self.problem.eval_F)(x_push, log_p) # (B,)
                if self.jko_len > 1: # if 1, then ignore the proximal term
                    loss += 1. / (2*self.jko_h) * ((x_push - x) ** 2).sum(-1) # (B,)

                return loss.mean()

            def train_step_impl(f_params, f_opt_state, x, log_p):
                loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
                    f_params, x, log_p)
                updates, f_opt_state = self.optimizer.update(
                    grads, f_opt_state, f_params)
                f_params = optax.apply_updates(f_params, updates)
                return f_params, f_opt_state, loss
            train_step_impl = jax.jit(train_step_impl)

            def train_step(train_dict):
                f_params = train_dict['f_params']
                f_opt_state = train_dict['f_opt_state']
                x = self.sample_till(
                    train_dict['rng'],
                    self.thp.train_batch_size,
                    train_dict['jko_step'],
                    need_log_p=need_log_p
                )
                if need_log_p:
                    x, log_p = x
                f_params, f_opt_state, loss = train_step_impl(
                    f_params, f_opt_state, x, log_p)
                return {
                    'f_params': f_params,
                    'f_opt_state': f_opt_state,
                    'loss': loss
                }

        self.train_step = train_step


    def run(self):
        self._create_functions()
        if not self.thp.is_val:
            self._train()
        else:
            self._validate()
