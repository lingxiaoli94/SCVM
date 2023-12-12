'''
Self-consistent velocity-matching solver in the generalized case.
'''

from scvm.solvers.flow_base import FlowBase

import jax
import jax.numpy as jnp
import wandb
from functools import partial

from scvm.auto.train_state import FlowScoreTrainState
from scvm.solvers.flow_view import PushforwardView, VelocityView

class SCVMS(FlowBase):
    def __init__(self, *,
                 use_ibp,
                 ode_score,
                 prox_weight,
                 smooth_weight,
                 smooth_decay,
                 splitting,
                 **kwargs):
        '''
        Args:
          use_ibp: whether to use integration by parts to avoid \nabla log p_t.
        '''
        super().__init__(**kwargs)
        self.use_ibp = use_ibp
        if ode_score:
            assert(not use_ibp)
            assert(issubclass(type(self.flow_view), VelocityView))
        self.ode_score = ode_score
        self.prox_weight = prox_weight
        self.smooth_weight = smooth_weight
        self.smooth_decay = smooth_decay
        self.splitting = splitting

    def create_train_state(self):
        if not self.splitting:
            return super().create_train_state()

        rng = jax.random.PRNGKey(self.seed)
        rng, flow_rng = jax.random.split(rng)
        f_params = self.flow.init(flow_rng,
                                *self.flow_view.get_init_args(
                                    self.problem.get_dim()
                                ))['params']

        # Note: we use s_param to denote not the score, but the splitted flow.
        return FlowScoreTrainState.create(
            rng=rng, f_params=f_params, s_params=f_params,
            tx=self.optimizer
        )


    def create_more_functions(self):
        info_in_axes = {'samples': 1, 'params': None}
        if self.problem.require_noise():
            info_in_axes['noise'] = 1
        if self.problem.require_log_p():
            info_in_axes['log_p_fn'] = None
        if self.problem.require_rng():
            info_in_axes['rng'] = None
        if self.use_ibp:
            info_in_axes['v_fn'] = None

        if self.use_ibp:
            self.v_dot_ibp_vmap = jax.vmap(
                jax.vmap(self.problem.compute_v_dot_ibp,
                         in_axes=(0, 0, info_in_axes)),
                in_axes=(0, None, None))

        if self.ode_score:
            self.v_goal_vmap = jax.vmap(
                jax.vmap(self.problem.compute_v_goal_with_score,
                         in_axes=(0, 0, 0, info_in_axes)),
                in_axes=(0, None, 0, None))

            score_fn = self.flow_view.forward_multi_t_with_score
            self.score_vmap = jax.vmap(
                score_fn,
                in_axes=(None, None, 0, 0)
            )
        else:
            if self.problem.impl_v_goal_batch():
                self.v_goal_vmap = jax.vmap(self.problem.compute_v_goal_batch,
                                            in_axes=(1, 0, info_in_axes),
                                            out_axes=1)
            else:
                self.v_goal_vmap = jax.vmap(
                    jax.vmap(self.problem.compute_v_goal,
                             in_axes=(0, 0, info_in_axes)),
                    in_axes=(0, None, None))



    def create_train_step(self):
        def loss_fn(params, last_params, batch, rng, itr):
            '''
            Args:
              t: (T,), array of increasing time.
              x0: (B, D), initial points.
            '''
            t, x0 = batch['t'], batch['x0']
            xt = self.forward_vmap(last_params, t, x0, False, False) # (B, T, D)
            xt = jax.lax.stop_gradient(xt)

            if self.thp.train_noise_level > 0:
                noise = batch['noise'] * self.thp.train_noise_level
                if self.thp.train_noise_scale_t:
                    scale = 1. - t / self.problem.get_total_time() # (T)
                    scale = jnp.expand_dims(scale, -1) # (T, 1)
                    noise = noise * scale
                xt = xt + noise

            if isinstance(self.flow_view, PushforwardView):
                # This is faster for PushforwarView.
                vt = self.v_from_vmap(params, t, x0, False) # (B, T, D)
            else:
                vt = self.v_at_vmap(params, t, xt, False) # (B, T, D)

            info = {
                'params': params,
                'samples': xt, # (B, T, D)
            }
            if self.problem.require_noise():
                rng, noise_rng = jax.random.split(rng)
                info['noise'] = jax.random.normal(noise_rng, [4, xt.shape[-2], xt.shape[-1]])
            if self.problem.require_log_p():
                info['log_p_fn'] = self.log_p_fn
            if self.problem.require_rng():
                info['rng'] = rng

            if self.use_ibp:
                info['v_fn'] = partial(self.flow_view.velocity_at,
                                       reverse=False)
                v_dot = self.v_dot_ibp_vmap(xt, t, info) # (B, T)
                loss = (vt * vt).sum(-1) - 2 * v_dot # (B, T)
                loss = loss.mean()
            else:
                if self.ode_score:
                    scores0 = jax.vmap(jax.grad(self.problem.prior.log_p))(x0)
                    scores = self.score_vmap(last_params, t, x0, scores0)[1]
                    scores = jax.lax.stop_gradient(scores)
                    v_goal = self.v_goal_vmap(xt, t, scores, info) # (B, T, D)
                else:
                    # Most general velocity matching branch.
                    v_goal = self.v_goal_vmap(xt, t, info) # (B, T, D)
                    v_goal = jax.lax.stop_gradient(v_goal)
                loss = jnp.sum((vt - v_goal) ** 2, -1).mean()

            if self.prox_weight > 0:
                # If train_flow_rep=1 then no effect
                assert(self.thp.train_flow_rep > 1)
                last_vt = self.v_at_vmap(last_params, t, xt, False) # (B, T, D)
                loss += self.prox_weight * ((vt - last_vt) ** 2).sum(-1).mean()

            loss += self.soft_init_loss(params, x0)

            if self.smooth_weight > 0:
                dvdx = jax.jacfwd(self.flow_view.velocity_at, argnums=2)
                dvdx = jax.vmap(dvdx, in_axes=(None, None, 0, None))
                dvdx = jax.vmap(dvdx, in_axes=(None, 0, 1, None), out_axes=1)

                dvdx_val = dvdx(params, t, xt, False) # (B, T, D, D)

                damp = self.smooth_weight * jnp.exp(-self.smooth_decay * itr)
                # loss += damp * (vt * vt).sum(-1).mean()
                loss += damp * (dvdx_val * dvdx_val).sum(-1).sum(-1).mean()
            return loss


        def splitting_loss_fn(params1, params2, batch):
            t, x0 = batch['t'], batch['x0']
            xt1 = self.forward_vmap(params1, t, x0, False, False) # (B, T, D)
            xt2 = self.forward_vmap(params2, t, x0, False, False) # (B, T, D)
            xt1 = jax.lax.stop_gradient(xt1)
            xt2 = jax.lax.stop_gradient(xt2)
            xt = jnp.concatenate([xt1, xt2], 0)

            vt1 = self.v_at_vmap(params1, t, xt, False) # (B, T, D)
            vt2 = self.v_at_vmap(params2, t, xt, False) # (B, T, D)
            loss = ((vt1 - vt2) ** 2).sum(-1).mean()
            return loss


        def train_inner_body(args):
            i, state, last_params, rng, itr, total_loss = args
            rng, j_rng, loss_rng = jax.random.split(rng, 3)
            batch = self.sample_batch(state, j_rng)
            if self.splitting:
                params = state.s_params
            else:
                params = state.f_params
            loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
                params, last_params, batch, loss_rng, itr)
            if self.splitting:
                state = state.apply_score_grad(grads=grads)
            else:
                state = state.apply_flow_grad(grads=grads)
            return (i + 1, state, last_params, rng, itr, total_loss + loss)


        def train_splitting_body(args):
            i, state, rng, total_loss = args
            rng, j_rng = jax.random.split(rng)
            batch = self.sample_batch(state, j_rng)
            loss, grads = jax.value_and_grad(splitting_loss_fn, argnums=0)(
                state.f_params, state.s_params, batch)
            state = state.apply_flow_grad(grads=grads)
            return (i + 1, state, rng, total_loss + loss)


        def train_step(state, rng, itr):
            avg_loss = 0
            last_params = state.f_params
            args = jax.lax.while_loop(
                lambda args: args[0] < self.thp.train_flow_rep,
                train_inner_body,
                (0, state, last_params, rng, itr, 0)
            )
            state, total_loss = args[1], args[-1]

            if self.splitting:
                args = jax.lax.while_loop(
                    lambda args: args[0] < self.thp.train_score_rep,
                    train_splitting_body,
                    (0, state, rng, 0)
                )
                state, splitting_loss = args[1], args[-1]

            diff_tree = jax.tree_util.tree_map(lambda p1, p2: ((p1 - p2)**2).sum(),
                                               last_params, state.f_params)
            param_diff_norm = jax.tree_util.tree_reduce(
                lambda p, s: p+s, diff_tree, 0)

            avg_loss = total_loss / self.thp.train_flow_rep
            loss_dict = {'loss': avg_loss,
                         'param_diff': param_diff_norm}
            if self.splitting:
                loss_dict['splitting_loss'] = splitting_loss
            return state, loss_dict

        train_step = jax.jit(train_step)
        return train_step


    def log_loss(self, loss_dict):
        loss = loss_dict['loss']
        log_dict = {'Flow Loss': loss,
                   'Param Diff': loss_dict['param_diff']}
        if self.splitting:
            log_dict['Splitting Loss'] = loss_dict['splitting_loss']
        wandb.log(log_dict,
                  step=self.global_step)
        return f'Step {self.global_step} | Loss: {loss}'


    def eval_vm(self, rng, timesteps):
        assert(not self.use_ibp)
        batch = self.sample_batch(self.state, rng)
        _, x0 = batch['t'], batch['x0']
        t = timesteps
        params = self.state.f_params
        xt = self.forward_vmap(params, t, x0, False, False) # (B, T, D)
        vt = self.v_from_vmap(params, t, x0, False) # (B, T, D)

        info = {
            'params': params,
            'samples': xt, # (B, T, D)
        }
        if self.problem.require_log_p():
            info['log_p_fn'] = self.log_p_fn
        v_goal = self.v_goal_vmap(xt, t, info) # (B, T ,D)

        return {
            'v_goal': v_goal,
            'vt': vt,
            'x0': x0,
            'xt': xt
        }


    def eval_multi_t(self, rng, timesteps, val_num_sample):
        B = self.thp.train_batch_size
        x0_arr = []
        xt_arr = []
        remain = val_num_sample
        while remain > 0:
            rng, batch_rng = jax.random.split(rng)
            batch = self.sample_batch(self.state, batch_rng)
            _, x0 = batch['t'], batch['x0']
            t = timesteps
            params = self.state.f_params
            xt = self.forward_vmap(params, t, x0, False, False) # (B, T, D)

            sz = min(remain, B)
            remain -= B
            x0_arr.append(x0[:sz])
            xt_arr.append(xt[:sz])

        x0 = jnp.concatenate(x0_arr, 0)
        xt = jnp.concatenate(xt_arr, 0)
        return {
            'x0': x0,
            'xt': xt
        }


    def eval_log_density(self, x, timesteps):
        # x: (B, D)
        params = self.state.f_params
        from tqdm import tqdm

        log_ps = []
        for t in tqdm(timesteps):
            dist_t = self.extract_solution(t)
            log_ps.append(dist_t.log_p_batch(x))
        return jnp.stack(log_ps, 1) # (B, T)
