'''
Adjoint solver from Shen et al. 2022.
'''

from scvm.solvers.flow_base import FlowBase

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint as jax_odeint
import numpy as np
import wandb

from scvm.solvers.models.vnn import VNN
from scvm.solvers.utils import jax_div
from scvm.problems.tFPE import TimeFPE
from scvm.problems.kl import KLDivergence
from scvm.auto.train_state import FlowTrainState


class AdjointSolver(FlowBase):
    def __init__(self,
                 use_vjp,
                 **kwargs):
        super().__init__(**kwargs)

        assert(isinstance(self.flow, VNN))
        assert(isinstance(self.problem, TimeFPE) or
               isinstance(self.problem, KLDivergence))
        if isinstance(self.problem, TimeFPE):
            assert(self.problem.grad_W is None)

        self.use_vjp = use_vjp
        self.tol = self.flow.ode_tol
        self.total_time = self.problem.get_total_time()


    def create_train_step(self):
        prior = self.problem.get_prior()
        v_fn = lambda params, t, x: self.flow.apply(
            {'params': params},
            t, x, False) # (params, (), (D,)) -> (D,)
        v_div = jax_div(v_fn, 2) # (params, (), (D,)) -> ()
        v_grad_div = jax.grad(v_div, 2) # (params, (), (D,)) -> (D,)

        D = self.problem.get_dim()
        def psi_fn(params, s, t):
            x, xi = s[:D], s[D:]

            fn = lambda x: v_fn(params, t, x) # (D,) -> (D,)
            if self.use_vjp:
                fn = lambda x: (v_fn(params, t, x) * xi).sum()
                xi_dt = -v_grad_div(params, t, x) - jax.grad(fn)(x)
            else:
                _, jvp = jax.jvp(fn, (x,), (xi,))
                xi_dt = -v_grad_div(params, t, x) - jvp

            return jnp.concatenate([
                v_fn(params, t, x),
                xi_dt
            ])

        def g_fn(params, s, t):
            x, xi = s[:D], s[D:]
            v_goal = self.problem.compute_v_goal_with_score(
                x, t, xi, None)
            v_pred = v_fn(params, t, x)
            return ((v_goal - v_pred) ** 2).sum()

        def forward_fn(params, x0, xi0):
            '''
            Args:
                x0: (D,)
                xi0: (D,)
            Returns:
                sT: (2*D,)
            '''
            s0 = jnp.concatenate([x0, xi0])
            psi_partial = lambda s, t: psi_fn(params, s, t)
            ts = jnp.array([0, self.total_time])
            sol = jax_odeint(psi_partial, s0, ts,
                             rtol=self.tol, atol=self.tol)
            return sol[1]

        def forward_pass(params, x0, xi0):
            '''
            Args:
              x0: (B, D)
              xi0: (B, D)
            Returns:
              sT (B, 2*D)
            '''
            return jax.vmap(forward_fn,
                            in_axes=(None, 0, 0))(params, x0, xi0)


        def backward_fn(params, sT, ts):
            '''
            Args:
              sT: (2*D,)
              ts: (T,) output times, increasing
            Returns:
              (s_ts, a_ts), (T, 2*D)
            '''

            def dstate_dt(state, t):
                s, a = state[:2*D], state[2*D:]
                dsdt = psi_fn(params, s, t) # (2D,)

                dgds = jax.grad(g_fn, 1)(params, s, t) # (2D,)
                fn = lambda s: psi_fn(params, s, t) # (2D,) -> (2D,)

                # This is the part where 3rd derivative of velocity field is needed!
                if self.use_vjp:
                    fn = lambda s: (psi_fn(params, s, t) * a).sum() # (2D,) -> (2D,)
                    jvp = jax.grad(fn)(s)
                else:
                    _, jvp = jax.jvp(fn, (s,), (a,)) # (2D,)

                return jnp.concatenate([dsdt, -dgds - jvp])


            aT = jnp.zeros([2 * D])
            stateT = jnp.concatenate([sT, aT])

            # Add time 0.
            # (t1, t2, t3) -> (0, T-t3, T-t2, T-t1) to be strictly increasing.
            ts = jnp.concatenate([jnp.zeros(1), self.total_time - jnp.flip(ts)])

            # Solve backward as if it's forward with reversed velocity.
            sol = jax_odeint(lambda state, t: -dstate_dt(state, self.total_time - t),
                             stateT, ts,
                             rtol=self.tol, atol=self.tol)
            return (jnp.flip(sol[1:, :2*D], 0),
                    jnp.flip(sol[1:, 2*D:], 0))


        def backward_pass(params, sT, ts):
            '''
            Args:
              sT: (B, 2*D)
              ts: (T,)
            Returns:
              (s_t, a_t), where s_t and a_t have shape (B, T, 2*D)
            '''
            return jax.vmap(
                backward_fn,
                in_axes=(None, 0, None)
            )(params, sT, ts)


        def loss_fn(params, x0, xi0, ts):
            '''
            Args:
              x0: (B, D)
              xi0: (B, D)
              ts: (T,)
            Returns:
              a scalar loss
            '''
            sT = forward_pass(params, x0, xi0)
            s_ts, a_ts = backward_pass(params, sT, ts) # (B, T, 2*D)
            s_ts = jax.lax.stop_gradient(s_ts)
            a_ts = jax.lax.stop_gradient(a_ts)

            psi_vmap = jax.vmap(jax.vmap(psi_fn,
                                         in_axes=(None, 0, 0)),
                                in_axes=(None, 0, None))
            g_vmap = jax.vmap(jax.vmap(g_fn,
                                       in_axes=(None, 0, 0)),
                              in_axes=(None, 0, None))
            loss = (a_ts * psi_vmap(params, s_ts, ts)).sum(-1)
            loss += g_vmap(params, s_ts, ts)
            return loss.mean()


        def train_step(state, rng, itr):
            batch = super(AdjointSolver, self).sample_batch(state, rng)
            x0, ts = batch['x0'], batch['t']
            # ts will be sorted with a leading zero
            xi0 = jax.vmap(jax.grad(self.problem.prior.log_p))(x0) # (B, D)

            params = state.f_params
            loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
                params, x0, xi0, ts)
            state = state.apply_flow_grad(grads=grads)
            return state, loss

        return jax.jit(train_step)


    def log_loss(self, loss):
        wandb.log({'Adjoint Loss': loss}, step=self.global_step)
        return f'Step {self.global_step} | Loss: {loss}'

