from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint as jax_odeint
import jaxopt
import flax

from scvm.solvers.models.vnn import VNN
from scvm.solvers.models.dummy import Dummy
from scvm.solvers.models.icnn import ICNN
from scvm.solvers.utils import jax_div

'''
An abstract class that presents a unified view for pushforward-based models
and velocity-based models (a.k.a. neural ODEs).
'''
class FlowView(ABC):
    def __init__(self, model):
        self.model = model


    def velocity_at(self, params, t, x, reverse):
        '''
        Args:
          t: ()
          x: (D,)
          reverse: a bool.
        Returns:
          (D,), velocity v(t, x).
        '''
        raise Exception('Not supported!')


    def velocity_from(self, params, t, x0, reverse):
        '''
        Args:
          t: ()
          x0: (D,)
          reverse: a bool.
        Returns:
          (D,), velocity v(t, forward(t, x0)).
        '''
        # This default implementation is not good for pushforward models.
        return self.velocity_at(params, t,
                                self.forward(params, t, x0, reverse, False),
                                reverse)


    @abstractmethod
    def forward(self, params, t, x0, reverse, ldj):
        '''
        Return x(t) with x(0) = x0 and d/dt x = v(x(t)).
        Args:
          t: ()
          x: (D,)
          reverse: a bool, whether to reverse the velocity.
          ldj: a bool, whether to include log |det T|.
        Returns:
          (D,), position xt.
          If ldj is True, then return a tuple (xt, ldj) where ldj
          is a scalar.
        '''
        pass


    def forward_multi_t(self, params, t, x0, reverse, ldj):
        '''
        Same as forward but with t being a sequence. For neural ODE this can
        be implemented more efficiently than directly vmapping forward.

        Args:
          t: (T), an increasing sequence of time.
          x: (D,)
          reverse: a bool, whether to reverse the velocity.
          ldj: a bool, whether to include log |det T|.
        Returns:
          (T, D,), position xt.
          If ldj is True, then return a tuple (xt, ldj) where ldj
          is (T,).
        '''
        return jax.vmap(self.forward,
                        in_axes=(None, 0, None, None, None))(
                            params, t, x0, reverse, ldj)


    def get_init_args(self, dim):
        return (0., jnp.zeros([dim]), False)


class PushforwardView(FlowView):
    def velocity_at(self, params, t, x, reverse):
        y = self.forward(params, t, x, not reverse, False)
        return self.velocity_from(params, t, y, reverse)


    def velocity_from(self, params, t, x0, reverse):
        xt_fn = lambda t: self.forward(params, t, x0, reverse, False)
        # xt_fn: () -> (D,)
        grad_fn = jax.jacfwd(xt_fn)
        v = grad_fn(t)
        return v


    def forward(self, params, t, x0, reverse, ldj):
        xt, ldj_t = self.model.apply({'params': params}, t, x0, reverse)
        if ldj:
            return xt, ldj_t
        return xt


class VelocityView(FlowView):
    def __init__(self, model,
                 ode_tol=None,
                 use_diffrax=None,
                 log_p_ode_mul=None):
        super().__init__(model)
        if ode_tol is None:
            self.tol = model.ode_tol
        else:
            self.tol = ode_tol
        if use_diffrax is None:
            self.use_diffrax = model.use_diffrax
        else:
            self.use_diffrax = use_diffrax
        if log_p_ode_mul is None:
            self.log_p_ode_mul = model.log_p_ode_mul
        else:
            self.log_p_ode_mul = log_p_ode_mul


    def velocity_at(self, params, t, x, reverse):
        return self.model.apply({'params': params},
                                t, x, reverse)


    def forward_multi_t(self, params, ts, x0, reverse, ldj):
        assert(x0.ndim == 1)
        if reverse:
            # Reversing multiple t's does not make sense here.
            assert(ts.shape[0] == 2)
            def v_fn(t, x):
                return self.velocity_at(params, ts[-1] + ts[0] - t, x, True)
        else:
            def v_fn(t, x):
                return self.velocity_at(params, t, x, False)

        if not self.use_diffrax:
            v_odeint_fn = lambda x, t: v_fn(t, x)
        else:
            from diffrax import diffeqsolve, Dopri5, Dopri8, Tsit5, ODETerm, SaveAt, PIDController

        # Note: odeint returns NaN if t[1] == t[0], etc. So we
        # use a hack here.
        ts = jnp.concatenate([jnp.array([ts[0]]), ts[1:] + 1e-32])

        if not ldj:
            if self.use_diffrax:
                term = ODETerm(lambda t, x, args: v_fn(t, x))
                solver = Dopri5()
                # solver = Tsit5()
                saveat = SaveAt(ts=ts)
                stepsize_controller = PIDController(rtol=self.tol, atol=self.tol)
                sol = diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=0.01, y0=x0,
                                  saveat=saveat, stepsize_controller=stepsize_controller)
                return sol.ys
            else:
                sol = jax_odeint(
                    v_odeint_fn,
                    x0, ts, rtol=self.tol, atol=self.tol)
                return sol
        else:
            if self.use_diffrax:
                def ldj_wrapper(t, y, args):
                    y, _ = y
                    fn = lambda y: v_fn(t, y)
                    f, vjp_fn = jax.vjp(fn, y)
                    (dfdy,) = jax.vmap(vjp_fn)(jnp.eye(y.shape[0]))
                    ldj = jnp.trace(dfdy)
                    return f, ldj * self.log_p_ode_mul
                term = ODETerm(ldj_wrapper)
                solver = Dopri5()
                saveat = SaveAt(ts=ts)
                stepsize_controller = PIDController(rtol=self.tol, atol=self.tol)
                sol = diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=0.1, y0=(x0, 0.),
                                  saveat=saveat, stepsize_controller=stepsize_controller)
                y, ldj = sol.ys
                return y, ldj / self.log_p_ode_mul
            else:
                v_div = jax_div(v_odeint_fn, argnums=0) # (D,), () -> ()
                func = lambda x_ex, t: jnp.append(
                    v_odeint_fn(x_ex[:-1], t),
                    v_div(x_ex[:-1], t) * self.log_p_ode_mul)
                sol = jax_odeint(
                    func,
                    jnp.append(x0, 0), ts, rtol=self.tol, atol=self.tol)
                return sol[:, :-1], sol[:, -1] / log_p_ode_mul


    def forward_multi_t_with_score(self, params, ts, x0, score0):
        '''
        Args:
          ts: (T), an increasing sequence of time.
          x0: (D,)
          score0: (D,)
        Returns:
          (T, D,), position xt, and (T, D), score st.
        '''
        assert(x0.ndim == 1)
        D = x0.shape[0]
        def v_fn(t, x):
            return self.velocity_at(params, t, x, False)

        # Note: odeint returns NaN if t[1] == t[0], etc. So we
        # use a hack here.
        ts = jnp.concatenate([jnp.array([ts[0]]), ts[1:] + 1e-32])

        v_div = jax_div(v_fn, argnums=1) # (), (D,) -> ()
        v_grad_div = jax.grad(v_div, argnums=1) # (), (D,) -> (D,)

        def x_score_wrapper(x, t):
            x, score = x[:D], x[D:]

            if True:
                fn = lambda x: (v_fn(t, x) * score).sum() # (D,) -> ()
                score_dt = -v_grad_div(t, x) - jax.grad(fn)(x)
            else:
                fn = lambda x: v_fn(t, x) # (D,) -> (D,)
                _, jvp = jax.jvp(fn, (x,), (score,))
                score_dt = -v_grad_div(t, x) - jvp

            return jnp.concatenate([
                v_fn(t, x),
                score_dt
            ])

        sol = jax_odeint(
            x_score_wrapper,
            jnp.concatenate([
                x0, score0
            ]), ts, rtol=self.tol, atol=self.tol)
        return sol[:, :D], sol[:, D:]


    def forward(self, params, t, x0, reverse, ldj):
        result = self.forward_multi_t(
            params, jnp.array([0., t]), x0, reverse, ldj)
        if not ldj:
            return result[-1, :]
        return result[0][-1, :], result[1][-1]

'''
Special view for ICNN, a special case of pushforward.
'''
class ICNNView(PushforwardView):
    def __init__(self, model):
        super().__init__(model)
        self.tol = model.icnn_tol


    def potential(self, params, t, x):
        return self.model.apply({'params': params}, t, x)


    def forward(self, params, t, x0, reverse, ldj):
        assert(isinstance(self.model, ICNN))
        potential_fn = lambda x: self.potential(params, t, x)
        forward_fn = jax.grad(potential_fn)
        hess_fn = jax.jacfwd(forward_fn)

        if reverse:
            y = self.backward(params, t, x0)
            if not ldj:
                return y
            H = hess_fn(y)
            _, logdet = jnp.linalg.slogdet(H)
            return y, -logdet

        x = forward_fn(x0)
        if not ldj:
            return x
        H = hess_fn(x0)
        _, logdet = jnp.linalg.slogdet(H)
        return x, logdet


    def backward(self, params, t, x):
        '''
        For backward, need to solve
        \nabla\phi^{-1}(x) = argmax_y (x^T y - \phi(y)).

        Args:
          t: ()
          x: (D,)
        Returns:
          y: (D,)

        To get gradient, need
        J(\nabla \phi^{-1})(x) = (H \phi (\nabla \phi^{-1}(x)))^{-1}
        '''
        def conjugate_obj(y, params, t, x):
            '''
            Find y such that \Phi(t, y) = x.
            This amounts to solving max_y xy - \phi(t, y), where \phi is the ICNN.

            Args:
              y: (D,)
              t: ()
              x: (D,)
            '''
            obj = (y * x).sum(-1) - self.potential(params, t, y)
            obj = -obj # maximize
            return obj

        solver = jaxopt.LBFGS(fun=conjugate_obj,
                              condition='wolfe',
                              linesearch='backtracking',
                              tol=self.tol)
        # TODO: use better initial points?
        y, info = solver.run(x, params, t, x)
        return y


    def get_init_args(self, dim):
        return (0., jnp.zeros([dim]))


'''
Special view for IRNN, a special case of pushforward.
'''
class IRNNView(PushforwardView):
    def forward(self, params, t, x0, reverse, ldj):
        (xt, ldj_t), _ = self.model.apply({'params': params['params'], 'lip': params['lip']}, t, x0, reverse, mutable=['lip'])
        if ldj:
            return xt, ldj_t
        return xt


'''
Special view for using the gradient of potential as the velocity field.
'''
class GPAVView(VelocityView):
    def __init__(self, model, transform_fn=lambda x:x):
        super().__init__(model)
        self.transform_fn = transform_fn

    def velocity_at(self, params, t, x, reverse):
        '''
        Override parent's implementation. We use
        -\nabla p_t as the velocity v_t.
        '''
        def neg_p_fn(params, t, x):
            p = self.model.apply({'params': params}, t, x)
            p = self.transform_fn(p)
            if reverse:
                p = -p
            return -p

        v_fn = jax.grad(neg_p_fn, 2)
        return v_fn(params, t, x)

'''
Special view for VMSSM with the score as the velocity field.
'''
class SAVView(VelocityView):
    def __init__(self, model, ode_tol):
        super().__init__(model, ode_tol)

    def velocity_at(self, params, t, x, reverse):
        v = self.model.apply({'params': params}, t, x)
        if reverse:
            v = -v
        return v


def create_flow_view(model):
    if isinstance(model, Dummy):
        return None
    if isinstance(model, VNN):
        return VelocityView(model)
    if isinstance(model, ICNN):
        return ICNNView(model)
    return PushforwardView(model)
