from cwgf.solvers.solver_base import SolverBase

from abc import abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
import time
from tqdm import trange
import wandb
from cwgf.problems.distribution import FuncDistribution
from cwgf.solvers.flow_view import create_flow_view
from cwgf.auto.train_state import FlowTrainState
from cwgf.auto.ckpt_manager import CkptManager

class FlowBase(SolverBase):
    def __init__(self, *,
                 thp,
                 flow,
                 optimizer,
                 **kwargs):
        super().__init__(**kwargs)

        self.thp = thp
        self.flow = flow
        self.flow_view = create_flow_view(flow)
        self.optimizer = optimizer
        self.ckpt_manager = CkptManager(self)


    def create_train_state(self):
        '''
        Subclasses can inherit this method to e.g. create a state with
        multiple models.
        '''
        rng = jax.random.PRNGKey(self.seed)
        rng, flow_rng = jax.random.split(rng)
        f_params = self.flow.init(flow_rng,
                                *self.flow_view.get_init_args(
                                    self.problem.get_dim()
                                ))['params']
        return FlowTrainState.create(
            rng=rng, f_params=f_params, tx=self.optimizer)


    @property
    def global_step(self):
        return self.state.step


    def _init_models(self):
        self.state = self.create_train_state()
        self.ckpt_manager.load()


    def _create_functions(self):
        '''
        Define common functions used by subclasses.
        '''
        forward_fn = self.flow_view.forward_multi_t
        forward_fn = jax.vmap(
            forward_fn,
            in_axes=(None, None, 0, None, None))
        self.forward_vmap = jax.jit(forward_fn,
                                    static_argnames=['reverse', 'ldj'])
        '''
        forward_vmap: (params, t, x0, reverse, ldj) -> xt or (xt, ldj_t):
          t: (T,)
          x0: (B, D)
          xt: (B, T, D)
          ldj_t: (B, T)
        '''

        v_at_vmap = self.flow_view.velocity_at
        v_at_vmap = jax.vmap(v_at_vmap, in_axes=(None, None, 0, None))
        v_at_vmap = jax.vmap(v_at_vmap, in_axes=(None, 0, 1, None),
                               out_axes=1)
        self.v_at_vmap = jax.jit(v_at_vmap, static_argnames='reverse')

        v_from_vmap = self.flow_view.velocity_from # (None, (), (D,)) -> (D,)
        v_from_vmap = jax.vmap(v_from_vmap, in_axes=(None, 0, None, None))
        v_from_vmap = jax.vmap(v_from_vmap, in_axes=(None, None, 0, None))
        self.v_from_vmap = jax.jit(v_from_vmap, static_argnames='reverse')

        '''
        v_at_vmap/v_from_vmap: (params, t, xt, reverse) -> v:
          t: (T,)
          xt: (B, T, D)
          v: (B, T, D)
        '''


        def log_p_fn(params, t, x):
            '''
            Args:
              t: (), scalar
              x: (D,)
            Returns:
              log_p: ()
            '''
            x0, ldj = self.flow_view.forward(
                params, t, x, True, True)
            log_p0 = self.problem.prior.log_p(x0)
            return log_p0 + ldj
        self.log_p_fn = log_p_fn
        self.log_p_vmap = jax.jit(jax.vmap(log_p_fn,
                                           in_axes=(None, None, 0)))
        '''
        log_p_vmap: (params, t, x) -> log_p:
          t: (), scalar
          x: (B, D)
          log_p: (B,)
        '''
        self.create_more_functions()
        self.train_step = self.create_train_step()


    def soft_init_loss(self, params, x0):
        if hasattr(self.flow, 'soft_init') and self.flow.soft_init > 0:
            y0 = jax.vmap(self.flow_view.forward,
                          in_axes=(None, None, 0, None, None))(
                              params, 0., x0, False, False) # (B, D)
            return self.flow.soft_init * jnp.sum((y0 - x0) ** 2, -1).mean()
        else:
            return 0.


    def create_more_functions(self):
        '''
        Subclasses may implement this to create more jitted functions.
        '''
        pass


    @abstractmethod
    def create_train_step(self):
        '''
        Each subclass must create a function train_step that takes positional
        arguments:
          state: the train state
          rng: jax's rng
          itr: training iteration
        and returns a tuple (state, loss), where loss can be of arbitrary type.
        '''
        pass


    def sample_batch(self, state, rng):
        '''
        Subclasses can override this to return more variables as a map.
        '''

        B = self.thp.train_batch_size
        T = self.thp.train_num_time
        total_time = self.problem.get_total_time()
        prior = self.problem.get_prior()

        rng, t_rng, x0_rng = jax.random.split(rng, 3)
        t = jax.random.uniform(t_rng, (T,))

        if self.thp.extend_final_t:
            total_time = total_time * 1.2

        if self.thp.train_even_time:
            th = total_time / T
            anchor = jnp.arange(T) * th # (T,)
            t = anchor + th * t
        else:
            t = jnp.sort(t)
            t = jnp.power(t, self.thp.train_time_pow) # use more time early on
            t = t * total_time
        # Note: jax's odeint has bug if the starting time is not zero.
        t = jnp.concatenate([jnp.zeros([1]), t])
        if self.thp.train_final_t:
            t = jnp.append(t, total_time)
        # jax.debug.print('sampled t: {}', t)
        x0 = prior.sample(x0_rng, B)

        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, [B, t.shape[0], x0.shape[-1]])
        return {
            't': t,
            'x0': x0,
            'noise': noise,
        }


    def post_train_step(self, rng):
        '''
        Args:
          rng: modified rng during the current training step
        '''
        self.state = self.state.replace(rng=rng, step=self.state.step+1)
        if self.thp.should_save(self.global_step):
            self.ckpt_manager.save()
        if self.thp.should_validate(self.global_step):
            self._validate()


    @abstractmethod
    def log_loss(self, loss):
        '''
        Subclasses need to know how to log training loss.
        Return a str to use in tqdm.trange.
        '''
        pass


    def _train(self):
        train_range = trange(self.global_step, self.thp.train_num_step)
        total_training_step = self.thp.train_num_step - self.global_step
        for i in train_range:
            start_time = time.time()
            rng, i_rng = jax.random.split(self.state.rng)
            self.state, loss = self.train_step(self.state, i_rng, i)
            self.post_train_step(rng)
            train_range.set_description(self.log_loss(loss))


    def run(self):
        self._create_functions()
        self._init_models()
        if not self.thp.is_val:
            self._train()
        else:
            self._validate()


    def extract_solution(self, t1):
        def sample_fn(rng, batch_size):
            x0 = self.problem.get_prior().sample(rng, batch_size)
            return self.forward_vmap(self.state.f_params,
                                     jnp.array([0, t1]),
                                     x0,
                                     False, False)[:, -1, :]

        return FuncDistribution(sample_fn,
                                log_p_batch_fn=lambda x:self.log_p_vmap(
                                    self.state.f_params,
                                    t1, x))
