from abc import ABC, abstractmethod

class ProblemBase(ABC):
    def get_prior(self):
        '''
        Returns:
          A Distribution instance.
        '''
        raise Exception('Not implemented!')


    # TODO: refactor this function
    @abstractmethod
    def get_total_time(self):
        pass


    # TODO: refactor this
    def get_dim(self):
        raise Exception('Not implemented!')


    def require_log_p(self):
        return False


    def require_rng(self):
        return False


    def require_noise(self):
        return False


    def impl_v_goal_batch(self):
        return False


    def eval_F(self, x, log_p=None):
        '''
        Evaluate the integrand of the functional at x.
        For now we assume the functional is an integral over samples.

        Args:
          x: (D,)
          p: () or None
          log_p: () or None
        Returns:
          A scalar.
        '''

        # Some methods do not need eval_F.
        raise Exception('eval_F not implemented but used!')


    def eval_dF(self, x, log_p=None):
        '''
        Evaluate the first variation of the functional at x.
        Args:
          x: (D,)
          p: () or None
          log_p: () or None
        Returns:
          A scalar.
        '''
        return None


    def compute_v_goal(self, x, t, info):
        '''
        Generalized velocity matching goal.
        Args:
          x: (D,)
          t: ()
          info: a dict, containing information about \mu_t:
            samples: (B, D)
            log_p_fn: a function (D,) -> (), only if require_log_p is True
        '''
        raise Exception('Not implemented!')


    def compute_v_dot_ibp(self, x, t, info):
        '''
        Compute \EE_{\mu_t}[<v_t(x), f_t(x; \mu_t)>] using integration
        by parts. Arguments are the same as those in compute_v_goal.

        Return just the dot product <v_t(x), f_t(x; \mu_t)>, a scalar, where
        the integration over \mu_t is done by the solver.
        '''
        raise Exception('Not implemented!')


    def compute_dual_stats(self, x_batch):
        raise Exception('Not implemented!')


    def dual_sample(self, rng, batch_size, stats):
        raise Exception('Not implemented!')


    def dual_potential(self, p_x, x):
        '''
        The output of the potential network might not be the actual potential.
        Here return the actual potential corresponding to p_x.
        '''
        raise Exception('Not implemented!')


    def dual_A(self, p_x, x, stats):
        '''
        Args:
          p_x: (), the evaluation p(x).
          x: (D,), a sample from the input measure.
          stats: A pytree, running statistics.
        Return:
          (), value of A, the first part of the dual formulation.
        '''
        raise Exception('Not implemented!')


    def dual_B(self, p_z, z, stats):
        raise Exception('Not implemented!')


    def SDE_sampler(self, rng, batch_size, num_steps):
        raise Exception('Not implemented!')
