import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from functools import partial


def exp_diff_norm(X, Y):
    '''
    Compute E[||X-Y||^2] given samples X, Y.
    Args:
      X: (B, D)
      Y: (B', D)
    '''
    norm_fn = lambda x, y: ((x-y) ** 2).sum()
    norm_vmap = jax.vmap(jax.vmap(norm_fn, in_axes=(0, None)),
                         in_axes=(None, 0), out_axes=1)
    return norm_vmap(X, Y).sum() / (X.shape[0] * Y.shape[0])


@partial(jax.jit, static_argnames=['dist', 'bandwidth'])
def compute_ksd(X, dist, bandwidth=1.0):
    score_fn = jax.grad(dist.log_p)

    '''
    Compute kernelized Stein discrepancy.
    Args:
      X: (B, D)
      score_fn: (D,) -> (D,)
    '''
    def kernel_fn(x, y):
        '''
        x, y: (D,)
        '''
        return jnp.exp(-((x - y) ** 2).sum() / (2*bandwidth**2))
    dkdx = jax.jacfwd(kernel_fn, argnums=0)
    dkdy = jax.jacfwd(kernel_fn, argnums=1)
    d2k = jax.jacrev(dkdy, argnums=0)

    def u_fn(x, y):
        '''
        x, y: (D,)
        '''
        score_x = score_fn(x)
        score_y = score_fn(y)
        tmp = (score_x * score_y).sum(-1) * kernel_fn(x, y)
        tmp += (score_x * dkdy(x, y)).sum(-1)
        tmp += (score_y * dkdx(x, y)).sum(-1)
        tmp += jnp.trace(d2k(x, y))
        return tmp

    B = X.shape[0]
    u_vmap = jax.vmap(
        jax.vmap(u_fn, in_axes=(None, 0)),
        in_axes=(0, None))
    u_all = u_vmap(X, X) # (B, B)
    mask = 1 - jnp.eye(B)
    return (u_all * mask).sum() / (B * (B - 1))


def compute_w2_gauss(mean1, cov1, mean2, cov2):
    tmp1 = ((mean1 - mean2) ** 2).sum()

    tmp2 = (cov1 @ cov2)
    tmp2 = scipy.linalg.sqrtm(tmp2)
    tmp2 = np.real(tmp2)
    tmp2 = np.trace(cov1 + cov2 - 2 * tmp2)
    dist = tmp1 + tmp2
    return dist


def compute_metric(dist1, dist2, *,
                   metric,
                   num_sample,
                   samples1=None, samples2=None,
                   seed=999,
                   ot_backend='pot',
                   dim = 1,
                   is_SDE = False):
    # TODO: add more metrics.
    '''
    Args:
      dist1, dist2: Intances of Distribution.
      metric: ['sym_kl', 'ed'].
    '''

    rng = jax.random.PRNGKey(seed)

    if metric == 'sym_kl':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)

        if is_SDE :
            log_p_11 = dist1.logpdf(jnp.transpose(samples1))
            log_p_21 = dist1.logpdf(jnp.transpose(samples2))
        else :
            log_p_11 = dist1.log_p_batch(samples1)
            log_p_21 = dist1.log_p_batch(samples2)
        log_p_12 = dist2.log_p_batch(samples1)
        log_p_22 = dist2.log_p_batch(samples2)

        kl_12 = (log_p_11 - log_p_12).mean()
        kl_21 = (log_p_22 - log_p_21).mean()
        return kl_12 + kl_21
    elif metric == 'sym_pos_kl':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)

        if is_SDE :
            log_p_11 = dist1.logpdf(jnp.transpose(samples1))
            log_p_21 = dist1.logpdf(jnp.transpose(samples2))
        else :
            log_p_11 = dist1.log_p_batch(samples1)
            log_p_21 = dist1.log_p_batch(samples2)
        log_p_12 = dist2.log_p_batch(samples1)
        log_p_22 = dist2.log_p_batch(samples2)

        return 0.5 * ((log_p_11 - log_p_12) ** 2 +
                      (log_p_21 - log_p_22) ** 2).mean()
    elif metric == 'ksd':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        return compute_ksd(samples1, dist2)
    elif metric == 'ot':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)
        if ot_backend == 'ott':
            raise Exception('This is buggy somehow.')
            import ott
            pc = ott.geometry.pointcloud.PointCloud(
                samples1, samples2, epsilon=1e-6)
            out = ott.solvers.linear.sinkhorn.sinkhorn(pc)
            return jnp.sum(out.matrix * out.geom.cost_matrix)
        else:
            import ot
            weights1 = np.ones(num_sample) / num_sample
            weights2 = np.ones(num_sample) / num_sample
            M = ot.dist(np.array(samples1), np.array(samples2))
            W = ot.emd2(weights1, weights2, M)
            return W
    elif metric == 'w2_gauss':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None:
            samples1 = dist1.sample(rng1, num_sample)
        mean1 = jnp.mean(samples1, axis=0)
        cov1 = jnp.cov(samples1, rowvar=False)
        if samples2 is None:
            from scvm.problems.distribution import Gaussian
            assert(isinstance(dist2, Gaussian))
            mean2 = dist2.mean
            cov2 = dist2.get_cov()
            # samples2 = dist2.sample(rng2, num)
            # mean2 = jnp.mean(samples2, axis=0)
            # cov2 = jnp.cov(samples2, rowvar=False)
        else:
            mean2 = jnp.mean(samples2, axis=0)
            cov2 = jnp.cov(samples2, rowvar=False)
        return compute_w2_gauss(mean1, cov1, mean2, cov2)
    elif metric == 'kl':
        if samples1 is None :
            samples1 = dist1.sample(rng, num_sample)
        log_p1 = dist1.log_p_batch(samples1)
        log_p2 = dist2.log_p_batch(samples1)
        return (log_p1 - log_p2).mean()
    elif metric == 'tv' :
        # we assume dist2 is an instance of Barenblatt distribution
        from scvm.problems.distribution import Barenblatt
        assert(isinstance(dist2, Barenblatt))
        bound = 1.25*dist2.bound
        n = int((50000)**(1/dim))
        batch_size = 1000
        x = jnp.array(jnp.meshgrid(*[jnp.linspace(-bound, bound, n)]*dim)).T.reshape(-1, dim)
        x = jnp.array_split(x,jnp.maximum(len(x)//batch_size,1))
        count = 0
        val = 0
        for i in range(len(x)):
            val += jnp.abs(dist1.p_batch(x[i])-dist2.p_batch(x[i])).sum()
            count += len(x[i])
        return val/count
    else :
        assert(metric == 'ed')
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)
        SS = exp_diff_norm(samples1, samples1)
        ST = exp_diff_norm(samples1, samples2)
        TT = exp_diff_norm(samples2, samples2)
        return (2 * ST - SS - TT)


@partial(jax.jit, static_argnames=['solver'])
def compute_consistency(*, rng, state, solver):
    print(solver.problem)
    from scvm.solvers.flow_base import FlowBase
    from scvm.solvers.flow_view import VelocityView, PushforwardView
    from scvm.problems.kl import KLDivergence
    from scvm.problems.tFPE import TimeFPE
    assert(isinstance(solver, FlowBase))
    assert(isinstance(solver.problem, KLDivergence) or
           isinstance(solver.problem, TimeFPE))
    problem = solver.problem
    total_time = problem.get_total_time()
    prior = problem.get_prior()

    rng, batch_rng = jax.random.split(rng)
    batch = solver.sample_batch(state, batch_rng)

    params = state.f_params

    t, x0 = batch['t'], batch['x0']
    xt = solver.forward_vmap(params, t, x0, False, False) # (B, T, D)

    v_goal_vmap = jax.vmap(
        jax.vmap(problem.compute_v_goal_with_score,
                 in_axes=(0, 0, 0, None)),
        in_axes=(0, None, 0, None))

    if isinstance(solver.flow_view, VelocityView):
        score_fn = solver.flow_view.forward_multi_t_with_score
        score_vmap = jax.vmap(
            score_fn,
            in_axes=(None, None, 0, 0)
        )

        vt = solver.v_at_vmap(params, t, xt, False) # (B, T, D)
        scores0 = jax.vmap(jax.grad(prior.log_p))(x0)
        scores = score_vmap(params, t, x0, scores0)[1]
    else:
        assert(isinstance(solver.flow_view, PushforwardView))
        vt = solver.v_from_vmap(params, t, x0, False) # (B, T, D)
        score_fn = jax.grad(solver.log_p_fn, argnums=2)
        score_vmap = jax.vmap(
            jax.vmap(score_fn,
                     in_axes=(None, 0, 0)),
            in_axes=(None, None, 0))
        scores = score_vmap(params, t, xt)

    v_goal = v_goal_vmap(xt, t, scores, None) # (B, T, D)

    loss = jnp.sum((vt - v_goal) ** 2, -1).mean()
    return loss
