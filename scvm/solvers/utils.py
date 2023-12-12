import jax
import jax.numpy as jnp
from flax import linen as nn

def jax_div(func, argnums):
    '''
    Divergence operator.
    Args:
      func: (...,D,...) -> (D,)
    Returns:
      Divergence of func, (...,D,...) -> ()
    '''

    # TODO: Hutchinson trace estimator.
    jac = jax.jacfwd(func, argnums=argnums)
    return lambda *a, **kw: jnp.trace(jac(*a, **kw))


def jax_div_hutchinson(func, argnums):
    '''
    Divergence operator with Hutchinson trace estimator.
    Args:
      func: (...,D,...) -> (D,)
    Returns:
      Divergence of func, taking an additional eps (M, D), where
      (M, D) are M directions for applying the Hutchinson trick.
    '''
    def div_fn(*args, **kwargs):
        grad_fn = lambda x, eps: (func(*(args[:argnums] + (x,) + args[argnums+1:])) * eps).sum(-1)
        grad_fn = jax.grad(grad_fn, argnums=0) # (D,) -> (D,)
        single_fn = lambda x, eps: (eps * grad_fn(x, eps)).sum(-1)
        eps = kwargs['eps']
        x = jnp.tile(args[argnums], [eps.shape[0], 1]) # (M, D)
        return jax.vmap(single_fn)(x, eps).mean()
    return div_fn


def tr_jac_pow(func, k, argnums):
    '''
    Trace of the power k of the jacobian of func
    Args:
      k: int
      func: (...,D,...) -> (D,)
    Returns:
      Divergence of func, (...,D,...) -> ()
    '''

    # TODO: Hutchinson trace estimator.
    jac = jax.jacfwd(func, argnums=argnums)
    return lambda *a, **kw: jnp.trace(jac(*a, **kw)**k)


def _while_loop_scan(cond_fun, body_fun, init_val, maxiter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    # When cond is met, we start doing no-ops.
    return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

  init = (init_val, cond_fun(init_val))
  return jax.lax.scan(_fun, init, None, length=maxiter)[0][0]



def _while_loop_flax_scan(cond_fun, body_fun, init_val, maxiter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    # When cond is met, we start doing no-ops.
    return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

  init = (init_val, cond_fun(init_val))
  return nn.scan(_fun, init, None, length=maxiter)[0][0]



def _while_loop_lax(cond_fun, body_fun, init_val, maxiter):
  """lax.while_loop based implementation (jit by default, no reverse-mode)."""
  def _cond_fun(_val):
    it, val = _val
    return jnp.logical_and(cond_fun(val), it <= maxiter - 1)

  def _body_fun(_val):
    it, val = _val
    val = body_fun(val)
    return it + 1, val

  return jax.lax.while_loop(_cond_fun, _body_fun, (0, init_val))[1]
