import flax
from typing import Any
import optax

class FlowTrainState(flax.struct.PyTreeNode):
    step: int
    rng: Any
    # Prefix "f_" indicates flow-specific fields.
    f_params: flax.core.FrozenDict[str, Any]
    f_opt_state: optax.OptState
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)


    def apply_flow_grad(self, *, grads, **kwargs):
        updates, new_f_opt_state = self.tx.update(
            grads, self.f_opt_state, self.f_params)
        new_f_params = optax.apply_updates(self.f_params, updates)
        return self.replace(
            f_params=new_f_params,
            f_opt_state=new_f_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, rng, f_params, tx, **kwargs):
        f_opt_state = tx.init(f_params)
        return cls(
            step=0,
            rng=rng,
            f_params=f_params,
            f_opt_state=f_opt_state,
            tx=tx,
            **kwargs,
        )


class GPAVTrainState(flax.struct.PyTreeNode):
    step: int
    rng: Any
    # Prefix "p_" indicates potential-specific fields.
    p_params: flax.core.FrozenDict[str, Any]
    p_opt_state: optax.OptState
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)

    @property
    def f_params(self):
        return self.p_params

    def apply_potential_grad(self, *, grads, **kwargs):
        updates, new_p_opt_state = self.tx.update(
            grads, self.p_opt_state, self.p_params)
        new_p_params = optax.apply_updates(self.p_params, updates)
        return self.replace(
            p_params=new_p_params,
            p_opt_state=new_p_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, rng, p_params, tx, **kwargs):
        p_opt_state = tx.init(p_params)
        return cls(
            step=0,
            rng=rng,
            p_params=p_params,
            p_opt_state=p_opt_state,
            tx=tx,
            **kwargs,
        )


class FlowPotentialTrainState(FlowTrainState):
    p_params: flax.core.FrozenDict[str, Any]
    p_opt_state: optax.OptState

    # For now assume potential uses the same optimizer as flow.
    def apply_potential_grad(self, *, grads, **kwargs):
        updates, new_p_opt_state = self.tx.update(
            grads, self.p_opt_state, self.p_params)
        new_p_params = optax.apply_updates(self.p_params, updates)
        return self.replace(
            p_params=new_p_params,
            p_opt_state=new_p_opt_state,
            **kwargs,
        )


    @classmethod
    def create(cls, *, rng, f_params, p_params, tx, **kwargs):
        f_opt_state = tx.init(f_params)
        p_opt_state = tx.init(p_params)
        return cls(
            step=0,
            rng=rng,
            f_params=f_params,
            f_opt_state=f_opt_state,
            p_params=p_params,
            p_opt_state=p_opt_state,
            tx=tx,
            **kwargs,
        )


class FlowScoreTrainState(FlowTrainState):
    s_params: flax.core.FrozenDict[str, Any]
    s_opt_state: optax.OptState

    # For now assume score uses the same optimizer as flow.
    def apply_score_grad(self, *, grads, **kwargs):
        updates, new_s_opt_state = self.tx.update(
            grads, self.s_opt_state, self.s_params)
        new_s_params = optax.apply_updates(self.s_params, updates)
        return self.replace(
            s_params=new_s_params,
            s_opt_state=new_s_opt_state,
            **kwargs,
        )


    @classmethod
    def create(cls, *, rng, f_params, s_params, tx, **kwargs):
        f_opt_state = tx.init(f_params)
        s_opt_state = tx.init(s_params)
        return cls(
            step=0,
            rng=rng,
            f_params=f_params,
            f_opt_state=f_opt_state,
            s_params=s_params,
            s_opt_state=s_opt_state,
            tx=tx,
            **kwargs,
        )


class JKOSingleTrainState(flax.struct.PyTreeNode):
    '''
    A training state for a single timestamp in a JKO scheme.
    '''
    step: int
    rng: Any
    params: flax.core.FrozenDict[str, Any]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)


    def apply_grad(self, i, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, rng, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            rng=rng,
            params=params,
            opt_state=opt_state,
            tx=tx,
            **kwargs,
        )
