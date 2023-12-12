from dataclasses import dataclass, field
from typing import Any
import optax
import flax

from cwgf.solvers.scvms import SCVMS
from cwgf.solvers.adjoint import AdjointSolver
from cwgf.solvers.jko import JKO
from cwgf.solvers.dfe import DFE

@dataclass
class ParamClsSetting:
    cls: Any
    param_dict: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class SolverSetting(ParamClsSetting):
    need_thp: bool = False
    need_flow: bool = False
    need_score: bool = False
    need_potential: bool = False
    need_optimizer: bool = False


g_solver_setting_dict = {
    'scvms': SolverSetting(
        cls=SCVMS,
        param_dict={
            'use_ibp': False,
            'ode_score': False,
            'prox_weight': 0.0,
            'smooth_weight': 0.0,
            'smooth_decay': 1.e-3,
            'splitting': False,
        },
        need_thp=True,
        need_flow=True,
        need_optimizer=True
    ),
    'adjoint': SolverSetting(
        cls=AdjointSolver,
        param_dict={
            'use_vjp': True,
        },
        need_thp=True,
        need_flow=True,
        need_optimizer=True
    ),
    'jko': SolverSetting(
        cls=JKO,
        param_dict={
            'jko_len': 20,
            'val_final_only': True,
            'jko_lr_decay': False,
            'pretrain': True,
        },
        need_thp=True,
        need_flow=True,
        need_potential=True,
        need_optimizer=True
    ),
    'dfe': SolverSetting(
        cls=DFE,
        param_dict={
            'num_particle': 1000,
            'dt': 1e-3,
            'num_inner_step': 25,
            'learning_rate': 1e-4,
            'save_dt': 1e-2,
            'pretrain_tol': 1e-4,
        },
    ),
}

from cwgf.solvers.thp import THP
g_thp_param_dict = {
    'train_batch_size': 1000,
    'train_num_time': 20,
    'train_flow_rep': 1,
    'train_potential_rep': 1,
    'train_score_rep': 1,
    # Below are advanced training modifications.
    'train_time_pow': 1.0,
    'train_even_time': True,
    'train_final_t': False,
    'extend_final_t': False,
    'train_noise_level': 0.0,
    'train_noise_scale_t': False,
    'ref_size': 500,
    'train_num_step': 5000,
    'val_freq': 1000,
    'save_freq': 1000,
    'is_val': False,
    'ckpt_name': 'ckpt',
}

from cwgf.solvers.models.tdmlp import TDMLP
from cwgf.solvers.models.mnf import MNF
from cwgf.solvers.models.vnn import VNN
from cwgf.solvers.models.icnn import ICNN
from cwgf.solvers.models.dummy import Dummy

g_model_setting_dict = {
    'tdmlp': ParamClsSetting(
        cls=TDMLP,
        param_dict={
            'num_layer': 2,
            'layer_size': 128,
            'activation_layer': 'celu',
            'soft_init' : 0.
        },
    ),
    'mnf': ParamClsSetting(
        cls=MNF,
        param_dict={
            'embed_time_dim': 64,
            'couple_mul': 2,
            'mask_type': 'loop',
            'activation_layer': 'celu',
            'soft_init': 0.,
            'ignore_time': False,
        }
    ),
    'icnn': ParamClsSetting(
        cls=ICNN,
        param_dict={
            'hidden_dims': [64, 128, 128, 64],
            'time_hidden_dims': [4, 8, 16, 32],
            # 'activation_layer': 'elu',
            'activation_layer': 'celu',
            'quadratic_rank': 20,
            'soft_init': 5.0,
            'ignore_time': False,
            'icnn_tol': 1e-2,
        }
    ),
    'vnn': ParamClsSetting(
        cls=VNN,
        param_dict={
            'num_layer': 3,
            'layer_size': 256,
            'activation_layer': 'silu',
            'kernel_var': 1.0,
            'ode_tol': 1e-4,
            'use_diffrax': True,
            'log_p_ode_mul': 1.0,
            'embed_time_dim': 64,
            'embed_space_dim': -1,
            'use_skip': True,
            'use_residual': False,
            'skip_only': False,
            'layer_norm': True,
        }
    ),
    'dummy': ParamClsSetting(
        cls=Dummy,
        param_dict={
        }
    ),
}

g_scheduler_setting_dict = {
    'constant': ParamClsSetting(
        cls=optax.constant_schedule,
        param_dict={
            'value': 1e-3,
        }
    ),
    'cosine_decay': ParamClsSetting(
        cls=optax.cosine_decay_schedule,
        param_dict={
            'init_value': 1e-3,
            'decay_steps': 10000,
            'alpha': 1e-2,
        }
    ),
    'multi_step_decay': ParamClsSetting(
        cls=optax.piecewise_constant_schedule,
        param_dict={
            'init_value': 1e-3,
            'boundaries_and_scales': {5000 : 0.1, 7500 : 0.1}
        }
    ),
}

g_optimizer_setting_dict = {
    'adam': ParamClsSetting(
        cls=optax.adam,
        param_dict={
            'learning_rate': 1e-3,
            'b1': 0.9,
            'b2': 0.999,
        }
    ),
    'adamw': ParamClsSetting(
        cls=optax.adamw,
        param_dict={
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'b1': 0.9,
            'b2': 0.999,
        }
    ),
}

class Registry:
    @staticmethod
    def get_thp_param_dict():
        return g_thp_param_dict

    @staticmethod
    def get_solver_setting(name):
        if isinstance(name, str):
            return g_solver_setting_dict[name]
        else:
            for k, v in g_solver_setting_dict.items():
                if isinstance(name, v.cls):
                    return v


    @staticmethod
    def get_model_setting(name):
        return g_model_setting_dict[name]


    @staticmethod
    def get_optimizer_setting(name):
        return g_optimizer_setting_dict[name]


    @staticmethod
    def get_scheduler_setting(name):
        return g_scheduler_setting_dict[name]
