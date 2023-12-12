'''
Ornstein-Uhlenbeck process.
'''

from pathlib import Path
import wandb
import time
import jax
import jax.numpy as jnp
import numpy as np
import math
from sklearn.datasets import make_spd_matrix

from cwgf.auto.ec import ExperimentCoordinator
from cwgf.problems.kl import KLDivergence
from cwgf.problems.distribution import Gaussian, StdGaussian, OUSolution
from cwgf.eval.cflow import wandb_log_animation
from cwgf.eval.metrics import compute_metric
from cwgf.eval.metrics import compute_consistency
from cwgf.eval.utils import save_dict_h5


if __name__ == '__main__':
    # from jax.config import config
    # config.update("jax_debug_nans", True)
    # from jax.config import config
    # config.update("jax_enable_x64", True)


    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'metric_kl': True,
        'metric_pos_kl': True,
        'metric_ed': True,
        'metric_ot': True,
        'metric_w2_gauss': True,
        'metric_target_kl': False,
        'metric_consistency': False,
        'metric_repeat': 10,
        'val_num_time': 20,
        'val_num_sample': 1000,
        'save_samples': False
    })
    ec.add_common_arguments({
        'dim': 2,
        'prob_seed': 999,
        'total_time': 2.0,
        'dual_mode': 'importance',
        'hutchinson': False,
    })


    ec_result = ec.parse_args()
    tmp_dict = ec_result.tmp_dict
    config = ec_result.config
    exp_dir = ec_result.exp_dir

    dim = config['dim']
    rng_state = np.random.RandomState(seed=config['seed'])
    total_time = config['total_time']
    val_num_time = tmp_dict['val_num_time']
    val_num_sample = tmp_dict['val_num_sample']
    timesteps = np.linspace(0, total_time, num=val_num_time+1)

    # M = rng_state.randn(dim, dim) # (D, D)
    # cov_sqrt = np.linalg.inv(M) # (D, D)
    # A = M.T @ M
    A = make_spd_matrix(dim, random_state=rng_state)
    cov_sqrt = np.linalg.cholesky(np.linalg.inv(A))

    # Want A = (cov_sqrt @ cov_sqrt.T)^{-1}:
    # (M.T @ M) = (M^{-1} @ M^{-T})^{-1} = M.T @ M.

    b = rng_state.randn(dim) # (D,)

    ou_sol = OUSolution(A, b)

    init_dist = StdGaussian(dim=dim)
    target_dist = Gaussian(mean=b, cov_sqrt=cov_sqrt)

    problem = KLDivergence(dim=dim,
                           prior=init_dist,
                           target_log_p=lambda X: target_dist.log_p(X),
                           sample_fn=lambda rng, B: target_dist.sample(rng, B),
                           total_time=total_time,
                           dual_mode=config['dual_mode'],
                           hutchinson=config['hutchinson'])

    solver = ec.create_solver(problem)

    metrics = []
    if tmp_dict['metric_kl']:
        metrics.append('sym_kl')
    if tmp_dict['metric_pos_kl']:
        metrics.append('sym_pos_kl')
    if tmp_dict['metric_ed']:
        metrics.append('ed')
    if tmp_dict['metric_ot']:
        metrics.append('ot')
    if tmp_dict['metric_w2_gauss']:
        metrics.append('w2_gauss')
    if tmp_dict['metric_consistency']:
        metrics.append('consistency')
    if tmp_dict['metric_target_kl']:
        metrics.append('target_kl')
        metrics.append('GT_target_kl')
    metric_name = {
        'sym_kl': 'Symmetric KL',
        'sym_pos_kl': 'Symmetric Positive Divergence',
        'ed': 'Energy Distance',
        'ot': 'Sinkhorn Distance',
        'w2_gauss': 'W2 (Gaussian)',
        'consistency': 'Consistency',
        'target_kl': 'Objective',
        'GT_target_kl': 'GT Objective',
    }
    metric_repeat = tmp_dict['metric_repeat']
    result_dir = exp_dir / 'results'

    def val_fn():
        wandb_log_animation(solver.global_step,
                            problem.total_time,
                            lambda t: solver.extract_solution(t),
                            exp_dir=exp_dir,
                            vis_range=[-5, 5],
                            num_timestep=val_num_time+1,
                            wandb_key='video')

        wandb_log_animation(solver.global_step,
                            problem.total_time,
                            lambda t: ou_sol.get_solution(t),
                            exp_dir=exp_dir,
                            vis_range=[-5, 5],
                            num_timestep=val_num_time+1,
                            wandb_key='gt_video')

        rng = jax.random.PRNGKey(88)
        save_dict = {}
        rng, target_rng = jax.random.split(rng)
        save_dict['timesteps'] = timesteps

        if tmp_dict['save_samples']:
            dist = solver.extract_solution(timesteps[-1])
            rng_tmp = jax.random.PRNGKey(999)
            save_dict['samples'] = dist.sample(rng_tmp, 1000)

        data_mean = []
        if 'target_kl' in metrics :
            ou_dist_end = ou_sol.get_solution(timesteps[-1])
        data_metric = {metric: [] for metric in metrics
                       if metric not in ['target_kl', 'GT_target_kl']}
        for t in timesteps:
            dist = solver.extract_solution(t)
            if dist is None:
                continue
            ou_dist = ou_sol.get_solution(t)
            cur_mean = []
            for metric in metrics:
                arr = []
                raw = []
                for seed in range(900, 900 + metric_repeat):
                    if metric == 'consistency':
                        m = compute_consistency(rng=jax.random.PRNGKey(seed), state=solver.state,
                                                solver=solver)
                        raw.append(m)
                    elif metric == 'target_kl':
                        m = compute_metric(
                            dist, ou_dist_end, metric='kl',
                            num_sample=val_num_sample, seed=seed)
                    elif metric == 'GT_target_kl':
                        m = compute_metric(
                            ou_dist, ou_dist_end, metric='kl',
                            num_sample=val_num_sample, seed=seed)
                    else:
                        m = compute_metric(
                                dist, ou_dist, metric=metric,
                                num_sample=val_num_sample, seed=seed)
                        raw.append(m)
                        m = math.log10(abs(m)+1e-12)
                    arr.append(m)
                if metric not in ['target_kl', 'GT_target_kl']:
                    data_metric[metric].append(raw)
                arr = np.array(arr)
                cur_mean.append(np.mean(arr))
            data_mean.append([t, *cur_mean])

        mean_table = wandb.Table(data=data_mean, columns=['t', *metrics])
        wandb.log({'mean_table': mean_table}, step=solver.global_step)

        for k, metric in enumerate(metrics):
            wandb.log({
                f'{metric}_plot': wandb.plot.line(
                    mean_table, 't', metric,
                    title=f'{metric_name[metric]} vs. Time'),
                f'final_{metric}': data_mean[-1][k+1],
            }, step=solver.global_step)

        for metric in metrics:
            data_metric[metric] = np.array(data_metric[metric]) # i,j is time i, seed j
        save_dict['metric_dict'] = data_metric
        save_dict_h5(save_dict,
                     result_dir / f'step-{solver.global_step}.h5',
                     create_dir=True)

    solver.set_custom_val_fn(val_fn)

    solver.run()
