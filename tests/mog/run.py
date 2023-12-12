'''
Mixture of Gaussians.
'''

from pathlib import Path
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import math
import time
from sklearn.datasets import make_spd_matrix

from cwgf.auto.ec import ExperimentCoordinator
from cwgf.problems.kl import KLDivergence
from cwgf.problems.distribution import Gaussian, Mixture
from cwgf.eval.cflow import \
    wandb_log_animation, wandb_log_image
from cwgf.eval.metrics import compute_metric
from cwgf.eval.metrics import compute_consistency
from cwgf.eval.utils import save_dict_h5


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'metric_kl': True,
        'metric_ed': True,
        'metric_ksd': False,
        'metric_ot': True,
        'metric_repeat': 10,
        'metric_consistency': False,
        'val_num_time': 10,
        'val_num_sample': 1000,
        'val_pow_timesteps': False
    })
    ec.add_common_arguments({
        'dim': 2,
        'debug_mode': False,
        'total_time': 2.0,
        'init_cov_sqrt': 4.0,
        'num_mixture': 5,
        'dual_mode': 'importance',
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
    vis_num_time = val_num_time + 1
    timesteps = np.linspace(0, total_time, num=val_num_time+1)
    if tmp_dict['val_pow_timesteps']:
        timesteps = np.power(timesteps / total_time, 3) * total_time

    if config['debug_mode']:
        assert(config['dim'] == 2)
        mixtures = []
        centers = [(-1, -1), (-1, 1), (1, 1)]
        for x, y in centers:
            mean = np.array([x, y], dtype=float)
            cov_sqrt = 0.1 * np.eye(dim)
            mixtures.append(Gaussian(mean=mean, cov_sqrt=cov_sqrt))
        init_dist = Mixture(mixtures, np.ones(len(mixtures)) / len(mixtures))

        target_dist = Gaussian(mean=np.zeros(2),
                               cov_sqrt=np.eye(2))

        vis_range=[-4, 4]
        vis_num_time = 100

    else:
        init_dist = Gaussian(mean=np.zeros(dim),
                             cov_sqrt=config['init_cov_sqrt']*np.eye(dim))

        num_mixture = config['num_mixture']
        mixtures = []
        for i in range(num_mixture):
            mean = rng_state.uniform(low=-5, high=5, size=[dim])
            cov_sqrt = np.eye(dim)
            mixtures.append(Gaussian(mean=mean, cov_sqrt=cov_sqrt))
        target_dist = Mixture(mixtures, np.ones(num_mixture) / num_mixture)

        vis_range=[-10, 10]

    problem = KLDivergence(dim=dim,
                           prior=init_dist,
                           target_log_p=lambda X: target_dist.log_p(X),
                           sample_fn=lambda rng, B: target_dist.sample(rng, B),
                           total_time=total_time,
                           dual_mode=config['dual_mode'])

    solver = ec.create_solver(problem)

    metrics = []
    if config['debug_mode']:
        metrics.append('w2_gauss')
    if tmp_dict['metric_kl']:
        metrics.append('sym_kl')
    if tmp_dict['metric_ed']:
        metrics.append('ed')
    if tmp_dict['metric_ksd']:
        metrics.append('ksd')
    if tmp_dict['metric_ot']:
        metrics.append('ot')
    if tmp_dict['metric_consistency']:
        metrics.append('consistency')
    metric_name = {
        'sym_kl': 'Symmetric KL',
        'ed': 'Energy Distance',
        'ksd': 'KSD',
        'ot': 'Sinkhorn Distance',
        'w2_gauss': 'W2 (Gaussian)',
        'consistency': 'Consistency',
    }
    metric_repeat = tmp_dict['metric_repeat']
    result_dir = exp_dir / 'results'

    start_time = time.time()
    def val_fn():
        val_start_time = time.time()
        rng = jax.random.PRNGKey(88)
        save_dict = {}
        rng, target_rng = jax.random.split(rng)
        save_dict['target_samples'] = target_dist.sample(target_rng, 5000)
        save_dict['timesteps'] = timesteps

        wandb_log_animation(solver.global_step,
                            problem.total_time,
                            lambda t: solver.extract_solution(t),
                            exp_dir=exp_dir,
                            vis_range=vis_range,
                            vis_size=1000,
                            num_timestep=vis_num_time,
                            wandb_key='video')

        wandb_log_image(solver.global_step,
                        target_dist,
                        vis_range=vis_range,
                        vis_size=1000,
                        wandb_key='gt_image')

        data = []
        data_metric = {metric: [] for metric in metrics}
        for i, t in enumerate(timesteps):
            dist = solver.extract_solution(t)
            if dist is None:
                continue
            cur = []
            for metric in metrics:
                s = 0
                raw = []
                for seed in range(900, 900 + metric_repeat):
                    if metric == 'consistency':
                        m = compute_consistency(rng=jax.random.PRNGKey(seed),
                                                state=solver.state,
                                                solver=solver)
                        raw.append(m)
                    else:
                        m = compute_metric(
                                dist, target_dist, metric=metric,
                                num_sample=val_num_sample, seed=seed)
                        raw.append(m)
                        m = math.log10(abs(m)+1e-12)
                    s += m
                data_metric[metric].append(raw)
                cur.append(s / metric_repeat)
            data.append([t, *cur])

            save_dict[f'samples_{i}'] = dist.sample(rng, 5000)

        table = wandb.Table(data=data, columns=['t', *metrics])

        for metric in metrics:
            data_metric[metric] = np.array(data_metric[metric]) # i,j is time i, seed j
        save_dict['metric_dict'] = data_metric

        for k, metric in enumerate(metrics):
            wandb.log({
                f'{metric}_plot': wandb.plot.line(
                    table, 't', metric,
                    title=f'{metric_name[metric]} vs. Time'),
                f'final_{metric}': data[-1][k+1],
            }, step=solver.global_step)

        save_dict_h5(save_dict,
                     result_dir / f'step-{solver.global_step}.h5',
                     create_dir=True)

        val_time = val_start_time - time.time()


    solver.set_custom_val_fn(val_fn)

    solver.run()

    total_time = time.time() - start_time
    total_val_time = solver.total_val_time
    print('Total train time: {}s, total run time: {}s'.format(total_time - total_val_time, total_val_time))
