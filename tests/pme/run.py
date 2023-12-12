'''
Porous Media Equation.
'''

from pathlib import Path
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
from cwgf.auto.ec import ExperimentCoordinator
from cwgf.problems.gen_ent import GeneralizedEntropy
from cwgf.problems.distribution import Gaussian, StdGaussian, Barenblatt
from cwgf.eval.cflow import wandb_log_animation
from cwgf.eval.metrics import compute_metric
from jax.config import config
from cwgf.eval.utils import save_dict_h5

def wandb_log_PME_animation(solver, pme_sol, save_dict):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    x = np.linspace(-1,1,100)
    x = np.expand_dims(x,axis=-1)
    out_est = []
    out_GT = []
    for i, t in enumerate(timesteps):
        dist = solver.extract_solution(t)
        if dist is None:
            break
        pme_dist = pme_sol.get_solution(t)
        est = np.exp(dist.log_p_batch(x))
        gt = np.exp(pme_dist.log_p_batch(x))
        out_est.append(est)
        out_GT.append(gt)
        save_dict[f'p_est{i}'] = est
        save_dict[f'p_GT{i}'] = gt
    out_est, out_GT = np.array(out_est), np.array(out_GT)

    fig, ax = plt.subplots()
    def animate(t):
        ax.clear()
        ax.plot(x, out_est[t], label = 'estimate')
        ax.plot(x, out_GT[t], label = 'truth')
        ax.set_ylim([0,15])
    ax.legend()
    ani = FuncAnimation(fig, animate, frames=len(timesteps),
                interval=50, repeat_delay=500)
    f = exp_dir / 'tmp_anim.gif'
    writergif = matplotlib.animation.PillowWriter(fps=30)
    ani.save(f, writer=writergif)
    wandb.log({'video': wandb.Video(str(f), fps=4, format='gif')},step=solver.global_step)

    return save_dict



class PMESolution:
    def __init__(self, t0, x0, m, dim, bound, C=0., init_samples=False, num_warmup_sampler=1000, stepsize_sampler=0.1, MH_code='pyro'):
        self.bound = bound
        self.t0 = t0
        self.x0 = x0
        self.m = m
        self.dim = dim
        self.init_samples = init_samples
        self.num_warmup_sampler = num_warmup_sampler
        self.stepsize_sampler = stepsize_sampler
        self.MH_code = MH_code
        self.C = C

    def get_solution(self, t):
        return Barenblatt(t, self.t0, self.x0, self.m, self.dim, self.bound, C=C, init_samples=self.init_samples,num_warmup_sampler=self.num_warmup_sampler, stepsize_sampler=self.stepsize_sampler, MH_code=self.MH_code)


if __name__ == '__main__':

    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'metric_kl': False,
        'metric_sym_pos_kl': True,
        'metric_ed': True,
        'metric_ot': True,
        'objective': True,
        'GT_objective': True,
        'metric_TV': True, 
        'metric_repeat': 10,
        'val_num_time': 20
    })
    ec.add_common_arguments({
        'm': 2,
        'dim': 1,
        'total_time': 0.025,
        'uniform_scale': 3.,
        'volume_scale': 1.5,
        'p0_bound' : 0.25,
        't0' : 1e-3,
        'C' : 0., 
        'init_samples': False,
        'eval_init_samples': False,
        'n_samples_eval' : 1000,
        'MH_code': 'my',
        'num_warmup_sampler': 1000,
        'stepsize_sampler': 0.1
    })

    ec_result = ec.parse_args()
    tmp_dict = ec_result.tmp_dict
    config = ec_result.config
    exp_dir = ec_result.exp_dir

    dim = config['dim']
    m = config['m']
    uniform_scale = config['uniform_scale']
    volume_scale = config['volume_scale']
    rng_state = np.random.RandomState(seed=config['seed'])
    total_time = config['total_time']
    val_num_time = tmp_dict['val_num_time']
    p0_bound = config['p0_bound']
    t0 = config['t0']
    C = config['C']
    init_samples = config['init_samples']
    eval_init_samples = config['eval_init_samples']
    n_samples_eval = config['n_samples_eval']
    timesteps = np.linspace(0, total_time, num=val_num_time+1)
    MH_code = config['MH_code']

    num_warmup_sampler = config['num_warmup_sampler']
    stepsize_sampler = config['stepsize_sampler']

    x0 = jnp.zeros((1,dim))
    pme_sol = PMESolution(t0, x0, m, dim, p0_bound, C=C, init_samples=False,num_warmup_sampler=num_warmup_sampler, stepsize_sampler=stepsize_sampler, MH_code=MH_code)
    init_dist = Barenblatt(0., t0, x0, m, dim, p0_bound, C=C, init_samples=init_samples,num_warmup_sampler=num_warmup_sampler, stepsize_sampler=stepsize_sampler, MH_code=MH_code)
    

    pme_dist_end = pme_sol.get_solution(timesteps[-1])
    problem = GeneralizedEntropy(dim=dim,
                           m = m,
                           prior=init_dist,
                           total_time=total_time,
                           uniform_scale=uniform_scale,
                           volume_scale=volume_scale)

    solver = ec.create_solver(problem)

    metrics = []
    if tmp_dict['metric_kl']:
        metrics.append('sym_kl')
    if tmp_dict['metric_sym_pos_kl']:
        metrics.append('sym_pos_kl')
    if tmp_dict['metric_ed']:
        metrics.append('ed')
    if tmp_dict['metric_ot']:
        metrics.append('ot')
    if tmp_dict['metric_TV']:
        metrics.append('tv')
    if tmp_dict['objective']:
        metrics.append('objective')
    if tmp_dict['GT_objective']:
        metrics.append('GT_objective')
    metric_name = {
        'sym_kl': 'Symmetric KL',
        'sym_pos_kl': 'Symmetric Positive KL',
        'ed': 'Energy Distance',
        'ot': 'Sinkhorn Distance',
        'objective': 'Objective',
        'GT_objective': 'GT Objective',
        'tv': 'Total Variation',
    }
    metric_repeat = tmp_dict['metric_repeat']

    result_dir = exp_dir / 'results'

    def val_fn():

        rng = jax.random.PRNGKey(88)

        save_dict = {}

        if dim == 1 : 
            save_dict = wandb_log_PME_animation(solver, pme_sol, save_dict) 

        save_dict['timesteps'] = timesteps
            
        data = []
        data_metric = {metric: [] for metric in metrics}
        c_neg_kl = 0
        for i, t in enumerate(timesteps):
            dist = solver.extract_solution(t)
            if dist is None:
                continue
            pme_dist = pme_sol.get_solution(t)
            cur = []
            if eval_init_samples : 
                dist_samples_all = dist.sample(jax.random.PRNGKey(900), n_samples_eval*metric_repeat)
                pme_samples_all = pme_dist.sample(jax.random.PRNGKey(900), n_samples_eval*metric_repeat)
            for metric in metrics :
                if metric == 'tv' : 
                    val = compute_metric(dist, pme_dist, metric='tv', num_sample=n_samples_eval, dim = dim)
                    raw = val 
                    val = math.log10(abs(val)+1e-15)
                    cur.append(val)
                else :
                    s = 0
                    raw = []
                    for seed in range(901, 901 + metric_repeat):
                        if eval_init_samples : 
                            dist_samples = jax.random.choice(jax.random.PRNGKey(seed),dist_samples_all,(n_samples_eval,1)).reshape(n_samples_eval,dim)
                            pme_samples = jax.random.choice(jax.random.PRNGKey(seed),pme_samples_all,(n_samples_eval,1)).reshape(n_samples_eval,dim)
                        else :
                            dist_samples = dist.sample(jax.random.PRNGKey(seed), n_samples_eval)
                            pme_samples = pme_dist.sample(jax.random.PRNGKey(seed), n_samples_eval)
                        if metric == 'objective' :
                            val = ((1/(m-1)) * (dist.p_batch(dist_samples)**(m-1))).mean()
                            raw.append(val)
                        elif metric == 'GT_objective' :
                            val = ((1/(m-1)) * (pme_dist.p_batch(pme_samples)**(m-1))).mean()
                            raw.append(val)
                        else :
                            val = compute_metric(dist, pme_dist, samples1 = dist_samples, samples2 = pme_samples, metric=metric, num_sample=n_samples_eval, seed=seed, dim = dim)
                            raw.append(val)
                            if val < 0 :
                                c_neg_kl += 1 
                            val = math.log10(abs(val)+1e-15)
                        s += val
                    cur.append(s / metric_repeat)
                data_metric[metric].append(raw)
            data.append([t, *cur])
            
            if eval_init_samples : 
                save_dict[f'samples_{i}'] = dist_samples_all
                save_dict[f'target_samples_{i}'] = pme_samples_all
            else :
                save_dict[f'samples_{i}'] = dist_samples
                save_dict[f'target_samples_{i}'] = pme_samples
    

        for metric in metrics:
            data_metric[metric] = np.array(data_metric[metric]) # i,j is time i, seed j
        save_dict['metric_dict'] = data_metric
        table = wandb.Table(data=data, columns=['t', *metrics])

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


    solver.set_custom_val_fn(val_fn)

    solver.run()
