'''
Time dependant Fokker Planck & Bird Flocking simulation 
'''

from pathlib import Path
import wandb
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import math
from sklearn.datasets import make_spd_matrix

from cwgf.auto.ec import ExperimentCoordinator
from cwgf.problems.tFPE import TimeFPE
from cwgf.problems.distribution import Gaussian, gaussian_unnormalized_log_p
from cwgf.eval.cflow import wandb_log_animation
from cwgf.eval.metrics import compute_metric
from cwgf.eval.utils import save_dict_h5
from cwgf.eval.metrics import compute_consistency


class OUSolution:
    def __init__(self, A, b):
        self.dim = A.shape[0]
        self.w, self.v = np.linalg.eig(A)
        self.A_inv = np.linalg.inv(A)
        self.b = b


    def get_solution(self, t):
        I = np.eye(self.dim)
        mean = (I - self.get_neg_exp(t)) @ self.b
        neg_2_exp = self.get_neg_exp(2 * t)
        cov = self.A_inv @ (I - neg_2_exp) + neg_2_exp
        cov_sqrt = np.linalg.cholesky(cov)
        return Gaussian(mean, cov_sqrt)


    def get_neg_exp(self, t):
        '''
        Returns: e^{-At}
        '''
        return self.v @ np.diag(np.exp(-self.w * t)) @ self.v.T


class timeOUSolution:
    def __init__(self, A_t, b_t, D_tx, m0, c0, timesteps):
        self.dim = A_t(0).shape[0]
        I = np.eye(self.dim)
        df_mean = lambda t,x : -A_t(t) @ (x - b_t(t))
        def df_cov(t,x) : 
            x = x.reshape((self.dim,self.dim))
            res = -A_t(t) @ x - x @ A_t(t).T + 2*D_tx(t,x)
            return res.reshape((-1,))
        self.mean = solve_ivp(df_mean, [0,timesteps[-1]], m0, dense_output=True)
        self.cov = solve_ivp(df_cov, [0,timesteps[-1]], c0.reshape((-1,)), dense_output=True)
        self.timesteps = timesteps

    def get_solution(self, t):
        cov_sqrt = np.linalg.cholesky(self.cov.sol(t).reshape((self.dim,self.dim)))
        return Gaussian(self.mean.sol(t), cov_sqrt)


if __name__ == '__main__':

    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'metric_sym_kl': False,
        'metric_sym_pos_kl': True,
        'metric_ed': False,
        'metric_ot': True,
        'metric_consistency': True,
        'metric_repeat': 10,
        'val_num_time': 20,
        'val_num_sample': 500,
        'log_animation': True,
        'eval_SDE': True,
        'save_samples': True
    })
    ec.add_common_arguments({
        'dim': 2,
        'total_time': 10.,
        'type_exp': 'time_OU'
    })

    ec_result = ec.parse_args()
    tmp_dict = ec_result.tmp_dict
    config = ec_result.config
    exp_dir = ec_result.exp_dir

    dim = config['dim']
    rng_state = np.random.RandomState(seed=0)
    total_time = config['total_time']
    val_num_time = tmp_dict['val_num_time']
    val_num_sample = tmp_dict['val_num_sample']
    timesteps = np.linspace(0, total_time, num=val_num_time+1)

    if config['type_exp'] == 'time_OU':
        W = None
        a = 3
        w = 1
        sigma = np.sqrt(0.25)
        cov_sqrt_t = lambda t : sigma*np.eye(dim)
        D = lambda t,x : cov_sqrt_t(t) @ cov_sqrt_t(t).T
        c0 = (sigma**2)*np.eye(dim)
        if dim == 2 : 
            A_t = lambda t : jnp.array([[1,0],[0,3]])
            beta_t = lambda t : a*jnp.array([jnp.cos(np.pi*w*t),jnp.sin(np.pi*w*t)])
        if dim == 3 : 
            A_t = lambda t : jnp.array([[1,0,0],[0,3,0], [0,0,1]])
            beta_t = lambda t : a*jnp.array([jnp.cos(np.pi*w*t),jnp.sin(np.pi*w*t),t])
        init_dist = Gaussian(mean=beta_t(0.), cov_sqrt = jnp.linalg.cholesky(c0))
        b = lambda t,x: A_t(t) @ (beta_t(t)-x)
        grad_W = None
    elif config['type_exp'] == 'birds':
        dim = 2
        a = 4
        w = 0.5
        sigma = np.sqrt(0.25)
        A_t = lambda t : jnp.eye(dim)
        beta_t = lambda t : a*jnp.array([jnp.cos(np.pi*w*t),0.5*jnp.sin(2*np.pi*w*t)])
        cov_sqrt_t = lambda t : sigma*np.eye(dim)
        D = lambda t,x : cov_sqrt_t(t) @ cov_sqrt_t(t).T
        c0 = (sigma**2)*np.eye(dim)
        init_dist = Gaussian(mean=0*beta_t(0.), cov_sqrt = jnp.linalg.cholesky(c0))
        b = lambda t,x: A_t(t) @ (beta_t(t)-x)
        alpha = 2
        W = lambda t,x,y : - alpha*jnp.sin(2*np.pi*w*t)*0.5*((jnp.expand_dims(x,0)-y)**2).sum(axis=-1).mean(axis=0) # t : () / x : (D,) / y : (B,D)
        grad_W = None
    else : 
        raise NotImplementedError

    if config['type_exp'] == 'time_OU':
        GT_sol = timeOUSolution(A_t, beta_t, D, beta_t(0.), c0, timesteps)
    else :
        GT_sol = None


    problem = TimeFPE(dim=dim,
                    prior=init_dist,
                    b = b,
                    D = D,
                    W = W,
                    grad_W = grad_W,
                    total_time=total_time)

    if tmp_dict['eval_SDE']:
        n_steps_SDE = 1000
        n_part = 100
        x_SDE, h_SDE = problem.SDE_sampler(jax.random.PRNGKey(0), n_part, n_steps_SDE)
        x_SDE = np.array(x_SDE)

    solver = ec.create_solver(problem)

    metrics = []
    metric_name = {}
    if tmp_dict['metric_consistency']:
        metrics.append('consistency')
        metric_name['consistency'] = 'Consistency'
    if tmp_dict['metric_sym_kl'] and GT_sol is not None :
        metrics.append('sym_kl')
        metric_name['sym_kl'] = 'Symmetric KL'
        if tmp_dict['eval_SDE']:
            metrics.append('sym_kl_SDE')
            metric_name['sym_kl_SDE'] = 'Symmetric KL SDE'
    if tmp_dict['metric_sym_pos_kl'] and GT_sol is not None :
        metrics.append('sym_pos_kl')
        metric_name['sym_pos_kl'] = 'Symmetric Positive KL'
        if tmp_dict['eval_SDE']:
            metrics.append('sym_pos_kl_SDE')
            metric_name['sym_pos_kl_SDE'] = 'Symmetric Positive KL SDE'
    if tmp_dict['metric_ed'] and GT_sol is not None :
        metrics.append('ed')
        metric_name['ed'] = 'Energy Distance'
        if tmp_dict['eval_SDE']:
            metrics.append('ed_SDE')
            metric_name['ed_SDE'] = 'Energy Distance SDE'
    if tmp_dict['metric_ot'] and GT_sol is not None :
        metrics.append('ot')
        metric_name['ot'] = 'Sinkhorn Distance'
        if tmp_dict['eval_SDE']:
            metrics.append('ot_SDE')
            metric_name['ot_SDE'] = 'Sinkhorn Distance SDE'

    metric_repeat = tmp_dict['metric_repeat']
    result_dir = exp_dir / 'results'

    def val_fn():

        rng = jax.random.PRNGKey(88)

        save_dict = {}

        save_dict['timesteps'] = timesteps

        if dim == 2 and tmp_dict['log_animation'] :
            if tmp_dict['eval_SDE']:
                SDE_vis = x_SDE[::(len(x_SDE)//100),:20]
                wandb_log_animation(solver.global_step,
                                    problem.total_time,
                                    ref_fn = beta_t,
                                    exp_dir=exp_dir,
                                    vis_range=[-4.25, 4.25],
                                    num_timestep=100,
                                    wandb_key='SDE_video',
                                    fade_trajectory = True,
                                    vis_size = 10,
                                    s = 50,
                                    x_all = SDE_vis,
                                    anim_name='SDE')


            wandb_log_animation(solver.global_step,
                                problem.total_time,
                                lambda t: solver.extract_solution(t),
                                ref_fn = beta_t,
                                exp_dir=exp_dir,
                                vis_range=[-4.25, 4.25],
                                num_timestep=100,
                                wandb_key='video',
                                fade_trajectory = True,
                                vis_size = 25,
                                s = 50,
                                anim_name='result')

            if GT_sol is not None :
                wandb_log_animation(solver.global_step,
                                    problem.total_time,
                                    lambda t: GT_sol.get_solution(t),
                                    ref_fn = beta_t,
                                    exp_dir=exp_dir,
                                    vis_range=[-4.25, 4.25],
                                    num_timestep=100,
                                    wandb_key='gt_video',
                                    fade_trajectory = True,
                                    vis_size = 25,
                                    s = 50,
                                    anim_name='GT')
        
        save_dict = {}
        rng, target_rng = jax.random.split(rng)
        save_dict['timesteps'] = timesteps
        if tmp_dict['eval_SDE']:
            save_dict['SDE_samples'] = x_SDE

        data = []
        data_metric = {metric: [] for metric in metrics}
        for i, t in enumerate(timesteps):
            dist = solver.extract_solution(t)
            if dist is None:
                continue
            if GT_sol is not None :
                GT_dist = GT_sol.get_solution(t)
            if tmp_dict['eval_SDE']:
                SDE_samples = x_SDE[round(t/h_SDE)]
                dist_SDE = jax.scipy.stats.gaussian_kde(jnp.transpose(SDE_samples))
            cur_mean = []
            for metric in metrics:
                s = 0
                raw = []
                arr = []
                for seed in range(900, 900 + metric_repeat):
                    if 'SDE' in metric : 
                        cur_SDE_samples = jax.random.choice(jax.random.PRNGKey(seed),SDE_samples,(val_num_sample,1)).reshape(val_num_sample,dim)
                        m = compute_metric(
                                dist_SDE, GT_dist, samples1=cur_SDE_samples, metric=metric[:-4],
                                num_sample=val_num_sample, seed=seed, is_SDE=True)
                        raw.append(m)
                        m = math.log10(abs(m)+1e-12)
                    elif metric == 'consistency':
                        m = compute_consistency(rng=jax.random.PRNGKey(seed), state=solver.state,
                                                solver=solver)
                        raw.append(m)
                    else :
                        m = compute_metric(
                                dist, GT_dist, metric=metric,
                                num_sample=val_num_sample, seed=seed)
                        raw.append(m)
                        m = math.log10(abs(m)+1e-12)
                    arr.append(m)
                arr = np.array(arr)
                cur_mean.append(np.mean(arr))
                data_metric[metric].append(raw)
            data.append([t, *cur_mean])

            if tmp_dict['save_samples']:
                save_dict[f'samples_{i}'] = dist.sample(rng, 5000)
                if GT_sol is not None :
                    save_dict[f'target_samples_{i}'] = GT_dist.sample(target_rng, 5000)

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
