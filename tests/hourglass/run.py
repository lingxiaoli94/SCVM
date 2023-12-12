'''
Hourglass animation, a special case of time-dependent FPE.
'''

from pathlib import Path
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import math
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib.animation import FuncAnimation

from scvm.auto.ec import ExperimentCoordinator
from scvm.problems.distribution import Gaussian
from scvm.problems.tFPE import TimeFPE
from scvm.eval.cflow import \
    wandb_log_animation, wandb_log_image
from scvm.eval.utils import save_dict_h5

def b_fn(t, x, *, force_scale, door_x=2, repulsion_scale=0.2):
    assert(x.shape[0] == 2)

    def dot(p, q):
        return (p * q).sum(-1)
    planks = [((0, 3), (3, 0.5)),
              ((1, 0), (1.5, 0)),
              ((-2, -4), (6, 0))]

    v = jnp.array([4, 0]) - x # attraction towards right

    for plank in planks:
        p, q = plank
        p = jnp.array(p)
        q = jnp.array(q)
        l = jnp.sqrt(dot(p - q, p - q))
        q_to_p = (p - q) / l
        r = dot(q_to_p, x - q)
        r = jnp.clip(r, 0, l)

        proj = r * q_to_p + q
        dist = ((x - proj) ** 2).sum(-1)
        dist_sqrt = jnp.sqrt(dist + 1e-8)
        v += force_scale * (x - proj) / dist_sqrt * jax.scipy.stats.norm.pdf(dist_sqrt, scale=repulsion_scale)

    return v


def D_fn(t, x):
    return jnp.eye(x.shape[0])


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'val_num_time': 10,
        'val_num_sample': 1000,
        'plot_flow': False,
        'save_traj': False
    })
    ec.add_common_arguments({
        'total_time': 2.0,
        'force_scale': 20.0,
        'cov_scale': 0.5
    })

    ec_result = ec.parse_args()
    tmp_dict = ec_result.tmp_dict
    config = ec_result.config
    exp_dir = ec_result.exp_dir

    cov_scale = config['cov_scale']
    init_dist = Gaussian(jnp.array([0., 0.]), cov_sqrt=cov_scale*jnp.eye(2))

    problem = TimeFPE(dim=2,
                      prior=init_dist,
                      b=partial(b_fn,
                                force_scale=config['force_scale']),
                      D=D_fn,
                      total_time=config['total_time'])

    rng_state = np.random.RandomState(seed=config['seed'])
    total_time = config['total_time']
    val_num_time = tmp_dict['val_num_time']
    val_num_sample = tmp_dict['val_num_sample']
    vis_num_time = val_num_time + 1
    timesteps = np.linspace(0, total_time, num=val_num_time+1)

    vis_range=[-5, 6]
    solver = ec.create_solver(problem)


    rng = jax.random.PRNGKey(88)

    rng, sde_rng = jax.random.split(rng)
    sde_rep = 10
    x_list_sde = problem.SDE_sampler(sde_rng, 5000, val_num_time*sde_rep,
                                     include_init=True)[0]
    x_list_sde = x_list_sde[::sde_rep]
    wandb_log_animation(0,
                        problem.total_time,
                        x_all=x_list_sde,
                        exp_dir=exp_dir,
                        vis_range=vis_range,
                        vis_size=5000,
                        s=1,
                        num_timestep=vis_num_time,
                        wandb_key='sde_video')

    result_dir = exp_dir / 'results'

    def val_fn():
        wandb_log_animation(solver.global_step,
                            problem.total_time,
                            lambda t: solver.extract_solution(t),
                            exp_dir=exp_dir,
                            vis_range=vis_range,
                            vis_size=5000,
                            s=1,
                            num_timestep=vis_num_time,
                            wandb_key='video')

        if tmp_dict['save_traj']:
            save_dict = {}
            save_dict['timesteps'] = timesteps
            eval_rng = jax.random.PRNGKey(99)
            eval_dict = solver.eval_multi_t(eval_rng, timesteps,
                                            val_num_sample=val_num_sample)
            save_dict.update(eval_dict)
            save_dict['em_xt'] = np.stack(x_list_sde, 1)
            save_dict_h5(save_dict,
                         result_dir / f'step-{solver.global_step}.h5',
                         create_dir=True)


        if not tmp_dict['plot_flow']:
            return

        rng = jax.random.PRNGKey(99)
        eval_dict = solver.eval_multi_t(rng, timesteps)

        x0 = eval_dict['x0']
        v0 = jnp.zeros_like(x0)
        xt = eval_dict['xt']
        import matplotlib.lines as lines
        def add_planks(ax):
            planks = [((0, 3), (3, 0.5)),
                          ((1, 0), (1.5, 0)),
                          ((-2, -4), (6, 0))]

            for plank in planks:
                p, q = plank
                p = np.array(p)
                q = np.array(q)
                ax.add_artist(lines.Line2D([p[0], q[0]], [p[1], q[1]], linewidth=4.0, color='darkviolet'))
        x_lim = (-2, 6)
        y_lim = (-3, 3)
        ratio = (y_lim[1] - y_lim[0]) / (x_lim[1] - x_lim[0])
        fig, ax = plt.subplots(figsize=(4, 4*ratio))
        ax.set_aspect('equal')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        colors = ss.rankdata(x0[:, 1])
        sc = ax.scatter(x0[:, 0], x0[:, 1], s=4, c=colors, cmap='winter')
        add_planks(ax)

        def update(i):
            sc.set_offsets(xt[:, i, :2])
            return sc

        ani = FuncAnimation(fig, update, frames=vis_num_time)
        f = exp_dir / 'flow_anim.gif'
        writergif = matplotlib.animation.PillowWriter(fps=10)
        ani.save(f, writer=writergif)
        wandb.log({'flow': wandb.Video(str(f), fps=1, format='gif')},
                  step=solver.global_step)
        plt.close(fig)

    solver.set_custom_val_fn(val_fn)

    solver.run()
