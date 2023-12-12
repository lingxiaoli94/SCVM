'''
Shortcuts to evaluate continuous-time flow solvers.
'''
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from functools import partial

def wandb_log_animation(step = 0,
                        total_time = 1.,
                        dist_fn = None,
                        *,
                        exp_dir = None,
                        ref_fn = None,
                        seed=999,
                        num_timestep=50,
                        vis_size=500,
                        vis_range=[-3, 3],
                        wandb_key='video',
                        fade_trajectory = False,
                        s=20,
                        x_all = None,
                        anim_name = 'tmp_anim',
                        x_ind=0,
                        y_ind=1):

    rng = jax.random.PRNGKey(seed)
    timesteps = jnp.linspace(0, total_time, num=num_timestep)

    if x_all is None :
        x_all = []
        for k, t in enumerate(timesteps):
            dist = dist_fn(t)
            if dist is None:
                num_timestep = k
                break
            xs = dist.sample(rng, vis_size)
            x_all.append(xs[:, [x_ind, y_ind]])

    if ref_fn is not None:
        ref_all = []
        for k, t in enumerate(timesteps):
            ref_all.append(ref_fn(t))

    x_all = jnp.array(x_all)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.set_xlim(*vis_range)
    ax.set_ylim(*vis_range)

    if fade_trajectory :
        global x_vals,y_vals, refs_vals, intensity, intensity_ref
        x_vals = []
        y_vals = []
        refs_vals = []
        intensity = []
        intensity_ref = []
        colors = [[0,0,1,0],[0,0,1,0.5],[0,0.2,0.4,1]]
        cmap = LinearSegmentedColormap.from_list("", colors)
        sc = ax.scatter(x_vals,y_vals, c=[], cmap=cmap, vmin=0,vmax=1,s=s)
        sc.set_array(intensity)
        if ref_fn is not None :
            colors = [[1,0,0,0],[1,0,0,0.5],[0.4,0.2,0.,1]]
            cmap = LinearSegmentedColormap.from_list("", colors)
            sc_ref = ax.scatter([], [],c = [], cmap=cmap, vmin=0,vmax=1, s=s)
            sc_ref.set_array(intensity_ref)
    else :
        sc = ax.scatter(x_all[0, :, 0], x_all[0, :, 1], s=s)

    def update(frame_id):
        if fade_trajectory :
            global x_vals,y_vals, refs_vals, intensity, intensity_ref
            new_xvals, new_yvals = x_all[frame_id][:,0], x_all[frame_id][:,1]
            x_vals.extend(new_xvals)
            y_vals.extend(new_yvals)
            sc.set_offsets(np.c_[x_vals,y_vals])
            intensity = np.concatenate((np.array(intensity)*0.8, np.ones(len(new_xvals))))
            sc.set_array(intensity)
            if ref_fn is not None :
                new_refs_val = [ref_all[frame_id][0],ref_all[frame_id][1]]
                refs_vals.append(new_refs_val)
                sc_ref.set_offsets(refs_vals)
                intensity_ref = np.concatenate((np.array(intensity_ref)*0.8, [1]))
                sc_ref.set_array(intensity_ref)
        else :
            sc.set_offsets(x_all[frame_id])
        return sc

    ani = FuncAnimation(fig, update, frames=num_timestep,
                        interval=200, repeat_delay=2000)
    f = exp_dir / (anim_name+'.gif')
    writergif = matplotlib.animation.PillowWriter(fps=10)
    ani.save(f, writer=writergif)
    wandb.log({wandb_key: wandb.Video(str(f), fps=1, format='gif')},
              step=step)
    plt.close(fig)


def wandb_log_image(step, dist, *,
                    seed=999,
                    vis_size=500,
                    vis_range=[-3, 3],
                    wandb_key='samples'):
    rng = jax.random.PRNGKey(seed)
    samples = dist.sample(rng, vis_size)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.set_xlim(*vis_range)
    ax.set_ylim(*vis_range)
    sc = ax.scatter(samples[:, 0], samples[:, 1], s=5)
    wandb.log({wandb_key: wandb.Image(fig)},
              step=step)
    plt.close(fig)
