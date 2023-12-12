'''
Spline of measures obtained by applying KDE to images.
'''

from pathlib import Path
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import math

from scvm.auto.ec import ExperimentCoordinator
from scvm.problems.distribution import Gaussian, \
    StdGaussian, FuzzyPointCloud
from scvm.problems.spline import Spline
from scvm.problems.kl import KLDivergence
from scvm.eval.cflow import \
    wandb_log_animation, wandb_log_image
from scvm.eval.metrics import compute_metric
from scvm.eval.utils import save_dict_h5


def build_dist_from_img(img, threshold=0.5, bandwidth=0.05,
                        exact_sample=False):
    arr = np.array(img) / 255
    W, H = arr.shape

    pc = []
    for x in range(W):
        for y in range(H):
            if arr[x, y] > threshold:
                pc.append([y / H, 1-x / W])
    pc = np.array(pc)
    return FuzzyPointCloud(pc, bandwidth=bandwidth,
                           exact_sample=exact_sample)


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'val_num_time': 10,
        'val_num_sample': 1000,
        'save_h5': False,
        'save_log_p': False
    })
    ec.add_common_arguments({
        'use_mnist': True,
        'use_ot_loss': True,
        'ot_threshold': 1e-4,
        'debiased': False,
        'ot_eps': 0.05,
        'pc_size': 5000,
        'total_time': 6.0,
        'key_indices': [0, 2],
        'time_bandwidth': 1.0,
        'multiplier': 1.0,
        'pc_bandwidth': 0.05,
        'gaussian_init': False,
        'exact_sample': False,
    })

    ec_result = ec.parse_args()
    tmp_dict = ec_result.tmp_dict
    config = ec_result.config
    exp_dir = ec_result.exp_dir

    init_dist = Gaussian(np.array([0.5, 0.5]), cov_sqrt=0.2*np.eye(2))

    if config['use_mnist']:
        # There is some import error with JAX and torch if import MINST earlier.
        # A JAX call must be made before importing MNIST.
        from torchvision.datasets import MNIST
        dataset = MNIST(root_dir / 'mnist_data',
                        download=True,
                        train=True)

        image_dict = {}
        for i in range(len(dataset)):
            img, label = dataset[i]
            if label not in image_dict:
                image_dict[label] = img
            if len(image_dict.keys()) == 10:
                break

        images = [image_dict[i] for i in range(10)]
        image_dists = [build_dist_from_img(img, bandwidth=config['pc_bandwidth'],
                                           exact_sample=config['exact_sample']) for img in images]
        key_dists = image_dists
    else:
        # Use 3D point clouds as key distributions.
        pts = np.load(root_dir / 'mesh_pcs_hhs.npy')
        print(f'pts: {pts.shape}')
        pts = pts[:, :config['pc_size'], :] # use a smaller set for speed
        key_dists = [FuzzyPointCloud(pts[i], bandwidth=config['pc_bandwidth'],
                                     exact_sample=config['exact_sample']) for i in range(pts.shape[0])]

    key_dists = [key_dists[i] for i in config['key_indices']]
    key_timestamps = np.linspace(0, config['total_time'], len(key_dists) + 1)[:len(key_dists)]

    if not config['gaussian_init']:
        init_dist = key_dists[0]

    problem = Spline(
        use_ot_loss=config['use_ot_loss'],
        ot_threshold=config['ot_threshold'],
        dim=2 if config['use_mnist'] else 3,
        prior=init_dist,
        key_timestamps=key_timestamps[1:],
        key_dists=key_dists[1:],
        time_bandwidth=config['time_bandwidth'],
        total_time=config['total_time'],
        multiplier=config['multiplier'],
        debiased=config['debiased'],
        ot_eps=config['ot_eps'])

    rng_state = np.random.RandomState(seed=config['seed'])
    total_time = config['total_time']
    val_num_time = tmp_dict['val_num_time']
    val_num_sample = tmp_dict['val_num_sample']
    vis_num_time = val_num_time + 1
    timesteps = np.linspace(0, total_time, num=val_num_time+1)

    if config['use_mnist']:
        vis_range = [-0.5, 1.5]
    else:
        vis_range = [-2, 2]
    solver = ec.create_solver(problem)

    for i in range(len(key_dists)):
        wandb_log_image(0,
                        key_dists[i],
                        vis_range=vis_range,
                        vis_size=1000,
                        wandb_key=f'key_dist_{i}')

    result_dir = exp_dir / 'results'
    def val_fn():
        wandb_log_animation(solver.global_step,
                            problem.total_time,
                            lambda t: solver.extract_solution(t),
                            exp_dir=exp_dir,
                            vis_range=vis_range,
                            vis_size=val_num_sample,
                            num_timestep=vis_num_time,
                            wandb_key='video',
                            s=5)
        wandb_log_animation(solver.global_step,
                            problem.total_time,
                            lambda t: solver.extract_solution(t),
                            exp_dir=exp_dir,
                            vis_range=vis_range,
                            vis_size=val_num_sample,
                            num_timestep=vis_num_time,
                            wandb_key='video_02',
                            s=5,
                            x_ind=0,
                            y_ind=2)
        wandb_log_animation(solver.global_step,
                            problem.total_time,
                            lambda t: solver.extract_solution(t),
                            exp_dir=exp_dir,
                            vis_range=vis_range,
                            vis_size=val_num_sample,
                            num_timestep=vis_num_time,
                            wandb_key='video_12',
                            s=5,
                            x_ind=1,
                            y_ind=2)

        save_dict = {}
        rng = jax.random.PRNGKey(88)
        rng, eval_rng = jax.random.split(rng)

        if tmp_dict['save_h5']:
            timesteps = jnp.linspace(0, total_time, 500)
            save_dict['timesteps'] = timesteps
            eval_dict = solver.eval_multi_t(eval_rng,
                                            timesteps,
                                            val_num_sample) # use train_batch_size
            save_dict.update(eval_dict)

            save_dict_h5(save_dict,
                         result_dir / f'step-{solver.global_step}.h5',
                         create_dir=True)


    solver.set_custom_val_fn(val_fn)

    solver.run()
