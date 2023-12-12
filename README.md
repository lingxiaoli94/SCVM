# Source code for Self-Consistent Velocity Matching of Probability Flows (NeurIPS 2023).

## Dependencies
- jax (please follow the official installation instructions; all other packages can be `pip` installed)
- optax
- flax
- jaxopt
- diffrax
- numpyro
- wandb
- scikit-learn
- tqdm
- matplotlib
- h5py
- POT

## Code structure
- `scvm/` folder is a Python package that involves all core implementation.
  - `scvm/problems/` contains classes for all problem type, with base class defined in `problem_base.py`.
  - `scvm/solvers` contains classes for all baseline solvers and the proposed solver SCVM, with base class defined in `solver_base.py`.
- `tests/` folder contains experiments from the paper. 
  - Each subfolder contains a `run.py` file which is the entry point of an experiment.

## How to run
First, install package `scvm` via `pip install . -e` or `conda develop .`

To run the mixture of Gaussians experiments with SCVM-NODE, while inside `tests/mog`, run
```
python run.py --exp_name=scvms_vnn_10d --project=uncategorized --dim=10 --num_mixture=10 --solver=scvms --use_ibp=true --flow=vnn --total_time=5.0 --train_num_step=10000 --train_batch_size=1000 --train_num_time=20 --val_freq=1000 --val_num_sample=1000 --optimizer=adam --b2=0.9 --scheduler=cosine_decay --alpha=0.001 --decay_steps=10000 --init_value=0.001 --wandb=true
```
By default, the logging will be done through WANDB, where various metrics and a video of the current probability flow will be logged at each validation step.

The other experiments can be run similarly---see the corresponding `run.py` for the experiment-specific command-line arguments:
- `tests/ou`: the Ornstein-Uhlenbeck process experiment
- `tests/pme`: the porous medium equation experiment
- `tests/tFPE`: the time-dependent Fokker-Planck equation experiment
- `tests/hourglass`: the obstacle flow experiment
- `tests/spline`: measure interpolation experiment
