Source code for Self-Consistent Velocity Matching of Probability Flows.

To run the mixture of Gaussians experiments with SCVM-NODE, while inside `tests/mog`, run
```
python run.py --exp_name=scvms_vnn_10d --project=uncategorized --dim=10 --num_mixture=10 --solver=scvms --use_ibp=true --flow=vnn --total_time=5.0 --train_num_step=10000 --train_batch_size=1000 --train_num_time=20 --val_freq=1000 --val_num_sample=1000 --optimizer=adam --b2=0.9 --scheduler=cosine_decay --alpha=0.001 --decay_steps=10000 --init_value=0.001 --wandb=true
```

The other experiments can be run similarly:
- `tests/ou`: the Ornstein-Uhlenbeck process experiment
- `tests/hourglass`: the obstacle flow experiment
- `tests/spline`: measure interpolation experiment
