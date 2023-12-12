'''
Temporary hyperparameter class.
This class contains hyperparameters that should not be saved to config files.
'''

from dataclasses import dataclass

@dataclass
class THP:
    train_batch_size: int
    train_num_time: int # used for VMS
    train_time_pow: float # power applied to timestamps during training
    train_even_time: bool # whether to use evenly spaced time
    train_noise_level: float # noise added to xt
    train_noise_scale_t: bool # whether to scale the noise inversely by t
    train_final_t: bool # whether to include total_time in each iteration
    train_flow_rep: int # number of repetitions for flow, used in GANs
    train_potential_rep: int # number of repetitions for potential
    train_score_rep: int # number of repetitions for score
    ref_size: int # used for interaction functionals
    train_num_step: int
    extend_final_t: bool
    val_freq: int
    save_freq: int
    is_val: bool
    ckpt_name: str


    def should_validate(self, step):
        # Whether to validate after training step "step".
        return step % self.val_freq == 0


    def should_save(self, step):
        # Whether to save after training step "step".
        return step % self.save_freq == 0
