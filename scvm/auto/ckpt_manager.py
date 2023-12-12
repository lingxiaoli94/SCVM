'''
A help class for saving/loading models.
'''

import flax
from flax.training import checkpoints
from pathlib import Path


class CkptManager:
    def __init__(self, solver):
        '''
        solver must have a field "state" that is serializable.
        '''
        from scvm.auto.registry import Registry

        setting = Registry.get_solver_setting(solver)
        assert(setting.need_thp and
               setting.need_optimizer)
        self.solver = solver


    def _get_ckpt_dir(self):
        return self.solver.exp_dir / self.solver.thp.ckpt_name


    def load(self):
        ckpt_path = self._get_ckpt_dir()
        print(f'ckpt_path: {ckpt_path}')
        if ckpt_path.exists():
            self.solver.state = checkpoints.restore_checkpoint(
                self._get_ckpt_dir(),
                target=self.solver.state)
            print('Restoring checkpoint at {}...'.format(ckpt_path))


    def save(self):
        checkpoints.save_checkpoint(
            self._get_ckpt_dir(),
            target=self.solver.state,
            step=self.solver.state.step,
            overwrite=True,
            keep=1)
