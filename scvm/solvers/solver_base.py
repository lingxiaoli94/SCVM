from abc import ABC, abstractmethod
from pathlib import Path
import functools
import time

class SolverBase(ABC):
    def __init__(self, *, problem, exp_dir, seed):
        '''
        Args:
          problem: An instance of ProblemBase.
          exp_dir: Directory to put all checkpoints/results in.
          seed: A random seed if the solver needs reproducibility.
        '''
        self.problem = problem
        self.exp_dir = Path(exp_dir)
        self.seed = seed

        self.total_val_time = 0


    def set_custom_val_fn(self, val_fn):
        '''
        Args:
          val_fn:
            A function that takes a reference to self (SolverBase instance).
            This function will be called during training and testing, but when
            it is called is up to the subclass.

            This function should be passed in from a run.py since the validation
            procedure depends on the application. This function can perform
            heterogeneous validation depending on the solver class.
        '''
        self.custom_val_fn = val_fn


    def _validate(self):
        start_time = time.time()
        self.custom_val_fn()
        end_time = time.time()
        self.total_val_time += end_time - start_time
        print(f'Validation at step {self.global_step} takes {end_time-start_time}s.')


    @abstractmethod
    def run(self):
        pass


    def extract_solution(self, t):
        '''
        Extract solution at time t as a distribution (an instance of Distribution
        class).

        Args:
          t: Time to extract the solution.
        '''
        return None
