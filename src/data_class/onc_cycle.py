from typing import Optional
from .dataclass import DataClass

class OneCycle(DataClass):
    cycle_min_lr: float = 3e-4  # Base learning rate used at the start and end of cycle.
    cycle_max_lr: float = 1e-3  # Learning rate used in the middle of the cycle. Can be smaller than cycle_min_lr
    decay_lr_rate: float = 1e-4  # Decay rate for learning rate.
    cycle_first_step_size: int = 2048  # Number of training iterations in the increasing half of a cycle.
    cycle_second_step_size: Optional[int] = None  # steps in second phase. None -> cycle_first_step_size
    cycle_first_stair_count: int = 0  # Number of stairs in first phase. 0 means staircase disabled
    cycle_second_stair_count: Optional[int] = None  # Number of stairs in second phase
    decay_step_size: int = 2  # Every how many steps to decay lr. 0 -> no decay
    cycle_momentum: bool = True  # Whether to cycle `momentum` inversely to learning rate.
    cycle_min_mom: float = 0.8  # Initial momentum which is the lower boundary in the cycle for each parameter group.
    cycle_max_mom: float = 0.9  # Upper momentum boundaries in the cycle for each parameter group.
    decay_mom_rate: float = 0  # Decay rate for momentum
    last_batch_iteration: int = -1  # The index of the last batch. This parameter is used when resuming a training job.