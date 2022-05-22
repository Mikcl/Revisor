from data_class import DataClass
from data_class.onc_cycle import OneCycle


class Optimizer(DataClass):
    type: str = "Adam"
    gradient_accumulation_steps: int = 1
    one_cycle: OneCycle = OneCycle()
    beta2: float = 0.95  # beta1 is controlled by one_cycle
    eps: float = 1e-8
    weight_decay: float = 0.01
