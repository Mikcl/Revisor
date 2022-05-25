from .dataclass import DataClass
from .onc_cycle import OneCycle


class AdaptiveGradientClipping(DataClass):
    gradient_clipping: float = 0.01
    zero_division_eps: float = 1e-6
    eps: float = 1e-3



class Optimizer(DataClass):
    type: str = "Adam"
    gradient_accumulation_steps: int = 1
    one_cycle: OneCycle = OneCycle()
    beta2: float = 0.95  # beta1 is controlled by one_cycle
    eps: float = 1e-8
    weight_decay: float = 0.01
    agc = AdaptiveGradientClipping()

