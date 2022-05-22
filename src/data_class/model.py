import torch

from data_class.dataclass import DataClass


class Model(DataClass):
    weight_sharing: bool = False
    checkpoint_path: str = "checkpoint.torch"
    steps_per_checkpoint: int = 0  # 0 -> disabled
    print_on_init: bool = True
    momentumnet_beta: float = 0.99  # The higher this is, the more numerically stable. BUT also lower impact per layer
    depth: int = 64
    batch_size: int = 128
    sequence_length: int = 256
    float16: bool = False
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    offloading: bool = False