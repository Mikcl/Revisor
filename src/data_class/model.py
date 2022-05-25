import torch

from .dataclass import DataClass


class Model(DataClass):
    weight_sharing: bool = False
    checkpoint_path: str = "checkpoint.torch"
    steps_per_checkpoint: int = 0  # 0 -> disabled
    print_on_init: bool = True
    features: int = 256
    momentumnet_beta: float = 0.99  # The higher this is, the more numerically stable. BUT also lower impact per layer
    depth: int = 64
    batch_size: int = 128
    sequence_length: int = 256
    float16: bool = False
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    offloading: bool = False

    # PROBABLY WANT TO DELETE
    conv_kernel_size: int = 7
    feature_shuffle: bool = False
    feed_forward_intermediate_factor: float = 2.
    norm_power: int = 2  # 1 = mean(abs(x)), 2 = std, ...
    bottleneck_group: int = 1  # not all group counts are possible. it has to be divide self.features without residual
    offloading: bool = False
    input_groups: int = 1
    output_groups: int = 1
    experts_in_input: int = 0  # 0 to disable MoE
    experts_in_output: int = 0
    moe_jitter_epsilon: float = 0.02
    expert_chunks: int = 1  # Increase it if not all MoE parameters fit onto the GPU