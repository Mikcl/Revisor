import numpy as np
import torch
import torch.utils.data.dataloader

import math
import random

from ..data_class.context import Context
from typing import Optional

from ..model import LinearAttentionLM
from ..linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper
from ..trainer import Trainer
from .formatting import pretty_print

DataLoaderIter = torch.utils.data.dataloader._BaseDataLoaderIter


def setup_torch(seed: int):
    torch._C._debug_set_autodiff_subgraph_inlining(False)  # skipcq: PYL-W0212
    torch._C._set_graph_executor_optimize(True)  # skipcq: PYL-W0212
    torch._C._set_backcompat_broadcast_warn(False)  # skipcq: PYL-W0212
    torch._C._set_backcompat_keepdim_warn(False)  # skipcq: PYL-W0212
    torch._C._set_cudnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_mkldnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_mkldnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_cudnn_benchmark(True)  # skipcq: PYL-W0212
    torch._C._set_cudnn_deterministic(False)  # skipcq: PYL-W0212
    torch._C._set_cudnn_allow_tf32(True)  # skipcq: PYL-W0212
    torch._C._set_cublas_allow_tf32(True)  # skipcq: PYL-W0212
    torch._C._jit_set_inline_everything_mode(True)  # skipcq: PYL-W0212

    torch._C._jit_set_profiling_executor(True)  # skipcq: PYL-W0212
    torch._C._jit_set_profiling_mode(True)  # skipcq: PYL-W0212
    torch._C._jit_override_can_fuse_on_cpu(False)  # skipcq: PYL-W0212
    torch._C._jit_override_can_fuse_on_gpu(True)  # skipcq: PYL-W0212
    torch._C._jit_set_texpr_fuser_enabled(True)  # skipcq: PYL-W0212
    torch._C._jit_set_nvfuser_enabled(False)  # skipcq: PYL-W0212

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model(ctx: Context, load_model: bool, data: Optional[torch.Tensor] = None) -> Trainer:
    
    mod = Trainer(ctx, AutoregressiveWrapper(LinearAttentionLM(ctx).to(dtype=torch.float16 if ctx.model.float16 else torch.float)),
                  data if data is None else None)

    if ctx.model.print_on_init:
        pretty_print(str(mod))

    parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, mod.parameters()))
    base = int(math.log10(parameters) / 3)
    pretty_print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')
    if load_model:
        mod.load()
    if not ctx.model.offloading:
        mod = mod.to(ctx.model.device)
    return mod


def encode(prompt: str) -> torch.Tensor:
    return torch.as_tensor(np.frombuffer(prompt.encode('UTF-8'), np.uint8))


def decode(output: torch.LongTensor) -> str:
    return ''.join(chr(c) for c in output.view(-1).unbind(0))