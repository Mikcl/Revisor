import copy
import typing

import torch
from torch.nn import functional as F

from deepspeed.runtime import lr_schedules

from .data_class.context import Context
from .optimizers.build import build_optimizer

class Trainer(torch.nn.Module):
    def __init__(self, ctx: Context, model: torch.nn.Module, data: typing.Optional[torch.Tensor]):
        super(Trainer, self).__init__()
        self.ctx = ctx
        self.model = torch.jit.trace(model, data) if data else model
        self.optimizer = build_optimizer(ctx, self.model.parameters())
        self.scheduler = lr_schedules.OneCycle(self.optimizer,
                                               ctx.optimizer.one_cycle.cycle_min_lr,
                                               ctx.optimizer.one_cycle.cycle_max_lr,
                                               ctx.optimizer.one_cycle.decay_lr_rate,
                                               ctx.optimizer.one_cycle.cycle_first_step_size,
                                               ctx.optimizer.one_cycle.cycle_second_step_size,
                                               ctx.optimizer.one_cycle.cycle_first_stair_count,
                                               ctx.optimizer.one_cycle.cycle_second_stair_count,
                                               ctx.optimizer.one_cycle.decay_step_size,
                                               ctx.optimizer.one_cycle.cycle_momentum,
                                               ctx.optimizer.one_cycle.cycle_min_mom,
                                               ctx.optimizer.one_cycle.cycle_max_mom,
                                               ctx.optimizer.one_cycle.decay_mom_rate,
                                               ctx.optimizer.one_cycle.last_batch_iteration)

    @torch.no_grad()
    def _to_device_detach(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.to(device=self.ctx.model.device, non_blocking=True).detach()

    def _forward_backward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        x_batch = self._to_device_detach(src)
        output = self.model(x_batch)
        loss = F.cross_entropy(output, self._to_device_detach(tgt))
        loss.backward()
        return loss.detach()

    @torch.no_grad()
    def _clip_gradient(self):
        for p in self.gradients():
            g_norm = p.grad.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.zero_division_eps)
            p_norm = p.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.eps)
            grad_scale = (p_norm / g_norm * self.ctx.optimizer.agc.gradient_clipping).clamp(max=1)
            p.grad.data.copy_(p.grad * grad_scale)

    def accumulated_step(self, data: torch.Tensor) -> torch.Tensor:
        loss = sum(self._forward_backward(s, t) for s, t in zip(*data))
        self._clip_gradient()
        return loss

    @torch.no_grad()
    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = None

    @torch.no_grad()
    def gradients(self) -> torch.nn.Parameter:
        for p in self.model.parameters():
            if p.grad is None:
                continue
            yield p

    def save(self):
        torch.save(self.state_dict(), self.ctx.model.checkpoint_path)

    def load(self):
        wrong_keys = self.load_state_dict(torch.load(self.ctx.model.checkpoint_path), strict=False)
        for key in wrong_keys.missing_keys + wrong_keys.unexpected_keys:
            if not any(k.startswith('_') for k in key.split('.')):
                if key in wrong_keys.missing_keys:
                    raise ValueError(f"{key} is missing in checkpoint but exists in model")
                if key in wrong_keys.unexpected_keys:
                    raise ValueError(f"{key} is missing in model but exists in checkpoint")


