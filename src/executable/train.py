import torch

from ..data_class.context import Context
from ..dataset import get_dataset
from ..utils.pytorch import get_model


def train_model(ctx: Context, steps=None, load_model: bool = False):

    data = get_dataset(ctx)
    data_len = len(data)
    data = iter(data)
    mod = get_model(ctx, load_model, next(data)[0])

    mean_loss = torch.zeros([], device=ctx.model.device, dtype=torch.float16 if ctx.model.float16 else torch.float)
    mean_max_loss = mean_loss.clone()

    i = 0
    while True:
        i += 1

        mean_loss += mod.accumulated_step(next(data))

        mod.optimizer.step()

        mod.zero_grad()
        mod.scheduler.step()
        for p in mod.optimizer.param_groups:  # OneCycle resets beta2 to 0.990
            p['betas'] = p['betas'][0], mod.ctx.optimizer.beta2
        with torch.no_grad():
            if i % 5 == 0:
                log(mean_loss, mean_max_loss,
                    mod.optimizer.param_groups[0]['lr'], mod.optimizer.param_groups[0]['betas'])
                mean_loss.zero_()
                mean_max_loss.zero_()
            if None:
                # TODO save model for larger training with checkpoints
                mod.save()
        if steps and i > steps:
            return