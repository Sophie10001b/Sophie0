import os
import math
import torch

from typing import Any, Dict, List, Optional, Tuple
from torch.optim.lr_scheduler import LRScheduler

class CosineLRSchedule(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup: Optional[int]=10000,
        max_lr: Optional[float]=1e-4,
        min_lr: Optional[float]=1e-6,
        max_steps: Optional[float]=150000
    ):
        self.warmup = warmup
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps

        super(CosineLRSchedule, self).__init__(optimizer)
    
    # override
    def get_lr(self) ->list[float]:
        step = max(1, self._step_count)
        if step <= self.warmup:
            scale = step / self.warmup
            return [min(lr * scale, self.max_lr) for lr in self.base_lrs]
        else:
            scale = (self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                    (1.0 + math.cos(((step - self.warmup) / (max(self.max_steps, step) - self.warmup)) * math.pi))) / self.max_lr
            if scale * self.max_lr < self.min_lr:
                scale = self.min_lr / self.max_lr
            return [min(lr * scale, self.max_lr) for lr in self.base_lrs]