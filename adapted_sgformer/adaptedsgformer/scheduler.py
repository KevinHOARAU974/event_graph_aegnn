from functools import partial
# import math
from typing import List

import numpy as np


class WarmupCosineScheduler:
    def __init__(self,
                 warmup_epochs: float,
                 num_iters_per_epoch: int,
                 tot_num_epochs: int,
                 min_lr_ratio: float=0.05,
                 warmup_lr_start: float=0):

        self.warmup_iters = int(num_iters_per_epoch * warmup_epochs)
        self.total_iters = int(tot_num_epochs * num_iters_per_epoch)

        self.min_lr_ratio = min_lr_ratio
        self.warmup_lr_start = warmup_lr_start

    def __call__(self, iters: int)->float:


        if iters < self.warmup_iters : 

            progress = iters / max(1, self.warmup_iters)

            # return self.warmup_lr_start + (1.0 - self.warmup_lr_start)*progress**2
            return self.warmup_lr_start + (1.0 - self.warmup_lr_start)*progress

        else:

            progress = (iters - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)

            progress = min(1.0, progress)

            return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + np.cos(np.pi * progress))

