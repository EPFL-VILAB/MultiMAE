# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class NoWeightingStrategy(nn.Module):
    """No weighting strategy
    """

    def __init__(self, **kwargs):
        super(NoWeightingStrategy, self).__init__()

    def forward(self, task_losses):
        return task_losses

class UncertaintyWeightingStrategy(nn.Module):
    """Uncertainty weighting strategy
    """

    def __init__(self, tasks):
        super(UncertaintyWeightingStrategy, self).__init__()

        self.tasks = tasks
        self.log_vars = nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, task_losses):
        losses_tensor = torch.stack(list(task_losses.values()))
        non_zero_losses_mask = (losses_tensor != 0.0)

        # calculate weighted losses
        losses_tensor = torch.exp(-self.log_vars) * losses_tensor + self.log_vars

        # if some loss was 0 (i.e. task was dropped), weighted loss should also be 0 and not just log_var as no information was gained
        losses_tensor *= non_zero_losses_mask

        # return dictionary of weighted task losses
        weighted_task_losses = task_losses.copy()
        weighted_task_losses.update(zip(weighted_task_losses, losses_tensor))
        return weighted_task_losses
