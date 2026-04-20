"""Learning rate schedulers and flow matching scheduler for Fizgig."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


# region Flow Matching Discrete Scheduler (for inference)

@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Euler scheduler for flow matching models.

    Modified from diffusers==0.29.2.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        n_tokens: Optional[int] = None,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self._step_index = None
        self._begin_index = None

        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, n_tokens: int = None):
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Pass one of `scheduler.timesteps` as a timestep."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        if self.config.solver == "euler":
            prev_sample = sample + model_output.to(torch.float32) * dt
        else:
            raise ValueError(f"Solver {self.config.solver} not supported.")

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


# endregion


# region RexLR Scheduler (for training)

class RexLR(torch.optim.lr_scheduler.LRScheduler):
    """Reflected Exponential (REX) learning rate scheduler.

    Reference: https://arxiv.org/abs/2107.04197
    Modified from: https://github.com/IvanVassi/REX_LR (Apache-2.0 License)
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        min_lr=0.0,
        num_steps=0,
        num_warmup_steps=0,
        rex_alpha=0.1,
        rex_beta=0.9,
        last_epoch=-1,
    ):
        if min_lr > max_lr:
            raise ValueError(f"min_lr ({min_lr}) must be <= max_lr ({max_lr})")
        if num_warmup_steps > num_steps:
            raise ValueError(f"num_warmup_steps ({num_warmup_steps}) must be <= num_steps ({num_steps})")

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        self.rex_alpha = rex_alpha
        self.rex_beta = rex_beta
        self.last_epoch = last_epoch

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Single warmup step
        if self.num_warmup_steps == 1 and self.last_epoch == 1:
            return [self.min_lr for _ in self.base_lrs]

        # Multi-step warmup: linear ramp from min_lr to max_lr
        if self.num_warmup_steps > 1 and 1 <= self.last_epoch <= (self.num_warmup_steps - 1):
            return [
                self.min_lr + (self.max_lr - self.min_lr) * (self.last_epoch - 1) / (self.num_warmup_steps - 1)
                for _ in self.base_lrs
            ]

        # Post-warmup REX decay
        step_after = self.last_epoch - self.num_warmup_steps
        remaining_steps = self.num_steps - self.num_warmup_steps

        if step_after >= remaining_steps or step_after == -1 or remaining_steps <= 0:
            return [self.min_lr for _ in self.base_lrs]

        rex_z = (remaining_steps - (step_after % remaining_steps)) / remaining_steps
        rex_factor = self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * (
            rex_z / (self.rex_alpha + self.rex_beta * rex_z)
        )

        return [base_lr * rex_factor for base_lr in self.base_lrs]

# endregion
