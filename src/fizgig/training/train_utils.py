"""Training utilities: loss tracking, checkpoint naming, state management."""

import argparse
import logging
import os
import shutil
from typing import Callable

import accelerate

logger = logging.getLogger(__name__)


# Checkpoint file naming patterns
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"


class LossRecorder:
    """Track per-step losses with a running moving average over the last epoch's worth of steps.

    Slots are indexed by the in-epoch step. A step that records no loss — e.g. an image the
    loss watch excluded from training — must be drop()ed so its slot leaves the average;
    otherwise skipped slots hold stale (or zero-padded) values that bias avr_loss, which the
    adaptive-LR watcher reads as a real signal."""

    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0
        self._empty: set[int] = set()  # slots not currently holding a live loss

    def _grow(self, step: int) -> None:
        while len(self.loss_list) <= step:
            self._empty.add(len(self.loss_list))
            self.loss_list.append(0.0)

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        self._grow(step)
        if step not in self._empty:
            self.loss_total -= self.loss_list[step]
        self._empty.discard(step)
        self.loss_list[step] = loss
        self.loss_total += loss

    def drop(self, *, step: int) -> None:
        """Mark an in-epoch step as not-trained (skipped/excluded): its slot leaves the average."""
        self._grow(step)
        if step not in self._empty:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = 0.0
            self._empty.add(step)

    @property
    def moving_average(self) -> float:
        n = len(self.loss_list) - len(self._empty)
        if n <= 0:
            return 0.0
        return self.loss_total / n


def get_epoch_ckpt_name(model_name: str, epoch_no: int) -> str:
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_step_ckpt_name(model_name: str, step_no: int) -> str:
    return STEP_FILE_NAME.format(model_name, step_no) + ".safetensors"


def get_last_ckpt_name(model_name: str) -> str:
    return model_name + ".safetensors"


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None
    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int):
    model_name = args.output_name
    logger.info(f"Saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"Removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator: accelerate.Accelerator, step_no: int):
    model_name = args.output_name
    logger.info(f"Saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"Removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator: accelerate.Accelerator):
    model_name = args.output_name
    logger.info("Saving final state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)


def get_sanitized_config_or_none(args: argparse.Namespace):
    """Return args dict for logging, with sensitive values filtered out. Returns None if --log_config is not set."""
    if not args.log_config:
        return None

    sensitive_args = {"wandb_api_key", "huggingface_token"}
    sensitive_path_args = {"dit", "vae", "text_encoder", "base_weights", "network_weights", "output_dir", "logging_dir"}

    filtered = {}
    for k, v in vars(args).items():
        if k in sensitive_args or k in sensitive_path_args:
            continue
        if v is None or isinstance(v, (bool, str, float, int)):
            filtered[k] = v
        elif isinstance(v, list):
            filtered[k] = str(v)
        else:
            filtered[k] = str(v)
    return filtered


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    """Return a linear interpolation function from (x1,y1) to (x2,y2)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b
