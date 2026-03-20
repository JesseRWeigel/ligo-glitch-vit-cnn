"""Training loop for ViT-B/16 on Gravity Spy spectrograms.

Identical to train_cnn.py except:
- Uses layer-wise LR decay parameter groups
- CosineWarmupScheduler adapted for multi-group LR scaling

All other components (focal loss, data loading, metrics) are reused
from Phase 2 infrastructure via import.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy, precision=fp16

import logging
import math

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score

logger = logging.getLogger(__name__)


# Reuse train_one_epoch and validate from CNN training -- they are model-agnostic.
from src.training.train_cnn import train_one_epoch, validate


class CosineWarmupSchedulerMultiGroup:
    """Cosine annealing with linear warmup for multi-group optimizers.

    Each param group has its own base_lr (set at optimizer construction).
    During warmup: lr_i = base_lr_i * (step / warmup_steps)
    After warmup: lr_i = base_lr_i * 0.5 * (1 + cos(pi * progress))

    This preserves the LR ratios from layer-wise decay throughout training.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        # Record initial LR for each group as the base
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self):
        """Return max LR across groups (head LR) for logging."""
        return max(pg["lr"] for pg in self.optimizer.param_groups)
