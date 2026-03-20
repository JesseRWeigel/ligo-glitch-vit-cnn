"""Training loop for CNN baseline on Gravity Spy spectrograms.

Implements:
- fp16 mixed precision via torch.amp
- AdamW optimizer with cosine annealing + linear warmup
- Focal loss with sqrt-inverse alpha
- Early stopping on val macro-F1
- Gradient clipping
- Per-epoch logging
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy, precision=fp16

import logging
import math
import time

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, num_classes,
                    gradient_clip_max_norm=1.0, scheduler=None):
    """Train for one epoch.

    Parameters
    ----------
    scheduler : CosineWarmupScheduler, optional
        If provided, stepped after each batch for per-step LR scheduling.

    Returns dict with 'loss' and 'macro_f1'.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        f1_metric.update(preds, labels)

    avg_loss = total_loss / max(n_batches, 1)
    macro_f1 = f1_metric.compute().item()
    return {"loss": avg_loss, "macro_f1": macro_f1}


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    """Validate on held-out set.

    Returns dict with 'loss', 'macro_f1', 'overall_accuracy', 'per_class_f1'.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average="none").to(device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        f1_metric.update(preds, labels)
        f1_per_class.update(preds, labels)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(n_batches, 1)
    macro_f1 = f1_metric.compute().item()
    per_class_f1 = f1_per_class.compute().cpu().tolist()
    overall_acc = correct / max(total, 1)

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "overall_accuracy": overall_acc,
        "per_class_f1": per_class_f1,
    }


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup, stepped per batch.

    During warmup: lr = base_lr * (step / warmup_steps)
    After warmup: cosine decay from base_lr to 0.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
