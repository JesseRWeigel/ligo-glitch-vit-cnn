"""ViT-B/16 classifier for Gravity Spy spectrogram classification.

Uses timm for pretrained ViT-B/16 weights with AugReg (ImageNet-21k -> 1k).
Implements layer-wise LR decay for fine-tuning per Steiner et al. 2022.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy, input_format=224x224_RGB_PNG_0to1

import logging

import timm
from timm.optim import param_groups_layer_decay

logger = logging.getLogger(__name__)

# Model ID priority list (best to worst pretrained weights)
MODEL_CANDIDATES = [
    ("vit_base_patch16_224.augreg_in21k_ft_in1k", "AugReg ImageNet-21k fine-tuned on ImageNet-1k"),
    ("vit_base_patch16_224.orig_in21k_ft_in1k", "Original ImageNet-21k fine-tuned on ImageNet-1k"),
    ("vit_base_patch16_224", "Default pretrained (fallback)"),
]


def build_vit_classifier(num_classes=23, pretrained=True):
    """Build ViT-B/16 classifier with best available pretrained weights.

    Parameters
    ----------
    num_classes : int
        Number of output classes (23 for Gravity Spy O3).
    pretrained : bool
        Whether to load pretrained weights.

    Returns
    -------
    model : nn.Module
        ViT-B/16 model with classification head replaced.
    model_id : str
        The timm model ID that was successfully loaded.
    pretrain_source : str
        Description of the pretrained weights used.
    """
    for model_id, source in MODEL_CANDIDATES:
        try:
            model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Loaded model: {model_id} ({source})")
            logger.info(f"Total params: {total_params / 1e6:.1f}M, Trainable: {trainable_params / 1e6:.1f}M")
            return model, model_id, source
        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")

    raise RuntimeError("Could not load any ViT-B/16 variant. Check timm installation and network.")


def get_layer_wise_lr_groups(model, base_lr, layer_decay=0.75, weight_decay=0.01):
    """Create optimizer parameter groups with layer-wise LR decay.

    Uses timm's built-in param_groups_layer_decay for correct ViT layer
    identification. ViT-B/16 has 12 transformer blocks:
      - Block i gets lr = base_lr * decay^(12 - i)
      - Block 0 (earliest): base_lr * 0.75^12 ~ 0.032 * base_lr
      - Block 11 (latest): base_lr * 0.75^1 = 0.75 * base_lr
      - Classification head: base_lr * 1.0 (no decay)
      - Patch/pos embed: same as block 0 (lowest LR)

    Parameters
    ----------
    model : nn.Module
        ViT model from timm.
    base_lr : float
        Base learning rate (applied to classification head).
    layer_decay : float
        Multiplicative decay per layer (0.75 per Steiner et al. 2022).
    weight_decay : float
        Weight decay for AdamW.

    Returns
    -------
    param_groups : list of dict
        Parameter groups for optimizer, each with 'params', 'lr', 'weight_decay'.
    """
    # timm's param_groups_layer_decay handles:
    #   - Identifying layer depth for each parameter
    #   - Excluding bias and LayerNorm from weight decay
    #   - Setting lr = base_lr * layer_decay^(num_layers - layer_index)
    param_groups = param_groups_layer_decay(
        model,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
        verbose=False,
    )

    # timm returns groups with 'lr_scale' but not absolute 'lr'.
    # Set absolute LR for each group.
    for group in param_groups:
        scale = group.get("lr_scale", 1.0)
        group["lr"] = base_lr * scale

    return param_groups


def log_lr_groups(param_groups):
    """Log the LR assigned to each parameter group for verification."""
    logger.info("Layer-wise LR groups:")
    for i, group in enumerate(param_groups):
        n_params = sum(p.numel() for p in group["params"])
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)
        logger.info(f"  Group {i:2d}: lr={lr:.2e}, wd={wd:.4f}, params={n_params / 1e6:.2f}M")
    # Summary
    lrs = [g["lr"] for g in param_groups if len(g["params"]) > 0]
    logger.info(f"  LR range: [{min(lrs):.2e}, {max(lrs):.2e}]")
