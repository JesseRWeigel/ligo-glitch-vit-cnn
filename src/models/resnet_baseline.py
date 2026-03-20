"""ResNet-50 baseline model for Gravity Spy classification.

Uses timm for pretrained BiT (Big Transfer) ResNet-50v2 weights.
Fallback hierarchy: ImageNet-21k+1k -> ImageNet-21k -> ImageNet-1k.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, input_format=224x224_RGB_PNG_0to1

import logging

import timm

logger = logging.getLogger(__name__)

# Model ID priority list (best to worst pretrained weights)
MODEL_CANDIDATES = [
    ("resnetv2_50x1_bit.goog_in21k_ft_in1k", "ImageNet-21k fine-tuned on ImageNet-1k"),
    ("resnetv2_50x1_bit.goog_in21k", "ImageNet-21k only"),
    ("resnet50.a1_in1k", "ImageNet-1k only (fallback)"),
]


def build_resnet50_baseline(num_classes=23, pretrained=True):
    """Build ResNet-50 baseline with best available pretrained weights.

    Parameters
    ----------
    num_classes : int
        Number of output classes (23 for Gravity Spy O3).
    pretrained : bool
        Whether to load pretrained weights.

    Returns
    -------
    model : nn.Module
        ResNet-50 model with classification head replaced.
    model_id : str
        The timm model ID that was successfully loaded.
    pretrain_source : str
        Description of the pretrained weights used.
    """
    for model_id, source in MODEL_CANDIDATES:
        try:
            model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
            logger.info(f"Loaded model: {model_id} ({source})")
            return model, model_id, source
        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")

    raise RuntimeError("Could not load any ResNet-50 variant. Check timm installation and network.")
