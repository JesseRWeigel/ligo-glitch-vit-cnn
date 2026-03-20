"""Standalone preprocessing for Gravity Spy spectrogram inference.

Extracts eval_transforms() from the training pipeline with NO training
dependencies (no wandb, no dataloader, no training-specific imports).

Preprocessing is locked to match training exactly:
  - Resize to 224x224
  - Normalize with ImageNet statistics
  - Convert to PyTorch tensor
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, input_format=224x224_RGB_PNG_0to1

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics for pretrained model normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def eval_transforms(image_size=224):
    """Evaluation transform -- resize + normalize only, no augmentation.

    This is identical to the eval_transforms used during training/validation.
    Input images are expected to be RGB numpy arrays with pixel values in [0, 255].
    Output tensors have pixel values normalized by ImageNet statistics.

    Parameters
    ----------
    image_size : int
        Target spatial dimension (default 224 for ViT-B/16 and ResNet-50v2).

    Returns
    -------
    transform : albumentations.Compose
        Evaluation transform pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_image(image_path, image_size=224):
    """Load an image file and apply evaluation transforms.

    Parameters
    ----------
    image_path : str
        Path to a PNG/JPG spectrogram image.
    image_size : int
        Target spatial dimension (default 224).

    Returns
    -------
    tensor : torch.Tensor
        Preprocessed image tensor of shape (3, image_size, image_size).
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # shape (H, W, 3), dtype uint8, values [0, 255]
    transform = eval_transforms(image_size)
    transformed = transform(image=img_np)
    return transformed["image"]  # torch.Tensor (3, 224, 224)
