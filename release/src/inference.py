#!/usr/bin/env python3
"""Standalone inference script for Gravity Spy glitch classification.

Classifies LIGO glitch spectrograms using either a ViT-B/16 or ResNet-50v2 BiT
model trained on O3 Gravity Spy data (23 classes).

Usage:
    python inference.py --model vit --image path/to/spectrogram.png
    python inference.py --model cnn --image path/to/spectrogram.png --top-k 5

Requirements: torch, timm, albumentations, numpy, Pillow
No training code imports required.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, input_format=224x224_RGB_PNG_0to1

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import timm

from preprocessing import load_image


def load_model(model_key, config_path=None, checkpoint_dir=None):
    """Load a model from config and checkpoint.

    Parameters
    ----------
    model_key : str
        Either 'vit' or 'cnn'.
    config_path : Path or None
        Path to model_config.json. Defaults to same directory as this script.
    checkpoint_dir : Path or None
        Path to checkpoints directory. Defaults to ../checkpoints/ relative to this script.

    Returns
    -------
    model : torch.nn.Module
        Loaded model in mode for inference.
    class_labels : list of str
        Ordered class label names (index -> label).
    """
    script_dir = Path(__file__).resolve().parent
    if config_path is None:
        config_path = script_dir / "model_config.json"
    if checkpoint_dir is None:
        checkpoint_dir = script_dir.parent / "checkpoints"

    with open(config_path) as f:
        config = json.load(f)

    if model_key not in config["models"]:
        available = list(config["models"].keys())
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {available}")

    model_cfg = config["models"][model_key]

    # Build architecture via timm (no pretrained weights -- we load our own)
    model = timm.create_model(
        model_cfg["model_name"],
        pretrained=False,
        num_classes=model_cfg["num_classes"],
    )

    # Load trained weights
    checkpoint_path = checkpoint_dir / model_cfg["checkpoint_file"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Expected file: {model_cfg['checkpoint_file']}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Handle both full training checkpoints (with 'model_state_dict' key)
    # and bare state_dict files
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    # Set to inference mode
    model.requires_grad_(False)
    model.eval()

    # Load class labels
    labels_path = script_dir / "class_labels.json"
    with open(labels_path) as f:
        class_labels = json.load(f)

    return model, class_labels


def predict(model, image_tensor, class_labels, top_k=3):
    """Run inference on a preprocessed image tensor.

    Parameters
    ----------
    model : torch.nn.Module
        Model in inference mode.
    image_tensor : torch.Tensor
        Preprocessed image of shape (3, 224, 224).
    class_labels : list of str
        Ordered class label names.
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    predictions : list of dict
        Top-k predictions, each with 'rank', 'class', 'probability'.
    """
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))  # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    top_k = min(top_k, len(class_labels))
    top_probs, top_indices = torch.topk(probs, top_k)

    predictions = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        predictions.append({
            "rank": rank,
            "class": class_labels[idx.item()],
            "probability": round(prob.item(), 6),
        })

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Gravity Spy glitch classification inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py --model vit --image spectrogram.png\n"
            "  python inference.py --model cnn --image spectrogram.png --top-k 5\n"
            "  python inference.py --model vit --image spectrogram.png --json\n"
        ),
    )
    parser.add_argument("--model", required=True, choices=["vit", "cnn"],
                        help="Model to use: vit (ViT-B/16) or cnn (ResNet-50v2 BiT)")
    parser.add_argument("--image", required=True, type=str,
                        help="Path to spectrogram image (PNG/JPG)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top predictions to show (default: 3)")
    parser.add_argument("--json", action="store_true",
                        help="Output predictions as JSON")
    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Load model
    model, class_labels = load_model(args.model)

    # Preprocess image
    image_tensor = load_image(str(image_path))

    # Run inference
    predictions = predict(model, image_tensor, class_labels, top_k=args.top_k)

    # Output results
    if args.json:
        result = {
            "model": args.model,
            "image": str(image_path),
            "predictions": predictions,
        }
        print(json.dumps(result, indent=2))
    else:
        model_name = "ViT-B/16" if args.model == "vit" else "ResNet-50v2 BiT"
        print(f"\nModel: {model_name}")
        print(f"Image: {image_path}")
        print(f"\nPredictions:")
        print(f"{'Rank':<6} {'Class':<25} {'Probability':<12}")
        print("-" * 43)
        for pred in predictions:
            print(f"{pred['rank']:<6} {pred['class']:<25} {pred['probability']:.4f}")
        print()


if __name__ == "__main__":
    main()
