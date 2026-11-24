"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for RetinaInsight.

Generates visual explanations highlighting retinal regions that positively contribute
to the screening prediction for any specified architecture.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
from PIL import Image

# Set KERAS_HOME before tensorflow
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir() and "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import tensorflow as tf

from configs.training_config import config
from src.data.dataset_loader import get_preprocessing_function
from src.models.transfer_models import TARGET_CONV_LAYERS

LOGGER = logging.getLogger(__name__)


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: tf.Tensor | np.ndarray,
    architecture: str,
    class_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute the Grad-CAM activation heatmap for a given input image.

    Args:
        model: Trained RetinaInsight classification model.
        image_tensor: Batch of 1 image of shape (1, 224, 224, 3) with values in [0, 255].
        architecture: Backbone architecture identifier.
        class_index: Optional target class index. If None, uses top predicted class.

    Returns:
        Tuple of (2D heatmap in [0, 1], predicted class probabilities, target class index).
    """
    arch = architecture.strip().lower()
    target_layer_name = TARGET_CONV_LAYERS.get(arch)
    if not target_layer_name:
        raise ValueError(f"Unknown architecture '{arch}' for Grad-CAM.")

    # Locate backbone model and classification head layers
    backbone = None
    head_layers = []
    found_backbone = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            found_backbone = True
        elif found_backbone:
            head_layers.append(layer)

    if backbone is None:
        raise ValueError(f"Could not locate backbone sub-model in {model.name}")

    target_conv_layer = backbone.get_layer(target_layer_name)

    # Sub-model mapping backbone inputs to [target_conv_activations, backbone_output]
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[target_conv_layer.output, backbone.output],
    )

    # Apply architecture-specific preprocessing
    prep_fn = get_preprocessing_function(arch)
    preprocessed_input = prep_fn(tf.cast(image_tensor, tf.float32))

    with tf.GradientTape() as tape:
        conv_outputs, backbone_out = backbone_grad_model(preprocessed_input)
        tape.watch(conv_outputs)

        # Forward pass through classification head
        x = backbone_out
        for head_layer in head_layers:
            x = head_layer(x)
        predictions = x

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))

        target_score = predictions[:, class_index]

    # Gradient of the target class score with respect to convolutional feature maps
    grads = tape.gradient(target_score, conv_outputs)

    # Global Average Pooling of gradients along spatial dimensions -> importance weights alpha_k
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Rectified Linear Unit (ReLU) to keep features that have positive impact on target class
    cam = tf.maximum(cam, 0.0)

    # Normalize to [0, 1] range
    max_val = tf.math.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val

    heatmap = cam.numpy()
    probabilities = predictions[0].numpy()
    return heatmap, probabilities, class_index


def overlay_heatmap_on_image(
    original_image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Superimpose a 2D Grad-CAM heatmap over the RGB fundus photograph.

    Args:
        original_image: PIL Image or RGB numpy array in range [0, 255].
        heatmap: 2D float array in range [0, 1].
        alpha: Blending weight for heatmap (1-alpha weight for original image).
        colormap_name: Matplotlib colormap name.

    Returns:
        RGB uint8 numpy array with shape (H, W, 3).
    """
    if isinstance(original_image, Image.Image):
        base_img = np.array(original_image.convert("RGB"), dtype=np.float32)
    else:
        base_img = np.array(original_image, dtype=np.float32)

    height, width = base_img.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = np.array(
        heatmap_pil.resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    # Apply colormap
    try:
        import matplotlib
        color_map = matplotlib.colormaps[colormap_name]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt
        color_map = plt.get_cmap(colormap_name)
    colored_cam = color_map(heatmap_resized)[:, :, :3] * 255.0  # Drop alpha channel

    # Blend
    superimposed = (alpha * colored_cam) + ((1.0 - alpha) * base_img)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed


def explain_prediction(
    model: tf.keras.Model,
    image_path: Path | str,
    architecture: str,
    class_index: int | None = None,
    alpha: float = 0.4,
) -> dict[str, Any]:
    """High-level explanation generator for a retinal image file.

    Loads image, produces predicted DR grade, calculates Grad-CAM heatmap,
    and returns superimposed overlay with clinical interpretation context.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    pil_image = Image.open(path).convert("RGB")
    resized_pil = pil_image.resize(config.IMAGE_SIZE)
    img_array = np.expand_dims(np.array(resized_pil, dtype=np.float32), axis=0)

    heatmap, probs, target_class = generate_gradcam_heatmap(
        model=model,
        image_tensor=img_array,
        architecture=architecture,
        class_index=class_index,
    )

    overlay = overlay_heatmap_on_image(resized_pil, heatmap, alpha=alpha)

    predicted_grade = int(np.argmax(probs))
    confidence = float(probs[predicted_grade])
    class_name = config.CLASS_NAMES.get(predicted_grade, "Unknown")

    return {
        "image_path": str(path),
        "architecture": architecture.lower(),
        "predicted_grade": predicted_grade,
        "predicted_name": class_name,
        "confidence": round(confidence, 4),
        "target_class_explained": target_class,
        "probabilities": [round(float(p), 4) for p in probs],
        "heatmap": heatmap,
        "overlay_image": overlay,
        "original_resized": np.array(resized_pil, dtype=np.uint8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanation for a retinal fundus image.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to retinal image file.")
    parser.add_argument("--architecture", type=str, default="mobilenetv2", choices=config.SUPPORTED_ARCHITECTURES)
    parser.add_argument("--model-path", type=str, default=None, help="Optional path to model file.")
    parser.add_argument("--output", type=str, default=None, help="Output path to save explanation image.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_path = args.model_path or (config.SAVED_MODELS_DIR / f"{args.architecture}_final.keras")
    model = tf.keras.models.load_model(model_path)

    explanation = explain_prediction(
        model=model,
        image_path=args.image_path,
        architecture=args.architecture,
    )

    LOGGER.info(
        "Predicted: Grade %d (%s) with %.2f%% confidence",
        explanation["predicted_grade"],
        explanation["predicted_name"],
        explanation["confidence"] * 100,
    )

    output_path = args.output or (
        config.FIGURES_DIR / f"gradcam_{Path(args.image_path).stem}_{args.architecture}.png"
    )
    Image.fromarray(explanation["overlay_image"]).save(output_path)
    LOGGER.info("Saved Grad-CAM overlay to %s", output_path)


if __name__ == "__main__":
    main()
