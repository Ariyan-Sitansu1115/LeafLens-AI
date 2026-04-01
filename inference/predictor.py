"""Inference helper for single-image classification.

Provides a reusable `Predictor` class that loads a PyTorch EfficientNet model
and performs single-image inference using the validation transform from
`preprocessing.image_transforms`.

Key features:
- Load model from a checkpoint path (supports full model or state_dict).
- Load labels from a JSON file (list or dict form supported).
- Accept image file path or raw image bytes for prediction.
- Returns predicted class name and softmax confidence score.

Designed for production use: clear errors, device handling, and small API.
"""
from __future__ import annotations

import io
import importlib
import json
import logging
from typing import Any, Dict, Optional, Union

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.image_transforms import preprocess_image, get_val_transform

# Try importing the project's model factory. If not available, fall back to torchvision.
try:
    from models.efficientnet_model import create_model
except Exception:
    create_model = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Predictor:
    """Simple predictor for single-image classification.

    Example:
        p = Predictor(model_path="models/rice_model.pt", labels_path="config/labels.json")
        label, conf = p.predict_from_file("/path/to/image.jpg")
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        device: Optional[Union[str, torch.device]] = None,
        model_variant: str = "b0",
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device in (None, "cuda") else (device or "cpu"))
        self.model_variant = model_variant

        self.labels = self._load_labels(self.labels_path)
        self.num_classes = len(self.labels)

        self.model = self._load_model(self.model_path)
        self.model.eval()

    def _load_labels(self, path: str) -> list:
        """Load labels from JSON file.
        
        Supports multiple formats:
        - List of class names: ["class1", "class2", ...]
        - Dict mapping indices to names: {"0": "class1", "1": "class2", ...}
        - Dict mapping string keys to names: {"class1_key": "class1", ...}
        
        Args:
            path: Path to labels JSON file
            
        Returns:
            Sorted list of class labels
            
        Raises:
            FileNotFoundError: If labels file not found
            json.JSONDecodeError: If JSON is invalid
            TypeError: If content is not a list or dict
            ValueError: If labels list is empty
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.exception("Labels file not found: %s", path)
            raise
        except json.JSONDecodeError:
            logger.exception("Invalid JSON in labels file: %s", path)
            raise

        # Support either a list of names or a dict mapping indices/keys to names
        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict):
            # Sort by key if keys look like indices, otherwise use insertion order
            try:
                # keys may be strings of ints (like "0", "1", etc.)
                sorted_items = sorted(data.items(), key=lambda kv: int(kv[0]))
                labels = [v for _, v in sorted_items]
            except (ValueError, TypeError):
                # keys are not numeric strings, use insertion order
                labels = list(data.values())
        else:
            raise TypeError("labels.json must contain a list or dict of class names")

        if not labels:
            raise ValueError("No labels found in labels file")

        logger.info(f"Loaded {len(labels)} class labels from {path}")
        return labels

    def _load_model(self, path: str) -> nn.Module:
        """Load model from checkpoint with robust format detection.
        
        Supports multiple checkpoint formats:
        1. Full nn.Module instance saved directly
        2. Checkpoint dict with 'model_state_dict' key
        3. Checkpoint dict with 'state_dict' key  
        4. Raw state_dict dict
        
        Dynamically rebuilds EfficientNet-B0 based on num_classes and loads
        weights from state_dict.
        
        Args:
            path: Path to model checkpoint (.pt or .pth file)
            
        Returns:
            PyTorch model on the configured device in eval mode
            
        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model loading or instantiation fails
        """
        # Load checkpoint (be permissive — support full model, state_dict, or wrapped dict)
        try:
            checkpoint = torch.load(path, map_location=self.device)
            logger.info(f"Loaded checkpoint from {path} (device: {self.device})")
        except FileNotFoundError:
            logger.exception("Model file not found: %s", path)
            raise
        except Exception:
            logger.exception("Failed to load model checkpoint: %s", path)
            raise

        # Case 1: checkpoint is already an nn.Module (full model saved directly)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint.to(self.device)
            logger.info(f"Loaded model as nn.Module instance")
            return model

        # Case 2 & 3: checkpoint is a dict with state_dict or model_state_dict keys
        # or raw state_dict
        state_dict = None
        idx_to_class = None
        
        if isinstance(checkpoint, dict):
            # Try common keys for state_dict
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    logger.info(f"Found state_dict under key '{key}'")
                    break
            
            # Try to extract class mapping if available
            for class_key in ("idx_to_class", "class_names", "classes"):
                if class_key in checkpoint:
                    idx_to_class = checkpoint[class_key]
                    logger.info(f"Found class mapping under key '{class_key}'")
                    break
            
            # If no state_dict found, assume the entire dict is the state_dict
            if state_dict is None:
                state_dict = checkpoint
                logger.info("Treating entire checkpoint dict as state_dict")
        else:
            # Checkpoint is not a dict, treat it as state_dict
            state_dict = checkpoint

        # If we still don't have a state_dict, fail
        if state_dict is None:
            raise RuntimeError("Could not extract state_dict from checkpoint")

        # Use the project's create_model factory to construct model with correct num_classes
        if not create_model:
            raise RuntimeError(
                "Model factory (create_model) not available. "
                "Cannot construct model architecture from state_dict. "
                "Ensure models.efficientnet_model module is importable."
            )

        try:
            model = create_model(
                num_classes=self.num_classes,
                variant=self.model_variant,
                pretrained=False,
                device=str(self.device)
            )
            logger.info(f"Created EfficientNet-{self.model_variant.upper()} model with {self.num_classes} classes")
        except Exception:
            logger.exception("Failed to create model architecture")
            raise RuntimeError("Cannot instantiate model architecture")

        # Attempt to load state_dict into the model
        try:
            # Remove potential 'module.' prefixes from DataParallel models
            if isinstance(state_dict, dict):
                new_state = {}
                for k, v in state_dict.items():
                    new_key = k
                    if k.startswith("module."):
                        new_key = k[len("module."):]
                    new_state[new_key] = v
                model.load_state_dict(new_state, strict=False)
                logger.info("Loaded state_dict with prefix removal (DataParallel compatibility)")
            else:
                model.load_state_dict(state_dict)  # type: ignore
                logger.info("Loaded state_dict")
        except Exception:
            logger.exception("Failed to load state_dict into model, attempting non-strict load")
            try:
                model.load_state_dict(state_dict, strict=False)  # type: ignore
                logger.info("Loaded state_dict (non-strict mode)")
            except Exception:
                logger.exception("Non-strict state_dict load also failed")
                raise RuntimeError("Failed to load model weights from state_dict")

        return model.to(self.device)

    def _prepare_tensor(self, image: Union[str, bytes, Image.Image]) -> torch.Tensor:
        # Accept bytes, file path, or PIL image. Use preprocess_image helper which returns
        # a batched tensor on the requested device.
        if isinstance(image, bytes):
            try:
                img = Image.open(io.BytesIO(image)).convert("RGB")
            except Exception:
                logger.exception("Invalid image bytes provided")
                raise
            return preprocess_image(img, input_size=224, device=self.device)

        if isinstance(image, Image.Image):
            return preprocess_image(image, input_size=224, device=self.device)

        if isinstance(image, str):
            # Let preprocess_image open the path
            return preprocess_image(image, input_size=224, device=self.device)

        raise TypeError("image must be a file path (str), bytes, or PIL.Image.Image")

    def predict(self, image: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Run inference for a single image and return a JSON-serializable dict.

        Args:
            image: file path, raw image bytes, or PIL.Image

        Returns:
            dict: {"label": str, "confidence": float}
        """
        tensor = self._prepare_tensor(image)

        with torch.no_grad():
            outputs = self.model(tensor)

            # Some models return (logits,) or (logits, aux). Handle both.
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs

            if not isinstance(logits, torch.Tensor):
                raise TypeError("Model output is not a tensor")

            probs = F.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

            idx = int(top_idx.item())
            prob = float(top_prob.item())

            try:
                label = self.labels[idx]
            except Exception:
                logger.exception("Predicted index %s out of range for labels (len=%d)", idx, len(self.labels))
                raise IndexError("Predicted index out of range for labels")

            return {"label": label, "confidence": prob}

    # Convenience wrappers
    def predict_from_file(self, image_path: str) -> Dict[str, Any]:
        return self.predict(image_path)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        return self.predict(image_bytes)


# ===== WEATHER FORECAST MODULE =====
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


# ===== MODEL LOADING =====
def _get_forecast_artifact_paths() -> tuple[Path, Path]:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_path = (project_root / "app" / "models" / "forecast_model.h5").resolve()
    scaler_path = (project_root / "app" / "models" / "scaler.save").resolve()
    return model_path, scaler_path


@lru_cache(maxsize=1)
def _load_forecast_artifacts():
    logger.info("Loading weather forecast artifacts")
    load_model = None
    import_errors = []

    for module_name in ("tensorflow.keras.models", "keras.models"):
        try:
            module = importlib.import_module(module_name)
            load_model = getattr(module, "load_model")
            break
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")

    if load_model is None:
        raise ImportError(
            "TensorFlow/Keras is required for weather forecasting inference. "
            "Install with: pip install tensorflow. "
            f"Import attempts failed -> {' | '.join(import_errors)}"
        )

    model_path, scaler_path = _get_forecast_artifact_paths()

    if not model_path.exists():
        logger.error("Forecast model missing at %s", model_path)
        raise FileNotFoundError(f"Forecast model not found at: {model_path}")
    if not scaler_path.exists():
        logger.error("Forecast scaler missing at %s", scaler_path)
        raise FileNotFoundError(f"Forecast scaler not found at: {scaler_path}")

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("Weather forecast artifacts loaded successfully")
        return model, scaler
    except Exception as exc:
        logger.exception("Failed to load weather forecast artifacts")
        raise RuntimeError("Unable to load forecast model/scaler artifacts") from exc


def build_sequence(today_data, history_data):
    """Build a valid (7, 2) input sequence from history and today's sensor data.

    Args:
        today_data: Current day weather features [temp, humidity].
        history_data: Historical weather rows with 2 features each.

    Returns:
        np.ndarray: Sequence with shape (7, 2) as [last_6_history_days + today].
    """
    try:
        history = np.asarray(history_data)
        today = np.asarray(today_data)
    except Exception as exc:
        raise TypeError("today_data and history_data must be array-like") from exc

    if history.ndim != 2 or history.shape[1] != 2:
        raise ValueError(
            f"history_data must be 2D with shape (N, 2); got {history.shape}"
        )
    if history.shape[0] < 6:
        raise ValueError(
            f"history_data must contain at least 6 rows; got {history.shape[0]}"
        )

    if today.shape not in {(2,), (1, 2)}:
        raise ValueError(
            f"today_data must have shape (2,) or (1, 2); got {today.shape}"
        )

    try:
        history = history.astype(np.float32)
        today = today.astype(np.float32).reshape(1, 2)
    except (TypeError, ValueError) as exc:
        raise ValueError("today_data and history_data must contain numeric values") from exc

    sequence = np.vstack([history[-6:], today])

    if sequence.shape != (7, 2):
        raise ValueError(f"Failed to build sequence with shape (7, 2); got {sequence.shape}")

    return sequence


def _validate_weather_input_sequence(input_sequence):
    if not isinstance(input_sequence, (list, np.ndarray)):
        raise TypeError(
            "input_sequence must be a list or numpy.ndarray with shape (7, 2)"
        )

    sequence = np.asarray(input_sequence)
    if sequence.shape != (7, 2):
        raise ValueError(
            f"input_sequence must have shape (7, 2), but got {sequence.shape}"
        )

    try:
        sequence = sequence.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_sequence must contain only numeric values") from exc

    if not np.isfinite(sequence).all():
        raise ValueError("input_sequence must contain only finite numeric values")

    return sequence


# ===== PREDICTION LOGIC =====
def predict_weather_3_days(input_sequence):
    logger.info("Weather prediction started")
    try:
        model, scaler = _load_forecast_artifacts()
        sequence = _validate_weather_input_sequence(input_sequence)

        dummy_prcp = np.zeros((sequence.shape[0], 1), dtype=np.float32)
        input_full = np.hstack([sequence, dummy_prcp])

        normalized_sequence = scaler.transform(input_full)
        rolling_window = normalized_sequence.copy()
        normalized_predictions = []

        for _ in range(3):
            model_input = rolling_window[:, :2]
            next_day_norm = model.predict(model_input[np.newaxis, :, :], verbose=0)[0]
            next_day_norm = np.asarray(next_day_norm, dtype=np.float32).reshape(-1)

            if next_day_norm.shape != (3,):
                raise ValueError(
                    f"Forecast model output must have shape (3,), got {next_day_norm.shape}"
                )

            normalized_predictions.append(next_day_norm)
            rolling_window = np.vstack([rolling_window[1:], next_day_norm])

        predictions = scaler.inverse_transform(np.asarray(normalized_predictions, dtype=np.float32))

        results = []
        for day_idx, values in enumerate(predictions, start=1):
            results.append(
                {
                    "day": day_idx,
                    "temp": float(values[0]),
                    "humidity": float(values[1]),
                    "prcp": float(values[2]),
                }
            )

        logger.info("Weather prediction completed successfully")
        return results
    except Exception as exc:
        logger.exception("Weather prediction failed")
        return {
            "forecast": [],
            "message": "Prediction temporarily unavailable",
        }


if __name__ == "__main__":
    # Quick CLI for manual testing (not a replacement for production usage)
    import argparse

    parser = argparse.ArgumentParser(description="Run single-image inference with a trained EfficientNet model")
    parser.add_argument("image", help="Path to image file to classify")
    parser.add_argument("--model", default="models/rice_model.pt", help="Path to model checkpoint")
    parser.add_argument("--labels", default="config/labels.json", help="Path to labels JSON file")
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    p = Predictor(model_path=args.model, labels_path=args.labels, device=args.device)
    result = p.predict_from_file(args.image)
    print(json.dumps(result))
