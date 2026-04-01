from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[reportMissingImports]
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore[reportMissingImports]
    from tensorflow.keras.models import Sequential  # type: ignore[reportMissingImports]
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow/Keras is required for forecast training. "
        "Install with: pip install tensorflow"
    ) from exc

FEATURES = ["temp", "humidity", "prcp"]
INPUT_FEATURES = ["temp", "humidity"]
TARGET_FEATURES = ["temp", "humidity", "prcp"]
SEQ_LEN = 7
TRAIN_RATIO = 0.8

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_weather_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Weather dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        raise ValueError("Dataset must contain a 'time' column")

    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required features: {missing_features}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df = df[["time", *FEATURES]].copy()
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if len(df) <= SEQ_LEN:
        raise ValueError(
            f"Not enough valid rows to create sequences. Required > {SEQ_LEN}, found {len(df)}"
        )

    return df


def _create_sequences(values: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] < len(TARGET_FEATURES):
        raise ValueError(
            "values must be a 2D array with at least 3 columns: temp, humidity, prcp"
        )

    x_data, y_data = [], []
    for idx in range(seq_len, len(values)):
        x_data.append(values[idx - seq_len : idx, : len(INPUT_FEATURES)])
        y_data.append(values[idx, : len(TARGET_FEATURES)])

    if not x_data:
        raise ValueError("No sequences were created. Check data size and sequence length.")

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)


def _build_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(len(TARGET_FEATURES)),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_forecast_model(
    csv_path: Path,
    model_path: Path,
    scaler_path: Path,
    sequence_length: int = SEQ_LEN,
    epochs: int = 100,
    batch_size: int = 32,
) -> None:
    logger.info("Loading weather dataset from %s", csv_path)
    try:
        df = _load_weather_dataframe(csv_path)

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df[FEATURES].values)
        if not np.isfinite(scaled_values).all():
            raise ValueError("Scaled weather values contain NaN or infinite numbers")

        x_all, y_all = _create_sequences(scaled_values, sequence_length)
        if x_all.shape[1:] != (sequence_length, len(INPUT_FEATURES)):
            raise ValueError(
                f"Invalid X shape {x_all.shape}; expected (samples, {sequence_length}, {len(INPUT_FEATURES)})"
            )
        if y_all.shape[1] != len(TARGET_FEATURES):
            raise ValueError(
                f"Invalid y shape {y_all.shape}; expected (samples, {len(TARGET_FEATURES)})"
            )
        if not np.isfinite(x_all).all() or not np.isfinite(y_all).all():
            raise ValueError("Training sequences contain NaN or infinite values")

        logger.info("Prepared training tensors: X=%s, y=%s", x_all.shape, y_all.shape)

        split_index = int(len(x_all) * TRAIN_RATIO)
        split_index = max(1, min(split_index, len(x_all) - 1))

        x_train, x_val = x_all[:split_index], x_all[split_index:]
        y_train, y_val = y_all[:split_index], y_all[split_index:]

        model = _build_model((sequence_length, len(INPUT_FEATURES)))
        if model.output_shape[-1] != len(TARGET_FEATURES):
            raise ValueError(
                f"Model output shape {model.output_shape} is incompatible; expected last dim {len(TARGET_FEATURES)}"
            )
        logger.info("Model output shape: %s", model.output_shape)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        logger.info("Starting forecast model training")
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Forecast model training completed")

        train_loss_history = history.history.get("loss", [])
        if len(train_loss_history) >= 2 and train_loss_history[-1] > train_loss_history[0]:
            logger.warning(
                "Training loss did not decrease overall (start=%.6f, end=%.6f)",
                float(train_loss_history[0]),
                float(train_loss_history[-1]),
            )

        eval_loss, eval_mae = model.evaluate(x_val, y_val, verbose=0)
        logger.info("Validation metrics - loss: %.6f, mae: %.6f", eval_loss, eval_mae)
        print(f"Validation loss: {float(eval_loss):.6f}")
        print(f"Validation MAE: {float(eval_mae):.6f}")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving forecast model to %s", model_path)
        model.save(model_path)
        logger.info("Saving scaler to %s", scaler_path)
        joblib.dump(scaler, scaler_path)

        print(f"Forecast model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    except Exception as exc:
        logger.exception("Forecast training pipeline failed")
        raise RuntimeError("Forecast training pipeline failed") from exc


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "data" / "final_weather_datset.csv"
    if not dataset_path.exists():
        fallback_dataset_path = root_dir / "data" / "final_weather_dataset.csv"
        if fallback_dataset_path.exists():
            logger.warning(
                "Primary dataset path not found (%s). Falling back to %s",
                dataset_path,
                fallback_dataset_path,
            )
            dataset_path = fallback_dataset_path
    output_model_path = root_dir / "app" / "models" / "forecast_model.h5"
    output_scaler_path = root_dir / "app" / "models" / "scaler.save"

    train_forecast_model(
        csv_path=dataset_path,
        model_path=output_model_path,
        scaler_path=output_scaler_path,
    )


if __name__ == "__main__":
    main()
