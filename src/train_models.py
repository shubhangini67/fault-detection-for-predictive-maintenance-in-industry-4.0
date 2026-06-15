"""Edge-deployable model training: Random Forest classifier and vibration autoencoder."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve_path
from src.data_loader import build_autoencoder_dataset, build_supervised_dataset


def _models_dir(config: dict) -> Path:
    path = resolve_path(config["paths"]["models_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scenario_prefix(fault_key: str) -> str:
    return re.sub(r"[\s\-_]+", "_", fault_key.strip()).strip("_")


def build_autoencoder(input_dim: int, config: dict) -> tf.keras.Model:
    arch = config["training"]["autoencoder"]["architecture"]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(arch[0], activation="relu"),
            tf.keras.layers.Dense(arch[1], activation="relu"),
            tf.keras.layers.Dense(arch[2], activation="relu"),
            tf.keras.layers.Dense(input_dim),
        ],
        name="vibration_autoencoder",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_random_forest(
    fault_key: str,
    config: dict | None = None,
    sample_size: int | None = 5000,
) -> dict:
    """Train lightweight tree ensemble for edge inference (no GPU required)."""
    cfg = config or load_config()
    x, y, scaler = build_supervised_dataset(fault_key, cfg, sample_size)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    rf_cfg = cfg["training"]["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        criterion=rf_cfg["criterion"],
        random_state=cfg["data"]["random_state"],
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    models_dir = _models_dir(cfg)
    prefix = _scenario_prefix(fault_key)
    joblib.dump(clf, models_dir / f"rf_{prefix}.joblib")
    joblib.dump(scaler, models_dir / f"scaler_{prefix}.joblib")

    return {
        "model": "random_forest",
        "scenario": fault_key,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "edge_ready": True,
        "artifact": str(models_dir / f"rf_{prefix}.joblib"),
    }


def train_autoencoder(
    fault_key: str,
    config: dict | None = None,
    sample_size: int | None = 5000,
) -> dict:
    """Train reconstruction-based anomaly detector for edge deployment."""
    cfg = config or load_config()
    x_train, x_test, _y_train, y_test, scaler = build_autoencoder_dataset(
        fault_key, cfg, sample_size
    )

    ae_cfg = cfg["training"]["autoencoder"]
    model = build_autoencoder(x_train.shape[1], cfg)
    history = model.fit(
        x_train,
        x_train,
        epochs=ae_cfg["epochs"],
        batch_size=ae_cfg["batch_size"],
        validation_split=ae_cfg["validation_split"],
        verbose=0,
        shuffle=True,
    )

    recon = model.predict(x_test, verbose=0)
    mae = np.mean(np.abs(recon - x_test), axis=1)
    threshold = ae_cfg["reconstruction_threshold"]
    y_pred = (mae > threshold).astype(int)

    models_dir = _models_dir(cfg)
    prefix = _scenario_prefix(fault_key)
    keras_path = models_dir / f"autoencoder_{prefix}.keras"
    model.save(keras_path)
    joblib.dump(scaler, models_dir / f"ae_scaler_{prefix}.joblib")
    joblib.dump(
        {"threshold": threshold, "mae_mean": float(mae.mean())},
        models_dir / f"ae_meta_{prefix}.joblib",
    )

    return {
        "model": "autoencoder",
        "scenario": fault_key,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "reconstruction_threshold": threshold,
        "final_val_loss": float(history.history["val_loss"][-1]),
        "edge_ready": True,
        "artifact": str(keras_path),
    }


def train_all(fault_key: str, sample_size: int | None = 5000) -> dict:
    """Train all edge models for a fault scenario and write manifest."""
    cfg = load_config()
    print("    Random Forest...", flush=True)
    rf_result = train_random_forest(fault_key, cfg, sample_size)
    print("    Autoencoder...", flush=True)
    ae_result = train_autoencoder(fault_key, cfg, sample_size)
    results = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "scenario": fault_key,
        "random_forest": rf_result,
        "autoencoder": ae_result,
    }

    manifest_path = resolve_path(cfg["paths"]["artifacts_manifest"])
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    manifest[fault_key] = results
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return results
