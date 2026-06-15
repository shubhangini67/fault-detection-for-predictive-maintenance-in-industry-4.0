"""On-device inference engine for industrial edge deployment."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.maintenance import recommend_maintenance, recommendation_to_dict
from src.scenario_utils import scenario_prefix as _scenario_prefix


def _tensorflow():
    import tensorflow as tf

    return tf


@dataclass
class EdgeInferenceResult:
    fault_detected: bool
    confidence: float
    model_type: str
    reconstruction_error: float | None
    maintenance: dict
    edge_metadata: dict


class EdgePredictor:
    """Unified edge inference facade for industrial fault detection."""

    def __init__(self, scenario_key: str, config: dict | None = None):
        self.config = config or load_config()
        self.scenario_key = scenario_key
        self.prefix = _scenario_prefix(scenario_key)
        self.models_dir = resolve_path(self.config["paths"]["models_dir"])
        self.features = self.config["data"]["features"]

    def _load_rf(self):
        clf_path = self.models_dir / f"rf_{self.prefix}.joblib"
        scaler_path = self.models_dir / f"scaler_{self.prefix}.joblib"
        if not clf_path.exists():
            raise FileNotFoundError(f"Random Forest not trained for {self.scenario_key}")
        return joblib.load(clf_path), joblib.load(scaler_path)

    def _load_autoencoder_bundle(self):
        keras_path = self.models_dir / f"autoencoder_{self.prefix}.keras"
        scaler_path = self.models_dir / f"ae_scaler_{self.prefix}.joblib"
        meta_path = self.models_dir / f"ae_meta_{self.prefix}.joblib"
        if not keras_path.exists():
            raise FileNotFoundError(f"Autoencoder not trained for {self.scenario_key}")
        model = _tensorflow().keras.models.load_model(keras_path)
        scaler = joblib.load(scaler_path)
        meta = joblib.load(meta_path)
        return model, scaler, meta

    def predict_random_forest(self, df: pd.DataFrame) -> EdgeInferenceResult:
        clf, scaler = self._load_rf()
        x = scaler.transform(df[self.features].values)
        proba = clf.predict_proba(x)
        labels = clf.predict(x)

        fault_rate = float(labels.mean())
        fault_detected = fault_rate > 0.5
        confidence = float(proba[:, 1].mean()) if fault_detected else float(proba[:, 0].mean())

        maintenance = recommend_maintenance(
            self.scenario_key, fault_detected, confidence, model_name="edge_random_forest"
        )

        return EdgeInferenceResult(
            fault_detected=fault_detected,
            confidence=confidence,
            model_type="random_forest",
            reconstruction_error=None,
            maintenance=recommendation_to_dict(maintenance),
            edge_metadata={
                "inference_location": "edge_gateway",
                "samples_analyzed": len(df),
                "fault_sample_rate": fault_rate,
            },
        )

    def predict_autoencoder(self, df: pd.DataFrame) -> EdgeInferenceResult:
        model, scaler, meta = self._load_autoencoder_bundle()
        x = scaler.transform(df[self.features].values)
        recon = model.predict(x, verbose=0)
        mae = np.mean(np.abs(recon - x), axis=1)
        threshold = meta["threshold"]

        fault_flags = mae > threshold
        fault_rate = float(fault_flags.mean())
        fault_detected = fault_rate > 0.5
        mean_mae = float(mae.mean())
        confidence = (
            min(1.0, mean_mae / (threshold + 1e-6))
            if fault_detected
            else 1.0 - min(1.0, mean_mae / threshold)
        )

        maintenance = recommend_maintenance(
            self.scenario_key, fault_detected, confidence, model_name="edge_autoencoder"
        )

        return EdgeInferenceResult(
            fault_detected=fault_detected,
            confidence=confidence,
            model_type="autoencoder",
            reconstruction_error=mean_mae,
            maintenance=recommendation_to_dict(maintenance),
            edge_metadata={
                "inference_location": "edge_gateway",
                "reconstruction_threshold": threshold,
                "samples_analyzed": len(df),
                "anomaly_sample_rate": fault_rate,
            },
        )

    def predict_tflite(self, df: pd.DataFrame, variant: str = "fp16") -> EdgeInferenceResult:
        tflite_path = self.models_dir / f"autoencoder_{self.prefix}_{variant}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(
                f"TFLite model not found: {tflite_path}. Run scripts/quantize.py"
            )

        _, scaler, meta = self._load_autoencoder_bundle()
        x = scaler.transform(df[self.features].values).astype(np.float32)

        interpreter = _tensorflow().lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        mae_values = []
        for row in x:
            interpreter.set_tensor(input_index, row.reshape(1, -1))
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            mae_values.append(float(np.mean(np.abs(output - row.reshape(1, -1)))))

        mae_arr = np.array(mae_values)
        threshold = meta["threshold"]
        fault_detected = float((mae_arr > threshold).mean()) > 0.5
        mean_mae = float(mae_arr.mean())
        confidence = (
            min(1.0, mean_mae / (threshold + 1e-6))
            if fault_detected
            else 1.0 - min(1.0, mean_mae / threshold)
        )

        maintenance = recommend_maintenance(
            self.scenario_key, fault_detected, confidence, model_name=f"edge_tflite_{variant}"
        )

        return EdgeInferenceResult(
            fault_detected=fault_detected,
            confidence=confidence,
            model_type=f"tflite_{variant}",
            reconstruction_error=mean_mae,
            maintenance=recommendation_to_dict(maintenance),
            edge_metadata={
                "inference_location": "edge_node",
                "quantized_model": str(tflite_path.name),
                "reconstruction_threshold": threshold,
                "samples_analyzed": len(df),
            },
        )
