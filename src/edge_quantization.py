"""Edge AI model quantization for on-device industrial inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import load_config, resolve_path
from src.data_loader import build_autoencoder_dataset
from src.scenario_utils import scenario_prefix as _scenario_prefix


def _file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024 if path.exists() else 0.0


def _benchmark_tflite(interpreter: tf.lite.Interpreter, input_shape: tuple, runs: int = 100) -> float:
    """Return mean inference latency in milliseconds."""
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    sample = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(10):
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()

    start = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000 / runs
    return float(elapsed_ms)


def quantize_autoencoder(
    fault_key: str,
    config: dict | None = None,
    sample_size: int = 5000,
) -> dict:
    """
    Export TensorFlow Lite models for edge deployment.

    Produces three variants aligned with industrial Edge AI practice:
    - float32 baseline (reference accuracy)
    - float16 (GPU/NPU-friendly, reduced memory footprint)
    - int8 dynamic range (MCU/ARM Cortex-A class gateways)
    """
    cfg = config or load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    prefix = _scenario_prefix(fault_key)
    keras_path = models_dir / f"autoencoder_{prefix}.keras"

    if not keras_path.exists():
        raise FileNotFoundError(
            f"Train models first: missing {keras_path}. Run scripts/train.py"
        )

    x_train, _, _, _, _ = build_autoencoder_dataset(fault_key, cfg, sample_size)
    model = tf.keras.models.load_model(keras_path)
    input_shape = (1, x_train.shape[1])

    results: dict = {"scenario": fault_key, "variants": {}}

    # Float32 TFLite (default conversion)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = models_dir / f"autoencoder_{prefix}_fp32.tflite"
    fp32_path.write_bytes(tflite_fp32)

    interp = tf.lite.Interpreter(model_content=tflite_fp32)
    interp.allocate_tensors()
    results["variants"]["float32"] = {
        "path": str(fp32_path),
        "size_kb": round(_file_size_kb(fp32_path), 2),
        "latency_ms": round(_benchmark_tflite(interp, input_shape), 4),
        "quantization": "none",
        "edge_use_case": "Reference deployment / validation on edge gateway",
    }

    # Float16 quantization (Edge TPU / NPU compatible path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    fp16_path = models_dir / f"autoencoder_{prefix}_fp16.tflite"
    fp16_path.write_bytes(tflite_fp16)

    interp = tf.lite.Interpreter(model_content=tflite_fp16)
    interp.allocate_tensors()
    results["variants"]["float16"] = {
        "path": str(fp16_path),
        "size_kb": round(_file_size_kb(fp16_path), 2),
        "latency_ms": round(_benchmark_tflite(interp, input_shape), 4),
        "quantization": "float16",
        "edge_use_case": "Industrial edge gateway with GPU/NPU acceleration",
    }

    # INT8 dynamic range quantization (smallest footprint)
    def representative_dataset():
        indices = np.random.choice(len(x_train), min(cfg["edge_ai"]["quantization"]["representative_samples"], len(x_train)), replace=False)
        for idx in indices:
            yield [x_train[idx : idx + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    try:
        tflite_int8 = converter.convert()
        int8_path = models_dir / f"autoencoder_{prefix}_int8.tflite"
        int8_path.write_bytes(tflite_int8)

        interp = tf.lite.Interpreter(model_content=tflite_int8)
        interp.allocate_tensors()
        results["variants"]["int8_dynamic"] = {
            "path": str(int8_path),
            "size_kb": round(_file_size_kb(int8_path), 2),
            "latency_ms": round(_benchmark_tflite(interp, input_shape), 4),
            "quantization": "int8_dynamic_range",
            "edge_use_case": "Resource-constrained PLC / ARM edge node",
        }
    except Exception as exc:
        results["variants"]["int8_dynamic"] = {
            "error": str(exc),
            "edge_use_case": "INT8 conversion skipped — use float16 on this platform",
        }

    if results["variants"].get("float32") and results["variants"].get("float16"):
        fp32_size = results["variants"]["float32"]["size_kb"]
        fp16_size = results["variants"]["float16"]["size_kb"]
        if fp32_size > 0:
            results["compression_ratio_fp16"] = round(fp32_size / max(fp16_size, 0.01), 2)

    manifest_path = models_dir / f"quantization_{prefix}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
