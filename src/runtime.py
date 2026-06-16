"""Runtime capability checks for cloud vs local deployments."""

from __future__ import annotations


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def tflite_available() -> bool:
    try:
        from tflite_runtime.interpreter import Interpreter  # noqa: F401

        return True
    except ImportError:
        return tensorflow_available()


def available_engines() -> list[str]:
    engines = ["Random Forest"]
    if tflite_available():
        engines.append("TFLite FP16")
    if tensorflow_available():
        engines.extend(["Autoencoder", "Ensemble"])
    return engines
