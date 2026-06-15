"""Runtime capability checks for cloud vs local deployments."""

from __future__ import annotations


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def available_engines() -> list[str]:
    engines = ["Random Forest", "TFLite FP16"]
    if tensorflow_available():
        engines.extend(["Autoencoder", "Ensemble"])
    return engines
