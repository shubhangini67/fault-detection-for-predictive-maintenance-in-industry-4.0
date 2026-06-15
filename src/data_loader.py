"""Dataset loading and preprocessing for vibration sensor streams."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import load_config, resolve_path


def list_fault_scenarios(config: dict | None = None) -> list[str]:
    cfg = config or load_config()
    return [k for k in cfg["fault_profiles"] if k != "NB"]


@lru_cache(maxsize=16)
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_scenario(
    fault_key: str,
    config: dict | None = None,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normal baseline and fault CSV for a given scenario."""
    cfg = config or load_config()
    datasets_dir = resolve_path(cfg["paths"]["datasets_dir"])
    features = cfg["data"]["features"]

    normal = _load_csv(str(datasets_dir / "NB.csv")).copy()
    normal["Fault"] = cfg["fault_profiles"]["NB"]["label"]

    fault_path = datasets_dir / f"{fault_key}.csv"
    if not fault_path.exists():
        raise FileNotFoundError(f"Fault dataset not found: {fault_path}")

    faulty = _load_csv(str(fault_path)).copy()
    faulty["Fault"] = cfg["fault_profiles"][fault_key]["label"]

    if sample_size:
        normal = normal.sample(n=min(sample_size, len(normal)), random_state=0)
        faulty = faulty.sample(n=min(sample_size, len(faulty)), random_state=0)

    return normal[features + ["Fault"]], faulty[features + ["Fault"]]


def build_supervised_dataset(
    fault_key: str,
    config: dict | None = None,
    sample_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Merge normal + fault data for supervised edge classifier training."""
    cfg = config or load_config()
    normal, faulty = load_scenario(fault_key, cfg, sample_size)
    dataset = pd.concat([normal, faulty], ignore_index=True)

    x = dataset[cfg["data"]["features"]].values
    y = dataset["Fault"].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y, scaler


def build_autoencoder_dataset(
    fault_key: str,
    config: dict | None = None,
    sample_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare train/test tensors for on-edge anomaly autoencoder."""
    cfg = config or load_config()
    test_size = cfg["data"]["test_size"]
    random_state = cfg["data"]["random_state"]
    features = cfg["data"]["features"]

    normal, faulty = load_scenario(fault_key, cfg, sample_size)

    x_train_n, x_test_n = train_test_split(normal, test_size=test_size, random_state=random_state)
    x_train_f, x_test_f = train_test_split(faulty, test_size=test_size, random_state=random_state)

    train = pd.concat([x_train_n, x_train_f], ignore_index=True)
    test = pd.concat([x_test_n, x_test_f], ignore_index=True)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[features])
    test_scaled = scaler.transform(test[features])

    y_train = train["Fault"].values
    y_test = test["Fault"].values
    return train_scaled, test_scaled, y_train, y_test, scaler


def load_stream_chunk(
    fault_key: str,
    offset: int = 0,
    chunk_size: int = 500,
    config: dict | None = None,
) -> pd.DataFrame:
    """Simulate an edge telemetry buffer (sliding window over sensor data)."""
    cfg = config or load_config()
    pool_size = cfg.get("dashboard", {}).get("stream_pool_size", 10000)
    normal, faulty = load_scenario(fault_key, cfg, sample_size=pool_size)
    combined = pd.concat([normal, faulty], ignore_index=True)
    end = min(offset + chunk_size, len(combined))
    return combined.iloc[offset:end].reset_index(drop=True)
