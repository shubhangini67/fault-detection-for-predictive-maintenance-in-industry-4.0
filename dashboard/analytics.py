"""Analytics helpers: health scoring, event log, rolling statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.edge_predictor import EdgeInferenceResult, EdgePredictor

SEVERITY_PENALTY = {"none": 0, "early": 18, "critical": 42}


def compute_health_score(
    fault_detected: bool,
    confidence: float,
    severity: str,
    reconstruction_error: float | None = None,
    threshold: float = 0.1,
) -> float:
    score = 100.0
    if fault_detected:
        score -= SEVERITY_PENALTY.get(severity, 20)
        score -= min(30, confidence * 30)
    else:
        score -= (1 - confidence) * 12
    if reconstruction_error is not None:
        score -= min(20, (reconstruction_error / max(threshold, 1e-6)) * 10)
    return float(np.clip(score, 5, 100))


def rolling_rms(series: pd.Series, window: int = 25) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(lambda x: np.sqrt(np.mean(x**2)))


def build_vibration_heatmap(chunk: pd.DataFrame, segments: int = 20) -> pd.DataFrame:
    n = len(chunk)
    seg_size = max(1, n // segments)
    rows = []
    for i in range(segments):
        start = i * seg_size
        end = min((i + 1) * seg_size, n)
        if start >= n:
            break
        seg = chunk.iloc[start:end]
        rows.append(
            {
                "Segment": f"T{i + 1}",
                "DE RMS": rolling_rms(seg["DE"]).iloc[-1],
                "FE RMS": rolling_rms(seg["FE"]).iloc[-1],
                "Fault %": seg["Fault"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def run_model_suite(predictor: EdgePredictor, chunk: pd.DataFrame) -> dict[str, EdgeInferenceResult]:
    results: dict[str, EdgeInferenceResult] = {}
    try:
        results["Random Forest"] = predictor.predict_random_forest(chunk)
    except FileNotFoundError:
        pass
    try:
        results["Autoencoder"] = predictor.predict_autoencoder(chunk)
    except FileNotFoundError:
        pass
    try:
        results["TFLite FP16"] = predictor.predict_tflite(chunk, variant="fp16")
    except FileNotFoundError:
        pass
    return results


def consensus_label(results: dict[str, EdgeInferenceResult]) -> tuple[str, float]:
    if not results:
        return "UNKNOWN", 0.0
    votes = sum(1 for r in results.values() if r.fault_detected)
    total = len(results)
    ratio = votes / total
    label = "FAULT" if ratio > 0.5 else "NORMAL"
    return label, ratio


def simulate_event_log(
    chunk: pd.DataFrame,
    predictor: EdgePredictor,
    windows: int = 8,
) -> pd.DataFrame:
    """Scan buffer in windows to produce an alert timeline."""
    n = len(chunk)
    win = max(20, n // windows)
    events = []
    for i in range(windows):
        start = i * win
        end = min(start + win, n)
        if start >= n:
            break
        window = chunk.iloc[start:end]
        try:
            r = predictor.predict_random_forest(window)
            events.append(
                {
                    "Window": f"W{i + 1:02d}",
                    "Offset": start,
                    "Status": "ALERT" if r.fault_detected else "OK",
                    "Confidence": round(r.confidence * 100, 1),
                    "Priority": r.maintenance["priority"],
                }
            )
        except FileNotFoundError:
            break
    return pd.DataFrame(events)


def edge_kpis(results: dict[str, EdgeInferenceResult], quant_report: dict | None) -> dict:
    latencies = []
    sizes = []
    if quant_report:
        for v in quant_report.get("variants", {}).values():
            if "error" not in v:
                latencies.append(v.get("latency_ms", 0))
                sizes.append(v.get("size_kb", 0))
    primary = next(iter(results.values()), None)
    return {
        "models_deployed": len(results),
        "edge_latency_ms": min(latencies) if latencies else None,
        "model_size_kb": min(sizes) if sizes else None,
        "inference_confidence": primary.confidence if primary else 0,
        "consensus_fault_rate": sum(1 for r in results.values() if r.fault_detected) / max(len(results), 1),
    }
