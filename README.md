# Industrial Equipment Fault Detection System

**Edge AI–powered predictive maintenance for Industry 4.0**

An end-to-end platform that detects bearing faults from vibration telemetry, runs **on-device inference** at the industrial edge, and turns model outputs into **CMMS-ready maintenance recommendations** — without relying on the cloud.

Built on research from [*Bearing Fault Detection Using Comparative Analysis of Random Forest, ANN, and Autoencoder Methods*](https://link.springer.com/chapter/10.1007/978-981-16-1089-9_14), extended from notebook experiments into a deployable Edge AI stack with training, quantization, inference, and an operations dashboard.

---

## Problem Statement

Unplanned equipment downtime in manufacturing and process industries costs billions annually. Bearing failures on rotating machinery (motors, pumps, compressors, conveyors) are a leading cause — often detected too late, after damage has spread to shafts, housings, and adjacent assets.

**Key challenges:**

| Challenge | Impact |
|-----------|--------|
| **Reactive maintenance** | Failures occur before teams can intervene |
| **Cloud-dependent analytics** | Latency, connectivity gaps, and data-sovereignty limits on factory floors |
| **Alert fatigue** | Raw sensor thresholds generate false positives without ML context |
| **Action gap** | Detecting a fault does not automatically produce a maintenance plan |
| **Resource constraints** | Edge gateways and PLCs cannot run full-size deep learning models |

Plants need a system that detects faults **locally**, **fast**, and **actionably** — turning vibration data into prioritized work orders, not just charts.

---

## Solution

This project delivers a **complete Edge AI fault detection pipeline** for industrial rotating equipment:

1. **Ingest** drive-end (DE) and fan-end (FE) vibration features from sensor telemetry
2. **Train** lightweight models optimized for edge deployment (Random Forest + Autoencoder)
3. **Quantize** neural models to TFLite (FP32 / FP16 / INT8) for constrained hardware
4. **Infer on-device** with sub-millisecond latency — no cloud round-trip required
5. **Recommend maintenance** via a rule-based CMMS module (priority, actions, spare parts, downtime)
6. **Monitor operations** through a Streamlit dashboard with live telemetry simulation

The platform supports five fault scenarios — normal baseline plus inner-race and outer-race defects at early (0.007 in) and critical (0.021 in) severity levels — enabling staged response from monitoring to emergency shutdown.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INDUSTRIAL EDGE LAYER                               │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ Sensors  │───▶│  Telemetry   │───▶│  Edge AI    │───▶│  Maintenance  │  │
│  │ DE / FE  │    │  Buffer      │    │  Inference  │    │  Engine       │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────┬───────┘  │
│                                           ▲                      │          │
│                                           │                      ▼          │
│                                    ┌──────┴──────┐         ┌───────────┐    │
│                                    │ TFLite / RF │         │   CMMS    │    │
│                                    │  Artifacts  │         │ Work Order│    │
│                                    └─────────────┘         └───────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                           ▲
                              OTA deploy after retrain + quantize
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MLOps / DEVELOPMENT LAYER                           │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ Datasets │───▶│   Training   │───▶│ Quantization│───▶│  Dashboard    │  │
│  │ (CSV)    │    │  RF + AE     │    │  TFLite     │    │  (Streamlit)  │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

| Stage | Component | Output |
|-------|-----------|--------|
| **Acquisition** | DE/FE accelerometers | Normalized vibration features |
| **Buffering** | Edge telemetry window | Sliding sample batch for inference |
| **Classification** | Random Forest or Autoencoder | Fault / Normal + confidence score |
| **Quantized inference** | TFLite FP16/INT8 model | Same output, smaller footprint |
| **Action** | Maintenance module | Priority, actions, spare parts, downtime |
| **Integration** | CMMS export | Work order title + technical notes |

### Project Structure

```
├── config/settings.yaml          # Platform & edge AI configuration
├── Datasets/                     # Vibration CSV data (DE, FE)
├── dashboard/app.py                # Streamlit operations dashboard
├── scripts/
│   ├── train.py                    # Train edge-deployable models
│   └── quantize.py                 # TFLite quantization pipeline
├── src/
│   ├── data_loader.py              # Sensor data ingestion & stream simulation
│   ├── train_models.py             # Random Forest + Autoencoder training
│   ├── edge_predictor.py           # On-device inference API
│   ├── edge_quantization.py        # Quantization + latency benchmarking
│   └── maintenance.py              # CMMS recommendation engine
├── models/                         # Trained artifacts (generated at runtime)
└── 1.–9. */                        # Original research notebooks (9 algorithms)
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|------------|------|
| **Language** | Python 3.10+ | Core platform |
| **ML / Edge AI** | scikit-learn, TensorFlow, TFLite | Training, inference, quantization |
| **Data** | pandas, NumPy | Telemetry processing |
| **Dashboard** | Streamlit, Plotly | Operations UI & analytics |
| **Config** | YAML | Fault profiles, training, edge settings |
| **Serialization** | joblib, Keras, TFLite | Model artifact persistence |
| **Interoperability** | ONNX / skl2onnx *(roadmap)* | Cross-platform edge deployment |

### Models

| Model | Type | Edge Role |
|-------|------|-----------|
| **Random Forest** | Supervised classifier | CPU-friendly, deterministic, no GPU required |
| **Autoencoder** | Unsupervised anomaly detector | Reconstruction-error fault detection |
| **TFLite variants** | Quantized autoencoder | FP32 / FP16 / INT8 deployment tiers |

### Dataset

Industrial bearing vibration data (~121,000 samples per file) with two sensor channels:

| Column | Description |
|--------|-------------|
| `DE` | Drive-end accelerometer (normalized) |
| `FE` | Fan-end accelerometer (normalized) |

| File | Fault Type | Severity |
|------|------------|----------|
| `NB.csv` | Normal baseline | — |
| `IR - 7.csv` | Inner race | Early (0.007 in) |
| `IR - 21.csv` | Inner race | Critical (0.021 in) |
| `OR - 7.csv` | Outer race | Early (0.007 in) |
| `OR - 21.csv` | Outer race | Critical (0.021 in) |

---

## Edge AI and Quantization Benefits

### Why Edge AI?

| Benefit | Description |
|---------|-------------|
| **Low latency** | Inference completes on the gateway in sub-milliseconds — fast enough for real-time alerting |
| **Offline operation** | Plants keep detecting faults during network outages or air-gapped environments |
| **Data sovereignty** | Raw vibration data never leaves the factory floor |
| **Cost efficiency** | No per-inference cloud fees; scales with hardware, not API calls |
| **Deterministic behavior** | Random Forest + fixed TFLite graphs produce reproducible edge outputs |

### Why Quantization?

Post-training quantization compresses the trained Keras autoencoder into **TensorFlow Lite** artifacts sized for industrial edge nodes:

| Variant | Format | Target Hardware | Benefit |
|---------|--------|-----------------|---------|
| **float32** | `.tflite` | Edge gateway (reference) | Baseline accuracy for validation |
| **float16** | `.tflite` | GPU/NPU edge gateway | Reduced memory footprint, accelerated math |
| **int8_dynamic** | `.tflite` | ARM PLC / constrained IPC | Smallest model size, fastest inference on CPU |

The quantization pipeline (`src/edge_quantization.py`) exports all three variants, benchmarks **model size (KB)** and **inference latency (ms)**, and writes a comparison report to `models/quantization_*.json` — giving operators a clear deployment tier for each device class.

### Edge AI Terminology

| Term | Meaning |
|------|---------|
| **Edge AI** | ML inference on an industrial gateway, IPC, or PLC companion — not in the cloud |
| **On-device inference** | Fault classification runs locally with offline capability |
| **Telemetry buffer** | Sliding window of vibration samples held in edge memory |
| **Post-training quantization** | Compressing FP32 weights to FP16/INT8 after training |
| **OTA model update** | Over-the-air deployment of new `.tflite` files to the edge fleet |

---

## Quick Start (Demo)

### 1. Environment

```bash
git clone <your-repo-url>
cd fault-detection-for-predictive-maintenance-in-industry-4.0
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Edge Models

```bash
python scripts/train.py --scenario "IR - 7" --sample-size 3000
```

Produces `models/rf_IR_7.joblib` and `models/autoencoder_IR_7.keras`.

### 3. Quantize for Edge Deployment

```bash
python scripts/quantize.py --scenario "IR - 7"
```

Exports TFLite variants and writes `models/quantization_IR_7.json`.

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** — train, quantize, run inference, and view maintenance recommendations from the UI.

### Deploy to Production

See **[DEPLOY.md](DEPLOY.md)** for Docker, Streamlit Cloud, and Render.com instructions.

```bash
docker compose up --build -d   # → http://localhost:8501
```

### Programmatic API

```python
from src.edge_predictor import EdgePredictor
from src.data_loader import load_stream_chunk

predictor = EdgePredictor("IR - 7")
chunk = load_stream_chunk("IR - 7", chunk_size=500)

result = predictor.predict_random_forest(chunk)
print(result.fault_detected, result.confidence, result.maintenance["priority"])

result = predictor.predict_tflite(chunk, variant="fp16")
print(result.reconstruction_error, result.maintenance["cmms_work_order_title"])
```

---

## Maintenance Recommendation Module

When a fault is detected, `src/maintenance.py` generates actionable output for maintenance teams:

- **Priority** — ROUTINE → SCHEDULED → URGENT → CRITICAL
- **Recommended actions** — MONITOR, INSPECT, LUBRICATE, ALIGN, REPLACE_BEARING, SHUTDOWN
- **Spare parts list** — OEM-aligned components
- **Estimated downtime** — Hours for planned intervention
- **CMMS work order title** — Ready for SAP PM, Maximo, or similar systems
- **Technical notes** — Inspection steps, lubrication checks, alignment tolerances

Low-confidence edge alerts (< 75%) trigger a secondary manual inspection recommendation before major intervention.

---

## Future Scope

| Area | Planned Enhancement |
|------|---------------------|
| **Live ingestion** | MQTT / OPC-UA connectors for real sensor streams |
| **Fleet management** | Multi-asset dashboard with edge node health monitoring |
| **Model export** | ONNX export for Random Forest and cross-runtime deployment |
| **Edge orchestration** | Kubernetes / K3s manifests for gateway fleet deployment |
| **Advanced models** | LSTM autoencoders and ensemble voting from research notebooks |
| **Digital twin** | Simulated degradation curves linked to maintenance scheduling |
| **Feedback loop** | Technician-confirmed labels fed back into retraining pipeline |
| **Explainability** | SHAP feature attribution for DE/FE contribution to fault decisions |

---

## Research Foundation

The original comparative study notebooks (9 algorithms across IR/OR fault sizes) are preserved in numbered folders:

1. Random Forest Classification  
2. Artificial Neural Networks  
3. Autoencoder  
4. LSTM + Autoencoder  
5. K-Means  
6. Isolation Forest  
7. One-Class SVM  
8. Gaussian Distribution  
9. Principal Component Analysis  

---

## Team & Citation

**Research basis:**

> Bearing Fault Detection Using Comparative Analysis of Random Forest, ANN, and Autoencoder Methods. Springer, 2021.

---

## License

See repository license file for terms of use.
