# AURA

> Auditable, Unified, Resilient Architecture for federated learning security.

AURA is a full-stack platform for inspecting, scoring, and auditing federated model updates before they are accepted into a training ecosystem. It combines a FastAPI backend, a React dashboard, federated learning simulation modules, SHAP-based explainability, anomaly detection, and ledger-style evidence tracking in one repository.

`FastAPI` `React 19` `TypeScript` `Flower` `PyTorch` `SHAP` `scikit-learn` `SQLite`

---

## Executive Summary

AURA is built around one practical question:

**How do you trust a model update in federated learning when the raw data never leaves the hospital?**

This repository answers that by combining three layers:

| Layer | Responsibility | Key Modules |
| --- | --- | --- |
| Federated Learning | Simulate distributed training across hospital clients | `fl_client/`, `fl_server/`, `scripts/simulate_fl.py` |
| Security Analysis | Evaluate model behavior, generate SHAP fingerprints, detect anomalies | `aura_backend/`, `detector/`, `sentinel/` |
| Audit and Operations | Persist evidence, expose APIs, surface results in a UI | `aura_backend/`, `ledger_stub/`, `aura-frontend/` |

The current primary runtime is the unified backend in `aura_backend/`, which exposes the API consumed by the dashboard and orchestrates the end-to-end analysis pipeline.

---
## SCREENSHOTS
<img width="1919" height="875" alt="image" src="https://github.com/user-attachments/assets/d4e6364f-b44a-4f33-9b66-37f70c09c82d" />
<img width="1919" height="871" alt="image" src="https://github.com/user-attachments/assets/793b6969-2811-46a0-aef3-10db778d1014" />
<img width="1919" height="865" alt="image" src="https://github.com/user-attachments/assets/3d187003-e6c7-40f3-a4d4-72e75845f559" />
<img width="1919" height="866" alt="image" src="https://github.com/user-attachments/assets/5e98fd2c-78af-48cc-8acb-0b24d63e6cf3" />

## What AURA Does

- Accepts federated model submissions from hospital-style participants.
- Evaluates submitted models against a golden validation dataset.
- Produces explainable fingerprints using SHAP-derived feature importance signals.
- Scores behavior with anomaly-detection logic to approve or reject suspicious updates.
- Stores evidence artifacts and ledger records for auditability.
- Exposes analytics, dataset inspection, evidence review, and security telemetry through a frontend dashboard.
- Includes standalone research and legacy microservice modules for Sentinel and ledger simulation.

---

## System Flow

```text
Hospital Client
    |
    v
FL Training / Generated Pipeline
    |
    v
AURA Backend (FastAPI)
    |
    +--> Model intake and hashing
    +--> Golden-set evaluation
    +--> SHAP-based fingerprint extraction
    +--> Isolation Forest anomaly scoring
    +--> Evidence report generation
    +--> SQLite ledger logging
    |
    v
React Dashboard
    |
    +--> Upload and monitoring
    +--> Attack evidence review
    +--> Ledger verification
    +--> Analytics and security views
```

---

## Platform Components

### 1. Unified Backend

`aura_backend/` is the main application entry point and the center of gravity for the project.

It provides:

- Model submission and interrogation APIs
- Golden-set inspection and hospital dataset exploration
- Evidence retrieval by session or transaction
- Pipeline execution and generated-model download endpoints
- Ledger history, stats, and verification
- Analytics, security, and dashboard summary endpoints
- Optional API-key protection with `AURA_API_KEY`

Important note: the backend configuration currently enforces `AURA_MODE=real`. Older documentation that refers to a demo mode is outdated relative to the current code.

### 2. Frontend Dashboard

`aura-frontend/` is a React + TypeScript + Vite application that acts as the operator console.

Implemented pages include:

- Dashboard
- Upload
- Sentinel Monitor
- Dataset Explorer
- Attack Evidence
- Reports
- Ledger
- Analytics
- Security

### 3. Federated Learning Core

The FL runtime is implemented through Flower-based server and client modules:

- `fl_server/server.py` starts the coordinator
- `fl_client/client_1.py`, `client_2.py`, `client_3.py` simulate hospitals
- `fl_client/models/base_model.py` contains baseline models
- `scripts/simulate_fl.py` launches a multi-client simulation flow

### 4. Security and Explainability

Security analysis is spread across:

- `detector/` for Isolation Forest training and inference
- `sentinel/` for the standalone Sentinel API and SHAP utilities
- `aura_backend/real_handler.py` and `pipeline_runner.py` for the integrated backend pipeline

### 5. Ledger and Audit Trail

Auditability is handled in two ways:

- `aura_backend/ledger_manager.py` writes transaction data to SQLite for the unified backend
- `ledger_stub/` provides a standalone blockchain-style ledger simulation service for integration experiments and tests

---

## Repository Map

```text
aura/
|-- aura_backend/        Main FastAPI backend and integrated analysis pipeline
|-- aura-frontend/       React dashboard and operator UI
|-- fl_client/           Flower clients and local training utilities
|-- fl_server/           Flower server and aggregation strategy
|-- detector/            Isolation Forest detector and trainer
|-- sentinel/            Standalone Sentinel API and SHAP tooling
|-- ledger_stub/         Standalone ledger simulation service
|-- attack_simulation/   Poisoning and attack-evaluation utilities
|-- scripts/             Run helpers for simulation, detector training, and services
|-- tests/               Unit and integration tests
|-- data/                Golden-set and training data assets
|-- docs/                Architecture notes
|-- fingerprints/        Fingerprint artifacts
|-- fl_logs/             FL update logs
|-- xai_reports/         Generated explainability outputs
```

---

## Tech Stack

| Area | Stack |
| --- | --- |
| Backend | FastAPI, Uvicorn, Pydantic, NumPy, Pandas |
| ML / FL | PyTorch, Flower |
| Explainability | SHAP |
| Detection | scikit-learn Isolation Forest |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Axios, Recharts, Framer Motion |
| Persistence | SQLite, filesystem artifacts |
| Testing | Pytest |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 20+
- `pip` and `npm`

### 1. Clone and prepare Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure and run the backend

```powershell
Copy-Item aura_backend\.env.example aura_backend\.env
cd aura_backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend defaults worth knowing:

- `AURA_MODE=real` is required by the current backend config
- `AURA_CORS_ORIGINS` defaults to `http://localhost:5173`
- `AURA_API_KEY` is optional; if set, clients must send it as `x-api-key`

### 3. Configure and run the frontend

Open a second terminal:

```powershell
cd aura-frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Frontend environment variables:

- `VITE_API_BASE_URL=http://localhost:8000`
- `VITE_AURA_API_KEY=` for optional backend auth
- `VITE_DASHBOARD_DEMO=false`

### 4. Open the system

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Backend OpenAPI docs: `http://localhost:8000/docs`

---

## Common Workflows

### Run the integrated dashboard + backend

Use this for the main product experience.

```powershell
# terminal 1
cd aura_backend
uvicorn main:app --reload --port 8000

# terminal 2
cd aura-frontend
npm run dev
```

### Run a federated learning simulation

```powershell
python -m scripts.simulate_fl
```

This starts the Flower server and three simulated clients.

### Train or refresh the anomaly detector

```powershell
python -m scripts.train_detector
```

### Run attack simulation experiments

```powershell
python -m scripts.run_attack_simulation --quick
```

or the full workflow:

```powershell
python -m scripts.run_attack_simulation
```

### Start standalone legacy services

These are not required for the unified backend flow, but they remain useful for experiments and tests.

```powershell
python -m scripts.start_sentinel --reload
python -m ledger_stub.main
```

---

## API Highlights

The unified backend exposes a broad surface area. The most important groups are:

| Area | Example Endpoints |
| --- | --- |
| System | `/system/mode`, `/system/health`, `/status` |
| Model Submission | `/sentinel/submit_update`, `/sentinel/report/{submission_id}`, `/sentinel/reports` |
| Dataset and Evidence | `/dataset/inspection`, `/api/dataset/hospital/{hospital_id}`, `/api/evidence/{session_id}` |
| Pipeline Runs | `/sentinel/pipeline/start`, `/sentinel/pipeline/status/{run_id}`, `/sentinel/pipeline/download/{run_id}` |
| Ledger | `/ledger/transactions`, `/ledger/transaction/{tx_id}`, `/ledger/stats`, `/ledger/verify/{tx_id}` |
| Operations | `/analytics/daily`, `/analytics/attacks`, `/security/threats`, `/stats/summary` |

---

## Testing

The repository includes coverage across the main subsystems:

- FL core behavior
- detector integration
- SHAP integration
- backend setup
- ledger integration
- attack simulation
- real-model flows

Run the suite with:

```powershell
pytest tests -q
```

---

## Key Artifacts and Outputs

| Path | Purpose |
| --- | --- |
| `aura_backend/received_models/` | Uploaded and generated model files |
| `aura_backend/xai_reports/` | Backend evidence reports |
| `aura_backend/aura_ledger.db` | Unified backend ledger database |
| `fingerprints/` | Saved behavioral fingerprint vectors and metadata |
| `fl_logs/` | Local FL update logs |
| `attack_simulation/results/` | Attack evaluation reports and plots |
| `xai_reports/` | Top-level explainability artifacts used by other modules |

---

## Notes for Contributors

- The repo contains both the current integrated backend and older standalone services; prefer `aura_backend/` plus `aura-frontend/` for the main product path.
- Some architecture markdown files still describe removed demo-mode behavior; use the source code as the authority.
- The backend expects golden-set and detector artifacts, but parts of the pipeline include fallbacks when assets are missing.
- The project is research-oriented in places, so some modules are exploratory rather than hardened production services.

---

## Why This Repository Stands Out

AURA is not just a federated learning demo and not just a dashboard. It is a stitched-together security workflow:

- distributed training behavior
- model validation
- explainable evidence generation
- anomaly scoring
- audit logging
- operator-facing observability

That combination is the core value of the project.

