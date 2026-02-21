# AURA - Federated Learning Security Interrogation Framework

AURA provides:
- A FastAPI backend for FL model interrogation (golden-set eval, SHAP, anomaly detection, ledger logging)
- A React frontend dashboard and Sentinel monitor UI

## Repository Layout

- `aura_backend/` - FastAPI backend
- `aura-frontend/` - React + Vite frontend
- `fl_client/`, `fl_server/`, `sentinel/` - FL/security components
- `data/` - local dataset artifacts

## Quick Start

### 1) Backend

```powershell
cd aura_backend
python -m venv .venv
.venv\Scripts\activate
pip install -r ..\requirements.txt
copy .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Frontend

```powershell
cd aura-frontend
npm install
copy .env.example .env
npm run dev
```

## Environment Files

- Backend example: `aura_backend/.env.example`
- Frontend example: `aura-frontend/.env.example`

## Notes

- `.env`, virtual envs, node modules, DB/runtime outputs, and generated model/XAI artifacts are git-ignored.
- Sentinel monitor flow runs against backend endpoints; dashboard can use demo preview data via `VITE_DASHBOARD_DEMO`.
