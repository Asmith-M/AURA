# AURA - Federated Learning Security Interrogation Framework

AURA provides:
- A FastAPI backend for FL model interrogation (golden-set eval, SHAP, anomaly detection, ledger logging)
- A React frontend dashboard and Sentinel monitor UI

## Project Structure

```
aura/
├── aura_backend/                    # FastAPI backend for FL security interrogation
│   ├── app/                         # Core backend application
│   ├── deploy/                      # Deployment configurations
│   ├── demo_handler.py              # Demo mode handler
│   ├── real_handler.py              # Production handler
│   ├── main.py                      # FastAPI entry point
│   ├── pipeline_runner.py           # XAI/security analysis pipeline
│   ├── session_storage.py           # Session/ledger storage
│   ├── config.py                    # Configuration settings
│   ├── requirements.txt             # Python dependencies
│   ├── ARCHITECTURE.md              # Backend architecture docs
│   ├── QUICKSTART.md                # Backend quick start guide
│   ├── run_demo.sh / run_demo.bat   # Demo execution scripts
│   ├── run_real.sh / run_real.bat   # Production execution scripts
│   ├── detector/                    # Anomaly detection models
│   ├── golden_set/                  # Validation reference data
│   ├── received_models/             # Received FL model artifacts
│   ├── xai_reports/                 # SHAP/XAI analysis outputs
│   └── __pycache__/                 # Python cache
│
├── aura-frontend/                   # React + Vite dashboard
│   ├── src/
│   │   ├── components/              # Reusable UI components
│   │   ├── pages/                   # Page-level components (Dashboard, Monitor)
│   │   ├── assets/                  # Images, icons, styles
│   │   ├── utils/                   # Helper functions
│   │   ├── App.jsx                  # Root component
│   │   └── main.jsx                 # Entry point
│   ├── public/                      # Static assets
│   ├── index.html                   # HTML template
│   ├── vite.config.ts               # Vite bundler configuration
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   ├── postcss.config.js            # PostCSS configuration
│   ├── tsconfig.json                # TypeScript configuration
│   ├── eslint.config.js             # ESLint configuration
│   ├── package.json                 # Node.js dependencies
│   └── README.md                    # Frontend documentation
│
├── attack_simulation/               # Adversarial attack generation & testing
│   ├── attack_types.py              # Attack scenario definitions
│   ├── poison_generators.py         # Poisoning attack generators
│   ├── evaluation_metrics.py        # Attack effectiveness metrics
│   └── test_scenarios.py            # Test case scenarios
│
├── detector/                        # Centralized anomaly detection
│   ├── detector.py                  # Main detector interface
│   ├── isolation_forest.py          # Isolation forest implementation
│   ├── trainer.py                   # Model training utilities
│   ├── training_data/               # Training datasets
│   ├── logs/                        # Detector execution logs
│   └── plots/                       # Visualization outputs
│
├── fl_client/                       # Federated learning clients
│   ├── client_base.py               # Base client implementation
│   ├── client_1.py, client_2.py, client_3.py  # Individual client instances
│   ├── models/                      # Client-side models
│   └── utils/                       # Client utilities
│
├── fl_server/                       # Federated learning server
│   └── (Server orchestration logic)
│
├── sentinel/                        # Security monitoring and alerting
│   └── (Real-time threat detection)
│
├── config/                          # Global configuration
│   ├── settings.py                  # Main settings module
│   ├── blockchain_config.json       # Blockchain configuration
│   └── __pycache__/                 # Python cache
│
├── data/                            # Datasets and data artifacts
│   ├── MNIST/                       # MNIST dataset
│   ├── train_data/                  # Training data
│   ├── test_data/                   # Test data
│   ├── golden_set.npy               # Reference validation set
│   └── golden_labels.npy            # Reference labels
│
├── docs/                            # Project documentation
│   └── architecture.md              # Architecture documentation
│
├── fingerprints/                    # Model fingerprint artifacts
│   ├── fingerprint_*.npy            # Stored fingerprints
│   └── metadata_*.json              # Fingerprint metadata
│
├── fl_logs/                         # Federated learning execution logs
│   └── update_*.json                # Model update records
│
├── scripts/                         # Utility scripts
│   └── (Helper utilities & automation)
│
├── tests/                           # Test suites
│   └── (Unit and integration tests)
│
├── xai_reports/                     # Explainability analysis outputs
│   └── (SHAP reports, visualizations)
│
├── aura_env/                        # Python virtual environment
│   ├── Scripts/                     # Executable scripts
│   ├── Lib/                         # Python packages
│   ├── Include/                     # Header files
│   └── pyvenv.cfg                   # Virtual environment config
│
├── requirements.txt                 # Root dependencies
├── README.md                        # This file
└── __init__.py                      # Package marker
```

## Quick Project Overview

```
aura/
├── aura_backend/          # FastAPI backend server
│   ├── main.py
│   ├── pipeline_runner.py
│   ├── config.py
│   ├── detector/
│   ├── golden_set/
│   └── xai_reports/
│
├── aura-frontend/         # React + Vite frontend
│   ├── src/
│   ├── public/
│   └── package.json
│
├── attack_simulation/     # Adversarial attack testing
├── detector/              # Anomaly detection models
├── fl_client/             # FL client instances
├── fl_server/             # FL server logic
├── sentinel/              # Security monitoring
├── config/                # Configuration files
├── data/                  # Datasets
├── docs/                  # Documentation
├── fingerprints/          # Model fingerprints
├── fl_logs/               # FL execution logs
├── scripts/               # Utility scripts
├── tests/                 # Test suites
└── requirements.txt       # Dependencies
```

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
