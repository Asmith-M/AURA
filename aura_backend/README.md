# AURA Backend - Federated Learning Security System

FastAPI backend with dual operating modes for demonstrating and executing federated learning security analysis.

## Architecture

```
aura_backend/
├── main.py                  # FastAPI application
├── config.py                # Configuration and mode management
├── demo_handler.py          # Demo mode logic
├── real_handler.py          # Real mode pipeline
├── ledger_manager.py        # SQLite ledger implementation
├── session_storage.py       # In-memory session storage
├── demo_data/               # Pre-defined demo responses
│   ├── demo_approved.json
│   ├── demo_rejected.json
│   └── demo_anomaly.json
├── received_models/         # Uploaded models (real mode)
├── xai_reports/             # SHAP analysis reports
├── detector/                # Isolation Forest detector
│   └── model.pkl
├── golden_set/              # Validation dataset
│   ├── golden_test.npy
│   └── golden_labels.npy
└── aura_ledger.db          # SQLite ledger database
```

## Operating Modes

### Demo Mode (default)
- Returns pre-defined realistic mock data
- No backend processing
- Perfect for UI screenshots and demonstrations
- Instant responses

**Activate:**
```bash
export AURA_MODE=demo  # Linux/Mac
$env:AURA_MODE="demo"  # Windows PowerShell
```

### Real Mode
- Executes full ML pipeline:
  1. Model saving and hashing
  2. Architecture analysis
  3. Golden set evaluation
  4. SHAP fingerprinting
  5. Anomaly detection
  6. Ledger logging
- Requires detector model and golden set
- Actual processing time

**Activate:**
```bash
export AURA_MODE=real  # Linux/Mac
$env:AURA_MODE="real"  # Windows PowerShell
```

## API Endpoints

### System Endpoints
- `GET /system/mode` - Get current operating mode
- `GET /system/health` - Health check with component status

### Sentinel Endpoints
- `POST /sentinel/submit_update` - Submit model for analysis
  - **Demo mode:** Form data with `hospital_id`
  - **Real mode:** Form data with `hospital_id` + file upload

### Session Endpoints
- `GET /session/{session_id}` - Retrieve session by ID
- `GET /sessions/list` - List recent sessions

### Ledger Endpoints
- `GET /ledger/transactions` - Get recent transactions
- `GET /ledger/transaction/{tx_id}` - Get specific transaction

### Statistics
- `GET /stats/summary` - System statistics

## Response Schema

Both modes return identical JSON schema:

```json
{
  "session_id": "string",
  "hospital_id": "string",
  "timestamp": "ISO8601",
  "processing_time": "float",
  "model_profile": {
    "architecture": "string",
    "parameters": "string",
    "parameter_count": "int",
    "dataset": "string",
    "model_size_mb": "float",
    "model_hash": "string"
  },
  "golden_test": {
    "accuracy": "float",
    "confusion_matrix": [[int]],
    "avg_confidence": "float",
    ...
  },
  "fingerprint": {
    "lung_opacity": "float",
    "cardiomegaly": "float",
    ...
  },
  "shap_analysis": {...},
  "anomaly_analysis": {
    "anomaly_score": "float",
    "threshold": "float",
    "verdict": "APPROVED|REJECTED",
    ...
  },
  "verdict": "APPROVED|REJECTED",
  "verdict_reasoning": "string",
  "ledger": {
    "tx_id": "string",
    "timestamp": "ISO8601",
    "blockchain_hash": "string",
    "evidence_hash": "string"
  },
  "warnings": ["string"],
  "recommendations": ["string"]
}
```

## Installation

1. **Install dependencies:**
```bash
pip install fastapi uvicorn python-multipart numpy torch scikit-learn
```

2. **Set operating mode:**
```bash
export AURA_MODE=demo  # or real
```

3. **Run server:**
```bash
cd aura_backend
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage Examples

### Demo Mode
```bash
# Set mode
export AURA_MODE=demo

# Submit update (returns random demo scenario)
curl -X POST "http://localhost:8000/sentinel/submit_update" \
  -F "hospital_id=HOSP1"
```

### Real Mode
```bash
# Set mode
export AURA_MODE=real

# Submit update with model file
curl -X POST "http://localhost:8000/sentinel/submit_update" \
  -F "hospital_id=HOSP1" \
  -F "model_file=@path/to/model.pth"
```

### Check Mode
```bash
curl http://localhost:8000/system/mode
# {"mode": "demo"}
```

### Health Check
```bash
curl http://localhost:8000/system/health
```

### Get Session
```bash
curl http://localhost:8000/session/DEMO-APPROVED-001
```

## Frontend Integration

Frontend should:
1. **Fetch mode on startup:** `GET /system/mode`
2. **Display mode badge:** "Demo Mode" or "Live Mode"
3. **Use same API calls** regardless of mode
4. **No simulation logic in frontend** - all handled by backend

## Configuration

Edit `config.py` to customize:
- `ANOMALY_THRESHOLD` - Detection threshold (default: 0.72)
- `SHAP_SAMPLE_SIZE` - SHAP samples (default: 50)
- `DEMO_PROCESSING_DELAY` - Demo delay in seconds (default: 1.0)
- File paths and directories

## Logging

All operations are logged with timestamps:
```
2026-02-20 10:15:23 - INFO - Received submission from HOSP1 in demo mode
2026-02-20 10:15:24 - INFO - Demo response generated: DEMO-APPROVED-abc123
```

## Database Schema

**Transactions Table:**
```sql
CREATE TABLE transactions (
    tx_id TEXT PRIMARY KEY,
    hospital_id TEXT NOT NULL,
    update_hash TEXT NOT NULL,
    verdict TEXT NOT NULL,
    evidence_hash TEXT,
    anomaly_score REAL,
    timestamp TEXT NOT NULL,
    metadata TEXT
)
```

## Development

**Add new demo scenarios:**
1. Create JSON file in `demo_data/`
2. Add to `DEMO_FILES` dict in `demo_handler.py`
3. Adjust selection weights if needed

**Extend real mode pipeline:**
1. Modify functions in `real_handler.py`
2. Add new processing steps
3. Update response schema

## Troubleshooting

**"Demo files not found"**
- Ensure `demo_data/*.json` files exist
- Check file permissions

**"Detector not available"**
- In real mode, check `detector/model.pkl` exists
- System will use fallback synthetic detection

**"Golden set not found"**
- In real mode, check `golden_set/*.npy` files
- System will generate synthetic results

## License

Part of AURA - Federated Learning Security System
