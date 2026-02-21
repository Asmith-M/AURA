# AURA Backend - Quick Start Guide

## 🚀 Getting Started in 2 Minutes

### Step 1: Install Dependencies
```bash
cd aura_backend
pip install -r requirements.txt
```

### Step 2: Choose Operating Mode

#### Option A: Demo Mode (Recommended for UI Testing)
**Windows:**
```powershell
$env:AURA_MODE="demo"
python main.py
```
Or simply run:
```powershell
.\run_demo.bat
```

**Linux/Mac:**
```bash
export AURA_MODE=demo
python main.py
```
Or simply run:
```bash
chmod +x run_demo.sh
./run_demo.sh
```

#### Option B: Real Mode (For Actual Processing)
**Windows:**
```powershell
$env:AURA_MODE="real"
python main.py
```
Or simply run:
```powershell
.\run_real.bat
```

**Linux/Mac:**
```bash
export AURA_MODE=real
python main.py
```
Or simply run:
```bash
chmod +x run_real.sh
./run_real.sh
```

### Step 3: Verify Server is Running
Open browser: http://localhost:8000

You should see:
```json
{
  "service": "AURA - Federated Learning Security System",
  "version": "2.0.0",
  "mode": "demo",
  "status": "operational",
  ...
}
```

### Step 4: Test with Sample Request

**Demo Mode Test:**
```bash
curl -X POST "http://localhost:8000/sentinel/submit_update" \
  -F "hospital_id=HOSP1"
```

**Or using PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/sentinel/submit_update" `
  -Method POST `
  -Form @{hospital_id="HOSP1"}
```

You'll get back a complete analysis with verdict, SHAP fingerprint, anomaly score, etc.

### Step 5: Run Test Suite
```bash
python test_backend.py
```

## 📋 Common Commands

### Check Current Mode
```bash
curl http://localhost:8000/system/mode
```

### Health Check
```bash
curl http://localhost:8000/system/health
```

### Get Recent Sessions
```bash
curl http://localhost:8000/sessions/list?limit=10
```

### Get Ledger Transactions
```bash
curl http://localhost:8000/ledger/transactions?limit=10
```

### Get Statistics
```bash
curl http://localhost:8000/stats/summary
```

## 🎯 What Each Mode Does

### Demo Mode
- ✅ Returns pre-defined realistic mock data instantly
- ✅ Perfect for UI screenshots and presentations
- ✅ No dependencies on detector models or datasets
- ✅ Randomly returns approved/rejected/anomaly scenarios
- ✅ Consistent, reproducible results

**Use when:**
- Testing frontend UI
- Creating screenshots
- Demonstrating to stakeholders
- No real model analysis needed

### Real Mode
- ✅ Executes full ML pipeline
- ✅ Loads actual PyTorch models
- ✅ Runs inference on golden validation set
- ✅ Computes SHAP explanations
- ✅ Performs anomaly detection
- ✅ Logs to blockchain ledger

**Use when:**
- Analyzing actual model submissions
- Production deployment
- Real security assessments needed

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Anomaly detection threshold
ANOMALY_THRESHOLD = 0.72  # Models above this are rejected

# SHAP configuration
SHAP_SAMPLE_SIZE = 50     # Number of samples for SHAP
SHAP_MAX_EVALS = 500      # Max SHAP evaluations

# Demo processing delay (seconds)
DEMO_PROCESSING_DELAY = 1.0
```

## 🐛 Troubleshooting

### Port 8000 Already in Use
Change the port:
```bash
uvicorn main:app --port 8001
```

### "Module not found" errors
Install dependencies:
```bash
pip install -r requirements.txt
```

### Demo files not loading
Verify files exist in `demo_data/`:
- demo_approved.json
- demo_rejected.json
- demo_anomaly.json

### Real mode: "Detector not available"
This is normal if detector model doesn't exist. System will use fallback synthetic detection.

## 📚 API Documentation

Once server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🎨 Frontend Integration

Your frontend should:

1. **Fetch mode on startup:**
```javascript
const response = await fetch('http://localhost:8000/system/mode');
const { mode } = await response.json();
console.log('Backend mode:', mode);
```

2. **Display mode badge:**
```jsx
{mode === 'demo' ? '🎭 Demo Mode' : '🔴 Live Mode'}
```

3. **Submit model updates:**
```javascript
const formData = new FormData();
formData.append('hospital_id', 'HOSP1');
// In real mode, also append model file:
// formData.append('model_file', modelFile);

const response = await fetch('http://localhost:8000/sentinel/submit_update', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Session:', result.session_id);
console.log('Verdict:', result.verdict);
```

## 🚢 Production Deployment

### Using Gunicorn (Linux)
```bash
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV AURA_MODE=real
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t aura-backend .
docker run -p 8000:8000 -e AURA_MODE=demo aura-backend
```

## 📊 Response Schema Reference

All endpoints return this unified schema:

```typescript
{
  session_id: string;           // Unique session identifier
  hospital_id: string;          // Hospital ID
  timestamp: string;            // ISO 8601 timestamp
  processing_time: number;      // Processing duration in seconds
  
  model_profile: {
    architecture: string;       // CNN-ResNet18, etc.
    parameters: string;         // "11,689,512"
    parameter_count: number;    // 11689512
    dataset: string;            // Training dataset info
    model_size_mb: number;      // 44.6
    model_hash: string;         // SHA256 hash
  };
  
  golden_test: {
    accuracy: number;           // 0.0 - 1.0
    precision: number;
    recall: number;
    f1_score: number;
    samples_tested: number;     // 100
    confusion_matrix: number[][];
    avg_confidence: number;
    class_distribution: object;
    per_class_accuracy: object;
  };
  
  fingerprint: {
    lung_opacity: number;       // Feature importances 0-1
    cardiomegaly: number;
    consolidation: number;
    edema: number;
    age_bias: number;
    gender_bias: number;
    pixel_artifact: number;
    texture_variance: number;
    edge_sensitivity: number;
    background_noise: number;
  };
  
  shap_analysis: {
    top_features: Array<{
      feature: string;
      importance: number;
      direction: "positive" | "negative" | "suspicious" | "neutral";
    }>;
    feature_interactions: object;
    baseline_value: number;
    samples_analyzed: number;
  };
  
  anomaly_analysis: {
    anomaly_score: number;      // 0-1, higher = more anomalous
    normalized_score: number;
    threshold: number;          // 0.72 default
    distance_from_normal: number;
    distance_sigma: number;
    outlier_probability: number;
    nearest_neighbors_count: number;
    isolation_depth: number;
    verdict: "APPROVED" | "REJECTED";
    confidence: number;
  };
  
  verdict: "APPROVED" | "REJECTED";
  verdict_reasoning: string;    // Human-readable explanation
  
  ledger: {
    tx_id: string;              // Transaction ID
    block_height: number;       // Blockchain block number
    timestamp: string;          // ISO 8601
    blockchain_hash: string;    // 0x...
    evidence_hash: string;      // SHA256 of SHAP report
    immutable: boolean;         // true
  };
  
  warnings: string[];           // Security warnings
  recommendations: string[];    // Suggested actions
}
```

## ✅ Success!

Your AURA backend is now running! 🎉

Next steps:
1. Test with `test_backend.py`
2. Integrate with your frontend
3. Switch between demo and real modes as needed
4. Deploy to production when ready

For more details, see [README.md](README.md)
