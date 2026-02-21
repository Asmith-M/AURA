cat > docs/architecture.md << 'EOF'
# AURA Architecture Documentation

## System Overview

AURA is a Trust-as-a-Service framework that secures Federated Learning in healthcare environments using explainable AI and blockchain auditing.

## Architecture Layers

### 1. Federated Learning Layer
- **Flower Framework**: Coordinates distributed training
- **PyTorch Models**: Local model training
- **Privacy Preservation**: Raw data never leaves hospitals

### 2. Security Layer
- **XAI Sentinel**: Behavioral analysis of model updates
- **SHAP Fingerprinting**: Feature importance analysis
- **Isolation Forest**: Anomaly detection

### 3. Audit Layer
- **Hyperledger Fabric Simulation**: Immutable transaction logging
- **Evidence Storage**: XAI reports with cryptographic hashes

## Data Flow
Hospital → FL Client → Sentinel Interception → SHAP Analysis → Isolation Forest → Ledger Recording


## Security Guarantees

- Distinguishes malicious outliers from legitimate medical innovations
- Provides explainable rejection evidence
- Maintains privacy of sensitive medical data
