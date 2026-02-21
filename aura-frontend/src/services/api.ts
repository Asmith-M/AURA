import axios from 'axios';
import type {
  AttackMetric,
  BehavioralReport,
  DailyMetric,
  DashboardStats,
  HospitalPerformance,
  LedgerFilters,
  LedgerStats,
  SecurityRecommendation,
  SentinelLog,
  Threat,
  Transaction,
  VerificationResult,
  Fingerprint,
  PipelineRunStatus,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

interface BackendFeatureInteractions {
  mean_abs_importance?: number;
  [key: string]: unknown;
}

interface BackendGoldenEval {
  accuracy?: number;
  samples_tested?: number;
  confusion_matrix?: unknown[];
  [key: string]: unknown;
}

interface BackendShapAnalysis {
  method?: string;
  feature_interactions?: BackendFeatureInteractions;
  [key: string]: unknown;
}

interface BackendAnomalyAnalysis {
  anomaly_score?: number;
  normalized_score?: number;
  confidence?: number;
  distance_sigma?: number;
  outlier_probability?: number;
  [key: string]: unknown;
}

interface BackendSession {
  session_id: string;
  hospital_id: string;
  timestamp: string;
  golden_test?: BackendGoldenEval;
  golden_eval?: BackendGoldenEval;
  fingerprint?: Record<string, unknown>;
  shap_fingerprint?: Record<string, unknown>;
  shap_analysis?: BackendShapAnalysis;
  anomaly_analysis?: BackendAnomalyAnalysis;
  anomaly_score?: number;
  verdict: string;
  evidence_hash?: string;
}

interface BackendLedgerTransaction {
  tx_id: string;
  hospital_id: string;
  update_hash: string;
  verdict: string;
  evidence_hash: string;
  anomaly_score: number;
  timestamp: string;
}

function parseHospitalId(value: unknown): number {
  const text = String(value ?? '');
  const digits = text.replace(/\D/g, '');
  return digits ? parseInt(digits, 10) : 0;
}

function buildBackendHospitalId(value: unknown): string {
  const text = String(value ?? '').trim();
  if (/^HOSP\d+$/i.test(text)) {
    return text.toUpperCase();
  }
  const digits = text.replace(/\D/g, '');
  return digits ? `HOSP${digits}` : text;
}

function toFingerprint(session: BackendSession): Fingerprint {
  const source = session.shap_fingerprint || session.fingerprint || {};
  const values = Object.values(source)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));

  if (values.length === 0) {
    return {
      mean_importance_global: 0,
      std_importance_global: 0,
      max_importance_global: 0,
      min_importance_global: 0,
      entropy_mean: 0,
      variance_stability: 0,
      feature_consistency: 0,
      prediction_stability: 0,
    };
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  const max = Math.max(...values);
  const min = Math.min(...values);

  return {
    mean_importance_global: mean,
    std_importance_global: std,
    max_importance_global: max,
    min_importance_global: min,
    entropy_mean: Math.max(0, session.shap_analysis?.feature_interactions?.mean_abs_importance ?? mean),
    variance_stability: Number(session.anomaly_analysis?.distance_sigma ?? std),
    feature_consistency: Number(session.anomaly_analysis?.confidence ?? 0.5),
    prediction_stability: Number(session.anomaly_analysis?.outlier_probability ?? 0.5),
  };
}

function mapSessionToBehavioralReport(session: BackendSession): BehavioralReport {
  const anomaly = session.anomaly_analysis || {};
  const golden = session.golden_eval || session.golden_test || {};
  const anomalyScore = Number(session.anomaly_score ?? anomaly.anomaly_score ?? 0);

  return {
    submission_id: session.session_id,
    hospital_id: parseHospitalId(session.hospital_id),
    analysis_result: {
      fingerprint: toFingerprint(session),
      test_accuracy: Number(golden.accuracy ?? 0),
      sample_size: Number(golden.samples_tested ?? 0),
      analysis_timestamp: session.timestamp,
      shap_method_used: String(session.shap_analysis?.method ?? 'unknown'),
    },
    detection_result: {
      verdict: String(session.verdict),
      is_anomalous: String(session.verdict) === 'REJECTED',
      anomaly_score: anomalyScore,
      normalized_score: Number(anomaly.normalized_score ?? anomalyScore),
      confidence: Number(anomaly.confidence ?? 0),
      raw_prediction: String(session.verdict) === 'REJECTED' ? 1 : -1,
    },
    fingerprint_id: String(session.evidence_hash ?? session.session_id),
    test_accuracy: Number(golden.accuracy ?? 0),
    raw_predictions: {
      confusion_matrix: golden.confusion_matrix ?? [],
    },
    timestamp: session.timestamp,
  };
}

function mapTransaction(tx: BackendLedgerTransaction): Transaction {
  return {
    transaction_id: tx.tx_id,
    hospital_id: parseHospitalId(tx.hospital_id),
    update_hash: tx.update_hash,
    verdict: tx.verdict as Transaction['verdict'],
    evidence_hash: tx.evidence_hash,
    anomaly_score: Number(tx.anomaly_score ?? 0),
    normalized_score: Number(tx.anomaly_score ?? 0) * 100,
    confidence: undefined,
    timestamp: tx.timestamp,
  };
}

export const systemAPI = {
  getMode: async (): Promise<{ mode: string }> => {
    const response = await apiClient.get('/system/mode');
    return response.data;
  },
  getHealth: async (): Promise<Record<string, unknown>> => {
    const response = await apiClient.get('/system/health');
    return response.data;
  },
};

export const sentinelAPI = {
  submitUpdate: async (formData: FormData): Promise<BehavioralReport> => {
    const backendData = new FormData();

    const hospitalIdRaw = formData.get('hospital_id');
    backendData.append('hospital_id', buildBackendHospitalId(hospitalIdRaw));

    const modelFile = formData.get('model_file') ?? formData.get('file');
    if (modelFile instanceof File) {
      backendData.append('model_file', modelFile);
    }

    const response = await apiClient.post('/sentinel/submit_update', backendData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    return mapSessionToBehavioralReport(response.data as BackendSession);
  },

  getReport: async (submissionId: string): Promise<BehavioralReport> => {
    const response = await apiClient.get(`/sentinel/report/${submissionId}`);
    return mapSessionToBehavioralReport(response.data as BackendSession);
  },

  getDetectionStats: async (): Promise<DashboardStats> => {
    const response = await apiClient.get('/sentinel/detection_stats');
    return response.data as DashboardStats;
  },

  getReports: async (limit?: number): Promise<BehavioralReport[]> => {
    const response = await apiClient.get('/sentinel/reports', {
      params: { limit: limit ?? 20 },
    });

    const rows = Array.isArray(response.data)
      ? response.data
      : Array.isArray(response.data?.reports)
      ? response.data.reports
      : [];

    return rows.map((row: BackendSession) => mapSessionToBehavioralReport(row));
  },

  getLatestLogs: async (): Promise<SentinelLog> => {
    const response = await apiClient.get('/sentinel/logs/latest');
    return response.data as SentinelLog;
  },

  startPipeline: async (payload: {
    hospital_id: string;
    rounds?: number;
    local_epochs?: number;
    max_samples_per_hospital?: number;
    attack_mode?: boolean;
  }): Promise<PipelineRunStatus> => {
    const response = await apiClient.post('/sentinel/pipeline/start', payload);
    return response.data as PipelineRunStatus;
  },

  getPipelineStatus: async (runId: string): Promise<PipelineRunStatus> => {
    const response = await apiClient.get(`/sentinel/pipeline/status/${runId}`);
    return response.data as PipelineRunStatus;
  },

  getLatestPipeline: async (): Promise<PipelineRunStatus | null> => {
    const response = await apiClient.get('/sentinel/pipeline/latest');
    return (response.data?.run ?? null) as PipelineRunStatus | null;
  },

  getPipelineModelDownloadUrl: (runId: string): string => {
    return `${API_BASE_URL}/sentinel/pipeline/download/${encodeURIComponent(runId)}`;
  },
};

export const ledgerAPI = {
  getHistory: async (hospitalId: number, filters?: LedgerFilters): Promise<Transaction[]> => {
    const txs = await ledgerAPI.getTransactions({ ...filters, hospital_id: hospitalId });
    return txs;
  },

  getTransactions: async (filters?: LedgerFilters): Promise<Transaction[]> => {
    const response = await apiClient.get('/ledger/transactions', {
      params: { limit: filters?.per_page ?? 50 },
    });

    let txs = ((response.data?.transactions ?? []) as BackendLedgerTransaction[]).map(mapTransaction);

    if (filters?.hospital_id) {
      txs = txs.filter((tx) => tx.hospital_id === filters.hospital_id);
    }
    if (filters?.verdict && filters.verdict !== 'ALL') {
      txs = txs.filter((tx) => tx.verdict === filters.verdict);
    }
    if (filters?.start_date) {
      txs = txs.filter((tx) => new Date(tx.timestamp) >= new Date(filters.start_date!));
    }
    if (filters?.end_date) {
      txs = txs.filter((tx) => new Date(tx.timestamp) <= new Date(filters.end_date!));
    }

    return txs;
  },

  getTransaction: async (txId: string): Promise<Transaction> => {
    const response = await apiClient.get(`/ledger/transaction/${txId}`);
    return mapTransaction(response.data as BackendLedgerTransaction);
  },

  verifyTransaction: async (txId: string): Promise<VerificationResult> => {
    const response = await apiClient.get(`/ledger/verify/${txId}`);
    return response.data as VerificationResult;
  },

  getLedgerStats: async (): Promise<LedgerStats> => {
    const response = await apiClient.get('/ledger/stats');
    return response.data as LedgerStats;
  },
};

export const analyticsAPI = {
  getDailyMetrics: async (days: number = 30): Promise<DailyMetric[]> => {
    const response = await apiClient.get('/analytics/daily', { params: { days } });
    return response.data as DailyMetric[];
  },

  getAttackMetrics: async (): Promise<AttackMetric[]> => {
    const response = await apiClient.get('/analytics/attacks');
    return response.data as AttackMetric[];
  },

  getHospitalPerformance: async (): Promise<HospitalPerformance[]> => {
    const response = await apiClient.get('/analytics/hospitals');
    return response.data as HospitalPerformance[];
  },
};

export const securityAPI = {
  getThreats: async (status?: string): Promise<Threat[]> => {
    const response = await apiClient.get('/security/threats', {
      params: status ? { status } : undefined,
    });
    return response.data as Threat[];
  },

  getRecommendations: async (): Promise<SecurityRecommendation[]> => {
    const response = await apiClient.get('/security/recommendations');
    return response.data as SecurityRecommendation[];
  },
};

export { apiClient };
