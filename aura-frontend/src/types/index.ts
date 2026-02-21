/**
 * Type definitions for the AURA Federated Learning Security Dashboard
 */

// Enums and type aliases
export type Verdict = 'APPROVED' | 'REJECTED' | 'ERROR' | 'PROCESSING';
export type PrivacyLevel = 'PUBLIC' | 'PRIVATE' | 'CONFIDENTIAL';
export type AttackType = 'DATA_POISONING' | 'MODEL_POISONING' | 'INFERENCE_ATTACK' | 'BYZANTINE';
export type TimeRange = '7d' | '30d' | '90d';
export type AlertType = 'success' | 'error' | 'warning' | 'info';
export type ChartType = 'line' | 'bar' | 'pie' | 'area';
export type ThreatSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type ThreatStatus = 'ACTIVE' | 'RESOLVED' | 'INVESTIGATING';
export type Priority = 'LOW' | 'MEDIUM' | 'HIGH';

// Core entity interfaces
export interface ModelWeights {
  hospital_id: number;
  model_weights: number[][];
  metadata: Record<string, unknown>;
}

export interface Transaction {
  transaction_id: string;
  hospital_id: number;
  update_hash: string;
  verdict: Verdict;
  evidence_hash?: string;
  anomaly_score: number;
  normalized_score?: number;
  confidence?: number;
  timestamp: string;
}

export interface Fingerprint {
  mean_importance_global: number;
  std_importance_global: number;
  max_importance_global: number;
  min_importance_global: number;
  entropy_mean: number;
  variance_stability: number;
  feature_consistency: number;
  prediction_stability: number;
}

export interface AnalysisResult {
  fingerprint: Fingerprint;
  test_accuracy: number;
  sample_size: number;
  analysis_timestamp: string;
  shap_method_used: string;
}

export interface DetectionResult {
  verdict: string;
  is_anomalous: boolean;
  anomaly_score: number;
  normalized_score: number;
  confidence: number;
  raw_prediction: number;
}

export interface BehavioralReport {
  submission_id: string;
  hospital_id: number;
  analysis_result: AnalysisResult;
  detection_result: DetectionResult;
  fingerprint_id: string;
  test_accuracy: number;
  raw_predictions: Record<string, unknown>;
  timestamp: string;
}

// Dashboard stats
export interface DashboardStats {
  total_submissions: number;
  approved: number;
  rejected: number;
  active_sessions: number;
  processing: number;
  approval_rate: number;
  security_effectiveness: number;
  uptime: string;
}

// Analytics data
export interface DailyMetric {
  date: string;
  submissions: number;
  approved: number;
  rejected: number;
}

export interface AttackMetric {
  type: AttackType;
  count: number;
}

export interface HospitalPerformance {
  hospital_id: number;
  hospital_name: string;
  submissions: number;
  approval_rate: number;
  avg_accuracy: number;
}

// Security data
export interface Threat {
  id: string;
  type: AttackType;
  severity: ThreatSeverity;
  hospital_id: number;
  description: string;
  detected_at: string;
  status: ThreatStatus;
}

export interface SecurityRecommendation {
  id: string;
  priority: Priority;
  title: string;
  description: string;
  action_required: string;
}

// Upload form data
export interface UploadFormData {
  hospital_id: number;
  model_description: string;
  privacy_level: PrivacyLevel;
  file: File | null;
}

// Ledger filter options
export interface LedgerFilters {
  hospital_id?: number;
  verdict?: Verdict | 'ALL';
  start_date?: string;
  end_date?: string;
  page?: number;
  per_page?: number;
}

// Blockchain info
export interface BlockchainInfo {
  total_blocks: number;
  latest_hash: string;
  pending_transactions: number;
  last_updated: string;
}

// Component props
export interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  color?: 'primary' | 'success' | 'warning' | 'danger';
}

export interface ChartContainerProps {
  title: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}

export interface AlertProps {
  type: AlertType;
  message: string;
  onClose?: () => void;
  autoClose?: boolean;
  duration?: number;
}

export interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export interface NavbarProps {
  onMenuClick: () => void;
  systemMode?: 'real' | 'unknown';
}

// API response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
}

export interface VerificationResult {
  verified: boolean;
  message: string;
}

export interface LedgerStats {
  total_transactions: number;
  approved: number;
  rejected: number;
  security_rate: number;
  blockchain_info: BlockchainInfo;
}

// Hook return types
export interface UseApiReturn<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

// Sentinel Monitor types
export interface ProcessingStep {
  id: string;
  label: string;
  completed: boolean;
  timestamp?: string;
}

export interface SentinelLog {
  model_id: string;
  hospital_id: number;
  golden_set_accuracy: number;
  shap_status: 'COMPLETED' | 'FAILED' | 'PROCESSING';
  isolation_score: number;
  verdict: Verdict;
  ledger_tx_id: string;
  timestamp: string;
  processing_steps: ProcessingStep[];
}

export type PipelineRunState = 'queued' | 'running' | 'completed' | 'failed';

export interface PipelineRunLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  progress: number;
  step?: string;
}

export interface PipelineDatasetHospitalInfo {
  hospital_id: number;
  display_name?: string;
  samples: number;
  class_distribution: Record<string, number>;
  input_shape: number[];
  data_slice_start?: number;
  noise_std?: number;
}

export interface PipelineDatasetInfo {
  dataset_name: string;
  dataset_variant?: string;
  judge_summary?: string;
  selected_hospital?: string;
  participating_hospitals?: number[];
  federated_strategy?: string;
  preprocessing_pipeline?: string[];
  class_labels?: Record<string, string>;
  total_samples: number;
  input_shape: number[];
  hospitals: PipelineDatasetHospitalInfo[];
  golden_set_samples: number;
  golden_set_class_distribution: Record<string, number>;
  golden_set_data_path?: string;
  golden_set_labels_path?: string;
}

export interface PipelineClientMetric {
  hospital_id: number;
  train_loss: number;
  eval_loss: number;
  eval_accuracy: number;
  samples: number;
}

export interface PipelineRoundMetric {
  round: number;
  avg_train_loss: number;
  avg_eval_accuracy: number;
  client_metrics: PipelineClientMetric[];
}

export interface PipelineTrainingSummary {
  rounds: number;
  local_epochs: number;
  max_samples_per_hospital: number;
  round_metrics: PipelineRoundMetric[];
}

export interface PipelineArtifacts {
  model_file_name: string;
  model_file_path: string;
  model_download_url: string;
  model_sha256: string;
  model_size_mb: number;
  attack_mode?: boolean;
  session_id?: string;
  ledger_tx?: string;
}

export interface PipelineRunConfig {
  rounds: number;
  local_epochs: number;
  max_samples_per_hospital: number;
  attack_mode?: boolean;
  seed?: number;
  scenario?: string;
}

export interface PipelineRunStatus {
  run_id: string;
  hospital_id: string;
  status: PipelineRunState;
  progress: number;
  current_step: string;
  message: string;
  started_at: string;
  updated_at: string;
  completed_at?: string | null;
  error?: string | null;
  logs: PipelineRunLog[];
  dataset_info?: PipelineDatasetInfo | null;
  training_summary?: PipelineTrainingSummary | null;
  artifacts?: PipelineArtifacts | null;
  final_session?: Record<string, unknown> | null;
  attack_mode?: boolean;
  config: PipelineRunConfig;
}
