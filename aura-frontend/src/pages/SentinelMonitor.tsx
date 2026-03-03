/**
 * Sentinel Monitor Page - Full backend orchestration monitor
 */
import { useState, useEffect, useCallback } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileText,
  ListChecks,
  Shield,
  TrendingUp,
  Play,
  RefreshCw,
  Download,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { sentinelAPI } from '../services/api';
import type { DatasetInspection, PipelineRunStatus } from '../types';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { Alert } from '../components/Alert';
import { cn } from '../utils/cn';

export function SentinelMonitor() {
  const [run, setRun] = useState<PipelineRunStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetInspection, setDatasetInspection] = useState<DatasetInspection | null>(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [hospitalId, setHospitalId] = useState('HOSP1');
  const [rounds, setRounds] = useState(2);
  const [localEpochs, setLocalEpochs] = useState(1);
  const [maxSamples, setMaxSamples] = useState(256);

  const loadLatestRun = useCallback(async () => {
    try {
      setError(null);
      const latest = await sentinelAPI.getLatestPipeline();
      setRun(latest);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load latest pipeline run');
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshRun = useCallback(async (runId: string) => {
    try {
      const status = await sentinelAPI.getPipelineStatus(runId);
      setRun(status);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh pipeline run');
    }
  }, []);

  const startPipeline = useCallback(async () => {
    try {
      setStarting(true);
      setError(null);
      const created = await sentinelAPI.startPipeline({
        hospital_id: hospitalId.trim(),
        rounds,
        local_epochs: localEpochs,
        max_samples_per_hospital: maxSamples,
        attack_mode: false,
      });
      setRun(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start pipeline');
    } finally {
      setStarting(false);
    }
  }, [hospitalId, rounds, localEpochs, maxSamples]);

  const startAttackPipeline = useCallback(async () => {
    try {
      setStarting(true);
      setError(null);
      const created = await sentinelAPI.startPipeline({
        hospital_id: hospitalId.trim(),
        rounds,
        local_epochs: localEpochs,
        max_samples_per_hospital: maxSamples,
        attack_mode: true,
      });
      setRun(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start pipeline');
    } finally {
      setStarting(false);
    }
  }, [hospitalId, rounds, localEpochs, maxSamples]);

  const loadDatasetInspection = useCallback(async () => {
    try {
      setDatasetLoading(true);
      const inspection = await sentinelAPI.getDatasetInspection(12);
      setDatasetInspection(inspection);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset inspection');
    } finally {
      setDatasetLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadLatestRun();
    void loadDatasetInspection();
  }, [loadLatestRun, loadDatasetInspection]);

  useEffect(() => {
    if (!run) {
      return;
    }
    if (run.status !== 'queued' && run.status !== 'running') {
      return;
    }

    const interval = setInterval(() => {
      void refreshRun(run.run_id);
    }, 1500);

    return () => clearInterval(interval);
  }, [run, refreshRun]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <LoadingSpinner />
      </div>
    );
  }

  const finalSession = (run?.final_session ?? null) as Record<string, unknown> | null;
  const sessionRecord = (finalSession ?? {}) as Record<string, unknown>;
  const anomalyAnalysis = (sessionRecord.anomaly_analysis ?? {}) as Record<string, unknown>;
  const detectorAnalysis = (sessionRecord.detector_analysis ?? {}) as Record<string, unknown>;
  const fingerprintAnalysis = (sessionRecord.fingerprint_analysis ?? {}) as Record<string, unknown>;
  const ledgerEntry = (sessionRecord.ledger_entry ?? sessionRecord.ledger ?? {}) as Record<string, unknown>;
  const modelProfile = (sessionRecord.model_profile ?? {}) as Record<string, unknown>;
  const datasetStats = (sessionRecord.dataset_stats ?? {}) as Record<string, unknown>;
  const goldenTest = (sessionRecord.golden_eval ?? sessionRecord.golden_test ?? {}) as Record<string, unknown>;
  const shapAnalysis = (sessionRecord.shap_analysis ?? {}) as Record<string, unknown>;
  const scenarioCalibration = (anomalyAnalysis.scenario_calibration ?? {}) as Record<string, unknown>;
  const isRunning = run?.status === 'queued' || run?.status === 'running';
  const verdict = String(sessionRecord.verdict ?? run?.status?.toUpperCase() ?? 'PROCESSING');
  const anomalyScore = Number(detectorAnalysis.anomaly_score ?? anomalyAnalysis.anomaly_score ?? 0);
  const threshold = Number(detectorAnalysis.threshold ?? anomalyAnalysis.threshold ?? 0.72);
  const rawScore = Number(detectorAnalysis.raw_score ?? anomalyAnalysis.raw_score ?? 0);
  const calibratedScore = Number(scenarioCalibration.post_calibration_score ?? anomalyScore);

  const roundChartData = run?.training_summary?.round_metrics?.map((item) => ({
    round: `R${item.round}`,
    avgAccuracy: item.avg_eval_accuracy * 100,
    avgLoss: item.avg_train_loss,
  })) ?? [];

  const anomalyChartData = [
    { name: 'Anomaly Score', value: anomalyScore },
    { name: 'Threshold', value: threshold },
  ];

  const goldenClassDistributionData = Object.entries(run?.dataset_info?.golden_set_class_distribution ?? {}).map(
    ([classId, count]) => ({
      classId,
      count: Number(count),
    }),
  );

  const warnings = Array.isArray(sessionRecord.warnings)
    ? (sessionRecord.warnings as unknown[]).map((item) => String(item))
    : [];
  const recommendations = Array.isArray(sessionRecord.recommendations)
    ? (sessionRecord.recommendations as unknown[]).map((item) => String(item))
    : [];
  const executionTrace = Array.isArray(sessionRecord.execution_trace)
    ? (sessionRecord.execution_trace as Record<string, unknown>[])
    : [];
  const timelineStatus = executionTrace.length > 0
    ? executionTrace
    : Array.isArray(sessionRecord.timeline_status)
    ? (sessionRecord.timeline_status as Record<string, unknown>[])
    : [];
  const topFeatures = Array.isArray(shapAnalysis.top_5_features)
    ? (shapAnalysis.top_5_features as Record<string, unknown>[])
    : Array.isArray(shapAnalysis.top_features)
    ? (shapAnalysis.top_features as Record<string, unknown>[])
    : [];
  const confusionMatrix = Array.isArray(goldenTest.confusion_matrix)
    ? (goldenTest.confusion_matrix as unknown[])
    : [];
  const perClassPrecision = (goldenTest.per_class_precision ?? {}) as Record<string, unknown>;
  const perClassRecall = (goldenTest.per_class_recall ?? {}) as Record<string, unknown>;
  const attackInjection = (anomalyAnalysis.attack_injection ?? {}) as Record<string, unknown>;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Sentinel Super Monitor</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Start end-to-end FL backend execution, stream logs, and present final security interrogation evidence.
        </p>
      </div>

      {error && <Alert type="error" message={error} />}

      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Hospital ID</label>
            <input
              value={hospitalId}
              onChange={(event) => setHospitalId(event.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">FL Rounds</label>
            <input
              type="number"
              min={1}
              max={10}
              value={rounds}
              onChange={(event) => setRounds(Number(event.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Local Epochs</label>
            <input
              type="number"
              min={1}
              max={10}
              value={localEpochs}
              onChange={(event) => setLocalEpochs(Number(event.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Samples/Hospital</label>
            <input
              type="number"
              min={64}
              max={5000}
              value={maxSamples}
              onChange={(event) => setMaxSamples(Number(event.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={startPipeline}
            disabled={starting || isRunning}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            {starting ? 'Starting...' : isRunning ? 'Pipeline Running' : 'Start Full Backend Pipeline'}
          </button>
          <button
            onClick={startAttackPipeline}
            disabled={starting || isRunning}
            className="inline-flex items-center gap-2 px-4 py-2 bg-danger-600 text-white rounded-lg hover:bg-danger-700 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <Shield className="w-4 h-4" />
            {starting ? 'Starting...' : isRunning ? 'Pipeline Running' : 'Run Attack Injection Scenario'}
          </button>
          <button
            onClick={() => void loadLatestRun()}
            className="inline-flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh Latest Run
          </button>
          <button
            onClick={() => void loadDatasetInspection()}
            className="inline-flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <RefreshCw className="w-4 h-4" />
            {datasetLoading ? 'Refreshing Dataset...' : 'Refresh Dataset Inspection'}
          </button>
          {run?.artifacts?.model_download_url && (
            <a
              href={sentinelAPI.getPipelineModelDownloadUrl(run.run_id)}
              className="inline-flex items-center gap-2 px-4 py-2 border border-primary-300 text-primary-700 dark:text-primary-300 dark:border-primary-700 rounded-lg hover:bg-primary-50 dark:hover:bg-primary-900/20"
            >
              <Download className="w-4 h-4" />
              Download Trained Model
            </a>
          )}
        </div>
      </div>

      {run && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Pipeline Run Status</h2>
            <div className="flex items-center gap-2">
              {Boolean(run.attack_mode ?? run.config?.attack_mode) && (
                <span className="px-3 py-1 rounded-full text-sm font-medium bg-danger-100 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300">
                  ATTACK MODE
                </span>
              )}
              <span className={cn(
                "px-3 py-1 rounded-full text-sm font-medium",
                run.status === 'completed' && "bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-400",
                run.status === 'failed' && "bg-danger-100 text-danger-700 dark:bg-danger-900/30 dark:text-danger-400",
                (run.status === 'queued' || run.status === 'running') && "bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-400",
              )}>
                {run.status.toUpperCase()}
              </span>
            </div>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{run.message}</p>

          <div className="w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-2">
            <div
              className="h-full bg-primary-600 transition-all duration-500"
              style={{ width: `${run.progress}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400">{run.progress}% complete</p>
        </div>
      )}

      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Live Backend Logs</h2>
        <div className="space-y-2 max-h-72 overflow-y-auto">
          {(run?.logs ?? []).map((entry) => (
            <div key={entry.id} className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-900 dark:text-white">{entry.message}</p>
                <span className="text-xs text-gray-500 dark:text-gray-400">{new Date(entry.timestamp).toLocaleTimeString()}</span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{entry.step} | {entry.progress}%</p>
            </div>
          ))}
          {(!run || run.logs.length === 0) && (
            <p className="text-sm text-gray-500 dark:text-gray-400">No run logs yet. Click Start to execute the full backend pipeline.</p>
          )}
        </div>
      </div>

      {run?.dataset_info && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-4">
              <FileText className="w-5 h-5 text-primary-600" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Scenario & Dataset Evidence</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Dataset: {run.dataset_info.dataset_name}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Variant: {run.dataset_info.dataset_variant ?? 'N/A'}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Selected Hospital: {run.dataset_info.selected_hospital ?? run.hospital_id}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Federated Strategy: {run.dataset_info.federated_strategy ?? 'FedAvg'}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">Summary: {run.dataset_info.judge_summary ?? 'N/A'}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Total FL Samples: {run.dataset_info.total_samples}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">Golden Set Samples: {run.dataset_info.golden_set_samples}</p>
            {run.dataset_info.preprocessing_pipeline && run.dataset_info.preprocessing_pipeline.length > 0 && (
              <div className="mb-4">
                <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-2">Preprocessing Pipeline</p>
                <div className="space-y-1">
                  {run.dataset_info.preprocessing_pipeline.map((step, index) => (
                    <p key={`${step}_${index}`} className="text-xs text-gray-600 dark:text-gray-400">
                      {index + 1}. {step}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">FL Training Progress</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={roundChartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                <XAxis dataKey="round" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="avgAccuracy" stroke="#16a34a" strokeWidth={2} />
                <Line type="monotone" dataKey="avgLoss" stroke="#2563eb" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Golden Set Class Distribution</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={goldenClassDistributionData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                <XAxis dataKey="classId" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#0ea5e9" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="lg:col-span-3 bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Hospital Contribution Breakdown</h3>
            <div className="space-y-2">
              {run.dataset_info.hospitals.map((item) => (
                <div key={item.hospital_id} className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/50">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">{item.display_name ?? `Hospital ${item.hospital_id}`}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Samples: {item.samples}
                    {' | '}
                    Slice Start: {item.data_slice_start ?? 0}
                    {' | '}
                    Noise: {Number(item.noise_std ?? 0).toFixed(4)}
                    {' | '}
                    Classes: {Object.entries(item.class_distribution).map(([k, v]) => `${k}:${v}`).join(', ')}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {datasetInspection && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Dataset Anomaly Inspection</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
            <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
              <p className="text-xs text-gray-500 dark:text-gray-400">Total Samples</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">{datasetInspection.total_samples}</p>
            </div>
            <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
              <p className="text-xs text-gray-500 dark:text-gray-400">Anomaly Count / Percentage</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {datasetInspection.anomaly_count} / {datasetInspection.anomaly_percentage.toFixed(2)}%
              </p>
            </div>
            <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
              <p className="text-xs text-gray-500 dark:text-gray-400">Anomaly Types</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">
                {Array.isArray(datasetInspection.anomaly_type) ? datasetInspection.anomaly_type.join(', ') : String(datasetInspection.anomaly_type)}
              </p>
            </div>
          </div>

          <div className="mb-5">
            <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Class Distribution</p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {Object.entries(datasetInspection.class_distribution).map(([key, value]) => `${key}:${value}`).join(', ')}
            </p>
          </div>

          <div className="mb-5">
            <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Anomaly Sample Indices</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 break-all">
              {datasetInspection.anomaly_indices.join(', ')}
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
            {datasetInspection.sample_preview.map((sample) => (
              <div key={`preview_${sample.index}`} className="p-2 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
                {sample.image_base64 ? (
                  <img src={sample.image_base64} alt={`Sample ${sample.index}`} className="w-full h-auto rounded border border-gray-200 dark:border-gray-700" />
                ) : (
                  <div className="w-full h-24 rounded bg-gray-200 dark:bg-gray-700" />
                )}
                <p className="text-xs mt-1 text-gray-700 dark:text-gray-300">
                  #{sample.index} | y={sample.label}
                </p>
                <p className={cn(
                  "text-[11px] font-medium",
                  sample.is_anomalous ? "text-danger-600 dark:text-danger-400" : "text-success-600 dark:text-success-400",
                )}>
                  {sample.is_anomalous ? `ANOMALY (${sample.anomaly_tags.join(', ') || 'untyped'})` : 'CLEAN'}
                </p>
              </div>
            ))}
          </div>

          <div>
            <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Statistical Summary</p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Global mean/std: {datasetInspection.statistical_summary.global_mean.toFixed(6)} / {datasetInspection.statistical_summary.global_std.toFixed(6)}
              {' | '}
              Clean mean/std: {datasetInspection.statistical_summary.clean_mean.toFixed(6)} / {datasetInspection.statistical_summary.clean_std.toFixed(6)}
              {' | '}
              Anomalous mean/std: {datasetInspection.statistical_summary.anomalous_mean.toFixed(6)} / {datasetInspection.statistical_summary.anomalous_std.toFixed(6)}
              {' | '}
              Signal delta: {datasetInspection.statistical_summary.signal_delta_mean_abs.toFixed(6)}
            </p>
          </div>
        </div>
      )}

      {run?.status === 'completed' && finalSession && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-success-600" />
                <h4 className="text-sm text-gray-600 dark:text-gray-400">Final Verdict</h4>
              </div>
              <p className="text-xl font-bold text-gray-900 dark:text-white">{verdict}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-5 h-5 text-warning-600" />
                <h4 className="text-sm text-gray-600 dark:text-gray-400">Anomaly Score</h4>
              </div>
              <p className="text-xl font-bold text-gray-900 dark:text-white">{anomalyScore.toFixed(3)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-primary-600" />
                <h4 className="text-sm text-gray-600 dark:text-gray-400">Golden Accuracy</h4>
              </div>
              <p className="text-xl font-bold text-gray-900 dark:text-white">
                {(Number(goldenTest.accuracy ?? 0) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-2">
                <Database className="w-5 h-5 text-primary-600" />
                <h4 className="text-sm text-gray-600 dark:text-gray-400">Ledger TX</h4>
              </div>
              <p className="text-sm font-mono text-gray-900 dark:text-white break-all">
                {String(ledgerEntry.tx_id ?? sessionRecord.ledger_tx ?? '')}
              </p>
            </div>
          </div>

          <details className="mt-4 bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg border border-gray-200 dark:border-gray-700">
            <summary className="cursor-pointer text-sm font-semibold text-gray-900 dark:text-white">
              Score Breakdown
            </summary>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              IF Raw Score: {rawScore.toFixed(6)} → Calibrated: {calibratedScore.toFixed(3)} → Final: {anomalyScore.toFixed(3)}
            </p>
            {Boolean(attackInjection.enabled) && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Evaluation Override Applied: {Boolean(attackInjection.forced_reject) ? 'Yes' : 'No'}
              </p>
            )}
          </details>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Anomaly Threshold Comparison</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={anomalyChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#2563eb" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Model Artifact Metadata</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Architecture: {String(modelProfile.architecture ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Parameter Count: {String(modelProfile.parameter_count ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Model Size (MB): {String(modelProfile.model_size_mb ?? modelProfile.model_size ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Weight Hash: {String(modelProfile.weight_hash ?? modelProfile.model_hash ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Hospital ID: {String(modelProfile.hospital_id ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Training Round: {String(modelProfile.training_round ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Samples Tested: {String(datasetStats.samples_tested ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Evidence Hash: {String(sessionRecord.evidence_hash ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Detector Mode: {String(anomalyAnalysis.detector_mode ?? detectorAnalysis.detector_mode ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Raw Score: {Number(detectorAnalysis.raw_score ?? anomalyAnalysis.raw_score ?? 0).toFixed(6)}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Detector Confidence: {Number(detectorAnalysis.detector_confidence ?? anomalyAnalysis.detector_confidence ?? anomalyAnalysis.confidence ?? 0).toFixed(3)}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Distance From Baseline: {Number(detectorAnalysis.distance_from_baseline ?? anomalyAnalysis.distance_from_baseline ?? 0).toFixed(6)}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Decision Boundary: {JSON.stringify(detectorAnalysis.decision_boundary ?? anomalyAnalysis.decision_boundary ?? {})}</p>
              {Boolean(scenarioCalibration.enabled) && (
                <>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    Scenario Risk Bias: {Number(scenarioCalibration.risk_bias ?? 0).toFixed(3)}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    Calibrated Score: {Number(scenarioCalibration.post_calibration_score ?? anomalyScore).toFixed(3)}
                  </p>
                </>
              )}
              <p className="text-sm text-gray-600 dark:text-gray-400">Model File: {String(run.artifacts?.model_file_name ?? '')}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-4">
                <ListChecks className="w-5 h-5 text-primary-600" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Pipeline Timeline</h3>
              </div>
              <div className="space-y-2 max-h-72 overflow-y-auto">
                {timelineStatus.map((item, index) => (
                  <div key={`${String(item.step ?? 'step')}_${index}`} className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">{String(item.step ?? 'unknown_step')}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {String(item.status ?? 'completed')}
                      {' | '}
                      {item.timestamp ? new Date(String(item.timestamp)).toLocaleTimeString() : 'N/A'}
                    </p>
                    {Boolean(item.details) && (
                      <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-1 break-all">
                        {JSON.stringify(item.details)}
                      </p>
                    )}
                  </div>
                ))}
                {timelineStatus.length === 0 && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No timeline steps available.</p>
                )}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">SHAP Top Features</h3>
              <div className="space-y-2">
                {topFeatures.map((feature, index) => (
                  <div key={`${String(feature.feature ?? 'feature')}_${index}`} className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">{String(feature.feature ?? 'feature')}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Importance: {Number(feature.importance ?? 0).toFixed(6)}
                      {' | '}
                      Direction: {String(feature.direction ?? 'neutral')}
                    </p>
                  </div>
                ))}
                {topFeatures.length === 0 && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No SHAP top-feature data available.</p>
                )}
              </div>
            </div>
          </div>

          <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Verdict Narrative</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              {String(sessionRecord.verdict_reasoning ?? 'No narrative available.')}
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Per-Class Precision / Recall</h3>
              <div className="space-y-2 max-h-72 overflow-y-auto">
                {Object.keys(perClassPrecision).map((classId) => (
                  <div key={classId} className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">Class {classId}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Precision: {Number(perClassPrecision[classId] ?? 0).toFixed(3)}
                      {' | '}
                      Recall: {Number(perClassRecall[classId] ?? 0).toFixed(3)}
                    </p>
                  </div>
                ))}
                {Object.keys(perClassPrecision).length === 0 && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No per-class metrics available.</p>
                )}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Confusion Matrix</h3>
              <div className="space-y-2">
                {confusionMatrix.map((row, index) => (
                  <p key={`cm_row_${index}`} className="text-sm font-mono text-gray-700 dark:text-gray-300">
                    {JSON.stringify(row)}
                  </p>
                ))}
                {confusionMatrix.length === 0 && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No confusion matrix available.</p>
                )}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Fingerprint Vector Transparency</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 break-all">
                Raw Vector: {JSON.stringify(fingerprintAnalysis.raw_vector ?? [])}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 break-all">
                Normalized Vector: {JSON.stringify(fingerprintAnalysis.normalized_vector ?? [])}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 break-all">
                Baseline Centroid: {JSON.stringify(fingerprintAnalysis.baseline_centroid ?? [])}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                Euclidean Distance: {Number(fingerprintAnalysis.euclidean_distance ?? 0).toFixed(8)}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Mahalanobis Distance: {Number(fingerprintAnalysis.mahalanobis_distance ?? 0).toFixed(8)}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Detector + Ledger Transparency</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 break-all">
                Score Interpretation: {String(detectorAnalysis.score_interpretation_logic ?? anomalyAnalysis.score_interpretation_logic ?? 'N/A')}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 break-all">
                Past Score Distribution: {JSON.stringify(detectorAnalysis.score_distribution_past_sessions ?? anomalyAnalysis.score_distribution_past_sessions ?? {})}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">Ledger TX: {String(ledgerEntry.tx_id ?? 'N/A')}</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">Ledger Timestamp: {String(ledgerEntry.timestamp ?? 'N/A')}</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 break-all">Ledger Evidence Hash: {String(ledgerEntry.evidence_hash ?? 'N/A')}</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 break-all">Ledger Update Hash: {String(ledgerEntry.update_hash ?? 'N/A')}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle className="w-5 h-5 text-warning-600" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Warnings</h3>
              </div>
              <div className="space-y-2">
                {warnings.map((warning, index) => (
                  <p key={`${warning}_${index}`} className="text-sm text-gray-700 dark:text-gray-300">
                    {index + 1}. {warning}
                  </p>
                ))}
                {warnings.length === 0 && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No warnings generated.</p>
                )}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recommendations</h3>
              <div className="space-y-2">
                {recommendations.map((recommendation, index) => (
                  <p key={`${recommendation}_${index}`} className="text-sm text-gray-700 dark:text-gray-300">
                    {index + 1}. {recommendation}
                  </p>
                ))}
                {recommendations.length === 0 && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No recommendations generated.</p>
                )}
              </div>
            </div>
          </div>

          {Boolean(attackInjection.enabled) && (
            <div className="mt-6 bg-danger-50 dark:bg-danger-900/20 border border-danger-300 dark:border-danger-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-danger-700 dark:text-danger-300 mb-2">Evaluation Mode Audit</h3>
              <p className="text-sm text-danger-700 dark:text-danger-300">Strategy: {String(attackInjection.strategy ?? 'N/A')}</p>
              <p className="text-sm text-danger-700 dark:text-danger-300">Base Score: {Number(attackInjection.base_score ?? 0).toFixed(3)}</p>
              <p className="text-sm text-danger-700 dark:text-danger-300">Boost Applied: {Number(attackInjection.anomaly_boost ?? 0).toFixed(3)}</p>
              <p className="text-sm text-danger-700 dark:text-danger-300">
                Ground Truth Override: {Boolean(attackInjection.forced_reject) ? 'Active (evaluation mode)' : 'Inactive'}
              </p>
            </div>
          )}
        </>
      )}

      {run?.status === 'failed' && (
        <div className="bg-danger-50 dark:bg-danger-900/20 border border-danger-300 dark:border-danger-700 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-danger-700 dark:text-danger-300 mb-2">Pipeline Failed</h3>
          <p className="text-sm text-danger-700 dark:text-danger-300">{run.error || run.message}</p>
        </div>
      )}
    </div>
  );
}
