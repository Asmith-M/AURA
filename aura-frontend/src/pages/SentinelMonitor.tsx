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
import type { PipelineRunStatus } from '../types';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { Alert } from '../components/Alert';
import { cn } from '../utils/cn';

export function SentinelMonitor() {
  const [run, setRun] = useState<PipelineRunStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

  useEffect(() => {
    void loadLatestRun();
  }, [loadLatestRun]);

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
  const modelProfile = (sessionRecord.model_profile ?? {}) as Record<string, unknown>;
  const datasetStats = (sessionRecord.dataset_stats ?? {}) as Record<string, unknown>;
  const goldenTest = (sessionRecord.golden_test ?? {}) as Record<string, unknown>;
  const shapAnalysis = (sessionRecord.shap_analysis ?? {}) as Record<string, unknown>;
  const isRunning = run?.status === 'queued' || run?.status === 'running';
  const verdict = String(sessionRecord.verdict ?? run?.status?.toUpperCase() ?? 'PROCESSING');
  const anomalyScore = Number(anomalyAnalysis.anomaly_score ?? 0);
  const threshold = Number(anomalyAnalysis.threshold ?? 0.72);

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
  const timelineStatus = Array.isArray(sessionRecord.timeline_status)
    ? (sessionRecord.timeline_status as Record<string, unknown>[])
    : [];
  const topFeatures = Array.isArray(shapAnalysis.top_features)
    ? (shapAnalysis.top_features as Record<string, unknown>[])
    : [];
  const confusionMatrix = Array.isArray(goldenTest.confusion_matrix)
    ? (goldenTest.confusion_matrix as unknown[])
    : [];
  const perClassPrecision = (goldenTest.per_class_precision ?? {}) as Record<string, unknown>;
  const perClassRecall = (goldenTest.per_class_recall ?? {}) as Record<string, unknown>;
  const attackInjection = (anomalyAnalysis.attack_injection ?? {}) as Record<string, unknown>;
  const scenarioCalibration = (anomalyAnalysis.scenario_calibration ?? {}) as Record<string, unknown>;

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
                {String(sessionRecord.ledger_tx ?? '')}
              </p>
            </div>
          </div>

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
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Parameters: {String(modelProfile.parameters ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Model Hash: {String(modelProfile.model_hash ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Samples Tested: {String(datasetStats.samples_tested ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Evidence Hash: {String(sessionRecord.evidence_hash ?? 'N/A')}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Detector Mode: {String(anomalyAnalysis.detector_mode ?? 'N/A')}</p>
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
              <h3 className="text-lg font-semibold text-danger-700 dark:text-danger-300 mb-2">Attack Injection Diagnostics</h3>
              <p className="text-sm text-danger-700 dark:text-danger-300">Strategy: {String(attackInjection.strategy ?? 'N/A')}</p>
              <p className="text-sm text-danger-700 dark:text-danger-300">Base Score: {Number(attackInjection.base_score ?? 0).toFixed(3)}</p>
              <p className="text-sm text-danger-700 dark:text-danger-300">Boost Applied: {Number(attackInjection.anomaly_boost ?? 0).toFixed(3)}</p>
              <p className="text-sm text-danger-700 dark:text-danger-300">Forced Reject: {String(Boolean(attackInjection.forced_reject))}</p>
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
