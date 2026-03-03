import { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ledgerAPI, sentinelAPI } from '../services/api';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { formatDate, formatScore, getVerdictColor } from '../utils/format';
import type { EvidencePayload, Transaction, VerificationResult } from '../types';

export function AttackEvidence() {
  const [searchParams] = useSearchParams();
  const [payload, setPayload] = useState<EvidencePayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [recentRejected, setRecentRejected] = useState<Transaction[]>([]);
  const [verifyResult, setVerifyResult] = useState<VerificationResult | null>(null);

  const sessionId = searchParams.get('session');
  const txId = searchParams.get('tx');

  const loadEvidence = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (sessionId) {
        const data = await sentinelAPI.getEvidence(sessionId);
        setPayload(data);
      } else if (txId) {
        const data = await sentinelAPI.getEvidenceByTransaction(txId);
        setPayload(data);
      } else {
        const txs = await ledgerAPI.getTransactions({ verdict: 'REJECTED', per_page: 20 });
        setRecentRejected(txs);
        setPayload(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load evidence report');
    } finally {
      setLoading(false);
    }
  }, [sessionId, txId]);

  useEffect(() => {
    void loadEvidence();
  }, [loadEvidence]);

  const chartData = useMemo(
    () =>
      (payload?.behavioral_fingerprint_comparison ?? []).slice(0, 10).map((item) => ({
        feature: item.feature,
        baseline: item.baseline,
        submitted: item.submitted,
      })),
    [payload],
  );

  const handleVerify = useCallback(async () => {
    if (!payload?.immutable_record.tx_id) {
      return;
    }
    try {
      const result = await ledgerAPI.verifyTransaction(payload.immutable_record.tx_id);
      setVerifyResult(result);
    } catch (err) {
      setVerifyResult({ verified: false, message: err instanceof Error ? err.message : 'Verification failed' });
    }
  }, [payload]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <LoadingSpinner />
      </div>
    );
  }

  if (!payload) {
    return (
      <div className="max-w-5xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Attack Evidence</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Select a rejected transaction from the ledger to inspect full evidence.
          </p>
        </div>
        {error && <Alert type="error" message={error} />}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg space-y-3">
          {recentRejected.map((tx) => (
            <a
              key={tx.transaction_id}
              href={`/attack-evidence?tx=${encodeURIComponent(tx.transaction_id)}`}
              className="block p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-900/40"
            >
              <p className="text-sm font-mono text-gray-900 dark:text-white">{tx.transaction_id}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Hospital {tx.hospital_id} | Score {formatScore(tx.anomaly_score)} | {formatDate(tx.timestamp)}
              </p>
            </a>
          ))}
          {recentRejected.length === 0 && (
            <p className="text-sm text-gray-500 dark:text-gray-400">No rejected transactions found yet.</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
        <p className="text-lg font-semibold text-gray-900 dark:text-white">{payload.accusation}</p>
        <div className="mt-3 flex flex-wrap gap-3 items-center">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getVerdictColor(String(payload.verdict))}`}>
            {payload.verdict}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">Hospital: {payload.hospital_id}</span>
          <span className="text-sm text-gray-600 dark:text-gray-400">TX: {payload.ledger_tx_id}</span>
          <span className="text-sm text-gray-600 dark:text-gray-400">{formatDate(payload.timestamp)}</span>
        </div>
      </div>

      {error && <Alert type="error" message={error} />}

      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">Dataset Contamination Evidence</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
          Total tested: {payload.dataset_contamination_evidence.total_samples_tested} | Anomalies detected:{' '}
          {payload.dataset_contamination_evidence.anomalies_detected} (
          {payload.dataset_contamination_evidence.anomaly_percentage.toFixed(2)}%)
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
          {payload.dataset_contamination_evidence.sample_preview.map((sample) => (
            <div key={`evidence_sample_${sample.index}`} className="p-1 rounded border border-gray-200 dark:border-gray-700">
              <img src={sample.image_base64} alt={`Evidence sample ${sample.index}`} className="w-full h-auto rounded" />
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">Behavioral Deviation from Clean Baseline</h3>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
            <XAxis dataKey="feature" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="baseline" fill="#9ca3af" />
            <Bar dataKey="submitted" fill="#ef4444" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Anomaly Diagnosis</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Score {formatScore(payload.anomaly_diagnosis.score)} vs threshold {formatScore(payload.anomaly_diagnosis.threshold)}.
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Anomalous percentile estimate: {payload.anomaly_diagnosis.anomalous_percentile_estimate.toFixed(2)}%
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Detector mode: {payload.anomaly_diagnosis.detector_mode}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Immutable Record</h3>
          <p className="text-sm font-mono text-gray-900 dark:text-white break-all mb-2">{payload.immutable_record.tx_id}</p>
          <p className="text-xs font-mono text-gray-600 dark:text-gray-400 break-all mb-1">
            Evidence Hash: {payload.immutable_record.evidence_hash}
          </p>
          <p className="text-xs font-mono text-gray-600 dark:text-gray-400 break-all mb-3">
            Update Hash: {payload.immutable_record.update_hash}
          </p>
          <button
            onClick={() => void handleVerify()}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700"
          >
            Verify Integrity
          </button>
          {verifyResult && (
            <p className="mt-3 text-sm text-gray-700 dark:text-gray-300">
              {verifyResult.verified ? (
                <span className="inline-flex items-center gap-1"><CheckCircle2 className="w-4 h-4 text-emerald-500" /> Evidence Intact</span>
              ) : (
                <span className="inline-flex items-center gap-1"><AlertTriangle className="w-4 h-4 text-amber-500" /> {verifyResult.message}</span>
              )}
            </p>
          )}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">Recommended Actions</h3>
        <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
          {payload.recommendations.map((item, idx) => (
            <li key={`${item}_${idx}`} className="flex items-start gap-2">
              <ShieldAlert className="w-4 h-4 mt-0.5 text-danger-500" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
