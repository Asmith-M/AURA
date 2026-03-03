import { useCallback, useEffect, useMemo, useState } from 'react';
import { Copy, Link2, ShieldCheck } from 'lucide-react';
import { ledgerAPI } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { formatDate, formatScore, getVerdictColor } from '../utils/format';
import type { LedgerFilters, LedgerStats, Transaction, VerificationResult, Verdict } from '../types';

function copyText(value: string): void {
  void navigator.clipboard?.writeText(value);
}

export function Ledger() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [stats, setStats] = useState<LedgerStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filters, setFilters] = useState<LedgerFilters>({ verdict: 'ALL', per_page: 200 });
  const [verifyMap, setVerifyMap] = useState<Record<string, VerificationResult>>({});

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [txData, statsData] = await Promise.all([
        ledgerAPI.getTransactions(filters),
        ledgerAPI.getLedgerStats(),
      ]);
      setTransactions(txData);
      setStats(statsData);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const handleVerify = useCallback(async (txId: string) => {
    const result = await ledgerAPI.verifyTransaction(txId);
    setVerifyMap((prev) => ({ ...prev, [txId]: result }));
  }, []);

  const timelineStats = useMemo(() => {
    if (transactions.length === 0) {
      return { genesis: null as string | null, latest: null as string | null };
    }
    const sorted = [...transactions].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    return {
      genesis: sorted[0]?.timestamp ?? null,
      latest: sorted[sorted.length - 1]?.timestamp ?? null,
    };
  }, [transactions]);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Blockchain Ledger</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Immutable audit records with on-demand evidence integrity verification.
        </p>
      </div>

      {stats && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 shadow-lg">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Ledger Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
            <div><p className="text-gray-500 dark:text-gray-400">Total</p><p className="font-semibold text-gray-900 dark:text-white">{stats.total_transactions}</p></div>
            <div><p className="text-gray-500 dark:text-gray-400">Approved</p><p className="font-semibold text-emerald-600">{stats.approved}</p></div>
            <div><p className="text-gray-500 dark:text-gray-400">Rejected</p><p className="font-semibold text-rose-600">{stats.rejected}</p></div>
            <div><p className="text-gray-500 dark:text-gray-400">Genesis Block Time</p><p className="font-semibold text-gray-900 dark:text-white">{timelineStats.genesis ? formatDate(timelineStats.genesis) : 'N/A'}</p></div>
            <div><p className="text-gray-500 dark:text-gray-400">Latest Block Time</p><p className="font-semibold text-gray-900 dark:text-white">{timelineStats.latest ? formatDate(timelineStats.latest) : 'N/A'}</p></div>
          </div>
        </div>
      )}

      <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 shadow-lg grid grid-cols-1 md:grid-cols-5 gap-3">
        <select
          value={filters.hospital_id ?? ''}
          onChange={(event) =>
            setFilters((prev) => ({
              ...prev,
              hospital_id: event.target.value ? Number(event.target.value) : undefined,
            }))
          }
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        >
          <option value="">All Hospitals</option>
          {Array.from({ length: 10 }, (_, idx) => idx + 1).map((id) => (
            <option key={`hospital_filter_${id}`} value={id}>
              Hospital {id}
            </option>
          ))}
        </select>

        <select
          value={filters.verdict ?? 'ALL'}
          onChange={(event) => setFilters((prev) => ({ ...prev, verdict: event.target.value as Verdict | 'ALL' }))}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        >
          <option value="ALL">All Verdicts</option>
          <option value="APPROVED">Approved</option>
          <option value="REJECTED">Rejected</option>
          <option value="PROCESSING">Processing</option>
          <option value="ERROR">Error</option>
        </select>

        <input
          type="date"
          value={filters.start_date ?? ''}
          onChange={(event) => setFilters((prev) => ({ ...prev, start_date: event.target.value || undefined }))}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        />
        <input
          type="date"
          value={filters.end_date ?? ''}
          onChange={(event) => setFilters((prev) => ({ ...prev, end_date: event.target.value || undefined }))}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        />
        <button
          onClick={() => setFilters({ verdict: 'ALL', per_page: 200 })}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200"
        >
          Reset
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center min-h-[40vh]">
          <LoadingSpinner />
        </div>
      ) : (
        <div className="relative space-y-4">
          {transactions.map((tx, index) => {
            const verify = verifyMap[tx.transaction_id];
            return (
              <div key={tx.transaction_id} className="relative pl-8">
                {index < transactions.length - 1 && (
                  <div className="absolute left-3 top-10 bottom-[-20px] w-[2px] bg-gray-300 dark:bg-gray-700" />
                )}
                <div className="absolute left-0 top-6 w-6 h-6 rounded-full bg-primary-600 text-white flex items-center justify-center">
                  <Link2 className="w-3 h-3" />
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 shadow-lg">
                  <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Transaction ID</p>
                      <p className="font-mono text-sm text-gray-900 dark:text-white break-all">{tx.transaction_id}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 text-xs rounded-full bg-gray-100 dark:bg-gray-900/40 text-gray-700 dark:text-gray-300">
                        Hospital {tx.hospital_id}
                      </span>
                      <span className={`px-2 py-1 text-xs rounded-full ${getVerdictColor(tx.verdict)}`}>{tx.verdict}</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Anomaly Score</p>
                      <p className="font-semibold text-gray-900 dark:text-white">{formatScore(tx.anomaly_score)}</p>
                    </div>
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Timestamp</p>
                      <p className="font-semibold text-gray-900 dark:text-white">{formatDate(tx.timestamp)}</p>
                    </div>
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Update Hash</p>
                      <p className="font-mono text-xs text-gray-900 dark:text-white break-all">{tx.update_hash}</p>
                    </div>
                  </div>

                  <div className="mt-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Evidence Hash</p>
                      <button onClick={() => copyText(tx.evidence_hash ?? '')} className="text-xs inline-flex items-center gap-1 text-primary-600 hover:text-primary-700">
                        <Copy className="w-3 h-3" />
                        Copy
                      </button>
                    </div>
                    <p className="font-mono text-xs text-gray-900 dark:text-white break-all">{tx.evidence_hash}</p>
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    <button
                      onClick={() => void handleVerify(tx.transaction_id)}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 text-sm"
                    >
                      <ShieldCheck className="w-4 h-4" />
                      Verify Integrity
                    </button>
                    {tx.verdict === 'REJECTED' && (
                      <a
                        href={`/attack-evidence?tx=${encodeURIComponent(tx.transaction_id)}`}
                        className="px-3 py-2 rounded-lg border border-danger-300 text-danger-700 dark:text-danger-300 dark:border-danger-700 hover:bg-danger-50 dark:hover:bg-danger-900/20 text-sm"
                      >
                        View Evidence
                      </a>
                    )}
                  </div>

                  {verify && (
                    <p className="mt-2 text-xs text-gray-600 dark:text-gray-300">
                      {verify.verified ? '✓ Evidence Intact' : `✗ ${verify.message}`}
                    </p>
                  )}
                </div>
              </div>
            );
          })}

          {transactions.length === 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-8 text-center border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400">
              No transactions match the active filters.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
