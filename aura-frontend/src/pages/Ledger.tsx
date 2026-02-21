/**
 * Ledger page - Blockchain transaction history
 */
import { useState, useEffect, useCallback } from 'react';
import { Database, CheckCircle, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { ledgerAPI } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { formatDate, truncateHash, getVerdictColor } from '../utils/format';
import type { Transaction, LedgerStats, Verdict, LedgerFilters } from '../types';

export function Ledger() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [stats, setStats] = useState<LedgerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<LedgerFilters>({
    hospital_id: undefined,
    verdict: 'ALL',
    page: 1,
    per_page: 20,
  });

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [txData, statsData] = await Promise.all([
        ledgerAPI.getTransactions(filters),
        ledgerAPI.getLedgerStats(),
      ]);
      setTransactions(txData);
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load ledger data:', error);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const handleVerify = async (txId: string) => {
    try {
      const result = await ledgerAPI.verifyTransaction(txId);
      alert(result.message);
    } catch {
      alert('Verification failed');
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Blockchain Ledger</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Immutable transaction history and blockchain verification
        </p>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <Database className="w-8 h-8 text-primary-600 mb-2" />
            <p className="text-sm text-gray-500 dark:text-gray-400">Total Transactions</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.total_transactions}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <CheckCircle className="w-8 h-8 text-success-600 mb-2" />
            <p className="text-sm text-gray-500 dark:text-gray-400">Approved</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.approved}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <XCircle className="w-8 h-8 text-danger-600 mb-2" />
            <p className="text-sm text-gray-500 dark:text-gray-400">Rejected</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.rejected}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Security Rate</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.security_rate.toFixed(1)}%
            </p>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Hospital ID
            </label>
            <select
              value={filters.hospital_id ?? ''}
              onChange={(e) =>
                setFilters({ ...filters, hospital_id: e.target.value ? parseInt(e.target.value) : undefined })
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="">All Hospitals</option>
              {Array.from({ length: 10 }, (_, i) => i + 1).map((id) => (
                <option key={id} value={id}>
                  Hospital {id}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Verdict
            </label>
            <select
              value={filters.verdict}
              onChange={(e) => setFilters({ ...filters, verdict: e.target.value as Verdict | 'ALL' })}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="ALL">All Verdicts</option>
              <option value="APPROVED">Approved</option>
              <option value="REJECTED">Rejected</option>
              <option value="PROCESSING">Processing</option>
              <option value="ERROR">Error</option>
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={() => setFilters({ hospital_id: undefined, verdict: 'ALL', page: 1, per_page: 20 })}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>

      {/* Transactions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="overflow-x-auto">
          {loading ? (
            <div className="flex items-center justify-center p-12">
              <LoadingSpinner size="lg" />
            </div>
          ) : (
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Transaction ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Hospital
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Hash
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Verdict
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Timestamp
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {transactions.map((tx) => (
                  <motion.tr
                    key={tx.transaction_id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="hover:bg-gray-50 dark:hover:bg-gray-700/50"
                  >
                    <td className="px-6 py-4 text-sm font-mono text-gray-900 dark:text-white">
                      {truncateHash(tx.transaction_id, 8, 6)}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                      Hospital {tx.hospital_id}
                    </td>
                    <td className="px-6 py-4 text-sm font-mono text-gray-500 dark:text-gray-400">
                      {truncateHash(tx.update_hash, 6, 4)}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getVerdictColor(tx.verdict)}`}>
                        {tx.verdict}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                      {tx.anomaly_score.toFixed(3)}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                      {formatDate(tx.timestamp)}
                    </td>
                    <td className="px-6 py-4">
                      <button
                        onClick={() => handleVerify(tx.transaction_id)}
                        className="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 font-medium"
                      >
                        Verify
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Blockchain Info */}
      {stats && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Blockchain Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Total Blocks</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {stats.blockchain_info.total_blocks}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Latest Hash</p>
              <p className="text-lg font-mono font-semibold text-gray-900 dark:text-white">
                {truncateHash(stats.blockchain_info.latest_hash)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Pending Transactions</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {stats.blockchain_info.pending_transactions}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
