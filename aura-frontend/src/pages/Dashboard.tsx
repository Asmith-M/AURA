/**
 * Dashboard page - Main overview with stats and charts
 */
import { useCallback, useEffect, useState } from 'react';
import { Activity, CheckCircle, XCircle, TrendingUp } from 'lucide-react';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { StatCard } from '../components/StatCard';
import { ChartContainer } from '../components/ChartContainer';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { analyticsAPI, ledgerAPI, sentinelAPI, systemAPI } from '../services/api';
import { dashboardDemoDailyMetrics, dashboardDemoStats, dashboardDemoTransactions } from '../data/dashboardDemo';
import { formatDate, formatPercentage, getVerdictColor } from '../utils/format';
import type { DailyMetric, DashboardStats, SystemStatus, Transaction } from '../types';

export function Dashboard() {
  const useDashboardDemo = (import.meta.env.VITE_DASHBOARD_DEMO ?? 'false') === 'true';
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [dailyMetrics, setDailyMetrics] = useState<DailyMetric[]>([]);
  const [recentSubmissions, setRecentSubmissions] = useState<Transaction[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const loadStats = useCallback(async () => {
    setLoading(true);
    try {
      if (useDashboardDemo) {
        setStats(dashboardDemoStats);
        setRecentSubmissions(dashboardDemoTransactions);
        setDailyMetrics(dashboardDemoDailyMetrics);
        setSystemStatus({ backend: 'online', model_loaded: true, last_run: new Date().toISOString() });
        return;
      }

      const [statsData, txData, dailyData, statusData] = await Promise.all([
        sentinelAPI.getDetectionStats(),
        ledgerAPI.getTransactions({ per_page: 10 }),
        analyticsAPI.getDailyMetrics(7),
        systemAPI.getStatus(),
      ]);
      setStats(statsData);
      setRecentSubmissions(txData);
      setDailyMetrics(dailyData);
      setSystemStatus(statusData);
    } catch (error) {
      console.error('Failed to load stats:', error);
    } finally {
      setLoading(false);
    }
  }, [useDashboardDemo]);

  useEffect(() => {
    void loadStats();
  }, [loadStats]);

  if (loading || !stats) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  const activityData = dailyMetrics.map((m) => ({
    date: new Date(m.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    submissions: m.submissions,
  }));

  const securityData = [
    { name: 'Approved', value: stats.approved, color: '#10b981' },
    { name: 'Rejected', value: stats.rejected, color: '#f43f5e' },
    { name: 'Processing', value: stats.processing, color: '#f59e0b' },
  ];

  return (
    <div className="space-y-6">
      {/* Welcome banner */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-xl bg-gradient-to-r from-primary-600 to-primary-800 p-8 text-white"
      >
        <div className="relative z-10">
          <h1 className="text-3xl font-bold mb-2">Welcome to AURA Dashboard</h1>
          <p className="text-primary-100">
            Federated Learning Security System - Real-time monitoring and analysis
          </p>
          {useDashboardDemo && (
            <p className="mt-2 text-xs text-primary-200">Dashboard currently showing curated demo preview data.</p>
          )}
        </div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-32 -mt-32" />
        <div className="absolute bottom-0 right-20 w-48 h-48 bg-white/10 rounded-full -mb-24" />
      </motion.div>

      {systemStatus && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 shadow-lg">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">System Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700">
              <p className="text-gray-500 dark:text-gray-400">Backend</p>
              <p className={`font-semibold ${systemStatus.backend === 'online' ? 'text-emerald-600' : 'text-red-600'}`}>
                {systemStatus.backend === 'online' ? 'Online' : 'Offline'}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700">
              <p className="text-gray-500 dark:text-gray-400">Isolation Forest</p>
              <p className={`font-semibold ${systemStatus.model_loaded ? 'text-emerald-600' : 'text-amber-600'}`}>
                {systemStatus.model_loaded ? 'Loaded' : 'Not Loaded'}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900/40 border border-gray-200 dark:border-gray-700">
              <p className="text-gray-500 dark:text-gray-400">Last Pipeline Run</p>
              <p className="font-semibold text-gray-900 dark:text-white">
                {systemStatus.last_run ? formatDate(systemStatus.last_run) : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Submissions"
          value={stats.total_submissions}
          icon={Activity}
          trend={{ value: 12.5, isPositive: true }}
          color="primary"
        />
        <StatCard
          title="Approved Models"
          value={stats.approved}
          icon={CheckCircle}
          trend={{ value: 8.3, isPositive: true }}
          color="success"
        />
        <StatCard
          title="Rejected Models"
          value={stats.rejected}
          icon={XCircle}
          trend={{ value: 3.2, isPositive: false }}
          color="danger"
        />
        <StatCard
          title="Attack Detection Rate"
          value={formatPercentage(stats.security_effectiveness, 0)}
          icon={TrendingUp}
          color="success"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity Chart */}
        <ChartContainer title="Recent Activity (7 Days)">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={activityData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis
                dataKey="date"
                className="text-xs"
                stroke="currentColor"
                style={{ fontSize: '12px' }}
              />
              <YAxis
                className="text-xs"
                stroke="currentColor"
                style={{ fontSize: '12px' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="submissions"
                stroke="#4f46e5"
                strokeWidth={2}
                dot={{ fill: '#4f46e5', r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        {/* Security Metrics Pie Chart */}
        <ChartContainer title="Security Metrics">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={securityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${percent ? (percent * 100).toFixed(0) : 0}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {securityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>

      {/* Recent Submissions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Ledger Activity</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Transaction ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Hospital
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Verdict
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Timestamp
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {recentSubmissions.map((submission) => (
                <tr
                  key={submission.transaction_id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700/50"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {submission.transaction_id.substring(0, 16)}...
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    Hospital {submission.hospital_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded-full ${getVerdictColor(
                        submission.verdict
                      )}`}
                    >
                      {submission.verdict}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {submission.anomaly_score.toFixed(3)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {formatDate(submission.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
