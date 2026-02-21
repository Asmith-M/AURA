/**
 * Analytics page - Time-based metrics and trends
 */
import { useState, useEffect, useCallback } from 'react';
import { Calendar, TrendingUp, Activity } from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { analyticsAPI } from '../services/api';
import { ChartContainer } from '../components/ChartContainer';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { TimeRange, DailyMetric, AttackMetric, HospitalPerformance } from '../types';

export function Analytics() {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [dailyMetrics, setDailyMetrics] = useState<DailyMetric[]>([]);
  const [attackMetrics, setAttackMetrics] = useState<AttackMetric[]>([]);
  const [hospitalPerformance, setHospitalPerformance] = useState<HospitalPerformance[]>([]);
  const [loading, setLoading] = useState(true);

  const loadAnalytics = useCallback(async () => {
    setLoading(true);
    try {
      const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
      const [daily, attacks, hospitals] = await Promise.all([
        analyticsAPI.getDailyMetrics(days),
        analyticsAPI.getAttackMetrics(),
        analyticsAPI.getHospitalPerformance(),
      ]);
      setDailyMetrics(daily);
      setAttackMetrics(attacks);
      setHospitalPerformance(hospitals);
    } catch (error) {
      console.error('Failed to load analytics:', error);
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    void loadAnalytics();
  }, [loadAnalytics]);

  const activityData = dailyMetrics.map((m) => ({
    date: new Date(m.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    submissions: m.submissions,
    approved: m.approved,
    rejected: m.rejected,
  }));

  const approvalRateData = dailyMetrics.map((m) => ({
    date: new Date(m.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    rate: m.submissions > 0 ? (m.approved / m.submissions) * 100 : 0,
  }));

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Analytics</h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Time-based metrics and performance trends
          </p>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2">
          {(['7d', '30d', '90d'] as TimeRange[]).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                timeRange === range
                  ? 'bg-primary-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              {range === '7d' ? '7 Days' : range === '30d' ? '30 Days' : '90 Days'}
            </button>
          ))}
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
              <Activity className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Total Submissions</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {dailyMetrics.reduce((sum, m) => sum + m.submissions, 0)}
              </p>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-success-100 dark:bg-success-900/20 rounded-lg">
              <TrendingUp className="w-6 h-6 text-success-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Avg Approval Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {(
                  (dailyMetrics.reduce((sum, m) => sum + m.approved, 0) /
                    dailyMetrics.reduce((sum, m) => sum + m.submissions, 0)) *
                  100
                ).toFixed(1)}
                %
              </p>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-danger-100 dark:bg-danger-900/20 rounded-lg">
              <Calendar className="w-6 h-6 text-danger-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Rejections</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {dailyMetrics.reduce((sum, m) => sum + m.rejected, 0)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Daily Events */}
        <ChartContainer title="Daily Events">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={activityData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis dataKey="date" style={{ fontSize: '12px' }} />
              <YAxis style={{ fontSize: '12px' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="submissions" stroke="#4f46e5" strokeWidth={2} />
              <Line type="monotone" dataKey="approved" stroke="#10b981" strokeWidth={2} />
              <Line type="monotone" dataKey="rejected" stroke="#f43f5e" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        {/* Attack Types */}
        <ChartContainer title="Attack Types Distribution">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={attackMetrics}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis dataKey="type" angle={-45} textAnchor="end" height={100} style={{ fontSize: '11px' }} />
              <YAxis style={{ fontSize: '12px' }} />
              <Tooltip />
              <Bar dataKey="count" fill="#f43f5e" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>

        {/* Approval Rates */}
        <ChartContainer title="Approval Rate Trend">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={approvalRateData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis dataKey="date" style={{ fontSize: '12px' }} />
              <YAxis style={{ fontSize: '12px' }} />
              <Tooltip />
              <Area type="monotone" dataKey="rate" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>

        {/* Top Hospitals */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Top Performing Hospitals
          </h3>
          <div className="space-y-3">
            {hospitalPerformance
              .sort((a, b) => b.approval_rate - a.approval_rate)
              .slice(0, 5)
              .map((hospital, index) => (
                <div
                  key={hospital.hospital_id}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/20 flex items-center justify-center text-primary-600 font-semibold text-sm">
                      #{index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">{hospital.hospital_name}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {hospital.submissions} submissions
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {hospital.approval_rate.toFixed(1)}%
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {(hospital.avg_accuracy * 100).toFixed(1)}% accuracy
                    </p>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
}
