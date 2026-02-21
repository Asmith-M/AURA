/**
 * Reports page - SHAP behavioral analysis reports
 */
import { useState, useEffect } from 'react';
import { FileText, Eye } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { sentinelAPI } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ChartContainer } from '../components/ChartContainer';
import { formatDate, formatScore, getVerdictColor } from '../utils/format';
import type { BehavioralReport } from '../types';

export function Reports() {
  const [reports, setReports] = useState<BehavioralReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<BehavioralReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      const data = await sentinelAPI.getReports(12);
      setReports(data);
    } catch (error) {
      console.error('Failed to load reports:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (selectedReport) {
    const fingerprint = selectedReport.analysis_result.fingerprint;
    const fingerprintData = [
      { name: 'Mean Importance', value: fingerprint.mean_importance_global },
      { name: 'Std Importance', value: fingerprint.std_importance_global },
      { name: 'Max Importance', value: fingerprint.max_importance_global },
      { name: 'Entropy Mean', value: fingerprint.entropy_mean / 3 }, // Normalize
      { name: 'Variance Stability', value: fingerprint.variance_stability },
      { name: 'Feature Consistency', value: fingerprint.feature_consistency },
      { name: 'Prediction Stability', value: fingerprint.prediction_stability },
    ];

    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Behavioral Report</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">
              Detailed SHAP analysis for submission {selectedReport.submission_id.substring(0, 16)}...
            </p>
          </div>
          <button
            onClick={() => setSelectedReport(null)}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Back to List
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Verdict</h3>
            <span
              className={`inline-block px-3 py-1 text-sm font-medium rounded-full ${getVerdictColor(
                selectedReport.detection_result.verdict
              )}`}
            >
              {selectedReport.detection_result.verdict}
            </span>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Anomaly Score</h3>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {formatScore(selectedReport.detection_result.anomaly_score)}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Test Accuracy</h3>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {(selectedReport.test_accuracy * 100).toFixed(2)}%
            </p>
          </div>
        </div>

        <ChartContainer title="Behavioral Fingerprint">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={fingerprintData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} style={{ fontSize: '11px' }} />
              <YAxis style={{ fontSize: '12px' }} />
              <Tooltip />
              <Bar dataKey="value" fill="#4f46e5" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Security Assessment</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Is Anomalous</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {selectedReport.detection_result.is_anomalous ? 'Yes' : 'No'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Confidence</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {(selectedReport.detection_result.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Sample Size</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {selectedReport.analysis_result.sample_size}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">SHAP Method</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {selectedReport.analysis_result.shap_method_used}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Behavioral Reports</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          SHAP analysis reports for all model submissions
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {reports.map((report) => (
          <motion.div
            key={report.submission_id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ y: -4 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 cursor-pointer"
            onClick={() => setSelectedReport(report)}
          >
            <div className="flex items-start justify-between mb-4">
              <FileText className="w-8 h-8 text-primary-600" />
              <span
                className={`px-2 py-1 text-xs font-medium rounded-full ${getVerdictColor(
                  report.detection_result.verdict
                )}`}
              >
                {report.detection_result.verdict}
              </span>
            </div>

            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
              Hospital {report.hospital_id}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4 font-mono">
              {report.submission_id.substring(0, 20)}...
            </p>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Accuracy</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  {(report.test_accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Anomaly Score</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  {formatScore(report.detection_result.anomaly_score)}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Timestamp</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  {formatDate(report.timestamp, false)}
                </span>
              </div>
            </div>

            <button className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400 rounded-lg hover:bg-primary-100 dark:hover:bg-primary-900/30">
              <Eye className="w-4 h-4" />
              View Details
            </button>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
