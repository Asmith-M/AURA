/**
 * Security page - Threat monitoring and security center
 */
import { useState, useEffect, useCallback } from 'react';
import { Shield, AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import { securityAPI } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { formatRelativeTime, getSeverityColor, getPriorityColor } from '../utils/format';
import type { Threat, SecurityRecommendation, ThreatStatus } from '../types';

export function Security() {
  const [threats, setThreats] = useState<Threat[]>([]);
  const [recommendations, setRecommendations] = useState<SecurityRecommendation[]>([]);
  const [statusFilter, setStatusFilter] = useState<ThreatStatus | 'ALL'>('ALL');
  const [loading, setLoading] = useState(true);

  const loadSecurityData = useCallback(async () => {
    setLoading(true);
    try {
      const [threatsData, recsData] = await Promise.all([
        securityAPI.getThreats(statusFilter),
        securityAPI.getRecommendations(),
      ]);
      setThreats(threatsData);
      setRecommendations(recsData);
    } catch (error) {
      console.error('Failed to load security data:', error);
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    void loadSecurityData();
  }, [loadSecurityData]);

  const activeThreats = threats.filter((t) => t.status === 'ACTIVE').length;
  const criticalThreats = threats.filter((t) => t.severity === 'CRITICAL').length;
  const resolvedThreats = threats.filter((t) => t.status === 'RESOLVED').length;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Security Center</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Real-time threat monitoring and security recommendations
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-danger-100 dark:bg-danger-900/20 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-danger-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Active Threats</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{activeThreats}</p>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-danger-100 dark:bg-danger-900/20 rounded-lg">
              <Shield className="w-6 h-6 text-danger-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Critical</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{criticalThreats}</p>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-success-100 dark:bg-success-900/20 rounded-lg">
              <CheckCircle className="w-6 h-6 text-success-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Resolved</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{resolvedThreats}</p>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
              <TrendingUp className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">System Health</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">98.5%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Threat Status Filter */}
      <div className="flex gap-2">
        {(['ALL', 'ACTIVE', 'INVESTIGATING', 'RESOLVED'] as (ThreatStatus | 'ALL')[]).map((status) => (
          <button
            key={status}
            onClick={() => setStatusFilter(status)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              statusFilter === status
                ? 'bg-primary-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
            }`}
          >
            {status}
          </button>
        ))}
      </div>

      {/* Threats Timeline */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Threat Detection Timeline</h3>
        <div className="space-y-4">
          {threats.map((threat, index) => (
            <motion.div
              key={threat.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-start gap-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
            >
              <div className="flex-shrink-0">
                <div className={`w-3 h-3 rounded-full mt-1.5 ${
                  threat.status === 'ACTIVE' ? 'bg-danger-600 animate-pulse' :
                  threat.status === 'INVESTIGATING' ? 'bg-warning-600' :
                  'bg-success-600'
                }`} />
              </div>
              <div className="flex-1">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSeverityColor(threat.severity)}`}>
                        {threat.severity}
                      </span>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">{threat.type}</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-1">{threat.description}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Hospital {threat.hospital_id} • {formatRelativeTime(threat.detected_at)}
                    </p>
                  </div>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    threat.status === 'ACTIVE' ? 'bg-danger-100 text-danger-600 dark:bg-danger-900/20' :
                    threat.status === 'INVESTIGATING' ? 'bg-warning-100 text-warning-600 dark:bg-warning-900/20' :
                    'bg-success-100 text-success-600 dark:bg-success-900/20'
                  }`}>
                    {threat.status}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Security Recommendations */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Security Recommendations</h3>
        <div className="space-y-4">
          {recommendations.map((rec) => (
            <div
              key={rec.id}
              className="flex items-start gap-4 p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
            >
              <div className="flex-shrink-0">
                <AlertTriangle className={`w-5 h-5 ${
                  rec.priority === 'HIGH' ? 'text-danger-600' :
                  rec.priority === 'MEDIUM' ? 'text-warning-600' :
                  'text-success-600'
                }`} />
              </div>
              <div className="flex-1">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-gray-900 dark:text-white">{rec.title}</h4>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getPriorityColor(rec.priority)}`}>
                    {rec.priority} Priority
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">{rec.description}</p>
                <p className="text-sm text-primary-600 dark:text-primary-400">
                  <strong>Action:</strong> {rec.action_required}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Health Monitoring */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">System Health Monitoring</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-success-50 dark:bg-success-900/10 rounded-lg border border-success-200 dark:border-success-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Sentinel API</span>
              <CheckCircle className="w-5 h-5 text-success-600" />
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">All services operational</p>
          </div>
          <div className="p-4 bg-success-50 dark:bg-success-900/10 rounded-lg border border-success-200 dark:border-success-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Blockchain Network</span>
              <CheckCircle className="w-5 h-5 text-success-600" />
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">Synchronized</p>
          </div>
          <div className="p-4 bg-success-50 dark:bg-success-900/10 rounded-lg border border-success-200 dark:border-success-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Detection Model</span>
              <CheckCircle className="w-5 h-5 text-success-600" />
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">Ready</p>
          </div>
        </div>
      </div>
    </div>
  );
}
