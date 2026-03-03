import { useCallback, useEffect, useMemo, useState } from 'react';
import { Database, Image as ImageIcon, Layers, ShieldCheck } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { sentinelAPI } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { Alert } from '../components/Alert';
import type { DatasetExplorerPayload } from '../types';

const HOSPITAL_OPTIONS = ['HOSP1', 'HOSP2', 'HOSP3'];

export function DatasetExplorer() {
  const [hospitalId, setHospitalId] = useState<string>('HOSP1');
  const [payload, setPayload] = useState<DatasetExplorerPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadDataset = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await sentinelAPI.getHospitalDataset(hospitalId, 12);
      setPayload(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load hospital dataset');
    } finally {
      setLoading(false);
    }
  }, [hospitalId]);

  useEffect(() => {
    void loadDataset();
  }, [loadDataset]);

  const chartData = useMemo(
    () =>
      Object.entries(payload?.class_distribution ?? {}).map(([classId, count]) => ({
        classId,
        count: Number(count),
      })),
    [payload],
  );

  if (loading && !payload) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Dataset Explorer</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Inspect hospital-specific slices and the shared golden validation set.
          </p>
        </div>
        <div className="min-w-[200px]">
          <label className="block text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Hospital</label>
          <select
            value={hospitalId}
            onChange={(event) => setHospitalId(event.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
          >
            {HOSPITAL_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option.replace('HOSP', 'Hospital ')}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <Alert type="error" message={error} />}

      {payload && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 shadow-lg">
              <div className="flex items-center gap-2 mb-2 text-primary-600">
                <Database className="w-5 h-5" />
                <h3 className="font-semibold text-gray-900 dark:text-white">Dataset Slice</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Samples: {payload.dataset_slice.sample_count}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Index Range: {payload.dataset_slice.index_range.start} - {payload.dataset_slice.index_range.end}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Noise Level: {payload.dataset_slice.noise_level.toFixed(4)}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Classes: {payload.dataset_slice.classes_represented.join(', ')}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 shadow-lg">
              <div className="flex items-center gap-2 mb-2 text-primary-600">
                <Layers className="w-5 h-5" />
                <h3 className="font-semibold text-gray-900 dark:text-white">Scenario</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">{payload.scenario.dataset_name}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">{payload.scenario.dataset_variant}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Participants: {payload.scenario.participant_hospitals.map((h) => `H${h}`).join(', ')}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 shadow-lg">
              <div className="flex items-center gap-2 mb-2 text-primary-600">
                <ShieldCheck className="w-5 h-5" />
                <h3 className="font-semibold text-gray-900 dark:text-white">Golden Validation</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Total Samples: {payload.golden_validation_set.sample_count}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Anomaly %: {payload.golden_validation_set.anomaly_percentage.toFixed(2)}%
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 shadow-lg">
              <div className="flex items-center gap-2 mb-2 text-primary-600">
                <ImageIcon className="w-5 h-5" />
                <h3 className="font-semibold text-gray-900 dark:text-white">Preprocessing</h3>
              </div>
              <div className="space-y-1">
                {payload.preprocessing_steps.map((step, idx) => (
                  <p key={`${step}_${idx}`} className="text-sm text-gray-600 dark:text-gray-400">
                    {idx + 1}. {step}
                  </p>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Class Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                <XAxis dataKey="classId" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#0ea5e9" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Hospital Sample Images</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {payload.sample_images.map((sample) => (
                <div key={`hospital_sample_${sample.index}`} className="p-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/40">
                  <img src={sample.image_base64} alt={`Hospital sample ${sample.index}`} className="w-full h-auto rounded" />
                  <p className="text-xs mt-1 text-gray-600 dark:text-gray-300">
                    #{sample.index} | y={sample.label}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Golden Validation Set (100 Shared Samples)
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              The same 100 samples are used to interrogate every submitted model.
            </p>
            <div className="max-h-[520px] overflow-y-auto grid grid-cols-4 md:grid-cols-8 lg:grid-cols-10 gap-2">
              {payload.golden_validation_set.sample_images.map((sample) => (
                <div key={`golden_${sample.index}`} className="p-1 rounded border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/40">
                  <img src={sample.image_base64} alt={`Golden sample ${sample.index}`} className="w-full h-auto rounded" />
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
