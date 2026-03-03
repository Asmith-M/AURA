/**
 * Upload page - Model upload interface with drag-and-drop
 */
import { useState, useCallback, type ChangeEvent } from 'react';
import { Upload as UploadIcon, FileText, CheckCircle, XCircle, Download, Eye } from 'lucide-react';
import { motion } from 'framer-motion';
import { sentinelAPI } from '../services/api';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { validateHospitalId, validateModelFile } from '../utils/validation';
import { formatScore } from '../utils/format';
import type { BehavioralReport, PrivacyLevel, UploadFormData, AlertType } from '../types';

type UploadStep = 'upload' | 'analysis' | 'detection' | 'ledger' | 'complete';

export function Upload() {
  const [formData, setFormData] = useState<UploadFormData>({
    hospital_id: 1,
    model_description: '',
    privacy_level: 'PRIVATE',
    file: null,
  });
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [simulationLogs, setSimulationLogs] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState<UploadStep>('upload');
  const [result, setResult] = useState<BehavioralReport | null>(null);
  const [alert, setAlert] = useState<{ type: AlertType; message: string } | null>(null);

  const steps = [
    { id: 'upload' as UploadStep, name: 'Upload', icon: UploadIcon },
    { id: 'analysis' as UploadStep, name: 'Analysis', icon: FileText },
    { id: 'detection' as UploadStep, name: 'Detection', icon: CheckCircle },
    { id: 'ledger' as UploadStep, name: 'Ledger', icon: CheckCircle },
  ];

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFormData((prev) => ({ ...prev, file: e.dataTransfer.files[0] }));
    }
  }, []);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFormData((prev) => ({ ...prev, file: e.target.files![0] }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate form
    const hospitalValidation = validateHospitalId(formData.hospital_id);
    if (!hospitalValidation.valid) {
      setAlert({ type: 'error', message: hospitalValidation.error! });
      return;
    }

    const fileValidation = validateModelFile(formData.file);
    if (!fileValidation.valid) {
      setAlert({ type: 'error', message: fileValidation.error! });
      return;
    }

    // Start upload process
    setUploading(true);
    setAlert(null);

    try {
      setSimulationLogs([]);
      setCurrentStep('analysis');
      const uploadFormData = new FormData();
      uploadFormData.append('hospital_id', formData.hospital_id.toString());
      if (formData.file) {
        uploadFormData.append('model_file', formData.file);
      }

      const report = await sentinelAPI.submitUpdate(uploadFormData);
      setCurrentStep('detection');
      setCurrentStep('ledger');
      setResult(report);
      setCurrentStep('complete');
      setAlert({ type: 'success', message: 'Model submitted successfully!' });
    } catch (error) {
      setAlert({
        type: 'error',
        message: error instanceof Error ? error.message : 'Upload failed',
      });
    } finally {
      setUploading(false);
    }
  };

  const simulateHospitalSubmission = useCallback(async (attackMode: boolean) => {
    const hospitalValidation = validateHospitalId(formData.hospital_id);
    if (!hospitalValidation.valid) {
      setAlert({ type: 'error', message: hospitalValidation.error! });
      return;
    }

    setUploading(true);
    setSimulating(true);
    setCurrentStep('analysis');
    setSimulationLogs([]);
    setAlert(null);

    try {
      const run = await sentinelAPI.startPipeline({
        hospital_id: `HOSP${formData.hospital_id}`,
        rounds: 2,
        local_epochs: 1,
        max_samples_per_hospital: 256,
        attack_mode: attackMode,
      });

      let finalRun = run;
      for (let attempts = 0; attempts < 240; attempts += 1) {
        if (finalRun.status === 'completed' || finalRun.status === 'failed') {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 1500));
        finalRun = await sentinelAPI.getPipelineStatus(run.run_id);
        setSimulationLogs(
          (finalRun.logs ?? []).slice(-7).map((entry) => `${entry.step ?? 'step'} @ ${entry.timestamp}`),
        );
      }

      if (finalRun.status !== 'completed' || !finalRun.final_session) {
        throw new Error(finalRun.error ?? 'Simulation did not complete successfully');
      }

      const sessionId = String((finalRun.final_session as Record<string, unknown>).session_id ?? '');
      if (!sessionId) {
        throw new Error('Completed simulation did not return a session_id');
      }

      const report = await sentinelAPI.getReport(sessionId);
      setCurrentStep('detection');
      setCurrentStep('ledger');
      setResult(report);
      setCurrentStep('complete');
      setAlert({
        type: 'success',
        message: attackMode
          ? 'Attack simulation completed and processed successfully.'
          : 'Hospital simulation completed and processed successfully.',
      });
    } catch (error) {
      setAlert({
        type: 'error',
        message: error instanceof Error ? error.message : 'Simulation failed',
      });
    } finally {
      setUploading(false);
      setSimulating(false);
    }
  }, [formData.hospital_id]);

  const getStepStatus = (stepId: UploadStep) => {
    const stepIndex = steps.findIndex((s) => s.id === stepId);
    const currentIndex = steps.findIndex((s) => s.id === currentStep);
    
    if (currentStep === 'complete') return 'complete';
    if (stepIndex < currentIndex) return 'complete';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Upload Model</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Submit your federated learning model for security analysis
        </p>
      </div>

      {alert && (
        <Alert
          type={alert.type}
          message={alert.message}
          onClose={() => setAlert(null)}
        />
      )}

      {/* Progress Steps */}
      {uploading && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => {
              const status = getStepStatus(step.id);
              const Icon = step.icon;
              
              return (
                <div key={step.id} className="flex items-center flex-1">
                  <div className="flex flex-col items-center flex-1">
                    <div
                      className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                        status === 'complete'
                          ? 'bg-success-600 text-white'
                          : status === 'active'
                          ? 'bg-primary-600 text-white animate-pulse'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-400'
                      }`}
                    >
                      {status === 'complete' ? <CheckCircle className="w-6 h-6" /> : <Icon className="w-6 h-6" />}
                    </div>
                    <span className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                      {step.name}
                    </span>
                  </div>
                  
                  {index < steps.length - 1 && (
                    <div className={`h-1 flex-1 mx-2 rounded ${
                      getStepStatus(steps[index + 1].id) === 'complete' || 
                      getStepStatus(steps[index + 1].id) === 'active'
                        ? 'bg-primary-600'
                        : 'bg-gray-200 dark:bg-gray-700'
                    }`} />
                  )}
                </div>
              );
            })}
          </div>
          {simulating && simulationLogs.length > 0 && (
            <div className="max-h-40 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-gray-50 dark:bg-gray-900/40">
              {simulationLogs.map((line, idx) => (
                <p key={`${line}_${idx}`} className="text-xs text-gray-600 dark:text-gray-300">
                  {line}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {!result ? (
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Simulate Hospital Submission</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Run the full backend pipeline without uploading a model file.
            </p>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => void simulateHospitalSubmission(false)}
                disabled={uploading}
                className="px-4 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
              >
                Simulate Clean Submission
              </button>
              <button
                type="button"
                onClick={() => void simulateHospitalSubmission(true)}
                disabled={uploading}
                className="px-4 py-2 rounded-lg bg-danger-600 text-white hover:bg-danger-700 disabled:opacity-50"
              >
                Simulate Attack Submission
              </button>
            </div>
          </div>

          {/* File Upload Area */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
                dragActive
                  ? 'border-primary-600 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-300 dark:border-gray-600 hover:border-primary-400'
              }`}
            >
              <UploadIcon className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                {formData.file ? formData.file.name : 'Drag and drop your model file here'}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">or click to browse</p>
              <input
                type="file"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
                accept=".pkl,.pth,.pt,.h5,.model"
              />
              <label
                htmlFor="file-upload"
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 cursor-pointer transition-colors"
              >
                <UploadIcon className="w-4 h-4" />
                Select File
              </label>
            </div>
          </div>

          {/* Form Fields */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Hospital ID
              </label>
              <select
                value={formData.hospital_id}
                onChange={(e) => setFormData({ ...formData, hospital_id: parseInt(e.target.value) })}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-600 focus:border-transparent"
              >
                {Array.from({ length: 10 }, (_, i) => i + 1).map((id) => (
                  <option key={id} value={id}>
                    Hospital {id}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Privacy Level
              </label>
              <select
                value={formData.privacy_level}
                onChange={(e) => setFormData({ ...formData, privacy_level: e.target.value as PrivacyLevel })}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-600 focus:border-transparent"
              >
                <option value="PUBLIC">Public</option>
                <option value="PRIVATE">Private</option>
                <option value="CONFIDENTIAL">Confidential</option>
              </select>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex gap-4">
            <button
              type="button"
              onClick={() => {
                setFormData({
                  hospital_id: 1,
                  model_description: '',
                  privacy_level: 'PRIVATE',
                  file: null,
                });
              }}
              className="flex-1 px-6 py-3 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={uploading}
              className="flex-1 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {uploading ? (
                <>
                  <LoadingSpinner size="sm" className="border-white border-t-transparent" />
                  Processing...
                </>
              ) : (
                <>
                  <UploadIcon className="w-5 h-5" />
                  Upload Model
                </>
              )}
            </button>
          </div>
        </form>
      ) : (
        /* Result Display */
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="space-y-6"
        >
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-4 mb-6">
              {result.detection_result.is_anomalous ? (
                <XCircle className="w-12 h-12 text-danger-600" />
              ) : (
                <CheckCircle className="w-12 h-12 text-success-600" />
              )}
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Submission {result.detection_result.is_anomalous ? 'Rejected' : 'Approved'}
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Your model has been analyzed and {result.detection_result.is_anomalous ? 'flagged as anomalous' : 'accepted'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Submission ID</p>
                <p className="font-mono text-sm text-gray-900 dark:text-white">{result.submission_id}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Fingerprint ID</p>
                <p className="font-mono text-sm text-gray-900 dark:text-white">{result.fingerprint_id}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Anomaly Score</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {formatScore(result.detection_result.anomaly_score)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Test Accuracy</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {(result.test_accuracy * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            <div className="mt-6 flex gap-4">
              <a href="/reports" className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700">
                <Eye className="w-4 h-4" />
                View Report
              </a>
              <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700">
                <Download className="w-4 h-4" />
                Download
              </button>
              <button
                onClick={() => {
                  setResult(null);
                  setCurrentStep('upload');
                  setFormData({
                    hospital_id: 1,
                    model_description: '',
                    privacy_level: 'PRIVATE',
                    file: null,
                  });
                }}
                className="flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Upload Another
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
