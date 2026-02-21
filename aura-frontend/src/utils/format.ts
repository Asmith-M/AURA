/**
 * Utility functions for formatting data
 */

/**
 * Format a number with commas for thousands
 */
export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num);
}

/**
 * Format a percentage value
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format a hash string by truncating it
 */
export function truncateHash(hash: string, startChars: number = 8, endChars: number = 6): string {
  if (hash.length <= startChars + endChars) {
    return hash;
  }
  return `${hash.substring(0, startChars)}...${hash.substring(hash.length - endChars)}`;
}

/**
 * Format a date string to a readable format
 */
export function formatDate(dateString: string, includeTime: boolean = true): string {
  const date = new Date(dateString);
  
  const options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  };
  
  if (includeTime) {
    options.hour = '2-digit';
    options.minute = '2-digit';
  }
  
  return new Intl.DateTimeFormat('en-US', options).format(date);
}

/**
 * Format a relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  const intervals = {
    year: 31536000,
    month: 2592000,
    week: 604800,
    day: 86400,
    hour: 3600,
    minute: 60,
  };
  
  for (const [unit, secondsInUnit] of Object.entries(intervals)) {
    const interval = Math.floor(seconds / secondsInUnit);
    if (interval >= 1) {
      return `${interval} ${unit}${interval === 1 ? '' : 's'} ago`;
    }
  }
  
  return 'just now';
}

/**
 * Format file size in bytes to human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Format a score to a fixed number of decimal places
 */
export function formatScore(score: number, decimals: number = 3): string {
  return score.toFixed(decimals);
}

/**
 * Get color class based on verdict
 */
export function getVerdictColor(verdict: string): string {
  switch (verdict.toUpperCase()) {
    case 'APPROVED':
      return 'text-success-600 bg-success-100 dark:bg-success-900/20';
    case 'REJECTED':
      return 'text-danger-600 bg-danger-100 dark:bg-danger-900/20';
    case 'PROCESSING':
      return 'text-warning-600 bg-warning-100 dark:bg-warning-900/20';
    case 'ERROR':
      return 'text-danger-600 bg-danger-100 dark:bg-danger-900/20';
    default:
      return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20';
  }
}

/**
 * Get color class based on severity
 */
export function getSeverityColor(severity: string): string {
  switch (severity.toUpperCase()) {
    case 'CRITICAL':
      return 'text-danger-600 bg-danger-100 dark:bg-danger-900/20';
    case 'HIGH':
      return 'text-danger-500 bg-danger-50 dark:bg-danger-900/10';
    case 'MEDIUM':
      return 'text-warning-600 bg-warning-100 dark:bg-warning-900/20';
    case 'LOW':
      return 'text-success-600 bg-success-100 dark:bg-success-900/20';
    default:
      return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20';
  }
}

/**
 * Get color class based on priority
 */
export function getPriorityColor(priority: string): string {
  return getSeverityColor(priority);
}
