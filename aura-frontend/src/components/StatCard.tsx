/**
 * Stat card component for displaying statistics with trend indicators
 */
import { TrendingUp, TrendingDown } from 'lucide-react';
import { motion } from 'framer-motion';
import type { StatCardProps } from '../types';
import { cn } from '../utils/cn';

export function StatCard({ title, value, icon: Icon, trend, color = 'primary' }: StatCardProps) {
  const getColorClasses = () => {
    switch (color) {
      case 'success':
        return 'bg-success-100 text-success-600 dark:bg-success-900/20 dark:text-success-400';
      case 'warning':
        return 'bg-warning-100 text-warning-600 dark:bg-warning-900/20 dark:text-warning-400';
      case 'danger':
        return 'bg-danger-100 text-danger-600 dark:bg-danger-900/20 dark:text-danger-400';
      default:
        return 'bg-primary-100 text-primary-600 dark:bg-primary-900/20 dark:text-primary-400';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 transition-all"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-3xl font-bold text-gray-900 dark:text-white mt-2">{value}</p>
          
          {trend && (
            <div className="flex items-center gap-1 mt-2">
              {trend.isPositive ? (
                <TrendingUp className="w-4 h-4 text-success-600" />
              ) : (
                <TrendingDown className="w-4 h-4 text-danger-600" />
              )}
              <span
                className={cn(
                  'text-sm font-medium',
                  trend.isPositive ? 'text-success-600' : 'text-danger-600'
                )}
              >
                {trend.value}%
              </span>
            </div>
          )}
        </div>
        
        <div className={cn('p-3 rounded-lg', getColorClasses())}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </motion.div>
  );
}
