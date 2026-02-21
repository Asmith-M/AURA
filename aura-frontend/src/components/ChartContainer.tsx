/**
 * Chart container component for wrapping charts with consistent styling
 */
import type { ChartContainerProps } from '../types';

export function ChartContainer({ title, children, actions }: ChartContainerProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>
        {actions}
      </div>
      <div className="w-full">{children}</div>
    </div>
  );
}
