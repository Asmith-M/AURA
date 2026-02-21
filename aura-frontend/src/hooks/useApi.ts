/**
 * Generic hook for making API calls with loading and error states
 */
import { useState, useCallback } from 'react';
import type { UseApiReturn } from '../types';

interface UseApiOptions<T> {
  onSuccess?: (data: T) => void;
  onError?: (error: string) => void;
}

export function useApi<T>(
  apiFunction: () => Promise<T>,
  options?: UseApiOptions<T>
): UseApiReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiFunction();
      setData(result);
      
      if (options?.onSuccess) {
        options.onSuccess(result);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      
      if (options?.onError) {
        options.onError(errorMessage);
      }
    } finally {
      setLoading(false);
    }
  }, [apiFunction, options]);

  return { data, loading, error, refetch };
}
