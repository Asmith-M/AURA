/**
 * Utility functions for validation
 */

/**
 * Validate hospital ID (must be a positive integer)
 */
export function validateHospitalId(id: number): { valid: boolean; error?: string } {
  if (!Number.isInteger(id)) {
    return { valid: false, error: 'Hospital ID must be an integer' };
  }
  
  if (id < 1 || id > 10) {
    return { valid: false, error: 'Hospital ID must be between 1 and 10' };
  }
  
  return { valid: true };
}

/**
 * Validate file upload (check file type and size)
 */
export function validateModelFile(file: File | null): { valid: boolean; error?: string } {
  if (!file) {
    return { valid: false, error: 'Please select a file to upload' };
  }
  
  // Check file extension (.pkl or .pth)
  const allowedExtensions = ['.pkl', '.pth', '.pt', '.h5', '.model'];
  const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  
  if (!allowedExtensions.includes(extension)) {
    return {
      valid: false,
      error: `Invalid file type. Allowed types: ${allowedExtensions.join(', ')}`,
    };
  }
  
  // Check file size (max 100MB)
  const maxSize = 100 * 1024 * 1024; // 100MB in bytes
  if (file.size > maxSize) {
    return {
      valid: false,
      error: 'File size must be less than 100MB',
    };
  }
  
  return { valid: true };
}

/**
 * Validate model description (must not be empty and reasonable length)
 */
export function validateDescription(description: string): { valid: boolean; error?: string } {
  if (!description || description.trim().length === 0) {
    return { valid: false, error: 'Description is required' };
  }
  
  if (description.length < 10) {
    return { valid: false, error: 'Description must be at least 10 characters' };
  }
  
  if (description.length > 500) {
    return { valid: false, error: 'Description must be less than 500 characters' };
  }
  
  return { valid: true };
}

/**
 * Validate date range
 */
export function validateDateRange(
  startDate: string | undefined,
  endDate: string | undefined
): { valid: boolean; error?: string } {
  if (!startDate && !endDate) {
    return { valid: true }; // No dates provided is valid
  }
  
  if (startDate && endDate) {
    const start = new Date(startDate);
    const end = new Date(endDate);
    
    if (start > end) {
      return { valid: false, error: 'Start date must be before end date' };
    }
    
    // Check if dates are not in the future
    const now = new Date();
    if (start > now || end > now) {
      return { valid: false, error: 'Dates cannot be in the future' };
    }
  }
  
  return { valid: true };
}

/**
 * Validate transaction ID format
 */
export function validateTransactionId(txId: string): { valid: boolean; error?: string } {
  if (!txId || txId.trim().length === 0) {
    return { valid: false, error: 'Transaction ID is required' };
  }
  
  // Transaction IDs should be alphanumeric with possible hyphens
  const pattern = /^[a-zA-Z0-9-]+$/;
  if (!pattern.test(txId)) {
    return { valid: false, error: 'Invalid transaction ID format' };
  }
  
  return { valid: true };
}
