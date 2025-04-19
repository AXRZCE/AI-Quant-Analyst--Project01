import React from 'react';
import { ModelInfo } from '../../api';

interface ModelPerformanceCardProps {
  modelInfo: ModelInfo;
}

const ModelPerformanceCard: React.FC<ModelPerformanceCardProps> = ({ modelInfo }) => {
  // Format metrics for display
  const formatMetric = (value: number): string => {
    // If it's a percentage (between 0 and 1)
    if (value >= 0 && value <= 1) {
      return `${(value * 100).toFixed(2)}%`;
    }
    // If it's a small decimal
    if (Math.abs(value) < 0.01) {
      return value.toExponential(2);
    }
    // Otherwise format with 2 decimal places
    return value.toFixed(2);
  };

  // Get color class based on metric name and value
  const getMetricColorClass = (name: string, value: number): string => {
    const lowerName = name.toLowerCase();
    
    // For accuracy, precision, recall, f1, r2 metrics (higher is better)
    if (
      lowerName.includes('accuracy') || 
      lowerName.includes('precision') || 
      lowerName.includes('recall') || 
      lowerName.includes('f1') ||
      lowerName.includes('r2') ||
      lowerName.includes('auc')
    ) {
      if (value >= 0.8) return 'text-green-600';
      if (value >= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    }
    
    // For error metrics (lower is better)
    if (
      lowerName.includes('error') || 
      lowerName.includes('loss') || 
      lowerName.includes('mae') || 
      lowerName.includes('mse') || 
      lowerName.includes('rmse')
    ) {
      if (value <= 0.1) return 'text-green-600';
      if (value <= 0.3) return 'text-yellow-600';
      return 'text-red-600';
    }
    
    // Default
    return 'text-gray-700';
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">{modelInfo.name}</h3>
          <p className="text-sm text-gray-500">Version: {modelInfo.version}</p>
        </div>
        <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
          {modelInfo.type}
        </span>
      </div>
      
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Performance Metrics</h4>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(modelInfo.metrics).map(([key, value]) => (
            <div key={key} className="flex flex-col">
              <span className="text-xs text-gray-500">{key}</span>
              <span className={`text-lg font-semibold ${getMetricColorClass(key, value)}`}>
                {formatMetric(value)}
              </span>
            </div>
          ))}
        </div>
      </div>
      
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Features</h4>
        <div className="flex flex-wrap gap-2">
          {modelInfo.features.map((feature) => (
            <span key={feature} className="px-2 py-1 text-xs font-medium rounded-full bg-gray-100 text-gray-800">
              {feature}
            </span>
          ))}
        </div>
      </div>
      
      <div className="text-xs text-gray-500 mt-4">
        Last updated: {new Date(modelInfo.last_updated).toLocaleString()}
      </div>
    </div>
  );
};

export default ModelPerformanceCard;
