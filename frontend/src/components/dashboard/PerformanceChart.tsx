import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

// Sample data - in a real app, this would come from an API
const sampleData = [
  {
    date: '2023-01-01',
    accuracy: 0.82,
    precision: 0.78,
    recall: 0.75,
    f1: 0.76,
    mse: 0.12
  },
  {
    date: '2023-02-01',
    accuracy: 0.84,
    precision: 0.80,
    recall: 0.77,
    f1: 0.78,
    mse: 0.11
  },
  {
    date: '2023-03-01',
    accuracy: 0.83,
    precision: 0.79,
    recall: 0.76,
    f1: 0.77,
    mse: 0.115
  },
  {
    date: '2023-04-01',
    accuracy: 0.85,
    precision: 0.82,
    recall: 0.79,
    f1: 0.80,
    mse: 0.10
  },
  {
    date: '2023-05-01',
    accuracy: 0.86,
    precision: 0.83,
    recall: 0.81,
    f1: 0.82,
    mse: 0.09
  },
  {
    date: '2023-06-01',
    accuracy: 0.87,
    precision: 0.84,
    recall: 0.82,
    f1: 0.83,
    mse: 0.085
  },
  {
    date: '2023-07-01',
    accuracy: 0.88,
    precision: 0.85,
    recall: 0.83,
    f1: 0.84,
    mse: 0.08
  }
];

interface PerformanceChartProps {
  data?: any[];
  metrics?: string[];
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ 
  data = sampleData,
  metrics = ['accuracy', 'precision', 'recall', 'f1'] 
}) => {
  // Define colors for each metric
  const metricColors: Record<string, string> = {
    accuracy: '#3B82F6', // blue
    precision: '#10B981', // green
    recall: '#F59E0B', // amber
    f1: '#8B5CF6', // purple
    mse: '#EF4444', // red
    rmse: '#EC4899', // pink
    mae: '#F97316', // orange
    r2: '#06B6D4' // cyan
  };

  // Format date for tooltip
  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString();
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
          <p className="font-medium text-gray-700">{formatDate(label)}</p>
          <div className="mt-2">
            {payload.map((entry: any, index: number) => (
              <p key={`item-${index}`} style={{ color: entry.color }} className="text-sm">
                {`${entry.name}: ${(entry.value * 100).toFixed(2)}%`}
              </p>
            ))}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis 
          dataKey="date" 
          tickFormatter={formatDate}
          stroke="#6B7280"
        />
        <YAxis 
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          domain={[0, 1]}
          stroke="#6B7280"
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        {metrics.map((metric) => (
          <Line
            key={metric}
            type="monotone"
            dataKey={metric}
            name={metric.charAt(0).toUpperCase() + metric.slice(1)}
            stroke={metricColors[metric] || '#000000'}
            activeDot={{ r: 8 }}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default PerformanceChart;
