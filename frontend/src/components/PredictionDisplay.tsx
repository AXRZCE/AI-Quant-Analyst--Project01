import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface ConfidenceInterval {
  lower: number;
  upper: number;
}

interface PredictionDisplayProps {
  prediction: number;
  confidenceInterval?: ConfidenceInterval;
  featureImportance?: Record<string, number>;
}

export default function PredictionDisplay({
  prediction,
  confidenceInterval,
  featureImportance
}: PredictionDisplayProps) {
  const pct = (prediction * 100).toFixed(2);
  const isPositive = prediction >= 0;
  const color = isPositive ? "text-green-600" : "text-red-600";
  const icon = isPositive ? "↑" : "↓";

  // Format feature importance data for chart
  const featureImportanceData = featureImportance ?
    Object.entries(featureImportance)
      .map(([feature, value]) => ({
        feature: feature.replace(/_/g, ' '),
        importance: Math.abs(value) // Use absolute value for visualization
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 5) : // Show top 5 features
    [];

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold mb-4">Prediction</h3>

      <div className="mb-6">
        <div className={`text-3xl font-bold ${color} flex items-center justify-center`}>
          <span className="mr-2">{icon}</span>
          <span>{pct}%</span>
        </div>
        <p className="text-gray-600 mt-2 text-center">
          Predicted return for next trading period
        </p>
      </div>

      {confidenceInterval && (
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Confidence Interval</h4>
          <div className="flex items-center justify-between">
            <span className="text-gray-700">{(confidenceInterval.lower * 100).toFixed(2)}%</span>
            <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden flex-grow mx-2">
              <div
                className="absolute top-0 left-0 h-full bg-blue-600 rounded-full"
                style={{
                  left: `${Math.max(0, (confidenceInterval.lower * 100) + 10)}%`,
                  width: `${Math.max(1, ((confidenceInterval.upper - confidenceInterval.lower) * 100))}%`
                }}
              ></div>
            </div>
            <span className="text-gray-700">{(confidenceInterval.upper * 100).toFixed(2)}%</span>
          </div>
          <p className="text-xs text-gray-500 mt-1 text-center">
            95% confidence interval
          </p>
        </div>
      )}

      {featureImportanceData.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Feature Importance</h4>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureImportanceData}
                layout="vertical"
                margin={{ top: 5, right: 5, bottom: 5, left: 5 }}
              >
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 12 }}
                  width={100}
                  tickFormatter={(value) => value.length > 12 ? `${value.substring(0, 12)}...` : value}
                />
                <Tooltip
                  formatter={(value: number) => [`${value.toFixed(4)}`, 'Importance']}
                  labelFormatter={(label) => label}
                />
                <Bar dataKey="importance" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-gray-500 mt-1 text-center">
            Relative importance of top features
          </p>
        </div>
      )}
    </div>
  );
}
