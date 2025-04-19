import React, { useState, useEffect } from 'react';
import { getModelInfo, getCacheStats, clearCache } from '../../api';
import ModelPerformanceCard from './ModelPerformanceCard';
import PerformanceChart from './PerformanceChart';
import { useAuth } from '../../context/AuthContext';

const Dashboard: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [cacheStats, setCacheStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [clearingCache, setClearingCache] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (!isAuthenticated) {
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        // Fetch model info
        const modelData = await getModelInfo();
        setModelInfo(modelData);

        // Fetch cache stats
        const cacheData = await getCacheStats();
        setCacheStats(cacheData);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [isAuthenticated]);

  const handleClearCache = async () => {
    if (!isAuthenticated) return;

    setClearingCache(true);
    try {
      await clearCache();
      // Refresh cache stats
      const cacheData = await getCacheStats();
      setCacheStats(cacheData);
    } catch (err) {
      console.error('Error clearing cache:', err);
      setError('Failed to clear cache. Please try again.');
    } finally {
      setClearingCache(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 text-center">
        <p className="text-gray-700">Please log in to view the dashboard.</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md" role="alert">
        <p className="font-bold">Error</p>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Dashboard</h2>
        <button
          onClick={handleClearCache}
          disabled={clearingCache}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 transition-colors"
        >
          {clearingCache ? 'Clearing...' : 'Clear Cache'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Model Performance Card */}
        {modelInfo && (
          <ModelPerformanceCard modelInfo={modelInfo} />
        )}

        {/* Cache Stats Card */}
        {cacheStats && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Cache Statistics</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Entries:</span>
                <span className="font-medium">{cacheStats.total_entries}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Hit Rate:</span>
                <span className="font-medium">{(cacheStats.hit_rate * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Memory Usage:</span>
                <span className="font-medium">{(cacheStats.memory_usage / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Oldest Entry:</span>
                <span className="font-medium">{new Date(cacheStats.oldest_entry * 1000).toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}

        {/* System Status Card */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">System Status</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">API Status:</span>
              <span className="px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-800">
                Online
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Last Update:</span>
              <span className="font-medium">{new Date().toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Environment:</span>
              <span className="font-medium">{import.meta.env.MODE}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Model Performance Over Time</h3>
        <div className="h-80">
          <PerformanceChart />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
