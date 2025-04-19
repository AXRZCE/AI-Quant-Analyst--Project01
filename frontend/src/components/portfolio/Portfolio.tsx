import React from 'react';
import PortfolioSummary from './PortfolioSummary';
import PortfolioHoldings from './PortfolioHoldings';
import { useAuth } from '../../context/AuthContext';

const Portfolio: React.FC = () => {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 text-center">
        <p className="text-gray-700">Please log in to view your portfolio.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Portfolio</h2>
        <div className="flex space-x-2">
          <button className="px-4 py-2 bg-white border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
            Export
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
            Add Position
          </button>
        </div>
      </div>

      <PortfolioSummary />
      
      <PortfolioHoldings />
    </div>
  );
};

export default Portfolio;
