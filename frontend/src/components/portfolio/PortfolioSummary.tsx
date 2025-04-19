import React from 'react';

// Sample portfolio data - in a real app, this would come from an API
const samplePortfolio = {
  totalValue: 125750.42,
  cashBalance: 15250.33,
  returns: {
    daily: 1.2,
    weekly: 3.5,
    monthly: -2.1,
    yearly: 12.4
  },
  allocation: [
    { category: 'Technology', value: 45000, percentage: 0.36 },
    { category: 'Healthcare', value: 25000, percentage: 0.20 },
    { category: 'Financials', value: 20000, percentage: 0.16 },
    { category: 'Consumer Discretionary', value: 15000, percentage: 0.12 },
    { category: 'Energy', value: 5500, percentage: 0.04 },
    { category: 'Cash', value: 15250.33, percentage: 0.12 }
  ]
};

interface PortfolioSummaryProps {
  portfolio?: typeof samplePortfolio;
}

const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({ portfolio = samplePortfolio }) => {
  // Format currency
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Get color class based on return value
  const getReturnColorClass = (value: number): string => {
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Portfolio Summary</h3>
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-500">Total Value</p>
            <p className="text-2xl font-bold text-gray-900">{formatCurrency(portfolio.totalValue)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Cash Balance</p>
            <p className="text-2xl font-bold text-gray-900">{formatCurrency(portfolio.cashBalance)}</p>
          </div>
        </div>
      </div>

      <div className="p-6 border-b border-gray-200">
        <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-4">Returns</h4>
        <div className="grid grid-cols-4 gap-4">
          {Object.entries(portfolio.returns).map(([period, value]) => (
            <div key={period}>
              <p className="text-xs text-gray-500 capitalize">{period}</p>
              <p className={`text-lg font-semibold ${getReturnColorClass(value)}`}>
                {value > 0 ? '+' : ''}{formatPercentage(value / 100)}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="p-6">
        <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-4">Allocation</h4>
        <div className="space-y-4">
          {portfolio.allocation.map((item) => (
            <div key={item.category} className="flex items-center">
              <div className="w-1/3">
                <p className="text-sm font-medium text-gray-700">{item.category}</p>
              </div>
              <div className="w-1/3">
                <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="absolute top-0 left-0 h-full bg-blue-600 rounded-full"
                    style={{ width: `${item.percentage * 100}%` }}
                  ></div>
                </div>
              </div>
              <div className="w-1/6 text-right">
                <p className="text-sm font-medium text-gray-700">{formatPercentage(item.percentage)}</p>
              </div>
              <div className="w-1/6 text-right">
                <p className="text-sm text-gray-500">{formatCurrency(item.value)}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PortfolioSummary;
