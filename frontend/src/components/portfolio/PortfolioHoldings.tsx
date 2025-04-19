import React, { useState } from 'react';

// Sample holdings data - in a real app, this would come from an API
const sampleHoldings = [
  {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    shares: 50,
    avgCost: 150.25,
    currentPrice: 175.50,
    marketValue: 8775.00,
    gain: 1262.50,
    gainPercent: 0.168,
    sector: 'Technology'
  },
  {
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    shares: 30,
    avgCost: 240.10,
    currentPrice: 265.75,
    marketValue: 7972.50,
    gain: 769.50,
    gainPercent: 0.107,
    sector: 'Technology'
  },
  {
    symbol: 'AMZN',
    name: 'Amazon.com Inc.',
    shares: 15,
    avgCost: 3100.50,
    currentPrice: 3250.25,
    marketValue: 48753.75,
    gain: 2246.25,
    gainPercent: 0.048,
    sector: 'Consumer Discretionary'
  },
  {
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    shares: 10,
    avgCost: 2500.75,
    currentPrice: 2750.50,
    marketValue: 27505.00,
    gain: 2497.50,
    gainPercent: 0.100,
    sector: 'Communication Services'
  },
  {
    symbol: 'JNJ',
    name: 'Johnson & Johnson',
    shares: 25,
    avgCost: 160.30,
    currentPrice: 155.40,
    marketValue: 3885.00,
    gain: -122.50,
    gainPercent: -0.031,
    sector: 'Healthcare'
  },
  {
    symbol: 'JPM',
    name: 'JPMorgan Chase & Co.',
    shares: 35,
    avgCost: 135.20,
    currentPrice: 145.75,
    marketValue: 5101.25,
    gain: 369.25,
    gainPercent: 0.078,
    sector: 'Financials'
  },
  {
    symbol: 'PG',
    name: 'Procter & Gamble Co.',
    shares: 20,
    avgCost: 140.50,
    currentPrice: 138.25,
    marketValue: 2765.00,
    gain: -45.00,
    gainPercent: -0.016,
    sector: 'Consumer Staples'
  }
];

interface PortfolioHoldingsProps {
  holdings?: typeof sampleHoldings;
}

const PortfolioHoldings: React.FC<PortfolioHoldingsProps> = ({ holdings = sampleHoldings }) => {
  const [sortField, setSortField] = useState<string>('marketValue');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [searchTerm, setSearchTerm] = useState<string>('');

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

  // Handle sort
  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  // Sort and filter holdings
  const sortedHoldings = [...holdings]
    .filter(holding => 
      holding.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      holding.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      holding.sector.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      const aValue = a[sortField as keyof typeof a];
      const bValue = b[sortField as keyof typeof b];
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue) 
          : bValue.localeCompare(aValue);
      }
      
      return sortDirection === 'asc' 
        ? (aValue as number) - (bValue as number) 
        : (bValue as number) - (aValue as number);
    });

  // Get sort icon
  const getSortIcon = (field: string) => {
    if (sortField !== field) return null;
    
    return sortDirection === 'asc' 
      ? <span className="ml-1">↑</span> 
      : <span className="ml-1">↓</span>;
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="p-6 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-800">Portfolio Holdings</h3>
          <div className="relative">
            <input
              type="text"
              placeholder="Search holdings..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 w-64"
            />
            <svg 
              className="absolute right-3 top-2.5 h-5 w-5 text-gray-400" 
              xmlns="http://www.w3.org/2000/svg" 
              viewBox="0 0 20 20" 
              fill="currentColor"
            >
              <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
            </svg>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th 
                scope="col" 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('symbol')}
              >
                Symbol {getSortIcon('symbol')}
              </th>
              <th 
                scope="col" 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('name')}
              >
                Name {getSortIcon('name')}
              </th>
              <th 
                scope="col" 
                className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('shares')}
              >
                Shares {getSortIcon('shares')}
              </th>
              <th 
                scope="col" 
                className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('avgCost')}
              >
                Avg Cost {getSortIcon('avgCost')}
              </th>
              <th 
                scope="col" 
                className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('currentPrice')}
              >
                Current Price {getSortIcon('currentPrice')}
              </th>
              <th 
                scope="col" 
                className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('marketValue')}
              >
                Market Value {getSortIcon('marketValue')}
              </th>
              <th 
                scope="col" 
                className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('gainPercent')}
              >
                Gain/Loss {getSortIcon('gainPercent')}
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedHoldings.map((holding) => (
              <tr key={holding.symbol} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-blue-600">{holding.symbol}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{holding.name}</div>
                  <div className="text-xs text-gray-500">{holding.sector}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                  {holding.shares.toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                  {formatCurrency(holding.avgCost)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                  {formatCurrency(holding.currentPrice)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                  {formatCurrency(holding.marketValue)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <div className={`text-sm font-medium ${holding.gain >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatCurrency(holding.gain)}
                  </div>
                  <div className={`text-xs ${holding.gainPercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {holding.gainPercent >= 0 ? '+' : ''}{formatPercentage(holding.gainPercent)}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PortfolioHoldings;
