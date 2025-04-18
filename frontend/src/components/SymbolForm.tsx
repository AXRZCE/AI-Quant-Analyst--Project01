import { useState } from "react";

interface SymbolFormProps {
  onRun: (symbol: string, days: number) => void;
  isLoading: boolean;
}

export default function SymbolForm({ onRun, isLoading }: SymbolFormProps) {
  const [symbol, setSymbol] = useState("AAPL");
  const [days, setDays] = useState(7);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onRun(symbol, days);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-4 mb-6">
      <div className="flex flex-col">
        <label htmlFor="symbol" className="text-sm font-medium text-gray-700 mb-1">
          Stock Symbol
        </label>
        <input
          id="symbol"
          className="border border-gray-300 p-2 rounded-md focus:ring-blue-500 focus:border-blue-500"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          placeholder="AAPL"
          required
        />
      </div>
      
      <div className="flex flex-col">
        <label htmlFor="days" className="text-sm font-medium text-gray-700 mb-1">
          Days of History
        </label>
        <input
          id="days"
          type="number"
          min={1}
          max={30}
          className="border border-gray-300 p-2 rounded-md w-24 focus:ring-blue-500 focus:border-blue-500"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          required
        />
      </div>
      
      <div className="flex items-end">
        <button
          type="submit"
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors duration-200 disabled:bg-blue-400"
          disabled={isLoading}
        >
          {isLoading ? "Loading..." : "Predict"}
        </button>
      </div>
    </form>
  );
}
