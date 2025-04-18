import { useState, useEffect } from "react";
import SymbolForm from "./components/SymbolForm";
import PriceChart from "./components/PriceChart";
import SentimentGauge from "./components/SentimentGauge";
import PredictionDisplay from "./components/PredictionDisplay";
import { predict, PredictionResponse, checkHealth } from "./api";

function App() {
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isApiAvailable, setIsApiAvailable] = useState(true);

  // Check API health on component mount
  useEffect(() => {
    const checkApiHealth = async () => {
      const isHealthy = await checkHealth();
      setIsApiAvailable(isHealthy);
    };
    
    checkApiHealth();
  }, []);

  const handleRun = async (symbol: string, days: number) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await predict(symbol, days);
      setData(result);
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError("Failed to fetch prediction. Please try again.");
      setData(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-blue-600 text-white p-6 shadow-md">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold">Project01 Quant Analyst</h1>
          <p className="mt-2 text-blue-100">AI-driven stock market analysis and prediction</p>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4">
        {!isApiAvailable ? (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6" role="alert">
            <p className="font-bold">API Unavailable</p>
            <p>The backend API is currently unavailable. Please check your connection or try again later.</p>
          </div>
        ) : null}

        <div className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Enter Stock Symbol</h2>
          <SymbolForm onRun={handleRun} isLoading={isLoading} />
        </div>

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-8" role="alert">
            <p className="font-bold">Error</p>
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        )}

        {data && !isLoading && (
          <div className="space-y-8">
            <PriceChart 
              timestamps={data.timestamps} 
              prices={data.prices} 
              ma5={data.ma_5} 
              rsi14={data.rsi_14} 
            />
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <SentimentGauge sentiment={data.sentiment} />
              <PredictionDisplay prediction={data.prediction} />
            </div>
          </div>
        )}
      </main>

      <footer className="bg-gray-800 text-white p-6 mt-12">
        <div className="container mx-auto">
          <p className="text-center">© 2025 Project01 Quant Analyst</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
