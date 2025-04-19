import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from "react-router-dom";
import SymbolForm from "./components/SymbolForm";
import PriceChart from "./components/PriceChart";
import SentimentGauge from "./components/SentimentGauge";
import PredictionDisplay from "./components/PredictionDisplay";
import LoginForm from "./components/auth/LoginForm";
import UserProfile from "./components/auth/UserProfile";
import Dashboard from "./components/dashboard/Dashboard";
import Portfolio from "./components/portfolio/Portfolio";
import { predict, advancedPredict, PredictionResponse, AdvancedPredictionResponse, checkHealth } from "./api";
import { AuthProvider, useAuth } from "./context/AuthContext";

// Protected route component
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  return <>{children}</>;
};

// Main content component
const MainContent = () => {
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [advancedData, setAdvancedData] = useState<AdvancedPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isApiAvailable, setIsApiAvailable] = useState(true);
  const [useAdvancedModel, setUseAdvancedModel] = useState(false);

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
      if (useAdvancedModel) {
        const result = await advancedPredict(symbol, days);
        setAdvancedData(result);
        setData(null);
      } else {
        const result = await predict(symbol, days);
        setData(result);
        setAdvancedData(null);
      }
    } catch (err: any) {
      console.error("Error fetching prediction:", err);

      // Extract error message from the response if available
      let errorMessage = "Failed to fetch prediction. Please try again.";

      if (err.response && err.response.data && err.response.data.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
      setData(null);
      setAdvancedData(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container mx-auto py-8 px-4">
      {!isApiAvailable ? (
        <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6" role="alert">
          <p className="font-bold">API Unavailable</p>
          <p>The backend API is currently unavailable. Please check your connection or try again later.</p>
        </div>
      ) : null}

      <div className="bg-white p-6 rounded-lg shadow-md mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Enter Stock Symbol</h2>
          <div className="flex items-center">
            <label className="inline-flex items-center mr-2">
              <input
                type="checkbox"
                checked={useAdvancedModel}
                onChange={() => setUseAdvancedModel(!useAdvancedModel)}
                className="form-checkbox h-5 w-5 text-blue-600"
              />
              <span className="ml-2 text-sm text-gray-700">Use Advanced Model</span>
            </label>
          </div>
        </div>
        <SymbolForm onRun={handleRun} isLoading={isLoading} />
      </div>

      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-8" role="alert">
          <p className="font-bold">Error</p>
          <p>{error}</p>
          <div className="mt-2">
            <p className="text-sm">Please try:</p>
            <ul className="list-disc list-inside text-sm ml-2">
              <li>Using a different stock symbol</li>
              <li>Reducing the number of days</li>
              <li>Checking your internet connection</li>
              <li>Trying again later</li>
            </ul>
          </div>
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
            <PredictionDisplay
              prediction={data.prediction}
              confidenceInterval={data.confidence_interval}
              featureImportance={data.feature_importance}
            />
          </div>
        </div>
      )}

      {advancedData && !isLoading && (
        <div className="space-y-8">
          <PriceChart
            timestamps={advancedData.historical_data.timestamps}
            prices={advancedData.historical_data.prices}
            ma5={advancedData.historical_data.ma_5}
            rsi14={advancedData.historical_data.rsi_14}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <SentimentGauge sentiment={advancedData.sentiment} />
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold mb-4">Advanced Prediction</h3>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-600">Latest Price:</span>
                  <span className="font-medium">${advancedData.latest_price.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Predicted Return:</span>
                  <span className={`font-medium ${advancedData.predicted_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {advancedData.predicted_return >= 0 ? '+' : ''}{(advancedData.predicted_return * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Predicted Price:</span>
                  <span className="font-medium">${advancedData.predicted_price.toFixed(2)}</span>
                </div>
                {advancedData.uncertainty !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Uncertainty:</span>
                    <span className="font-medium">{(advancedData.uncertainty * 100).toFixed(2)}%</span>
                  </div>
                )}
                {advancedData.confidence_interval && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Confidence Interval:</span>
                    <span className="font-medium">
                      ${advancedData.confidence_interval.lower.toFixed(2)} - ${advancedData.confidence_interval.upper.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
};

// Login page component
const LoginPage = () => {
  const { login, isAuthenticated } = useAuth();
  const [error, setError] = useState<string | null>(null);

  if (isAuthenticated) {
    return <Navigate to="/dashboard" />;
  }

  return (
    <div className="container mx-auto py-16 px-4">
      <div className="max-w-md mx-auto">
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">Sign In to Project01</h2>

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
            <p className="font-bold">Error</p>
            <p>{error}</p>
          </div>
        )}

        <LoginForm
          onSuccess={(token) => login(token)}
          onError={(err) => setError(err)}
        />
      </div>
    </div>
  );
};

// App component
function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <AppHeader />

          <Routes>
            <Route path="/" element={<MainContent />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <div className="container mx-auto py-8 px-4">
                  <Dashboard />
                </div>
              </ProtectedRoute>
            } />
            <Route path="/portfolio" element={
              <ProtectedRoute>
                <div className="container mx-auto py-8 px-4">
                  <Portfolio />
                </div>
              </ProtectedRoute>
            } />
          </Routes>

          <AppFooter />
        </div>
      </Router>
    </AuthProvider>
  );
}

// Header component
const AppHeader = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <header className="bg-blue-600 text-white shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center">
            <Link to="/" className="text-2xl font-bold">Project01 Quant Analyst</Link>
          </div>

          <div className="hidden md:flex items-center space-x-6">
            <nav className="flex space-x-6">
              <Link to="/" className="text-white hover:text-blue-200 transition-colors">Home</Link>
              {isAuthenticated && (
                <>
                  <Link to="/dashboard" className="text-white hover:text-blue-200 transition-colors">Dashboard</Link>
                  <Link to="/portfolio" className="text-white hover:text-blue-200 transition-colors">Portfolio</Link>
                </>
              )}
            </nav>

            {isAuthenticated ? (
              <UserProfile token="" onLogout={logout} />
            ) : (
              <Link
                to="/login"
                className="px-4 py-2 bg-white text-blue-600 rounded-md hover:bg-blue-50 transition-colors"
              >
                Sign In
              </Link>
            )}
          </div>

          <div className="md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="text-white focus:outline-none"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {isMobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-blue-500">
            <nav className="flex flex-col space-y-4">
              <Link
                to="/"
                className="text-white hover:text-blue-200 transition-colors"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Home
              </Link>
              {isAuthenticated && (
                <>
                  <Link
                    to="/dashboard"
                    className="text-white hover:text-blue-200 transition-colors"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Dashboard
                  </Link>
                  <Link
                    to="/portfolio"
                    className="text-white hover:text-blue-200 transition-colors"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Portfolio
                  </Link>
                </>
              )}

              {isAuthenticated ? (
                <button
                  onClick={() => {
                    logout();
                    setIsMobileMenuOpen(false);
                  }}
                  className="text-white hover:text-blue-200 transition-colors text-left"
                >
                  Sign Out
                </button>
              ) : (
                <Link
                  to="/login"
                  className="text-white hover:text-blue-200 transition-colors"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Sign In
                </Link>
              )}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

// Footer component
const AppFooter = () => {
  return (
    <footer className="bg-gray-800 text-white p-6 mt-12">
      <div className="container mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <p>© 2025 Project01 Quant Analyst</p>
          <div className="flex space-x-4 mt-4 md:mt-0">
            <a href="#" className="text-gray-400 hover:text-white transition-colors">Terms</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">Privacy</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">Contact</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default App;
