import axios from "axios";

// Create an axios instance with the base URL
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api",
});

// Define the prediction function
export interface PredictionRequest {
  symbol: string;
  days: number;
}

export interface PredictionResponse {
  timestamps: string[];
  prices: number[];
  ma_5: number[];
  rsi_14: number[];
  sentiment: {
    positive: number;
    neutral: number;
    negative: number;
  };
  prediction: number;
}

export async function predict(symbol: string, days: number): Promise<PredictionResponse> {
  const response = await api.post<PredictionResponse>("/predict", { symbol, days });
  return response.data;
}

// Health check function
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get("/health");
    return response.data.status === "ok";
  } catch (error) {
    console.error("Health check failed:", error);
    return false;
  }
}
