import axios, { AxiosRequestConfig } from "axios";

// Create an axios instance with the base URL
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api",
});

// Add request interceptor to include auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token && config.headers) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
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
  confidence_interval?: {
    lower: number;
    upper: number;
  };
  feature_importance?: Record<string, number>;
}

export interface AdvancedPredictionResponse {
  symbol: string;
  latest_price: number;
  predicted_return: number;
  predicted_price: number;
  confidence_interval?: {
    lower: number;
    upper: number;
  };
  uncertainty?: number;
  sentiment: {
    positive: number;
    neutral: number;
    negative: number;
  };
  historical_data: {
    timestamps: string[];
    prices: number[];
    ma_5: number[];
    rsi_14: number[];
    [key: string]: any;
  };
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_at: number;
}

export interface User {
  username: string;
  email: string;
  full_name: string;
  scopes: string[];
  is_active: boolean;
}

export interface ModelInfo {
  name: string;
  version: string;
  type: string;
  features: string[];
  metrics: Record<string, number>;
  last_updated: string;
}

// Authentication functions
export async function login(username: string, password: string): Promise<LoginResponse> {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);

  const response = await api.post<LoginResponse>('/token', formData);
  return response.data;
}

export async function getUserProfile(token?: string): Promise<User> {
  const config: AxiosRequestConfig = {};
  if (token) {
    config.headers = {
      Authorization: `Bearer ${token}`
    };
  }

  const response = await api.get<User>('/users/me', config);
  return response.data;
}

// Prediction functions
export async function predict(symbol: string, days: number): Promise<PredictionResponse> {
  const response = await api.post<PredictionResponse>("/predict", { symbol, days });
  return response.data;
}

export async function advancedPredict(symbol: string, days: number): Promise<AdvancedPredictionResponse> {
  const response = await api.post<AdvancedPredictionResponse>("/predict/advanced", { symbol, days });
  return response.data;
}

// Model information
export async function getModelInfo(): Promise<ModelInfo> {
  const response = await api.get<ModelInfo>("/model/info");
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

// Cache management
export async function getCacheStats(): Promise<any> {
  const response = await api.get("/cache/stats");
  return response.data;
}

export async function clearCache(): Promise<any> {
  const response = await api.post("/cache/clear");
  return response.data;
}
