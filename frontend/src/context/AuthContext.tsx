import React, { createContext, useState, useEffect, useContext } from 'react';
import { getUserProfile } from '../api';

interface AuthContextType {
  isAuthenticated: boolean;
  token: string | null;
  user: any | null;
  login: (token: string) => void;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  token: null,
  user: null,
  login: () => {},
  logout: () => {},
  isLoading: true,
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(localStorage.getItem('auth_token'));
  const [user, setUser] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      if (!token) {
        setIsLoading(false);
        return;
      }

      // Check if token is expired
      const expiry = localStorage.getItem('token_expiry');
      if (expiry && parseInt(expiry) < Date.now() / 1000) {
        // Token is expired
        logout();
        setIsLoading(false);
        return;
      }

      try {
        // Fetch user profile to validate token
        const userData = await getUserProfile(token);
        setUser(userData);
      } catch (error) {
        console.error('Auth validation failed:', error);
        logout();
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [token]);

  const login = (newToken: string) => {
    localStorage.setItem('auth_token', newToken);
    setToken(newToken);
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('token_expiry');
    setToken(null);
    setUser(null);
  };

  const value = {
    isAuthenticated: !!token,
    token,
    user,
    login,
    logout,
    isLoading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;
