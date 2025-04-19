import { useState, useEffect } from 'react';
import { getUserProfile } from '../../api';

interface UserProfileProps {
  token: string;
  onLogout: () => void;
}

interface UserData {
  username: string;
  email: string;
  full_name: string;
  scopes: string[];
  is_active: boolean;
}

const UserProfile = ({ token, onLogout }: UserProfileProps) => {
  const [userData, setUserData] = useState<UserData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const data = await getUserProfile(token);
        setUserData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load user profile');
        // If unauthorized, log out
        if (err instanceof Error && err.message.includes('401')) {
          handleLogout();
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserData();
  }, [token]);

  const handleLogout = () => {
    // Clear local storage
    localStorage.removeItem('auth_token');
    localStorage.removeItem('token_expiry');
    
    // Call logout callback
    onLogout();
  };

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  if (isLoading) {
    return (
      <div className="flex items-center space-x-2">
        <div className="h-8 w-8 rounded-full bg-gray-200 animate-pulse"></div>
        <div className="h-4 w-24 bg-gray-200 animate-pulse rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center space-x-2">
        <button 
          onClick={handleLogout}
          className="text-sm text-red-600 hover:text-red-800"
        >
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={toggleDropdown}
        className="flex items-center space-x-2 focus:outline-none"
        aria-expanded={isDropdownOpen}
        aria-haspopup="true"
      >
        <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-medium">
          {userData?.username.charAt(0).toUpperCase()}
        </div>
        <span className="text-sm font-medium text-gray-700 hidden md:block">
          {userData?.username}
        </span>
        <svg 
          className={`h-5 w-5 text-gray-400 transition-transform ${isDropdownOpen ? 'transform rotate-180' : ''}`} 
          xmlns="http://www.w3.org/2000/svg" 
          viewBox="0 0 20 20" 
          fill="currentColor"
        >
          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
      </button>

      {isDropdownOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10 ring-1 ring-black ring-opacity-5">
          <div className="px-4 py-2 border-b border-gray-100">
            <p className="text-sm font-medium text-gray-900">{userData?.full_name || userData?.username}</p>
            <p className="text-xs text-gray-500 truncate">{userData?.email}</p>
          </div>
          
          {userData?.scopes && userData.scopes.length > 0 && (
            <div className="px-4 py-2 border-b border-gray-100">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Permissions</p>
              <div className="mt-1 flex flex-wrap gap-1">
                {userData.scopes.map((scope) => (
                  <span key={scope} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                    {scope}
                  </span>
                ))}
              </div>
            </div>
          )}
          
          <a 
            href="#profile" 
            className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            onClick={(e) => {
              e.preventDefault();
              setIsDropdownOpen(false);
              // Navigate to profile page or open profile modal
            }}
          >
            Your Profile
          </a>
          
          <a 
            href="#settings" 
            className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            onClick={(e) => {
              e.preventDefault();
              setIsDropdownOpen(false);
              // Navigate to settings page or open settings modal
            }}
          >
            Settings
          </a>
          
          <button 
            onClick={handleLogout}
            className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-gray-100"
          >
            Sign Out
          </button>
        </div>
      )}
    </div>
  );
};

export default UserProfile;
