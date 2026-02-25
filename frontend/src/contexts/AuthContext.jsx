/**
 * Auth Context
 * =============
 * 
 * React context for authentication state management.
 * Provides login/register/logout functions and current user state.
 */
import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authApi, clearAuth } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(() => {
        try {
            const saved = localStorage.getItem('raglaw_user');
            return saved ? JSON.parse(saved) : null;
        } catch {
            return null;
        }
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Listen for auth expiry events from API client
    useEffect(() => {
        const handleExpired = () => {
            setUser(null);
            setError('Session expired. Please login again.');
        };
        window.addEventListener('auth:expired', handleExpired);
        return () => window.removeEventListener('auth:expired', handleExpired);
    }, []);

    const login = useCallback(async (username, password) => {
        setLoading(true);
        setError(null);
        try {
            const data = await authApi.login(username, password);
            setUser(data.user);
            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    const register = useCallback(async (username, password, email = null) => {
        setLoading(true);
        setError(null);
        try {
            const data = await authApi.register(username, password, email);
            setUser(data.user);
            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    const logout = useCallback(() => {
        authApi.logout();
        setUser(null);
        setError(null);
    }, []);

    const value = {
        user,
        loading,
        error,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        clearError: () => setError(null),
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}

export default AuthContext;
