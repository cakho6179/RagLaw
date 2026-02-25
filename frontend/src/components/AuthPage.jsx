/**
 * Auth Page Component
 * ====================
 * 
 * Login/Register page with toggle between modes.
 */
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import './AuthPage.css';

function AuthPage() {
    const { login, register, loading, error, clearError } = useAuth();
    const [mode, setMode] = useState('login'); // 'login' | 'register'
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [email, setEmail] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (mode === 'login') {
                await login(username, password);
            } else {
                await register(username, password, email || null);
            }
        } catch {
            // Error is handled by context
        }
    };

    const toggleMode = () => {
        setMode(mode === 'login' ? 'register' : 'login');
        clearError();
    };

    return (
        <div className="auth-page">
            <div className="auth-card">
                <div className="auth-header">
                    <div className="auth-logo">⚖️</div>
                    <h1>RagLaw</h1>
                    <p>Vietnamese Legal AI Assistant</p>
                </div>

                <div className="auth-tabs">
                    <button
                        className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
                        onClick={() => { setMode('login'); clearError(); }}
                    >
                        Đăng nhập
                    </button>
                    <button
                        className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
                        onClick={() => { setMode('register'); clearError(); }}
                    >
                        Đăng ký
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="auth-form">
                    <div className="form-group">
                        <label htmlFor="username">Tên đăng nhập</label>
                        <input
                            id="username"
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="Nhập tên đăng nhập"
                            required
                            autoFocus
                        />
                    </div>

                    {mode === 'register' && (
                        <div className="form-group">
                            <label htmlFor="email">Email (tùy chọn)</label>
                            <input
                                id="email"
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="email@example.com"
                            />
                        </div>
                    )}

                    <div className="form-group">
                        <label htmlFor="password">Mật khẩu</label>
                        <input
                            id="password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Nhập mật khẩu"
                            required
                            minLength={6}
                        />
                    </div>

                    {error && (
                        <div className="auth-error">
                            <span>⚠️</span> {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        className="auth-submit"
                        disabled={loading || !username || !password}
                    >
                        {loading ? '⏳ Đang xử lý...' : mode === 'login' ? '🔓 Đăng nhập' : '📝 Đăng ký'}
                    </button>
                </form>

                <div className="auth-footer">
                    <button className="auth-link" onClick={toggleMode}>
                        {mode === 'login'
                            ? 'Chưa có tài khoản? Đăng ký ngay'
                            : 'Đã có tài khoản? Đăng nhập'}
                    </button>

                    <button className="auth-skip" onClick={() => window.dispatchEvent(new Event('auth:skip'))}>
                        Tiếp tục không đăng nhập →
                    </button>
                </div>
            </div>
        </div>
    );
}

export default AuthPage;
