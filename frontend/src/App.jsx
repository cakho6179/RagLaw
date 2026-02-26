import { useState, useEffect } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import AuthPage from './components/AuthPage'
import ChatInterface from './components/ChatInterface'
import DocumentBrowser from './components/DocumentBrowser'
import './styles/index.css'
import './styles/App.css'

function AppContent() {
    const { isAuthenticated } = useAuth()
    const [theme, setTheme] = useState('dark')
    const [currentPage, setCurrentPage] = useState('chat') // 'chat' | 'documents'
    const [showAuth, setShowAuth] = useState(false)

    useEffect(() => {
        const savedTheme = localStorage.getItem('raglaw-theme') || 'dark'
        setTheme(savedTheme)
        document.documentElement.setAttribute('data-theme', savedTheme)
    }, [])

    // Listen for auth skip event
    useEffect(() => {
        const handleSkip = () => setShowAuth(false)
        window.addEventListener('auth:skip', handleSkip)
        return () => window.removeEventListener('auth:skip', handleSkip)
    }, [])

    // Hide auth page when authenticated
    useEffect(() => {
        if (isAuthenticated) setShowAuth(false)
    }, [isAuthenticated])

    const toggleTheme = () => {
        const newTheme = theme === 'dark' ? 'light' : 'dark'
        setTheme(newTheme)
        localStorage.setItem('raglaw-theme', newTheme)
        document.documentElement.setAttribute('data-theme', newTheme)
    }

    // Show auth page if requested
    if (showAuth && !isAuthenticated) {
        return <AuthPage />
    }

    return (
        <div className="app">
            {/* Page Content */}
            <main className="app-main">
                {currentPage === 'chat' ? (
                    <ChatInterface theme={theme} onThemeToggle={toggleTheme} />
                ) : (
                    <DocumentBrowser onBack={() => setCurrentPage('chat')} />
                )}
            </main>

            {/* Background Decorations */}
            <div className="bg-decoration">
                <div className="bg-blob blob-1"></div>
                <div className="bg-blob blob-2"></div>
            </div>
        </div>
    )
}

function UserMenu() {
    const { user, logout } = useAuth()
    const [open, setOpen] = useState(false)

    return (
        <div className="user-menu">
            <button className="user-menu-trigger" onClick={() => setOpen(!open)}>
                👤 {user?.username || 'User'}
            </button>
            {open && (
                <div className="user-menu-dropdown">
                    <div className="user-menu-info">
                        <strong>{user?.username}</strong>
                        {user?.email && <small>{user.email}</small>}
                    </div>
                    <button className="user-menu-item" onClick={() => { logout(); setOpen(false); }}>
                        🚪 Đăng xuất
                    </button>
                </div>
            )}
        </div>
    )
}

function App() {
    return (
        <AuthProvider>
            <AppContent />
        </AuthProvider>
    )
}

export default App
