import { useState, useEffect } from 'react'
import './Sidebar.css'

function Sidebar({ activeView, onViewChange, theme, onThemeToggle, isConnected, onSessionSelect, currentSessionId }) {
    const [sessions, setSessions] = useState([])
    const [showSessions, setShowSessions] = useState(false)

    const navItems = [
        { id: 'chat', icon: '💬', label: 'Hỏi đáp', desc: 'Chat với AI pháp luật' },
        { id: 'search', icon: '🔍', label: 'Tra cứu', desc: 'Tìm kiếm văn bản' },
        { id: 'evaluation', icon: '📊', label: 'Đánh giá', desc: 'Benchmark hệ thống' }
    ]

    useEffect(() => {
        if (activeView === 'chat') {
            loadSessions()
        }
    }, [activeView])

    const loadSessions = async () => {
        try {
            const res = await fetch('/api/chat/sessions')
            if (res.ok) {
                const data = await res.json()
                setSessions(data.sessions || [])
            }
        } catch (err) {
            console.error('Failed to load sessions:', err)
        }
    }

    const createNewSession = () => {
        onSessionSelect(null) // null = new session
        setShowSessions(false)
    }

    const selectSession = (sessionId) => {
        onSessionSelect(sessionId)
        setShowSessions(false)
    }

    const deleteSession = async (e, sessionId) => {
        e.stopPropagation()
        try {
            await fetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' })
            setSessions(prev => prev.filter(s => s.session_id !== sessionId))
            if (currentSessionId === sessionId) {
                onSessionSelect(null)
            }
        } catch (err) {
            console.error('Failed to delete session:', err)
        }
    }

    const formatTime = (timestamp) => {
        if (!timestamp) return ''
        return new Date(timestamp).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit' })
    }

    return (
        <nav className="sidebar">
            {/* Brand Header */}
            <div className="sidebar-brand">
                <div className="brand-logo">⚖️</div>
                <div className="brand-text">
                    <span className="brand-name">KAISER</span>
                    <span className="brand-sub">Legal AI</span>
                </div>
            </div>

            {/* Navigation */}
            <div className="sidebar-nav">
                {navItems.map(item => (
                    <button
                        key={item.id}
                        className={`nav-item ${activeView === item.id ? 'active' : ''}`}
                        onClick={() => {
                            onViewChange(item.id)
                            if (item.id === 'chat') setShowSessions(!showSessions)
                        }}
                    >
                        <span className="nav-icon">{item.icon}</span>
                        <div className="nav-text">
                            <span className="nav-label">{item.label}</span>
                            <span className="nav-desc">{item.desc}</span>
                        </div>
                        {item.id === 'chat' && sessions.length > 0 && (
                            <span className="nav-badge">{sessions.length}</span>
                        )}
                    </button>
                ))}
            </div>

            {/* Sessions Panel (appears when chat is active) */}
            {activeView === 'chat' && showSessions && (
                <div className="sessions-panel">
                    <div className="sessions-header">
                        <span>📂 Lịch sử chat</span>
                        <button className="btn-close" onClick={() => setShowSessions(false)}>×</button>
                    </div>

                    <button className="new-chat-btn" onClick={createNewSession}>
                        ✨ Chat mới
                    </button>

                    <div className="sessions-list">
                        {sessions.map(session => (
                            <div
                                key={session.session_id}
                                className={`session-item ${currentSessionId === session.session_id ? 'active' : ''}`}
                                onClick={() => selectSession(session.session_id)}
                            >
                                <span className="session-icon">💬</span>
                                <div className="session-info">
                                    <span className="session-title">{session.title || 'Cuộc trò chuyện'}</span>
                                    <span className="session-date">{formatTime(session.updated_at)}</span>
                                </div>
                                <button
                                    className="session-delete"
                                    onClick={(e) => deleteSession(e, session.session_id)}
                                >🗑️</button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Footer */}
            <div className="sidebar-footer">
                <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                    <span className="status-dot"></span>
                    <span className="status-text">{isConnected ? 'Đã kết nối' : 'Mất kết nối'}</span>
                </div>
                <button className="theme-toggle" onClick={onThemeToggle}>
                    {theme === 'dark' ? '🌞' : '🌙'}
                </button>
            </div>
        </nav>
    )
}

export default Sidebar
