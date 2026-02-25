import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './ChatInterface.css'

function ChatInterface({ theme, onThemeToggle }) {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [currentSessionId, setCurrentSessionId] = useState(null)
    const [selectedModel, setSelectedModel] = useState('gemini') // Default to 'gemini'
    const [selectedMode, setSelectedMode] = useState('flash')  // flash, pro, or gemma
    const [isConnected, setIsConnected] = useState(false)
    const [showSessions, setShowSessions] = useState(false)
    const [sessions, setSessions] = useState([])
    const [expandedSources, setExpandedSources] = useState({})
    const endOfMessagesRef = useRef(null)

    const suggestions = [
        "Quyền sở hữu tài sản theo Bộ luật Dân sự?",
        "Thủ tục đăng ký kết hôn?",
        "Hợp đồng lao động cần những gì?"
    ]

    useEffect(() => {
        checkConnection()
        loadSessions()
    }, [])

    useEffect(() => {
        endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const checkConnection = async () => {
        try {
            const res = await fetch('/api/health')
            setIsConnected(res.ok)
        } catch {
            setIsConnected(false)
        }
    }

    const loadSessions = async () => {
        try {
            const res = await fetch('/api/sessions')
            if (res.ok) {
                const data = await res.json()
                setSessions(data.sessions || [])
            }
        } catch (err) {
            console.error('Load sessions error:', err)
        }
    }

    const startNewChat = () => {
        setCurrentSessionId(null)
        setMessages([])
        setExpandedSources({})
        if (window.innerWidth < 768) setShowSessions(false)
    }

    const loadSession = async (sessionId) => {
        try {
            const res = await fetch(`/api/sessions/${sessionId}`)
            if (res.ok) {
                const data = await res.json()
                setCurrentSessionId(sessionId)
                setMessages(data.messages || [])
                setExpandedSources({})
                if (window.innerWidth < 768) setShowSessions(false)
            }
        } catch (err) {
            console.error('Load session error:', err)
        }
    }

    const toggleSource = (msgIndex, sourceIndex) => {
        const key = `${msgIndex}-${sourceIndex}`
        setExpandedSources(prev => ({ ...prev, [key]: !prev[key] }))
    }

    const deleteSession = async (e, sessionId) => {
        e.stopPropagation()
        try {
            await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' })
            setSessions(prev => prev.filter(s => s.session_id !== sessionId))
            if (currentSessionId === sessionId) {
                startNewChat()
            }
        } catch (err) {
            console.error('Delete session error:', err)
        }
    }

    const sendMessage = async (text = input) => {
        if (!text.trim() || isLoading) return

        const userMessage = { role: 'user', content: text }
        setMessages(prev => [...prev, userMessage])
        setInput('')
        setIsLoading(true)

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    session_id: currentSessionId,
                    model: selectedModel,
                    mode: selectedMode
                })
            })

            if (res.ok) {
                const data = await res.json()

                if (data.session_id && data.session_id !== currentSessionId) {
                    setCurrentSessionId(data.session_id)
                    loadSessions()
                }

                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.reply,
                    sources: data.sources || [],
                    isStreaming: false
                }])
            } else {
                throw new Error('Request failed')
            }
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Xin lỗi, đã xảy ra lỗi. Vui lòng thử lại!',
                isError: true,
                isStreaming: false
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const formatDate = (timestamp) => {
        if (!timestamp) return ''
        const date = new Date(timestamp)
        return date.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
    }

    return (
        <div className="chat-wrapper">
            {/* Session Drawer */}
            <div className={`session-drawer ${showSessions ? 'open' : ''}`}>
                <div className="drawer-header">
                    <h3>📂 Lịch sử chat</h3>
                    <button className="close-btn" onClick={() => setShowSessions(false)}>✕</button>
                </div>

                <button className="new-chat-btn" onClick={startNewChat}>
                    <span>✨</span> Cuộc trò chuyện mới
                </button>

                <div className="session-list">
                    {sessions.length === 0 ? (
                        <div className="empty-sessions">
                            <span>💬</span>
                            <p>Chưa có cuộc trò chuyện nào</p>
                        </div>
                    ) : (
                        sessions.map(session => (
                            <div
                                key={session.session_id}
                                className={`session-card ${currentSessionId === session.session_id ? 'active' : ''}`}
                                onClick={() => loadSession(session.session_id)}
                            >
                                <div className="session-icon">💬</div>
                                <div className="session-info">
                                    <span className="session-title">{session.title || 'Cuộc trò chuyện'}</span>
                                    <span className="session-date">{formatDate(session.updated_at)}</span>
                                </div>
                                <button
                                    className="delete-btn"
                                    onClick={(e) => deleteSession(e, session.session_id)}
                                    title="Xóa"
                                >🗑️</button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Overlay */}
            {showSessions && <div className="drawer-overlay" onClick={() => setShowSessions(false)} />}

            {/* Header */}
            <header className="chat-header">
                <div className="header-left">
                    <button
                        className="menu-btn"
                        onClick={() => setShowSessions(!showSessions)}
                        title="Lịch sử chat"
                    >
                        <span className="menu-icon">☰</span>
                        {sessions.length > 0 && <span className="badge">{sessions.length}</span>}
                    </button>
                    <div className="logo">⚖️</div>
                    <h1 className="brand-title">KAISER Legal AI</h1>
                </div>

                <div className="header-right">
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="model-select"
                    >
                        <option value="gemini">🌟 Gemini</option>
                        <option value="local">🏠 Local AI</option>
                    </select>

                    {selectedModel === 'gemini' && (
                        <div className="mode-toggle">
                            <button
                                className={`mode-btn ${selectedMode === 'flash' ? 'active' : ''}`}
                                onClick={() => setSelectedMode('flash')}
                                title="Tốc độ cao (Gemini Flash)"
                            >
                                ⚡ Flash
                            </button>
                            <button
                                className={`mode-btn ${selectedMode === 'pro' ? 'active' : ''}`}
                                onClick={() => setSelectedMode('pro')}
                                title="Phân tích sâu (Gemini Pro)"
                            >
                                🎯 Pro
                            </button>
                            <button
                                className={`mode-btn ${selectedMode === 'gemma' ? 'active' : ''}`}
                                onClick={() => setSelectedMode('gemma')}
                                title="Mô hình tối ưu hóa pháp luật"
                            >
                                🌟 Gemma
                            </button>
                        </div>
                    )}

                    <button className="icon-btn" onClick={onThemeToggle} title="Đổi theme">
                        {theme === 'dark' ? '🌞' : '🌙'}
                    </button>

                    <div className={`status-dot ${isConnected ? 'connected' : ''}`}
                        title={isConnected ? 'Đã kết nối' : 'Mất kết nối'} />
                </div>
            </header>

            {/* Main Chat */}
            <main className="chat-main">
                {messages.length === 0 ? (
                    <div className="welcome-section">
                        <div className="welcome-icon">⚖️</div>
                        <h2>Xin chào! Tôi là trợ lý pháp luật AI</h2>
                        <p>Hãy hỏi tôi về luật pháp Việt Nam</p>

                        <div className="suggestion-chips">
                            {suggestions.map((s, i) => (
                                <button
                                    key={i}
                                    className="chip"
                                    onClick={() => sendMessage(s)}
                                >
                                    {s}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="messages-list">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`msg ${msg.role} ${msg.isError ? 'error' : ''}`}>
                                <div className="msg-avatar">
                                    {msg.role === 'user' ? '👤' : '⚖️'}
                                </div>
                                <div className="msg-bubble">
                                    {msg.thoughts?.length > 0 && (
                                        <div className="msg-thoughts">
                                            <details>
                                                <summary>
                                                    <span className="brain-icon">🧠</span> Suy nghĩ của AI ({msg.thoughts.length} bước)
                                                </summary>
                                                <div className="thought-logs">
                                                    {msg.thoughts.map((t, i) => (
                                                        <div key={i} className={`thought-item ${t.agent.toLowerCase()}`}>
                                                            <strong className="thought-agent">[{t.agent}]</strong>
                                                            <span className="thought-text">{t.text}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </details>
                                        </div>
                                    )}

                                    <div className={`msg-text markdown-content ${msg.isStreaming ? 'streaming' : ''}`}>
                                        <ReactMarkdown>{msg.content || (msg.isStreaming ? 'Đang tạo câu trả lời...' : '')}</ReactMarkdown>
                                    </div>

                                    {!msg.isStreaming && msg.sources && msg.sources.length > 0 && (
                                        <div className="message-sources">
                                            <h4 className="sources-label">📚 TÀI LIỆU THAM KHẢO</h4>
                                            <div className="sources-grid">
                                                {msg.sources.map((src, sourceIdx) => {
                                                    const isExpanded = expandedSources[`${idx}-${sourceIdx}`];
                                                    const title = src.metadata?.document_title
                                                        || (src.metadata?.source ? src.metadata.source.split(/[\\/]/).pop() : null)
                                                        || (src.content ? src.content.split('\n')[0].substring(0, 60) + '...' : `Tài liệu #${sourceIdx + 1}`);

                                                    return (
                                                        <div key={sourceIdx} className={`source-card ${isExpanded ? 'expanded' : ''}`}>
                                                            <div
                                                                className="source-header"
                                                                onClick={() => toggleSource(idx, sourceIdx)}
                                                                title="Nhấn để xem chi tiết"
                                                            >
                                                                <span className="source-title">
                                                                    <span className="source-icon">📄</span>
                                                                    {title}
                                                                </span>
                                                                <span className="source-toggle-icon">
                                                                    {isExpanded ? '▲' : '▼'}
                                                                </span>
                                                            </div>
                                                            {isExpanded && (
                                                                <div className="source-content">
                                                                    {src.content}
                                                                </div>
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="msg assistant">
                                <div className="msg-avatar">⚖️</div>
                                <div className="typing">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        )}
                        <div ref={endOfMessagesRef} />
                    </div>
                )}
            </main>

            {/* Input */}
            <footer className="chat-input">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Nhập câu hỏi của bạn..."
                    rows={1}
                />
                <button
                    className="send-btn"
                    onClick={() => sendMessage()}
                    disabled={!input.trim() || isLoading}
                >
                    {isLoading ? '⏳' : '📤'}
                </button>
            </footer>
        </div>
    )
}

export default ChatInterface
