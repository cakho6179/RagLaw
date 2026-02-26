import { useState, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'
import './styles/index.css'
import './styles/App.css'

function App() {
    const [theme, setTheme] = useState('dark')

    useEffect(() => {
        const savedTheme = localStorage.getItem('raglaw-theme') || 'dark'
        setTheme(savedTheme)
        document.documentElement.setAttribute('data-theme', savedTheme)
    }, [])

    const toggleTheme = () => {
        const newTheme = theme === 'dark' ? 'light' : 'dark'
        setTheme(newTheme)
        localStorage.setItem('raglaw-theme', newTheme)
        document.documentElement.setAttribute('data-theme', newTheme)
    }

    return (
        <div className="app">
            <main className="app-main">
                <ChatInterface theme={theme} onThemeToggle={toggleTheme} />
            </main>

            <div className="bg-decoration">
                <div className="bg-blob blob-1"></div>
                <div className="bg-blob blob-2"></div>
            </div>
        </div>
    )
}

export default App
