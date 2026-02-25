import { useState } from 'react'
import './LegalSearch.css'

function LegalSearch() {
    const [searchQuery, setSearchQuery] = useState('')
    const [results, setResults] = useState([])
    const [isSearching, setIsSearching] = useState(false)
    const [selectedDoc, setSelectedDoc] = useState(null)

    const handleSearch = async (e) => {
        e.preventDefault()
        if (!searchQuery.trim() || isSearching) return

        setIsSearching(true)
        setResults([])

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: searchQuery })
            })

            const data = await res.json()
            setResults(data.results || [])
        } catch (err) {
            console.error('Search error:', err)
        } finally {
            setIsSearching(false)
        }
    }

    // Mock featured laws for demo
    const featuredLaws = [
        { id: 1, name: 'Bộ luật Dân sự 2015', icon: '📜', articles: 689 },
        { id: 2, name: 'Bộ luật Lao động 2019', icon: '👷', articles: 220 },
        { id: 3, name: 'Luật Doanh nghiệp 2020', icon: '🏢', articles: 218 },
        { id: 4, name: 'Bộ luật Hình sự 2015', icon: '⚖️', articles: 426 },
    ]

    return (
        <div className="search-container animate-fadeIn">
            {/* Header */}
            <div className="search-header">
                <h1 className="gradient-text">Tra Cứu Văn Bản Pháp Luật</h1>
                <p className="text-muted">Tìm kiếm văn bản quy phạm pháp luật Việt Nam</p>
            </div>

            {/* Search Box */}
            <form className="search-box" onSubmit={handleSearch}>
                <div className="search-input-wrapper glass">
                    <span className="search-icon">🔍</span>
                    <input
                        type="text"
                        className="search-input"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Tìm kiếm theo từ khóa, số hiệu văn bản..."
                        disabled={isSearching}
                    />
                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={!searchQuery.trim() || isSearching}
                    >
                        {isSearching ? 'Đang tìm...' : 'Tìm kiếm'}
                    </button>
                </div>
            </form>

            {/* Results or Featured */}
            {results.length > 0 ? (
                <div className="search-results">
                    <h3>{results.length} kết quả</h3>
                    <div className="results-list">
                        {results.map((result, i) => (
                            <div
                                key={i}
                                className="result-card card-hover"
                                onClick={() => setSelectedDoc(result)}
                            >
                                <div className="result-header">
                                    <span className="result-type">📄</span>
                                    <div className="result-info">
                                        <h4>{result.title || result.law_id || 'Văn bản'}</h4>
                                        <span className="result-meta">
                                            {result.article && `Điều ${result.article}`}
                                        </span>
                                    </div>
                                    <span className="result-score">
                                        {Math.round((result.score || 0) * 100)}%
                                    </span>
                                </div>
                                <p className="result-excerpt">
                                    {result.text?.substring(0, 200)}...
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="featured-section">
                    <h3>Bộ luật phổ biến</h3>
                    <div className="featured-grid">
                        {featuredLaws.map((law) => (
                            <div key={law.id} className="featured-card card-hover">
                                <span className="featured-icon">{law.icon}</span>
                                <h4>{law.name}</h4>
                                <span className="featured-articles">{law.articles} điều</span>
                            </div>
                        ))}
                    </div>

                    {/* Stats */}
                    <div className="search-stats">
                        <div className="stat-card glass">
                            <span className="stat-value">50,000+</span>
                            <span className="stat-label">Văn bản pháp luật</span>
                        </div>
                        <div className="stat-card glass">
                            <span className="stat-value">2,000+</span>
                            <span className="stat-label">Luật và Bộ luật</span>
                        </div>
                        <div className="stat-card glass">
                            <span className="stat-value">AI-Powered</span>
                            <span className="stat-label">Tìm kiếm thông minh</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Document Preview Modal */}
            {selectedDoc && (
                <div className="modal-overlay" onClick={() => setSelectedDoc(null)}>
                    <div className="modal-content glass" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>{selectedDoc.title || selectedDoc.law_id}</h3>
                            <button
                                className="modal-close"
                                onClick={() => setSelectedDoc(null)}
                            >
                                ✕
                            </button>
                        </div>
                        <div className="modal-body">
                            <pre>{selectedDoc.text}</pre>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default LegalSearch
