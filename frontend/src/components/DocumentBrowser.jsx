/**
 * Document Browser Component
 * ============================
 * 
 * Browse and search Vietnamese legal documents.
 * Supports pagination, type filtering, and document detail view.
 */
import { useState, useEffect, useCallback } from 'react';
import { documentsApi } from '../api/client';
import './DocumentBrowser.css';

function DocumentBrowser({ onBack }) {
    const [documents, setDocuments] = useState([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(0);
    const [searchQuery, setSearchQuery] = useState('');
    const [docTypes, setDocTypes] = useState([]);
    const [selectedType, setSelectedType] = useState('');
    const [selectedDoc, setSelectedDoc] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Load document types on mount
    useEffect(() => {
        documentsApi.getTypes()
            .then(types => setDocTypes(types))
            .catch(() => { });
    }, []);

    // Load documents
    const loadDocuments = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            let result;
            if (searchQuery.trim()) {
                result = await documentsApi.search(searchQuery, page, 20, selectedType || null);
            } else {
                result = await documentsApi.list(page, 20, selectedType || null);
            }
            setDocuments(result.documents || []);
            setTotal(result.total || 0);
            setTotalPages(result.total_pages || 0);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [searchQuery, page, selectedType]);

    useEffect(() => { loadDocuments(); }, [loadDocuments]);

    // Load document detail
    const openDocument = async (docId) => {
        setLoading(true);
        try {
            const doc = await documentsApi.get(docId);
            setSelectedDoc(doc);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = (e) => {
        e.preventDefault();
        setPage(1);
        loadDocuments();
    };

    // Detail view
    if (selectedDoc) {
        return (
            <div className="doc-browser">
                <div className="doc-detail">
                    <div className="doc-detail-header">
                        <button className="doc-back-btn" onClick={() => setSelectedDoc(null)}>
                            ← Quay lại
                        </button>
                        <h2>{selectedDoc.title || selectedDoc.id}</h2>
                        <div className="doc-meta-row">
                            {selectedDoc.document_type && (
                                <span className="doc-badge">{selectedDoc.document_type}</span>
                            )}
                            {selectedDoc.official_number && (
                                <span className="doc-meta-item">📋 {selectedDoc.official_number}</span>
                            )}
                            {selectedDoc.issued_date && (
                                <span className="doc-meta-item">📅 {selectedDoc.issued_date}</span>
                            )}
                            {selectedDoc.signer && (
                                <span className="doc-meta-item">✍️ {selectedDoc.signer}</span>
                            )}
                        </div>
                    </div>

                    <div className="doc-articles">
                        {selectedDoc.articles && selectedDoc.articles.length > 0 ? (
                            selectedDoc.articles.map((article, idx) => (
                                <div key={idx} className="doc-article">
                                    <h3 className="article-title">
                                        {article.article_id && <span className="article-id">{article.article_id}</span>}
                                        {article.title}
                                    </h3>
                                    <div className="article-text">{article.text}</div>
                                </div>
                            ))
                        ) : (
                            <div className="doc-no-articles">
                                <p>📄 Không có điều khoản chi tiết cho văn bản này.</p>
                                {selectedDoc.url && (
                                    <a href={selectedDoc.url} target="_blank" rel="noopener noreferrer"
                                        className="doc-external-link">
                                        🔗 Xem trên nguồn gốc
                                    </a>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    // List view
    return (
        <div className="doc-browser">
            <div className="doc-browser-header">
                <div className="doc-title-row">
                    {onBack && (
                        <button className="doc-back-btn" onClick={onBack}>← Chat</button>
                    )}
                    <h2>📚 Tra cứu văn bản pháp luật</h2>
                    <span className="doc-total">{total.toLocaleString()} văn bản</span>
                </div>

                <form className="doc-search-form" onSubmit={handleSearch}>
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Tìm kiếm theo tên hoặc số hiệu..."
                        className="doc-search-input"
                    />
                    <select
                        value={selectedType}
                        onChange={(e) => { setSelectedType(e.target.value); setPage(1); }}
                        className="doc-type-filter"
                    >
                        <option value="">Tất cả loại</option>
                        {docTypes.map(type => (
                            <option key={type} value={type}>{type}</option>
                        ))}
                    </select>
                    <button type="submit" className="doc-search-btn">🔍 Tìm</button>
                </form>
            </div>

            {error && <div className="doc-error">⚠️ {error}</div>}

            <div className="doc-list">
                {loading && documents.length === 0 ? (
                    <div className="doc-loading">⏳ Đang tải...</div>
                ) : documents.length === 0 ? (
                    <div className="doc-empty">Không tìm thấy văn bản nào.</div>
                ) : (
                    documents.map((doc) => (
                        <div
                            key={doc.id}
                            className="doc-card"
                            onClick={() => openDocument(doc.id)}
                        >
                            <div className="doc-card-header">
                                <h3 className="doc-card-title">{doc.title || doc.id}</h3>
                                {doc.document_type && (
                                    <span className="doc-badge">{doc.document_type}</span>
                                )}
                            </div>
                            <div className="doc-card-meta">
                                {doc.official_number && <span>📋 {doc.official_number}</span>}
                                {doc.issued_date && <span>📅 {doc.issued_date}</span>}
                                {doc.article_count > 0 && <span>📄 {doc.article_count} điều</span>}
                            </div>
                            {doc.snippet && (
                                <p className="doc-card-snippet">{doc.snippet}</p>
                            )}
                        </div>
                    ))
                )}
            </div>

            {totalPages > 1 && (
                <div className="doc-pagination">
                    <button
                        disabled={page <= 1}
                        onClick={() => setPage(p => Math.max(1, p - 1))}
                    >
                        ← Trước
                    </button>
                    <span>Trang {page} / {totalPages}</span>
                    <button
                        disabled={page >= totalPages}
                        onClick={() => setPage(p => p + 1)}
                    >
                        Sau →
                    </button>
                </div>
            )}
        </div>
    );
}

export default DocumentBrowser;
